import argparse
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_budgeted_llama.py"
SPINE = ROOT / "results" / "expert_vault" / "qwen2_full_vault" / "qwen2-spine.gguf"
FULL_INDEX = ROOT / "results" / "expert_vault" / "qwen2_full_vault" / "experts.pack.idx"
FULL_PACK = ROOT / "results" / "expert_vault" / "qwen2_full_vault" / "experts.pack"
PROMPTS = [
    "Explain cache locality in two sentences.",
    "Write a compact C++ binary search function.",
    "Solve 3x + 5 = 20 and state x.",
]


def env_arg(key, value):
    return ["--env", f"{key}={value}"]


def run_trial(case, trial, tokens, output_dir):
    size = case["size"]
    pack_dir = ROOT / "results" / "ability_packs" / f"qwen2_universal_{size}"
    name = f"{case['name']}_trial_{trial + 1}"
    command = [
        "python", str(RUNNER), "--name", name, "--model", str(SPINE),
        "--prompt", PROMPTS[trial % len(PROMPTS)], "--tokens", str(tokens),
        "--ctx", "256", "--ngl", "99", "--ub", "1", "--kv", "gpu",
        "--gpu-budget-mb", str(case["budget"]), "--require-evm-counters",
        "--timeout-s", "600", "--out-dir", str(output_dir),
        *env_arg("EVM_EXPERTS_PER_TENSOR", case["capacity"]),
        *env_arg("EVM_TARGET_EXPERT_COUNT", 64),
        *([] if case["pack_only"] else env_arg("EVM_CPU_BACKING", 1)),
        *([] if not case["pack_only"] else env_arg("EVM_GPU_PACK_ONLY", 1)),
        *env_arg("EVM_CUDA_STREAMING", 1),
        *env_arg("EVM_EXPERT_VAULT_INDEX", pack_dir / "experts.pack.idx" if case["pack_only"] else FULL_INDEX),
        *env_arg("EVM_ABILITY_PACK_INDEX", pack_dir / "experts.pack.idx"),
        *env_arg("EVM_ABILITY_PACK", pack_dir / "experts.pack"),
        *env_arg("EVM_ABILITY_PACK_ONLY", 1 if case["pack_only"] else 0),
        *env_arg("EVM_STRICT_BUDGET", 1),
        *env_arg("EVM_PREFILL_BATCH_THRESHOLD", 999),
    ]
    command += env_arg("EVM_EXPERT_VAULT_PACK", pack_dir / "experts.pack" if case["pack_only"] else FULL_PACK)
    command += ["--", "--mmap", "--ignore-eos"]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    row_path = output_dir / f"{name}.json"
    row = json.loads(row_path.read_text(encoding="utf-8")) if row_path.exists() else {}
    row["workflow_case"] = case["name"]
    row["trial"] = trial + 1
    row["valid"] = completed.returncode == 0 and row.get("returncode") == 0 and row.get("has_evm_counters", False)
    return row


def summarize(case, rows):
    valid = [row for row in rows if row["valid"]]
    values = [row["generation_tps"] for row in valid]
    return {
        "name": case["name"], "pack_size": case["size"], "resident_expert_pct": case["resident_pct"], "pack_only": case["pack_only"],
        "capacity": case["capacity"], "trials": len(rows), "valid_trials": len(valid),
        "mean_generation_tps": round(statistics.mean(values), 2) if values else 0,
        "stddev_generation_tps": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
        "mean_peak_vram_mb": round(statistics.mean(row["peak_gpu_memory_mb"] for row in valid)) if valid else 0,
        "mean_peak_rss_mb": round(statistics.mean(row["peak_process_rss_mb"] for row in valid)) if valid else 0,
        "mean_hit_rate_pct": round(statistics.mean(row["cache_hit_rate_pct"] for row in valid), 2) if valid else 0,
        "mean_substitutions": round(statistics.mean(row.get("pack_substitutions", 0) for row in valid), 1) if valid else 0,
        "pass": len(valid) == len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the production Qwen2 ability-pack matrix.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ability_packs" / "workflow_universal_runs")
    parser.add_argument("--cases", help="comma-separated case names; omitted runs the full matrix")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        {"name": "pack16_exact", "size": 16, "resident_pct": 25.0, "capacity": 24, "budget": 18000, "pack_only": False},
        {"name": "pack24_exact", "size": 24, "resident_pct": 37.5, "capacity": 32, "budget": 23500, "pack_only": False},
        {"name": "pack16_only", "size": 16, "resident_pct": 25.0, "capacity": 16, "budget": 15000, "pack_only": True},
        {"name": "pack24_only", "size": 24, "resident_pct": 37.5, "capacity": 24, "budget": 20000, "pack_only": True},
    ]
    selected_names = set(args.cases.split(",")) if args.cases else {case["name"] for case in cases}
    cases_to_run = [case for case in cases if case["name"] in selected_names]
    if not cases_to_run:
        raise SystemExit("no matching workflow cases")
    summary_path = args.out_dir / "ability_workflow_summary.json"
    existing = {}
    if summary_path.exists():
        existing = {row["name"]: row for row in json.loads(summary_path.read_text(encoding="utf-8"))}
    summaries = []
    for case in cases_to_run:
        rows = [run_trial(case, trial, args.tokens, args.out_dir) for trial in range(args.trials)]
        summary = summarize(case, rows)
        summaries.append(summary)
        status = "PASS" if summary["pass"] else "FAIL"
        print(f"{case['name']}: {summary['mean_generation_tps']:.2f} t/s | "
              f"{summary['mean_peak_vram_mb']} MB VRAM | {summary['valid_trials']}/{summary['trials']} | {status}")
    for row in summaries:
        existing[row["name"]] = row
    ordered = [existing[case["name"]] for case in cases if case["name"] in existing]
    summary_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if all(row["pass"] for row in summaries) else 1)


if __name__ == "__main__":
    main()
