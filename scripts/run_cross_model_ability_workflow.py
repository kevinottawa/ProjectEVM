import argparse
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_budgeted_llama.py"
PROMPTS = [
    "Explain cache locality in two sentences.",
    "Write a compact binary search function.",
    "Solve 3x + 5 = 20.",
]
CONFIGS = {
    "qwen1": {
        "spine": "results/ability_packs/qwen1_full_vault/spine.gguf",
        "full": "results/ability_packs/qwen1_full_vault",
        "expert_count": 60,
        "packs": {25.0: (15, "results/ability_packs/qwen1_universal_25"), 37.5: (23, "results/ability_packs/qwen1_universal_37p5")},
        "exact_budget": 10000, "only_budget": 7000,
    },
    "deepseek": {
        "spine": "results/ability_packs/deepseek_full_vault/spine.gguf",
        "full": "results/ability_packs/deepseek_full_vault",
        "expert_count": 64,
        "packs": {25.0: (16, "results/ability_packs/deepseek_universal_25"), 37.5: (24, "results/ability_packs/deepseek_universal_37p5")},
        "exact_budget": 16000, "only_budget": 11000,
    },
}


def add_env(command, key, value):
    command += ["--env", f"{key}={value}"]


def run_trial(model_name, config, percentage, count, pack_dir, exact, trial, args):
    mode = "exact" if exact else "only"
    label = str(percentage).replace(".", "p")
    name = f"{model_name}_p{label}_{mode}_trial_{trial + 1}"
    full_dir = ROOT / config["full"]
    selected_dir = ROOT / pack_dir
    capacity = count + 8 if exact else count
    command = [
        "python", str(RUNNER), "--name", name, "--model", str(ROOT / config["spine"]),
        "--prompt", PROMPTS[trial % len(PROMPTS)], "--tokens", str(args.tokens),
        "--ctx", "256", "--ngl", "99", "--ub", "1", "--kv", "gpu",
        "--gpu-budget-mb", str(config["exact_budget"] if exact else config["only_budget"]),
        "--require-evm-counters", "--timeout-s", "300", "--out-dir", str(args.out_dir),
    ]
    env = {
        "EVM_EXPERTS_PER_TENSOR": capacity,
        "EVM_TARGET_EXPERT_COUNT": config["expert_count"],
        "EVM_CPU_BACKING": 1 if exact else None,
        "EVM_GPU_PACK_ONLY": None if exact else 1,
        "EVM_CUDA_STREAMING": 1,
        "EVM_EXPERT_VAULT_INDEX": (full_dir if exact else selected_dir) / "experts.pack.idx",
        "EVM_EXPERT_VAULT_PACK": (full_dir if exact else selected_dir) / "experts.pack",
        "EVM_ABILITY_PACK_INDEX": selected_dir / "experts.pack.idx",
        "EVM_ABILITY_PACK_ONLY": 0 if exact else 1,
        "EVM_ROUTER_MASK_UNAVAILABLE": None if exact else 1,
        "EVM_STRICT_BUDGET": 1,
        "EVM_PREFILL_BATCH_THRESHOLD": 999,
    }
    for key, value in env.items():
        if value is not None:
            add_env(command, key, value)
    command += ["--", "--mmap", "--ignore-eos"]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    row_path = args.out_dir / f"{name}.json"
    row = json.loads(row_path.read_text(encoding="utf-8")) if row_path.exists() else {}
    return {
        "valid": completed.returncode == 0 and row.get("returncode") == 0 and row.get("has_evm_counters", False),
        "generation_tps": row.get("generation_tps", 0), "peak_vram_mb": row.get("peak_gpu_memory_mb", 0),
        "peak_rss_mb": row.get("peak_process_rss_mb", 0), "hit_rate_pct": row.get("cache_hit_rate_pct", 0),
        "substitutions": row.get("pack_substitutions", 0),
    }


def summarize(model_name, percentage, count, exact, rows):
    valid = [row for row in rows if row["valid"]]
    values = [row["generation_tps"] for row in valid]
    return {
        "model": model_name, "resident_expert_pct": percentage, "experts_per_layer": count,
        "mode": "exact_fallback" if exact else "pack_only", "trials": len(rows), "valid_trials": len(valid),
        "mean_generation_tps": round(statistics.mean(values), 2) if values else 0,
        "stddev_generation_tps": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
        "mean_peak_vram_mb": round(statistics.mean(row["peak_vram_mb"] for row in valid)) if valid else 0,
        "mean_peak_rss_mb": round(statistics.mean(row["peak_rss_mb"] for row in valid)) if valid else 0,
        "mean_hit_rate_pct": round(statistics.mean(row["hit_rate_pct"] for row in valid), 2) if valid else 0,
        "mean_substitutions": round(statistics.mean(row["substitutions"] for row in valid), 1) if valid else 0,
        "pass": len(valid) == len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-model exact and pack-only ability runtime matrix.")
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--percentages", default="25,37.5", help="comma-separated residency percentages")
    parser.add_argument("--modes", default="exact,only", help="comma-separated modes: exact,only")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[args.model]
    selected_percentages = {float(value) for value in args.percentages.split(",")}
    selected_modes = set(args.modes.split(","))
    summaries = []
    for percentage, (count, pack_dir) in config["packs"].items():
        if percentage not in selected_percentages:
            continue
        for exact in (True, False):
            if ("exact" if exact else "only") not in selected_modes:
                continue
            rows = [run_trial(args.model, config, percentage, count, pack_dir, exact, trial, args) for trial in range(args.trials)]
            summary = summarize(args.model, percentage, count, exact, rows)
            summaries.append(summary)
            print(f"{args.model} {percentage}% {summary['mode']}: {summary['mean_generation_tps']:.2f} t/s | {summary['valid_trials']}/{summary['trials']} | {'PASS' if summary['pass'] else 'FAIL'}")
    (args.out_dir / "ability_workflow_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if all(row["pass"] for row in summaries) else 1)


if __name__ == "__main__":
    main()
