import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-evm-batch-profiler.exe"
CONTRACT = ROOT / "config" / "mri" / "v2" / "workflow_quality_contract.json"
CONFIGS = {
    "qwen1": {"original": "models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf", "spine": "results/ability_packs/qwen1_full_vault/spine.gguf", "experts": 60, "count": 23},
    "deepseek": {"original": "models/DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf", "spine": "results/ability_packs/deepseek_full_vault/spine.gguf", "experts": 64, "count": 24},
    "qwen2": {"original": "models/qwen2-57b-a14b-instruct-q4_k_m.gguf", "spine": "results/expert_vault/qwen2_full_vault/qwen2-spine.gguf", "experts": 64, "count": 24,
              "baseline_mode": "exact_external_vault", "baseline_capacity": 8,
              "baseline_index": "results/expert_vault/qwen2_full_vault/experts.pack.idx",
              "baseline_pack": "results/expert_vault/qwen2_full_vault/experts.pack"},
}


def clean_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("EVM_")}


def run(model, manifest, env, tokens, out_dir):
    if out_dir.exists():
        for path in out_dir.iterdir():
            path.unlink()
    else:
        out_dir.mkdir(parents=True)
    command = [str(EXE), "-m", str(model), "--manifest", str(manifest), "--profile", str(out_dir / "routing.jsonl"),
               "--rows", str(out_dir / "rows.jsonl"), "--checkpoint", str(out_dir / "checkpoint.txt"), "-n", str(tokens),
               "-c", "256", "-ngl", "99", "--no-evm-profile", "--ignore-eos"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               encoding="utf-8", errors="ignore", env=env, timeout=1200)
    rows = [json.loads(line) for line in (out_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (out_dir / "rows.jsonl").exists() else []
    return completed.returncode, {row["prompt_id"]: row for row in rows}


def pack_env(config, pack, count):
    env = clean_env()
    env.update({"EVM_EXPERTS_PER_TENSOR": str(count), "EVM_TARGET_EXPERT_COUNT": str(config["experts"]),
                "EVM_GPU_PACK_ONLY": "1", "EVM_CUDA_STREAMING": "1", "EVM_EXPERT_VAULT_INDEX": str(pack / "experts.pack.idx"),
                "EVM_EXPERT_VAULT_PACK": str(pack / "experts.pack"), "EVM_ABILITY_PACK_INDEX": str(pack / "experts.pack.idx"),
                "EVM_ABILITY_PACK_ONLY": "1", "EVM_ROUTER_MASK_UNAVAILABLE": "1",
                "EVM_STRICT_BUDGET": "1", "EVM_PREFILL_BATCH_THRESHOLD": "999"})
    return env


def baseline_model_env(config):
    if config.get("baseline_mode") != "exact_external_vault":
        return ROOT / config["original"], clean_env(), "native_full_model"
    env = clean_env()
    env.update({"EVM_EXPERTS_PER_TENSOR": str(config["baseline_capacity"]), "EVM_TARGET_EXPERT_COUNT": str(config["experts"]),
                "EVM_CPU_BACKING": "1", "EVM_CUDA_STREAMING": "1", "EVM_EXPERT_VAULT_INDEX": str(ROOT / config["baseline_index"]),
                "EVM_EXPERT_VAULT_PACK": str(ROOT / config["baseline_pack"]), "EVM_STRICT_BUDGET": "1",
                "EVM_PREFILL_BATCH_THRESHOLD": "999"})
    return ROOT / config["spine"], env, "exact_external_vault"


def main():
    parser = argparse.ArgumentParser(description="Compare frequency and graph workflow packs against their original model.")
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--frequency-pack", type=Path, required=True)
    parser.add_argument("--graph-pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--frequency-label", default="frequency")
    parser.add_argument("--graph-label", default="graph")
    parser.add_argument("--frequency-count", type=int)
    parser.add_argument("--graph-count", type=int)
    args = parser.parse_args()
    config = CONFIGS[args.model]
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tasks = contract["tasks"]
    runtime_root = args.out.parent / (args.model + "_workflow_quality_runtime")
    manifest = runtime_root / "manifest.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    baseline_model, baseline_env, baseline_mode = baseline_model_env(config)
    labels = ((args.frequency_label, args.frequency_pack, args.frequency_count or config["count"]),
              (args.graph_label, args.graph_pack, args.graph_count or config["count"]))
    result_doc = {"format": "evm-workflow-pack-quality-v1", "model": args.model, "baseline_mode": baseline_mode, "tasks": [], "packs": {label: [] for label, _, _ in labels}}
    _, baseline_rows = run(baseline_model, manifest, baseline_env, args.tokens, runtime_root / "baseline")
    pack_rows = {}
    for label, folder, count in labels:
        _, pack_rows[label] = run(ROOT / config["spine"], manifest, pack_env(config, folder, count), args.tokens, runtime_root / label)
    for task in tasks:
        baseline = baseline_rows.get(task["id"], {"valid": False, "quality_pass": False})
        result_doc["tasks"].append({"id": task["id"], "domain": task["domain"], "pass": baseline.get("valid") and baseline.get("quality_pass"),
                                    "returncode": 0 if baseline.get("valid") else 1, "token_fingerprint_fnv1a64": baseline.get("token_fingerprint_fnv1a64", "")})
        for label, _, _ in labels:
            row = pack_rows[label].get(task["id"], {"valid": False, "quality_pass": False})
            result_doc["packs"][label].append({"id": task["id"], "domain": task["domain"], "pass": row.get("valid") and row.get("quality_pass"),
                                                "baseline_pass": baseline.get("valid") and baseline.get("quality_pass"),
                                                "returncode": 0 if row.get("valid") else 1,
                                                "token_fingerprint_fnv1a64": row.get("token_fingerprint_fnv1a64", "")})
    baseline_passes = sum(row["pass"] for row in result_doc["tasks"])
    result_doc["summary"] = {"baseline_passed": baseline_passes, "total": len(tasks)}
    for label, rows in result_doc["packs"].items():
        result_doc["summary"][label] = {"passed": sum(row["pass"] for row in rows),
                                         "retained_baseline_passes": sum(row["pass"] and row["baseline_pass"] for row in rows),
                                         "baseline_opportunities": baseline_passes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result_doc, indent=2) + "\n", encoding="utf-8")
    summary = result_doc["summary"]
    print(f"{args.model}: baseline {summary['baseline_passed']}/{summary['total']} | " + " | ".join(f"{label} {summary[label]['passed']}/{summary['total']}" for label, _, _ in labels))


if __name__ == "__main__":
    main()
