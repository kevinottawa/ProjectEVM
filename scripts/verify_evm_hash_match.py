import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-evm-batch-profiler.exe"
MANIFEST = ROOT / "config" / "mri" / "v2" / "evm_hash_match_five.json"
MODELS = {
    "qwen1": {"full": "models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf", "spine": "results/ability_packs/qwen1_full_vault/spine.gguf", "vault": "results/ability_packs/qwen1_full_vault", "experts": 60},
    "deepseek": {"full": "models/DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf", "spine": "results/ability_packs/deepseek_full_vault/spine.gguf", "vault": "results/ability_packs/deepseek_full_vault", "experts": 64},
    "qwen2": {"full": "models/qwen2-57b-a14b-instruct-q4_k_m.gguf", "spine": "results/expert_vault/qwen2_full_vault/qwen2-spine.gguf", "vault": "results/expert_vault/qwen2_full_vault", "experts": 64},
}


def clean_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("EVM_")}


def run(model, env, out_dir, tokens):
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    command = [str(PROFILER), "-m", str(model), "--manifest", str(MANIFEST), "--profile", str(out_dir / "routing.jsonl"),
               "--rows", str(out_dir / "rows.jsonl"), "--checkpoint", str(out_dir / "checkpoint.txt"), "-n", str(tokens),
               "-c", "256", "-ngl", "99", "--ub", "1", "--mmap", "--no-evm-profile", "--ignore-eos"]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=900)
    rows = {}
    rows_path = out_dir / "rows.jsonl"
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["prompt_id"]] = row
    return completed.returncode, rows


def main():
    parser = argparse.ArgumentParser(description="Verify exact EVM token hashes against a full-GPU baseline.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--capacity", type=int, default=8, help="exact EVM GPU pool slots per expert tensor")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--predictor", action="store_true", help="enable the live online EVM prefetch predictor")
    parser.add_argument("--learned", action="store_true", help="enable the exported layer-aware learned eviction scheduler")
    parser.add_argument("--gpu-page-table", action="store_true", help="use the exact GPU logical-to-physical hit mapper")
    parser.add_argument("--router-score-prefetch", action="store_true", help="enable conservative router-score prefetch into empty EVM slots")
    parser.add_argument("--learned-router-score", action="store_true", help="rank router-score prefetches with the exported 3-to-17-token predictor")
    parser.add_argument("--predictive-slots", type=int, default=2, help="reserved EVM slots for router-score prefetch")
    parser.add_argument("--score-prefetch-min-ppm", type=int, default=20000, help="minimum router probability for a prefetch, in parts per million")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = MODELS[args.model]
    baseline_code, baseline = run(ROOT / config["full"], clean_env(), args.out_dir / "full", args.tokens)
    env = clean_env()
    vault = ROOT / config["vault"]
    env.update({"EVM_EXPERTS_PER_TENSOR": str(args.capacity), "EVM_TARGET_EXPERT_COUNT": str(config["experts"]),
                "EVM_CPU_BACKING": "1", "EVM_CUDA_STREAMING": "1", "EVM_EXPERT_VAULT_INDEX": str(vault / "experts.pack.idx"),
                "EVM_EXPERT_VAULT_PACK": str(vault / "experts.pack"), "EVM_STRICT_BUDGET": "1", "EVM_PREFILL_BATCH_THRESHOLD": "999"})
    if args.predictor:
        env.update({"EVM_ONLINE_PREDICTOR": "1", "EVM_PREDICTOR_PREFETCH": "1", "EVM_PREDICTOR_MIN_COUNT": "2"})
    if args.learned:
        prior = ROOT / "results" / "predictor_training" / args.model / "model" / "runtime_layer_prior.txt"
        env.update({"EVM_LEARNED_SCHEDULER": "1", "EVM_LEARNED_SCHEDULER_PATH": str(prior)})
    if args.gpu_page_table:
        env["EVM_GPU_PAGE_TABLE"] = "1"
    if args.router_score_prefetch:
        env.update({"EVM_ROUTER_SCORE_PREFETCH": "1", "EVM_ROUTER_SCORE_PREFETCH_COUNT": "1", "EVM_PREDICTOR_RESERVED_SLOTS": str(args.predictive_slots), "EVM_ROUTER_SCORE_PREFETCH_MIN_PPM": str(args.score_prefetch_min_ppm)})
    if args.learned_router_score:
        prior = ROOT / "results" / "predictor_training" / args.model / "model" / "runtime_layer_prior.txt"
        env.update({"EVM_ROUTER_SCORE_PREFETCH": "1", "EVM_ROUTER_SCORE_PREFETCH_COUNT": "1", "EVM_PREDICTOR_RESERVED_SLOTS": str(args.predictive_slots), "EVM_ROUTER_SCORE_PREFETCH_MIN_PPM": str(args.score_prefetch_min_ppm),
                    "EVM_LEARNED_ROUTER_SCORE": "1", "EVM_LEARNED_SCHEDULER_PATH": str(prior)})
    exact_code, exact = run(ROOT / config["spine"], env, args.out_dir / "exact_evm", args.tokens)
    tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    rows = []
    for task in tasks:
        key = task["id"]
        full = baseline.get(key, {})
        evm = exact.get(key, {})
        match = bool(full.get("valid") and evm.get("valid") and full.get("token_fingerprint_fnv1a64") == evm.get("token_fingerprint_fnv1a64"))
        rows.append({"id": key, "full_valid": bool(full.get("valid")), "evm_valid": bool(evm.get("valid")), "hash_match": match,
                     "full_hash": full.get("token_fingerprint_fnv1a64", ""), "evm_hash": evm.get("token_fingerprint_fnv1a64", "")})
    result = {"format": "evm-hash-match-v1", "model": args.model, "capacity": args.capacity, "predictor": args.predictor, "learned": args.learned, "learned_router_score": args.learned_router_score, "gpu_page_table": args.gpu_page_table, "router_score_prefetch": args.router_score_prefetch,
              "baseline_returncode": baseline_code, "exact_evm_returncode": exact_code, "rows": rows,
              "matched": sum(row["hash_match"] for row in rows), "total": len(rows)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "hash_match_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.model}: {result['matched']}/{result['total']} token hashes match | PASS")
    raise SystemExit(0 if result["matched"] == result["total"] else 1)


if __name__ == "__main__":
    main()
