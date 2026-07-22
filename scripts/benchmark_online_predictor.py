import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_budgeted_llama.py"
MODELS = {
    "qwen1": {"spine": "results/ability_packs/qwen1_full_vault/spine.gguf", "vault": "results/ability_packs/qwen1_full_vault", "experts": 60, "budget": 7000},
    "deepseek": {"spine": "results/ability_packs/deepseek_full_vault/spine.gguf", "vault": "results/ability_packs/deepseek_full_vault", "experts": 64, "budget": 11000},
    "qwen2": {"spine": "results/expert_vault/qwen2_full_vault/qwen2-spine.gguf", "vault": "results/expert_vault/qwen2_full_vault", "experts": 64, "budget": 24000},
}
PROMPTS = (
    "Explain cache locality in two sentences.",
    "Give a concise explanation of binary search.",
    "Explain why asynchronous copies need valid source memory.",
)


def gpu_total_mb(fallback):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="ignore",
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return int(completed.stdout.strip().splitlines()[0])
    return fallback


def run_trial(model, config, mode, trial, args):
    name = f"{model}_{mode}_trial_{trial + 1}"
    env = [
        f"EVM_TARGET_EXPERT_COUNT={config['experts']}",
        "EVM_CPU_BACKING=1", "EVM_CUDA_STREAMING=1", f"EVM_EXPERT_VAULT_INDEX={ROOT / config['vault'] / 'experts.pack.idx'}",
        f"EVM_EXPERT_VAULT_PACK={ROOT / config['vault'] / 'experts.pack'}", "EVM_STRICT_BUDGET=1",
        "EVM_PREFILL_BATCH_THRESHOLD=999",
    ]
    gpu_budget_mb = args.gpu_budget_mb or config["budget"]
    if args.reserve_vram_gb:
        reserve_mb = args.reserve_vram_gb * 1024
        env += ["EVM_CAPACITY_PCT=100", f"EVM_CUDA_FREE_RESERVE_MB={reserve_mb}"]
        gpu_budget_mb = max(1, gpu_total_mb(config["budget"]) - reserve_mb)
    else:
        env += [f"EVM_EXPERTS_PER_TENSOR={args.capacity}"]
    if mode == "predictor":
        env += ["EVM_ONLINE_PREDICTOR=1", "EVM_PREDICTOR_PREFETCH=1", "EVM_PREDICTOR_MIN_COUNT=2", f"EVM_PREDICTOR_MIN_CONFIDENCE_PCT={args.confidence}"]
    elif mode == "learned":
        prior = ROOT / "results" / "predictor_training" / model / "model" / "runtime_layer_prior.txt"
        env += ["EVM_LEARNED_SCHEDULER=1", f"EVM_LEARNED_SCHEDULER_PATH={prior}"]
    elif mode == "gpu-page":
        env += ["EVM_GPU_PAGE_TABLE=1"]
    elif mode == "score-prefetch":
        env += ["EVM_ROUTER_SCORE_PREFETCH=1", "EVM_ROUTER_SCORE_PREFETCH_COUNT=1", f"EVM_PREDICTOR_RESERVED_SLOTS={args.predictive_slots}", f"EVM_ROUTER_SCORE_PREFETCH_MIN_PPM={args.score_prefetch_min_ppm}"]
    elif mode == "learned-score":
        prior = ROOT / "results" / "predictor_training" / model / "model" / "runtime_layer_prior.txt"
        env += ["EVM_ROUTER_SCORE_PREFETCH=1", "EVM_ROUTER_SCORE_PREFETCH_COUNT=1", f"EVM_PREDICTOR_RESERVED_SLOTS={args.predictive_slots}", f"EVM_ROUTER_SCORE_PREFETCH_MIN_PPM={args.score_prefetch_min_ppm}",
                "EVM_LEARNED_ROUTER_SCORE=1", f"EVM_LEARNED_SCHEDULER_PATH={prior}"]
    elif mode == "learned-evict":
        prior = ROOT / "results" / "predictor_training" / model / "model" / "runtime_layer_prior.txt"
        env += ["EVM_ROUTER_SCORE_PREFETCH=1", "EVM_ROUTER_SCORE_PREFETCH_COUNT=0", "EVM_PREDICTOR_RESERVED_SLOTS=0",
                "EVM_LEARNED_ROUTER_SCORE=1", f"EVM_LEARNED_SCHEDULER_PATH={prior}"]
    elif mode == "gpu-scheduler":
        prior = ROOT / "results" / "predictor_training" / model / "model" / "runtime_layer_prior.txt"
        env += ["EVM_GPU_PAGE_TABLE=1", "EVM_GPU_SCHEDULER=1", f"EVM_LEARNED_SCHEDULER_PATH={prior}"]
    command = [sys.executable, str(RUNNER), "--name", name, "--model", str(ROOT / config["spine"]),
               "--prompt", PROMPTS[trial % len(PROMPTS)], "--tokens", str(args.tokens), "--ctx", "256", "--ngl", "99", "--ub", "1",
               "--kv", "gpu", "--gpu-budget-mb", str(gpu_budget_mb), "--require-evm-counters", "--timeout-s", str(args.timeout_s),
               "--out-dir", str(args.out_dir)]
    for item in env:
        command += ["--env", item]
    command += ["--", "--mmap", "--ignore-eos"]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    path = args.out_dir / f"{name}.json"
    row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    row["valid"] = completed.returncode == 0 and row.get("returncode") == 0 and row.get("has_evm_counters", False)
    return row


def summarize(model, mode, rows, args):
    valid = [row for row in rows if row["valid"]]
    def average(key):
        return round(statistics.mean(row.get(key, 0) for row in valid), 2) if valid else 0
    return {"model": model, "mode": mode, "capacity": rows[0].get("evm_env", {}).get("EVM_EXPERTS_PER_TENSOR", "dynamic"), "reserve_vram_gb": args.reserve_vram_gb or "", "confidence_pct": args.confidence if mode == "predictor" else "",
            "trials": len(rows), "valid_trials": len(valid),
            "mean_generation_tps": average("generation_tps"), "mean_peak_vram_mb": round(average("peak_gpu_memory_mb")),
            "mean_hit_rate_pct": average("cache_hit_rate_pct"), "mean_bytes_transferred_mb": average("bytes_transferred_mb"),
            "mean_predictor_prefetches": average("predictor_prefetches"), "mean_predictor_hits": average("predictor_hits"),
            "mean_gpu_page_hits": average("gpu_page_hits"), "mean_gpu_page_misses": average("gpu_page_misses"),
            "mean_router_score_prefetches": average("router_score_prefetches"),
            "mean_process_rss_mb": round(average("peak_process_rss_mb")), "mean_system_ram_delta_mb": round(average("system_ram_used_delta_mb")),
            "mean_pagefile_delta_mb": round(average("pagefile_used_delta_mb"))}


def main():
    parser = argparse.ArgumentParser(description="Compare exact EVM LRU and online-predictor throughput.")
    parser.add_argument("--models", default="qwen1,deepseek")
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--modes", default="lru,predictor")
    parser.add_argument("--confidence", type=int, default=35)
    parser.add_argument("--predictive-slots", type=int, default=2, help="reserved EVM slots for score-ranked prefetches")
    parser.add_argument("--score-prefetch-min-ppm", type=int, default=20000, help="minimum router probability for a prefetch, in parts per million")
    parser.add_argument("--reserve-vram-gb", type=int, default=0, help="keep this many GiB free by dynamically sizing EVM pools")
    parser.add_argument("--gpu-budget-mb", type=int, default=0, help="override the profile GPU guardrail for a controlled capacity probe")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for model in (item.strip() for item in args.models.split(",") if item.strip()):
        config = MODELS[model]
        for mode in (item.strip() for item in args.modes.split(",") if item.strip()):
            rows = [run_trial(model, config, mode, trial, args) for trial in range(args.trials)]
            summary.append(summarize(model, mode, rows, args))
    (args.out_dir / "online_predictor_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(f"{row['model']} {row['mode']}: {row['mean_generation_tps']:.2f} t/s | {row['valid_trials']}/{row['trials']} | score prefetches {row['mean_router_score_prefetches']:.1f}")
    raise SystemExit(0 if all(row["valid_trials"] == row["trials"] for row in summary) else 1)


if __name__ == "__main__":
    main()
