import csv
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "controlled_final" / "fusion_aware_trials"
CSV_PATH = ROOT / "results" / "controlled_final" / "fusion_aware_trials.csv"

MODELS = [
    {
        "id": "qwen15_moe",
        "model": ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf",
        "target_experts": "60",
        "pool_mb": "512",
        "tokens": "16",
        "timeout": "300",
    },
    {
        "id": "deepseek_coder_v2_lite",
        "model": ROOT / "models" / "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf",
        "target_experts": "64",
        "pool_mb": "1024",
        "tokens": "16",
        "timeout": "420",
    },
    {
        "id": "qwen2_57b_a14b",
        "model": ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf",
        "target_experts": "64",
        "pool_mb": "2048",
        "tokens": "4",
        "timeout": "900",
    },
]


def run_trial(model, trial):
    name = f"{model['id']}_fusion_aware_trial_{trial:02d}"
    cmd = [
        "python",
        str(ROOT / "scripts" / "run_budgeted_llama.py"),
        "--name", name,
        "--model", str(model["model"]),
        "--tokens", model["tokens"],
        "--ctx", "256",
        "--kv", "gpu",
        "--gpu-budget-mb", "24000",
        "--require-evm-counters",
        "--timeout-s", model["timeout"],
        "--env", "EVM_CAPACITY_PCT=33",
        "--env", f"EVM_TARGET_EXPERT_COUNT={model['target_experts']}",
        "--env", "EVM_CPU_BACKING=1",
        "--env", "EVM_CUDA_STREAMING=1",
        "--env", f"EVM_EXPERT_POOL_MB={model['pool_mb']}",
        "--env", "EVM_CUDA_FREE_RESERVE_MB=1024",
        "--env", "EVM_PREFILL_CAPACITY_PCT=50",
        "--env", "EVM_STRICT_BUDGET=1",
        "--env", "EVM_FUSION_AWARE=1",
        "--",
        "--no-mmap",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(model["timeout"]) + 60,
    )
    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    start = proc.stdout.find("{")
    data = json.loads(proc.stdout[start:])
    data["trial"] = trial
    data["model_id"] = model["id"]
    data["summary_json"] = str(out_path)
    return data


def summarize(model_id, rows):
    valid = [
        r for r in rows
        if r["returncode"] == 0 and not r["timed_out"] and not r["budget_exceeded"] and r["has_evm_counters"]
    ]
    gen = [r.get("generation_tps", 0.0) for r in valid]
    peak = [r.get("peak_gpu_memory_mb", 0) for r in valid]
    hits = sum(r.get("cache_hits", 0) for r in rows)
    misses = sum(r.get("cache_misses", 0) for r in rows)
    total = hits + misses
    return {
        "model_id": model_id,
        "trial": "summary",
        "valid_trials": len(valid),
        "trials": len(rows),
        "generation_tps_mean": f"{statistics.mean(gen):.2f}" if gen else "0.00",
        "generation_tps_std": f"{statistics.stdev(gen):.2f}" if len(gen) > 1 else "0.00",
        "peak_gpu_memory_mb_mean": f"{statistics.mean(peak):.0f}" if peak else "0",
        "peak_gpu_memory_mb_max": max(peak) if peak else 0,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": f"{100.0 * hits / total:.2f}" if total else "0.00",
        "bytes_transferred_mb_mean": f"{statistics.mean([r.get('bytes_transferred_mb', 0.0) for r in valid]):.2f}" if valid else "0.00",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for model in MODELS:
        model_rows = []
        for trial in range(1, 4):
            print(f"running {model['id']} fusion-aware trial {trial}/3", flush=True)
            row = run_trial(model, trial)
            model_rows.append(row)
            all_rows.append(row)
        summaries.append(summarize(model["id"], model_rows))

    trial_fields = [
        "model_id", "trial", "returncode", "timed_out", "budget_exceeded",
        "peak_gpu_memory_mb", "elapsed_s", "prompt_tps", "generation_tps",
        "cache_hits", "cache_misses",
        "cache_hit_rate_pct", "bytes_transferred_mb", "has_evm_counters",
        "summary_json", "log",
    ]
    summary_fields = [
        "model_id", "trial", "valid_trials", "trials", "generation_tps_mean",
        "generation_tps_std", "peak_gpu_memory_mb_mean", "peak_gpu_memory_mb_max",
        "cache_hits", "cache_misses", "hit_rate_pct", "bytes_transferred_mb_mean",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=trial_fields + [f for f in summary_fields if f not in trial_fields])
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
