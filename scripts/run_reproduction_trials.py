import csv
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
OUT_DIR = ROOT / "results" / "final_proof" / "reproduction_trials"
CSV_PATH = ROOT / "results" / "final_proof" / "reproduction_trials.csv"
MODEL = ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
PROMPT = "Explain virtual memory in operating systems in two concise paragraphs."
TRIALS = 5


def parse_output(text):
    prompt_tps = 0.0
    generation_tps = 0.0
    match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    if match:
        prompt_tps = float(match.group(1))
        generation_tps = float(match.group(2))

    hits = sum(int(m.group(1)) for m in re.finditer(r"Cache Hits\s*:\s*(\d+)", text))
    misses = sum(int(m.group(1)) for m in re.finditer(r"Cache Misses\s*:\s*(\d+)", text))
    bytes_transferred = sum(int(m.group(1)) for m in re.finditer(r"Bytes Transferred:\s*(\d+)", text))
    return prompt_tps, generation_tps, hits, misses, bytes_transferred


def run_once(case_name, trial_idx, ngl, env_updates):
    env = os.environ.copy()
    env.update(env_updates)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        str(EXE),
        "-m", str(MODEL),
        "-p", PROMPT,
        "-n", "64",
        "-c", "256",
        "-ngl", str(ngl),
        "-ub", "4",
        "--temp", "0.0",
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        input="/exit\n",
        text=True,
        encoding="utf-8",
        errors="ignore",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=360,
    )
    elapsed = time.time() - t0
    text = proc.stdout + "\n" + proc.stderr
    log_path = OUT_DIR / f"{case_name}_trial_{trial_idx:02d}.log"
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    prompt_tps, generation_tps, hits, misses, bytes_transferred = parse_output(text)
    return {
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "cache_hits": hits,
        "cache_misses": misses,
        "bytes_transferred": bytes_transferred,
        "cuda_error": "CUDA error" in text or "illegal memory access" in text,
        "artifact": generation_tps > 1000.0 or "Instant EOS" in text,
        "log": str(log_path),
    }


def summarize(case_name, ngl, env_updates, trials):
    rows = []
    for i in range(1, trials + 1):
        row = run_once(case_name, i, ngl, env_updates)
        rows.append(row)
        print(
            f"{case_name} trial {i}/{trials}: "
            f"prompt={row['prompt_tps']:.2f} gen={row['generation_tps']:.2f} "
            f"hits={row['cache_hits']} misses={row['cache_misses']} "
            f"cuda_error={row['cuda_error']} artifact={row['artifact']}"
        )

    prompt = [r["prompt_tps"] for r in rows if not r["cuda_error"] and not r["artifact"] and r["returncode"] == 0]
    gen = [r["generation_tps"] for r in rows if not r["cuda_error"] and not r["artifact"] and r["returncode"] == 0]
    total_hits = sum(r["cache_hits"] for r in rows)
    total_misses = sum(r["cache_misses"] for r in rows)
    total = total_hits + total_misses
    return {
        "name": case_name,
        "trials": trials,
        "valid_trials": len(gen),
        "ngl": ngl,
        "prompt_tps_mean": f"{statistics.mean(prompt):.2f}" if prompt else "0.00",
        "prompt_tps_std": f"{statistics.stdev(prompt):.2f}" if len(prompt) > 1 else "0.00",
        "generation_tps_mean": f"{statistics.mean(gen):.2f}" if gen else "0.00",
        "generation_tps_std": f"{statistics.stdev(gen):.2f}" if len(gen) > 1 else "0.00",
        "cache_hits_total": total_hits,
        "cache_misses_total": total_misses,
        "hit_rate_pct": f"{100.0 * total_hits / total:.2f}" if total else "0.00",
        "bytes_transferred_avg_mb": f"{statistics.mean([r['bytes_transferred'] for r in rows]) / (1024 * 1024):.2f}",
        "cuda_error_trials": sum(1 for r in rows if r["cuda_error"]),
        "artifact_trials": sum(1 for r in rows if r["artifact"]),
        "env": " ".join(f"{k}={v}" for k, v in sorted(env_updates.items())),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        ("cpu_only_baseline", 0, {"EVM_DISABLE": "1"}),
        ("gpu_native_baseline", 99, {"EVM_DISABLE": "1"}),
        ("cpu_backed_evm_33", 99, {"EVM_CAPACITY_PCT": "33", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_CPU_BACKING": "1", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
        ("gpu_backed_evm_100", 99, {"EVM_CAPACITY_PCT": "100", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
        ("gpu_backed_evm_80", 99, {"EVM_CAPACITY_PCT": "80", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
        ("gpu_backed_evm_66", 99, {"EVM_CAPACITY_PCT": "66", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
        ("gpu_backed_evm_33", 99, {"EVM_CAPACITY_PCT": "33", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
    ]
    summaries = [summarize(*case, trials=TRIALS) for case in cases]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
