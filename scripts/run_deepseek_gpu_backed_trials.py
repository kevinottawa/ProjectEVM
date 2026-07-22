import csv
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
MODEL = ROOT / "models" / "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf"
OUT_DIR = ROOT / "results" / "final_proof" / "deepseek_gpu_backed_trials"
CSV_PATH = ROOT / "results" / "final_proof" / "deepseek_gpu_backed_trials.csv"
PROMPT = "Explain virtual memory in operating systems in one concise paragraph."


def gpu_used_mb():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0
    return int(proc.stdout.strip().splitlines()[0].strip())


def parse_output(text):
    perf = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    prompt_tps = float(perf.group(1)) if perf else 0.0
    generation_tps = float(perf.group(2)) if perf else 0.0
    hits = sum(int(m.group(1)) for m in re.finditer(r"Cache Hits\s*:\s*(\d+)", text))
    misses = sum(int(m.group(1)) for m in re.finditer(r"Cache Misses\s*:\s*(\d+)", text))
    bytes_transferred = sum(int(m.group(1)) for m in re.finditer(r"Bytes Transferred:\s*(\d+)", text))
    intercepts = len(re.findall(r"EVM:\s*intercepting", text))
    return prompt_tps, generation_tps, hits, misses, bytes_transferred, intercepts


def run_trial(trial):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["EVM_CAPACITY_PCT"] = "33"
    env["EVM_TARGET_EXPERT_COUNT"] = "64"
    env["EVM_PREFILL_BATCH_THRESHOLD"] = "1"
    env["EVM_DEBUG"] = "1"
    env.pop("EVM_CPU_BACKING", None)
    env.pop("EVM_DISABLE", None)

    cmd = [
        str(EXE),
        "-m",
        str(MODEL),
        "-p",
        PROMPT,
        "-n",
        "32",
        "-c",
        "256",
        "-ngl",
        "99",
        "-ub",
        "4",
        "--temp",
        "0.0",
    ]

    peak = {"value": gpu_used_mb()}
    stop = {"value": False}

    def poll():
        while not stop["value"]:
            peak["value"] = max(peak["value"], gpu_used_mb())
            time.sleep(0.25)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    start = time.time()
    proc = subprocess.run(
        cmd,
        input="/exit\n",
        text=True,
        encoding="utf-8",
        errors="ignore",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=420,
    )
    elapsed = time.time() - start
    stop["value"] = True
    thread.join(timeout=2)

    text = proc.stdout + "\n" + proc.stderr
    log_path = OUT_DIR / f"trial_{trial}.log"
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    prompt_tps, generation_tps, hits, misses, bytes_transferred, intercepts = parse_output(text)
    total = hits + misses
    return {
        "trial": trial,
        "returncode": proc.returncode,
        "elapsed_s": f"{elapsed:.1f}",
        "peak_gpu_memory_mb": peak["value"],
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": 100.0 * hits / total if total else 0.0,
        "bytes_transferred_mb": bytes_transferred / (1024 * 1024),
        "intercepts": intercepts,
        "cuda_error": "CUDA error" in text or "illegal memory access" in text,
        "artifact": generation_tps > 1000.0 or "Instant EOS" in text,
        "log": str(log_path),
    }


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in range(1, 6):
        print(f"running trial {trial}")
        rows.append(run_trial(trial))

    valid = [r for r in rows if r["returncode"] == 0 and not r["cuda_error"] and not r["artifact"]]
    total_hits = sum(r["cache_hits"] for r in valid)
    total_misses = sum(r["cache_misses"] for r in valid)
    total = total_hits + total_misses
    summary = {
        "trial": "summary",
        "returncode": "",
        "elapsed_s": f"{mean([float(r['elapsed_s']) for r in valid]):.1f}",
        "peak_gpu_memory_mb": max((r["peak_gpu_memory_mb"] for r in valid), default=0),
        "prompt_tps": f"{mean([r['prompt_tps'] for r in valid]):.2f}+/-{stdev([r['prompt_tps'] for r in valid]):.2f}",
        "generation_tps": f"{mean([r['generation_tps'] for r in valid]):.2f}+/-{stdev([r['generation_tps'] for r in valid]):.2f}",
        "cache_hits": total_hits,
        "cache_misses": total_misses,
        "hit_rate_pct": f"{100.0 * total_hits / total:.2f}" if total else "0.00",
        "bytes_transferred_mb": f"{mean([r['bytes_transferred_mb'] for r in valid]):.2f}",
        "intercepts": sum(r["intercepts"] for r in valid),
        "cuda_error": sum(1 for r in rows if r["cuda_error"]),
        "artifact": sum(1 for r in rows if r["artifact"]),
        "log": f"valid_trials={len(valid)}/{len(rows)}",
    }

    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["prompt_tps"] = f"{out['prompt_tps']:.2f}"
            out["generation_tps"] = f"{out['generation_tps']:.2f}"
            out["hit_rate_pct"] = f"{out['hit_rate_pct']:.2f}"
            out["bytes_transferred_mb"] = f"{out['bytes_transferred_mb']:.2f}"
            writer.writerow(out)
        writer.writerow(summary)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
