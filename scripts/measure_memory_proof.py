import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
MODEL = ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
OUT_DIR = ROOT / "results" / "final_proof"
PROMPT = "Explain virtual memory in operating systems in two concise paragraphs."


def gpu_used_mb():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0:
        return 0
    first = proc.stdout.strip().splitlines()[0]
    return int(first.strip())


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
    total = hits + misses
    return prompt_tps, generation_tps, hits, misses, 100.0 * hits / total if total else 0.0, bytes_transferred


def run_case(name, ngl, env_updates):
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

    peak = {"value": gpu_used_mb()}
    stop = {"value": False}

    def poll():
        while not stop["value"]:
            peak["value"] = max(peak["value"], gpu_used_mb())
            time.sleep(0.2)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
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
        timeout=720,
    )
    elapsed = time.time() - t0
    stop["value"] = True
    t.join(timeout=2.0)

    output = proc.stdout + "\n" + proc.stderr
    log_path = OUT_DIR / f"{name}.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")

    prompt_tps, generation_tps, hits, misses, hit_rate, bytes_transferred = parse_output(output)
    return {
        "name": name,
        "returncode": proc.returncode,
        "elapsed_s": f"{elapsed:.1f}",
        "ngl": ngl,
        "prompt_tps": f"{prompt_tps:.2f}",
        "generation_tps": f"{generation_tps:.2f}",
        "peak_gpu_memory_mb": peak["value"],
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": f"{hit_rate:.2f}",
        "bytes_transferred_mb": f"{bytes_transferred / (1024 * 1024):.2f}",
        "cuda_error": "CUDA error" in output or "illegal memory access" in output,
        "instant_eos_or_artifact": generation_tps > 1000.0 or "Instant EOS" in output,
        "env": " ".join(f"{k}={v}" for k, v in sorted(env_updates.items())),
        "log": str(log_path),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        ("qwen15_cpu_only_baseline", 0, {"EVM_DISABLE": "1"}),
        ("qwen15_gpu_native_baseline", 99, {"EVM_DISABLE": "1"}),
        ("qwen15_cpu_backed_evm_33_memory", 99, {"EVM_CAPACITY_PCT": "33", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_CPU_BACKING": "1", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
        ("qwen15_gpu_backed_evm_33_memory", 99, {"EVM_CAPACITY_PCT": "33", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
        ("qwen15_gpu_backed_evm_66_memory", 99, {"EVM_CAPACITY_PCT": "66", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"}),
    ]
    rows = []
    for name, ngl, env in cases:
        print(f"running {name}")
        rows.append(run_case(name, ngl, env))

    path = OUT_DIR / "memory_proof.csv"
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
