import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
OUT_DIR = ROOT / "results" / "final_proof" / "cross_model_memory"
CSV_PATH = ROOT / "results" / "final_proof" / "cross_model_memory.csv"
PROMPT = "Explain virtual memory in operating systems in one concise paragraph."


MODELS = [
    {
        "id": "qwen15_moe",
        "path": ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf",
        "target_experts": "60",
        "tokens": "32",
        "timeout": 360,
    },
    {
        "id": "deepseek_coder_v2_lite",
        "path": ROOT / "models" / "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf",
        "target_experts": "",
        "tokens": "32",
        "timeout": 420,
    },
    {
        "id": "qwen2_57b_a14b",
        "path": ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf",
        "target_experts": "64",
        "tokens": "32",
        "timeout": 900,
    },
]


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
    return int(proc.stdout.strip().splitlines()[0].strip())


def parse_output(text):
    match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    prompt_tps = float(match.group(1)) if match else 0.0
    generation_tps = float(match.group(2)) if match else 0.0
    hits = sum(int(m.group(1)) for m in re.finditer(r"Cache Hits\s*:\s*(\d+)", text))
    misses = sum(int(m.group(1)) for m in re.finditer(r"Cache Misses\s*:\s*(\d+)", text))
    bytes_transferred = sum(int(m.group(1)) for m in re.finditer(r"Bytes Transferred:\s*(\d+)", text))
    return prompt_tps, generation_tps, hits, misses, bytes_transferred


def run_case(model, mode):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if mode == "native_gpu":
        env["EVM_DISABLE"] = "1"
    elif mode == "cpu_backed_evm_33":
        env["EVM_CAPACITY_PCT"] = "33"
        env["EVM_CPU_BACKING"] = "1"
        env["EVM_PREFILL_BATCH_THRESHOLD"] = "1"
        if model["target_experts"]:
            env["EVM_TARGET_EXPERT_COUNT"] = model["target_experts"]
    else:
        raise ValueError(mode)

    cmd = [
        str(EXE),
        "-m", str(model["path"]),
        "-p", PROMPT,
        "-n", model["tokens"],
        "-c", "256",
        "-ngl", "99",
        "-ub", "4",
        "--temp", "0.0",
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
        timeout=model["timeout"],
    )
    elapsed = time.time() - start
    stop["value"] = True
    thread.join(timeout=2)

    text = proc.stdout + "\n" + proc.stderr
    log_path = OUT_DIR / f"{model['id']}_{mode}.log"
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    prompt_tps, generation_tps, hits, misses, bytes_transferred = parse_output(text)
    total = hits + misses
    return {
        "model": model["id"],
        "mode": mode,
        "returncode": proc.returncode,
        "elapsed_s": f"{elapsed:.1f}",
        "tokens": model["tokens"],
        "peak_gpu_memory_mb": peak["value"],
        "prompt_tps": f"{prompt_tps:.2f}",
        "generation_tps": f"{generation_tps:.2f}",
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": f"{100.0 * hits / total:.2f}" if total else "0.00",
        "bytes_transferred_mb": f"{bytes_transferred / (1024 * 1024):.2f}",
        "cuda_error": "CUDA error" in text or "illegal memory access" in text,
        "artifact": generation_tps > 1000.0 or "Instant EOS" in text,
        "env": " ".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith("EVM_")),
        "log": str(log_path),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        if not model["path"].exists():
            continue
        for mode in ("native_gpu", "cpu_backed_evm_33"):
            print(f"running {model['id']} {mode}")
            rows.append(run_case(model, mode))

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
