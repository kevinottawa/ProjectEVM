import csv
import json
import os
import re
import argparse
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
OUT_DIR = ROOT / "results" / "controlled_final"
CSV_PATH = OUT_DIR / "controlled_runtime.csv"
PROMPT = "Explain virtual memory in operating systems in one concise paragraph."


MODELS = [
    {
        "id": "qwen15_moe",
        "path": ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf",
        "target_experts": "60",
        "gpu_pool_mb": "512",
        "tokens": 32,
        "timeout": 360,
    },
    {
        "id": "deepseek_coder_v2_lite",
        "path": ROOT / "models" / "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf",
        "target_experts": "64",
        "gpu_pool_mb": "1024",
        "tokens": 16,
        "timeout": 540,
    },
    {
        "id": "qwen2_57b_a14b",
        "path": ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf",
        "target_experts": "64",
        "gpu_pool_mb": "2048",
        "tokens": 4,
        "timeout": 1200,
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
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0
    return int(proc.stdout.strip().splitlines()[0].strip())


def parse_output(text):
    match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    prompt_tps = float(match.group(1)) if match else 0.0
    generation_tps = float(match.group(2)) if match else 0.0
    hits = sum(int(m.group(1)) for m in re.finditer(r"Cache Hits\s*:\s*(\d+)", text))
    misses = sum(int(m.group(1)) for m in re.finditer(r"Cache Misses\s*:\s*(\d+)", text))
    evictions = sum(int(m.group(1)) for m in re.finditer(r"Evictions\s*:\s*(\d+)", text))
    bytes_transferred = sum(int(m.group(1)) for m in re.finditer(r"Bytes Transferred:\s*(\d+)", text))
    total = hits + misses
    return {
        "prompt_tps": f"{prompt_tps:.2f}",
        "generation_tps": f"{generation_tps:.2f}",
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_evictions": evictions,
        "hit_rate_pct": f"{100.0 * hits / total:.2f}" if total else "0.00",
        "bytes_transferred_mb": f"{bytes_transferred / (1024 * 1024):.2f}",
        "has_evm_counters": total > 0,
        "cuda_error": "CUDA error" in text or "illegal memory access" in text,
        "artifact": generation_tps > 1000.0 or "Instant EOS" in text,
    }


def run_case(model, mode):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    for key in list(env):
        if key.startswith("EVM_"):
            env.pop(key)

    extra_args = []
    require_counters = False

    if mode == "native_gpu":
        env["EVM_DISABLE"] = "1"
    elif mode == "cpu_backed_residency":
        env.update({
            "EVM_CAPACITY_PCT": "33",
            "EVM_CPU_BACKING": "1",
            "EVM_TARGET_EXPERT_COUNT": model["target_experts"],
        })
        extra_args.append("--no-mmap")
    elif mode == "gpu_backed_remap":
        env.update({
            "EVM_CAPACITY_PCT": "33",
            "EVM_TARGET_EXPERT_COUNT": model["target_experts"],
            "EVM_EXPERT_POOL_MB": model["gpu_pool_mb"],
            "EVM_CUDA_FREE_RESERVE_MB": "1024",
            "EVM_PREFILL_CAPACITY_PCT": "50",
            "EVM_STRICT_BUDGET": "1",
        })
        require_counters = True
    elif mode == "unified_streaming":
        env.update({
            "EVM_CAPACITY_PCT": "33",
            "EVM_CPU_BACKING": "1",
            "EVM_CUDA_STREAMING": "1",
            "EVM_TARGET_EXPERT_COUNT": model["target_experts"],
            "EVM_EXPERT_POOL_MB": model["gpu_pool_mb"],
            "EVM_CUDA_FREE_RESERVE_MB": "1024",
            "EVM_PREFILL_CAPACITY_PCT": "50",
            "EVM_STRICT_BUDGET": "1",
        })
        extra_args.append("--no-mmap")
        require_counters = True
    elif mode == "unified_fusion_aware":
        env.update({
            "EVM_CAPACITY_PCT": "33",
            "EVM_CPU_BACKING": "1",
            "EVM_CUDA_STREAMING": "1",
            "EVM_TARGET_EXPERT_COUNT": model["target_experts"],
            "EVM_EXPERT_POOL_MB": model["gpu_pool_mb"],
            "EVM_CUDA_FREE_RESERVE_MB": "1024",
            "EVM_PREFILL_CAPACITY_PCT": "50",
            "EVM_STRICT_BUDGET": "1",
            "EVM_FUSION_AWARE": "1",
        })
        extra_args.append("--no-mmap")
        require_counters = True
    else:
        raise ValueError(mode)

    cmd = [
        str(EXE),
        "-m", str(model["path"]),
        "-p", PROMPT,
        "-n", str(model["tokens"]),
        "-c", "256",
        "-ngl", "99",
        "-ub", "4",
        "--temp", "0.0",
        "--kv-offload",
        *extra_args,
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
    timed_out = False
    try:
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
        returncode = proc.returncode
        text = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="ignore")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="ignore")
        text = stdout + "\n" + stderr
    finally:
        elapsed = time.time() - start
        stop["value"] = True
        thread.join(timeout=2)

    run_dir = OUT_DIR / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{model['id']}_{mode}.log"
    log_path.write_text(text, encoding="utf-8", errors="ignore")

    metrics = parse_output(text)
    success = (
        returncode == 0 and
        not timed_out and
        not metrics["cuda_error"] and
        not metrics["artifact"] and
        (not require_counters or metrics["has_evm_counters"])
    )
    return {
        "model": model["id"],
        "mode": mode,
        "success": success,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": f"{elapsed:.1f}",
        "tokens": model["tokens"],
        "peak_gpu_memory_mb": peak["value"],
        "kv": "gpu",
        "require_counters": require_counters,
        **metrics,
        "env": json.dumps({k: env[k] for k in sorted(env) if k.startswith("EVM_")}, sort_keys=True),
        "command": " ".join(cmd),
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-qwen2", action="store_true", help="include slow/experimental Qwen2 rows")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = ["native_gpu", "cpu_backed_residency", "gpu_backed_remap", "unified_streaming", "unified_fusion_aware"]
    for model in MODELS:
        if not model["path"].exists():
            continue
        if model["id"] == "qwen2_57b_a14b" and not args.include_qwen2:
            continue
        for mode in modes:
            if model["id"] == "qwen2_57b_a14b" and mode == "gpu_backed_remap":
                continue
            print(f"running {model['id']} {mode}", flush=True)
            rows.append(run_case(model, mode))

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
