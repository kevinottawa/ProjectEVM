import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
MODEL = ROOT / "models" / "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf"
OUT_DIR = ROOT / "results" / "final_proof" / "deepseek_gpu_backed"
CSV_PATH = ROOT / "results" / "final_proof" / "deepseek_gpu_backed_probe.csv"
PROMPT = "Define virtual memory briefly."


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
    target_counts = sorted(set(int(m.group(1)) for m in re.finditer(r"experts=(\d+)", text)))
    total = hits + misses
    return {
        "prompt_tps": f"{prompt_tps:.2f}",
        "generation_tps": f"{generation_tps:.2f}",
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": f"{100.0 * hits / total:.2f}" if total else "0.00",
        "bytes_transferred_mb": f"{bytes_transferred / (1024 * 1024):.2f}",
        "intercepts": intercepts,
        "intercept_expert_counts": ";".join(str(v) for v in target_counts),
        "cuda_error": "CUDA error" in text or "illegal memory access" in text,
        "artifact": generation_tps > 1000.0 or "Instant EOS" in text,
    }


def run_case(name, target_experts):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["EVM_CAPACITY_PCT"] = "33"
    env["EVM_PREFILL_BATCH_THRESHOLD"] = "1"
    env["EVM_DEBUG"] = "1"
    env.pop("EVM_CPU_BACKING", None)
    env.pop("EVM_DISABLE", None)
    if target_experts:
        env["EVM_TARGET_EXPERT_COUNT"] = target_experts
    else:
        env.pop("EVM_TARGET_EXPERT_COUNT", None)

    cmd = [
        str(EXE),
        "-m",
        str(MODEL),
        "-p",
        PROMPT,
        "-n",
        "8",
        "-c",
        "128",
        "-ngl",
        "99",
        "-ub",
        "1",
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
            timeout=300,
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

    log_path = OUT_DIR / f"{name}.log"
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    parsed = parse_output(text)
    parsed.update(
        {
            "model": "deepseek_coder_v2_lite",
            "mode": name,
            "target_experts": target_experts or "",
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_s": f"{elapsed:.1f}",
            "peak_gpu_memory_mb": peak["value"],
            "env": " ".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith("EVM_")),
            "log": str(log_path),
        }
    )
    return parsed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in ("", "64", "160"):
        name = "gpu_backed_evm_33"
        if target:
            name += f"_target_{target}"
        else:
            name += "_no_target"
        print(f"running {name}")
        rows.append(run_case(name, target))

    fieldnames = [
        "model",
        "mode",
        "target_experts",
        "returncode",
        "timed_out",
        "elapsed_s",
        "peak_gpu_memory_mb",
        "prompt_tps",
        "generation_tps",
        "cache_hits",
        "cache_misses",
        "hit_rate_pct",
        "bytes_transferred_mb",
        "intercepts",
        "intercept_expert_counts",
        "cuda_error",
        "artifact",
        "env",
        "log",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
