import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
OUT_DIR = ROOT / "results" / "final_workflow" / "max_context"
CSV_PATH = ROOT / "results" / "final_workflow" / "max_context.csv"
PROMPT = "Briefly define virtual memory."
CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


MODELS = [
    {
        "id": "qwen15_moe",
        "path": ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf",
        "target_experts": "60",
        "timeout": 240,
    },
    {
        "id": "deepseek_coder_v2_lite",
        "path": ROOT / "models" / "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf",
        "target_experts": "64",
        "timeout": 360,
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


def parse_perf(text):
    match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    return (
        float(match.group(1)) if match else 0.0,
        float(match.group(2)) if match else 0.0,
    )


def run_case(model, mode, ctx_len):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("EVM_CPU_BACKING", None)
    env.pop("EVM_DISABLE", None)
    env.pop("EVM_TARGET_EXPERT_COUNT", None)
    env.pop("EVM_CAPACITY_PCT", None)
    env.pop("EVM_PREFILL_BATCH_THRESHOLD", None)

    if mode == "native_gpu":
        env["EVM_DISABLE"] = "1"
    elif mode == "cpu_backed_evm_33":
        env["EVM_CAPACITY_PCT"] = "33"
        env["EVM_CPU_BACKING"] = "1"
        env["EVM_PREFILL_BATCH_THRESHOLD"] = "1"
        env["EVM_TARGET_EXPERT_COUNT"] = model["target_experts"]
    else:
        raise ValueError(mode)

    cmd = [
        str(EXE),
        "-m",
        str(model["path"]),
        "-p",
        PROMPT,
        "-n",
        "8",
        "-c",
        str(ctx_len),
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{model['id']}_{mode}_ctx{ctx_len}.log"
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    prompt_tps, generation_tps = parse_perf(text)
    oom = "out of memory" in text.lower() or "cuda error" in text.lower() or "failed to allocate" in text.lower()
    artifact = generation_tps > 1000.0 or "Instant EOS" in text
    success = returncode == 0 and not timed_out and not oom and not artifact and generation_tps > 0.0

    return {
        "model": model["id"],
        "mode": mode,
        "ctx_len": ctx_len,
        "success": success,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": f"{elapsed:.1f}",
        "peak_gpu_memory_mb": peak["value"],
        "prompt_tps": f"{prompt_tps:.2f}",
        "generation_tps": f"{generation_tps:.2f}",
        "oom_or_cuda_error": oom,
        "artifact": artifact,
        "env": " ".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith("EVM_")),
        "log": str(log_path),
    }


def main():
    rows = []
    for model in MODELS:
        if not model["path"].exists():
            continue
        for mode in ("native_gpu", "cpu_backed_evm_33"):
            for ctx_len in CONTEXTS:
                print(f"running {model['id']} {mode} ctx={ctx_len}")
                row = run_case(model, mode, ctx_len)
                rows.append(row)
                if not row["success"]:
                    break

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
