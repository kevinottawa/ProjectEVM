import csv
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
OUT_DIR = ROOT / "results" / "final_proof"
PROMPT = "Explain virtual memory in operating systems in two concise paragraphs."


def parse_output(text):
    prompt_ts = 0.0
    gen_ts = 0.0
    hits = 0
    misses = 0
    bytes_transferred = 0

    cli_match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    if cli_match:
        prompt_ts = float(cli_match.group(1))
        gen_ts = float(cli_match.group(2))

    prompt_match = re.search(
        r"prompt eval time =\s*[\d.]+ ms /\s*\d+ tokens\s*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    if prompt_match and prompt_ts == 0.0:
        prompt_ts = float(prompt_match.group(1))

    eval_match = re.search(
        r"eval time =\s*[\d.]+ ms /\s*\d+ runs\s*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    if eval_match and gen_ts == 0.0:
        gen_ts = float(eval_match.group(1))

    for match in re.finditer(r"Cache Hits\s*:\s*(\d+)", text):
        hits += int(match.group(1))
    for match in re.finditer(r"Cache Misses\s*:\s*(\d+)", text):
        misses += int(match.group(1))
    for match in re.finditer(r"Bytes Transferred:\s*(\d+)", text):
        bytes_transferred += int(match.group(1))

    return prompt_ts, gen_ts, hits, misses, bytes_transferred


def run_case(name, model, extra_args=None, env_updates=None, timeout=360):
    env = os.environ.copy()
    env.update(env_updates or {})
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        str(EXE),
        "-m", str(model),
        "-p", PROMPT,
        "-n", "64",
        "-c", "256",
        "-ngl", "99",
        "-ub", "4",
        "--temp", "0.0",
    ]
    if extra_args:
        cmd.extend(extra_args)

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
        timeout=timeout,
    )
    elapsed = time.time() - t0
    full_output = proc.stdout + "\n" + proc.stderr
    log_path = OUT_DIR / f"{name}.log"
    log_path.write_text(full_output, encoding="utf-8", errors="ignore")

    prompt_ts, gen_ts, hits, misses, bytes_transferred = parse_output(full_output)
    total = hits + misses
    hit_rate = 100.0 * hits / total if total else 0.0
    cuda_error = "CUDA error" in full_output or "illegal memory access" in full_output
    instant_eos = gen_ts > 1000.0 or "Instant EOS" in full_output

    return {
        "name": name,
        "returncode": proc.returncode,
        "elapsed_s": f"{elapsed:.1f}",
        "prompt_tps": f"{prompt_ts:.2f}",
        "generation_tps": f"{gen_ts:.2f}",
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": f"{hit_rate:.2f}",
        "bytes_transferred_mb": f"{bytes_transferred / (1024 * 1024):.2f}",
        "cuda_error": cuda_error,
        "instant_eos_or_artifact": instant_eos,
        "log": str(log_path),
        "command": " ".join(cmd),
        "env": " ".join(f"{k}={v}" for k, v in sorted((env_updates or {}).items())),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qwen_small = ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
    qwen_large = ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf"

    cases = [
        {
            "name": "qwen15_baseline_native",
            "model": qwen_small,
            "env_updates": {"EVM_DISABLE": "1"},
        },
        {
            "name": "qwen15_evm_100",
            "model": qwen_small,
            "env_updates": {"EVM_CAPACITY_PCT": "100", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"},
        },
        {
            "name": "qwen15_evm_80",
            "model": qwen_small,
            "env_updates": {"EVM_CAPACITY_PCT": "80", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"},
        },
        {
            "name": "qwen15_evm_66",
            "model": qwen_small,
            "env_updates": {"EVM_CAPACITY_PCT": "66", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"},
        },
        {
            "name": "qwen15_evm_33",
            "model": qwen_small,
            "env_updates": {"EVM_CAPACITY_PCT": "33", "EVM_TARGET_EXPERT_COUNT": "60", "EVM_PREFILL_BATCH_THRESHOLD": "1"},
        },
    ]

    if qwen_large.exists():
        cases.append({
            "name": "qwen2_57b_cpu_backed_evm_33_smoke",
            "model": qwen_large,
            "env_updates": {"EVM_CAPACITY_PCT": "33", "EVM_TARGET_EXPERT_COUNT": "64", "EVM_PREFILL_BATCH_THRESHOLD": "1", "EVM_CPU_BACKING": "1"},
            "timeout": 720,
        })

    rows = []
    for case in cases:
        print(f"Running {case['name']}...")
        rows.append(run_case(**case))

    csv_path = OUT_DIR / "runtime_proof.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
