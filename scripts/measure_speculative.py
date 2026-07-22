import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
TARGET = ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf"
DRAFT = ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
OUT_DIR = ROOT / "results" / "final_proof"
LOG_PATH = OUT_DIR / "speculative_smoke.log"


def parse_llama_output(text):
    cli_match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    eval_match = re.search(
        r"eval time =\s*[\d.]+ ms /\s*\d+ runs\s*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)",
        text,
    )
    gen_tps = float(cli_match.group(2)) if cli_match else float(eval_match.group(1)) if eval_match else 0.0
    return {
        "generation_tps": gen_tps,
        "cuda_error": "CUDA error" in text or "illegal memory access" in text,
        "evm_counters": "[EVM Metrics]" in text,
    }


def run_speculative_smoke(timeout=720):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "EVM_CAPACITY_PCT": "66",
            "EVM_TARGET_EXPERT_COUNT": "64",
            "EVM_CPU_BACKING": "1",
            "EVM_PREFILL_BATCH_THRESHOLD": "1",
        }
    )

    cmd = [
        str(EXE),
        "-m",
        str(TARGET),
        "-md",
        str(DRAFT),
        "--spec-draft-n-max",
        "5",
        "-p",
        "Describe how a virtual memory system handles page faults and eviction.",
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
    LOG_PATH.write_text(full_output, encoding="utf-8", errors="ignore")
    stats = parse_llama_output(full_output)

    print(f"returncode={proc.returncode}")
    print(f"elapsed_s={elapsed:.1f}")
    print(f"generation_tps={stats['generation_tps']:.2f}")
    print(f"cuda_error={stats['cuda_error']}")
    print(f"evm_counters={stats['evm_counters']}")
    print(f"log={LOG_PATH}")


if __name__ == "__main__":
    run_speculative_smoke()
