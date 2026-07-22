import argparse
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_budgeted_llama.py"
MODEL = ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf"
DRAFT = ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
OUT_DIR = ROOT / "results" / "production_evm"

DISK_COMPACT_ENV = [
    "EVM_CAPACITY_PCT=13",
    "EVM_EXPERTS_PER_TENSOR=8",
    "EVM_TARGET_EXPERT_COUNT=64",
    "EVM_CPU_BACKING=1",
    "EVM_CUDA_STREAMING=1",
    "EVM_DISK_BACKING=1",
    "EVM_CUDA_FREE_RESERVE_MB=768",
    "EVM_PREFILL_BATCH_THRESHOLD=999",
    "EVM_STRICT_BUDGET=1",
]


CASES = {
    "cpu_backed": {
        "require_counters": False,
        "env": [
            "EVM_CAPACITY_PCT=33",
            "EVM_TARGET_EXPERT_COUNT=64",
            "EVM_CPU_BACKING=1",
        ],
    },
    "compact_async": {
        "require_counters": True,
        "env": [
            "EVM_CAPACITY_PCT=13",
            "EVM_EXPERTS_PER_TENSOR=8",
            "EVM_TARGET_EXPERT_COUNT=64",
            "EVM_CPU_BACKING=1",
            "EVM_CUDA_STREAMING=1",
            "EVM_CUDA_FREE_RESERVE_MB=768",
            "EVM_PREFILL_BATCH_THRESHOLD=999",
            "EVM_STRICT_BUDGET=1",
        ],
    },
    "disk_compact_async": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_TRIM=1", "EVM_DISK_TRIM_INTERVAL=1"],
    },
    "disk_compact_trim4": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_TRIM=1", "EVM_DISK_TRIM_INTERVAL=4"],
    },
    "disk_compact_trim8": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_TRIM=1", "EVM_DISK_TRIM_INTERVAL=8"],
    },
    "disk_compact_lazy": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": DISK_COMPACT_ENV,
    },
    "disk_staged2": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_STAGING=1", "EVM_DISK_STAGING_SLOTS=2", "EVM_DISK_TRIM=1"],
    },
    "disk_staged4": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_STAGING=1", "EVM_DISK_STAGING_SLOTS=4", "EVM_DISK_TRIM=1"],
    },
    "disk_staged8": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_STAGING=1", "EVM_DISK_STAGING_SLOTS=8", "EVM_DISK_TRIM=1"],
    },
    "disk_cache8g": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_TRIM=1", "EVM_DISK_CACHE_MB=8192"],
    },
    "disk_cache12g": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_TRIM=1", "EVM_DISK_CACHE_MB=12288"],
    },
    "disk_cache16g": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_DISK_TRIM=1", "EVM_DISK_CACHE_MB=16384"],
    },
    "disk_pool16": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_EXPERTS_PER_TENSOR=16", "EVM_DISK_TRIM=1"],
    },
    "disk_pool24": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_EXPERTS_PER_TENSOR=24", "EVM_DISK_TRIM=1"],
    },
    "disk_pool32": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_EXPERTS_PER_TENSOR=32", "EVM_DISK_TRIM=1"],
    },
    "disk_pool40": {
        "require_counters": True,
        "mmap": True,
        "ub": "1",
        "env": [*DISK_COMPACT_ENV, "EVM_EXPERTS_PER_TENSOR=40", "EVM_DISK_TRIM=1"],
    },
    "compact_speculative": {
        "require_counters": True,
        "speculative": True,
        "env": [
            "EVM_CAPACITY_PCT=13",
            "EVM_EXPERTS_PER_TENSOR=8",
            "EVM_TARGET_EXPERT_COUNT=64",
            "EVM_CPU_BACKING=1",
            "EVM_CUDA_STREAMING=1",
            "EVM_CUDA_FREE_RESERVE_MB=768",
            "EVM_PREFILL_BATCH_THRESHOLD=999",
            "EVM_STRICT_BUDGET=1",
        ],
    },
}


def extract_json(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("runner emitted no JSON summary")
    return json.loads(text[start:])


def run(case_name, trial, tokens, timeout):
    case = CASES[case_name]
    name = f"qwen2_{case_name}_t{trial:02d}"
    exe = ROOT / "llama.cpp" / "build" / "bin" / "Release" / (
        "llama-speculative.exe" if case.get("speculative") else "llama-cli.exe"
    )
    cmd = [
        "python", str(RUNNER),
        "--name", name,
        "--model", str(MODEL),
        "--tokens", str(tokens),
        "--ctx", "256",
        "--ub", case.get("ub", "1" if case_name.startswith("compact") else "4"),
        "--kv", "gpu",
        "--gpu-budget-mb", "24000",
        "--timeout-s", str(timeout),
        "--out-dir", str(OUT_DIR / "runs"),
        "--exe", str(exe),
    ]
    if case["require_counters"]:
        cmd.append("--require-evm-counters")
    for item in case["env"]:
        cmd.extend(["--env", item])
    cmd.extend(["--", "--mmap" if case.get("mmap") else "--no-mmap", "--ignore-eos"])
    if case.get("speculative"):
        cmd.extend([
            "--spec-type", "draft-simple",
            "--spec-draft-model", str(DRAFT),
            "--spec-draft-ngl", "99",
            "--spec-draft-n-max", "5",
            "--spec-draft-n-min", "1",
        ])

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout + 90,
    )
    try:
        row = extract_json(proc.stdout)
    except Exception:
        row = {
            "name": name,
            "returncode": proc.returncode,
            "generation_tps": 0.0,
            "peak_gpu_memory_mb": 0,
            "timed_out": True,
            "budget_exceeded": False,
            "has_evm_counters": False,
        }
    row["case"] = case_name
    row["trial"] = trial
    row["pass"] = (
        proc.returncode == 0
        and row.get("returncode") == 0
        and not row.get("timed_out")
        and not row.get("budget_exceeded")
        and row.get("generation_tps", 0.0) > 0.0
        and (not case["require_counters"] or row.get("has_evm_counters"))
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_name in args.cases:
        case_rows = [run(case_name, trial, args.tokens, args.timeout_s) for trial in range(1, args.trials + 1)]
        rows.extend(case_rows)
        valid = [row for row in case_rows if row["pass"]]
        speed = statistics.mean(row["generation_tps"] for row in valid) if valid else 0.0
        peak = max((row["peak_gpu_memory_mb"] for row in valid), default=0)
        host = max((row.get("peak_process_private_mb", 0) for row in valid), default=0)
        status = "PASS" if len(valid) == len(case_rows) else "FAIL"
        print(f"{case_name}: {speed:.2f} t/s | VRAM {peak:,} MB | host commit {host:,} MB | {status}")

    suffix = "_".join(args.cases)
    (OUT_DIR / f"production_benchmark_{suffix}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if args.cases == list(CASES):
        (OUT_DIR / "production_benchmark.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
