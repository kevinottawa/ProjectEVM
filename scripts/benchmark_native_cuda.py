import argparse
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_budgeted_llama.py"
MODEL = ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf"
OUT_DIR = ROOT / "results" / "native_cuda_tuning"

CASES = {
    "auto_ub4": {"ub": "4", "extra": ["--flash-attn", "auto", "--mmap"]},
    "fa_on_ub1": {"ub": "1", "extra": ["--flash-attn", "on", "--mmap"]},
    "fa_on_ub4": {"ub": "4", "extra": ["--flash-attn", "on", "--mmap"]},
    "fa_off_ub1": {"ub": "1", "extra": ["--flash-attn", "off", "--mmap"]},
}


def extract_json(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("runner emitted no JSON")
    return json.loads(text[start:])


def run_case(case_name, trial, tokens, timeout):
    case = CASES[case_name]
    name = f"qwen2_native_{case_name}_t{trial:02d}"
    cmd = [
        "python", str(RUNNER),
        "--name", name,
        "--model", str(MODEL),
        "--tokens", str(tokens),
        "--ctx", "256",
        "--ngl", "99",
        "--ub", case["ub"],
        "--kv", "gpu",
        "--gpu-budget-mb", "24500",
        "--timeout-s", str(timeout),
        "--out-dir", str(OUT_DIR / "runs"),
        "--env", "EVM_DISABLE=1",
        "--", "--ignore-eos", *case["extra"],
    ]
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
    row = extract_json(proc.stdout)
    row["case"] = case_name
    row["trial"] = trial
    row["pass"] = (
        proc.returncode == 0
        and row.get("returncode") == 0
        and not row.get("timed_out")
        and not row.get("budget_exceeded")
        and row.get("generation_tps", 0) > 0
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
        current = [run_case(case_name, trial, args.tokens, args.timeout_s) for trial in range(1, args.trials + 1)]
        rows.extend(current)
        valid = [row for row in current if row["pass"]]
        speed = statistics.mean(row["generation_tps"] for row in valid) if valid else 0.0
        vram = max((row["peak_gpu_memory_mb"] for row in valid), default=0)
        host = max((row.get("peak_process_private_mb", 0) for row in valid), default=0)
        status = "PASS" if len(valid) == len(current) else "FAIL"
        print(f"{case_name}: {speed:.2f} t/s | VRAM {vram:,} MB | host commit {host:,} MB | {status}")

    (OUT_DIR / "native_cuda_benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
