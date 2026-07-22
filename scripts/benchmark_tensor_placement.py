import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-bench.exe"


def gpu_mb():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return int(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else 0


def run_case(name, model, extra, repetitions, tokens, out_dir):
    command = [str(EXE), "-m", str(model), "-p", "0", "-n", str(tokens),
               "-r", str(repetitions), "-o", "json", *extra]
    baseline_gpu = gpu_mb()
    peak_gpu = baseline_gpu
    peak_rss = 0
    stop = False
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="ignore")
    tracked = psutil.Process(process.pid)

    def monitor():
        nonlocal peak_gpu, peak_rss
        while not stop:
            peak_gpu = max(peak_gpu, gpu_mb())
            try:
                peak_rss = max(peak_rss, tracked.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    stdout, stderr = process.communicate()
    stop = True
    thread.join(timeout=2)
    (out_dir / f"{name}.log").write_text(stderr, encoding="utf-8")
    try:
        payload = json.loads(stdout)
        row = payload[0] if isinstance(payload, list) else payload
    except (json.JSONDecodeError, IndexError):
        row = {}
    return {
        "name": name,
        "pass": process.returncode == 0 and bool(row),
        "generation_tps": round(float(row.get("avg_ts", 0)), 2),
        "stddev_tps": round(float(row.get("stddev_ts", 0)), 2),
        "peak_vram_mb": peak_gpu,
        "incremental_vram_mb": max(0, peak_gpu - baseline_gpu),
        "peak_process_rss_mb": round(peak_rss / 1048576),
        "tensor_override": row.get("tensor_buft_overrides", "none"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "tensor_placement"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("full_gpu", ["-ngl", "99"]),
        ("cpu_spine_gpu_experts", ["-ngl", "0", "-ot", ".*_exps.*=CUDA0"]),
        ("full_cpu_layers", ["-ngl", "0"]),
    ]
    rows = [run_case(name, Path(args.model), extra, args.repetitions, args.tokens, out_dir)
            for name, extra in cases]
    output = out_dir / "summary.json"
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        status = "PASS" if row["pass"] else "FAIL"
        print(f'{row["name"]}: {row["generation_tps"]:.2f} +/- {row["stddev_tps"]:.2f} t/s | '
              f'{row["peak_vram_mb"]} MB VRAM | {row["peak_process_rss_mb"]} MB RSS | {status}')
    raise SystemExit(0 if all(row["pass"] for row in rows) else 1)


if __name__ == "__main__":
    main()
