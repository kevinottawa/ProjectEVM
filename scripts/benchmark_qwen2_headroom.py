import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "benchmark_online_predictor.py"


def main():
    parser = argparse.ArgumentParser(description="Run exact Qwen2 EVM VRAM-headroom trials with compact summaries.")
    parser.add_argument("--reserves", default="4,6,8,12", help="comma-separated GiB held free on the GPU")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_valid = True
    for reserve in (int(value) for value in args.reserves.split(",") if value.strip()):
        run_dir = args.out_dir / f"reserve_{reserve}gb"
        command = [sys.executable, str(BENCHMARK), "--models", "qwen2", "--modes", "lru,learned",
                   "--reserve-vram-gb", str(reserve), "--trials", str(args.trials), "--tokens", str(args.tokens),
                   "--out-dir", str(run_dir)]
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        summary_path = run_dir / "online_predictor_summary.json"
        rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
        summaries.extend(rows)
        valid = all(row.get("valid_trials") == row.get("trials") for row in rows)
        print(f"reserve={reserve}GB valid={valid} rows={len(rows)}")
        if completed.returncode or not valid:
            all_valid = False
    (args.out_dir / "qwen2_headroom_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    for row in summaries:
        print(f"reserve={row['reserve_vram_gb']}GB {row['mode']}: {row['mean_generation_tps']:.2f} t/s | VRAM {row['mean_peak_vram_mb']} MB | RAM delta {row['mean_system_ram_delta_mb']} MB | pagefile delta {row['mean_pagefile_delta_mb']} MB")
    raise SystemExit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
