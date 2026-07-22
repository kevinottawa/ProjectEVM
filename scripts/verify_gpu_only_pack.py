import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify a GPU-resident EVM pack-only benchmark row.")
    parser.add_argument("rows", nargs="+", type=Path)
    args = parser.parse_args()
    failures = []
    summaries = []
    for path in args.rows:
        row = json.loads(path.read_text(encoding="utf-8"))
        env = row.get("evm_env", {})
        checks = {
            "valid_process": row.get("returncode") == 0 and not row.get("timed_out") and not row.get("budget_exceeded"),
            "cuda_counters": row.get("has_evm_counters") is True,
            "gpu_pack_mode": env.get("EVM_GPU_PACK_ONLY") == "1",
            "no_cpu_backing": "EVM_CPU_BACKING" not in env,
            "no_exact_fallback": env.get("EVM_ABILITY_PACK_ONLY") == "1",
            "gpu_kv": row.get("kv") == "gpu",
            "zero_pagefile_growth": row.get("pagefile_used_delta_mb", 1) == 0,
            "strict_budget": env.get("EVM_STRICT_BUDGET") == "1",
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append(f"{path.name}: {','.join(failed)}")
        summaries.append({"row": path.name, "status": "PASS" if not failed else "FAIL",
                          "generation_tps": row.get("generation_tps", 0),
                          "peak_vram_mb": row.get("peak_gpu_memory_mb", 0),
                          "peak_rss_mb": row.get("peak_process_rss_mb", 0),
                          "end_rss_mb": row.get("end_process_rss_mb", 0),
                          "pagefile_delta_mb": row.get("pagefile_used_delta_mb", 0)})
    for row in summaries:
        print(f"{row['row']}: {row['generation_tps']:.2f} t/s | {row['peak_vram_mb']} MB VRAM | RSS {row['peak_rss_mb']} peak/{row['end_rss_mb']} end MB | pagefile +{row['pagefile_delta_mb']} MB | {row['status']}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
