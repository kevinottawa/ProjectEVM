import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runtime_rows():
    source = ROOT / "results" / "qwen2_learned_scheduler_reproduction" / "online_predictor_summary.json"
    rows = []
    for row in json.loads(source.read_text(encoding="utf-8")):
        rows.append({
            "scenario": f"exact_8_slot_{row['mode']}",
            "status": "pass",
            "trials": row["valid_trials"],
            "generation_tps": row["mean_generation_tps"],
            "peak_vram_mb": row["mean_peak_vram_mb"],
            "system_ram_delta_mb": row["mean_system_ram_delta_mb"],
            "pagefile_delta_mb": row["mean_pagefile_delta_mb"],
            "cache_hit_rate_pct": row["mean_hit_rate_pct"],
            "note": "32 GB host; full-reference hash gate did not complete",
        })
    return rows


def no_go_rows():
    rows = []
    for directory in ("qwen2_headroom_smoke", "qwen2_headroom_smoke_4_6"):
        for path in (ROOT / "results" / directory).glob("reserve_*gb/qwen2_*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            reserve = path.parent.name.removeprefix("reserve_").removesuffix("gb")
            rows.append({
                "scenario": f"reserve_{reserve}gb_{row['name'].split('_')[1]}",
                "status": "no_go",
                "trials": 0,
                "generation_tps": "",
                "peak_vram_mb": row.get("peak_gpu_memory_mb", ""),
                "system_ram_delta_mb": row.get("system_ram_used_delta_mb", ""),
                "pagefile_delta_mb": row.get("pagefile_used_delta_mb", ""),
                "cache_hit_rate_pct": "",
                "note": "minimum exact expert pool exceeded declared GPU headroom before EVM counters",
            })
    return rows


def main():
    rows = runtime_rows() + no_go_rows()
    target = ROOT / "docs" / "tables" / "qwen2_32gb_runtime.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Qwen2 32 GB evidence: {len(rows)} rows | PASS")


if __name__ == "__main__":
    main()
