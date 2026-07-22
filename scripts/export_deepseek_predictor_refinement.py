import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("initial_35", ROOT / "results" / "online_predictor" / "online_predictor_summary.json"),
    ("confidence_50", ROOT / "results" / "deepseek_predictor_refine_cap8_c50" / "online_predictor_summary.json"),
    ("capacity10_confidence_50", ROOT / "results" / "deepseek_predictor_refine_cap10_c50" / "online_predictor_summary.json"),
)


def main():
    rows = []
    for label, path in SOURCES:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["model"] != "deepseek":
                continue
            rows.append({"configuration": label, "mode": row["mode"], "capacity": row.get("capacity", 8),
                         "confidence_pct": row.get("confidence_pct", 35 if row["mode"] == "predictor" else ""),
                         "trials": row["trials"], "generation_tps": row["mean_generation_tps"],
                         "peak_vram_mb": row["mean_peak_vram_mb"], "hit_rate_pct": row["mean_hit_rate_pct"],
                         "transfer_mb": row["mean_bytes_transferred_mb"], "prefetches": row["mean_predictor_prefetches"],
                         "predictor_hits": row["mean_predictor_hits"]})
    target = ROOT / "docs" / "tables" / "deepseek_predictor_refinement.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"DeepSeek predictor refinement: {len(rows)} rows | PASS")


if __name__ == "__main__":
    main()
