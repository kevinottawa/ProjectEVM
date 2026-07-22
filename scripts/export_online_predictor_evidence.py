import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def index(rows):
    return {(row["model"], row["mode"]): row for row in rows}


def main():
    baseline = index(json.loads((ROOT / "results" / "online_predictor" / "online_predictor_summary.json").read_text(encoding="utf-8")))
    tuned = index(json.loads((ROOT / "results" / "online_predictor_confidence" / "online_predictor_summary.json").read_text(encoding="utf-8")))
    rows = []
    for model in ("qwen1", "deepseek"):
        lru = baseline[(model, "lru")]
        predictor = tuned[(model, "predictor")]
        rows.append({"model": model, "lru_tps": lru["mean_generation_tps"], "predictor_tps": predictor["mean_generation_tps"],
                     "tps_change_pct": round(100 * (predictor["mean_generation_tps"] / lru["mean_generation_tps"] - 1), 2),
                     "lru_hit_rate_pct": lru["mean_hit_rate_pct"], "predictor_hit_rate_pct": predictor["mean_hit_rate_pct"],
                     "lru_transfer_mb": lru["mean_bytes_transferred_mb"], "predictor_transfer_mb": predictor["mean_bytes_transferred_mb"],
                     "predictor_prefetches": predictor["mean_predictor_prefetches"], "predictor_hits": predictor["mean_predictor_hits"],
                     "peak_vram_mb": predictor["mean_peak_vram_mb"]})
    target = ROOT / "docs" / "tables" / "online_predictor_small_models.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Online predictor evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
