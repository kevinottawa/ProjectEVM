import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = json.loads((ROOT / "results" / "learned_scheduler_runtime" / "online_predictor_summary.json").read_text(encoding="utf-8"))
    by_key = {(row["model"], row["mode"]): row for row in runtime}
    rows = []
    for model in ("deepseek", "qwen1"):
        lru = by_key[(model, "lru")]
        learned = by_key[(model, "learned")]
        hashes = json.loads((ROOT / "results" / "learned_scheduler_hash" / model / "hash_match_summary.json").read_text(encoding="utf-8"))
        rows.append({
            "model": model,
            "trials": learned["valid_trials"],
            "lru_tps": lru["mean_generation_tps"],
            "learned_tps": learned["mean_generation_tps"],
            "tps_change_pct": round(100 * (learned["mean_generation_tps"] / lru["mean_generation_tps"] - 1), 2),
            "lru_vram_mb": lru["mean_peak_vram_mb"],
            "learned_vram_mb": learned["mean_peak_vram_mb"],
            "lru_hit_rate_pct": lru["mean_hit_rate_pct"],
            "learned_hit_rate_pct": learned["mean_hit_rate_pct"],
            "exact_hash_matches": f"{hashes['matched']}/{hashes['total']}",
        })
    target = ROOT / "docs" / "tables" / "learned_scheduler_runtime.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Learned scheduler runtime evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
