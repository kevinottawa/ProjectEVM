import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("deepseek", "qwen1", "qwen2")


def main():
    rows = []
    for model in MODELS:
        base = ROOT / "results" / "predictor_training" / model
        capture = json.loads((base / "trace" / "capture_summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((base / "model" / "training_metadata.json").read_text(encoding="utf-8"))
        with (base / "model" / "layer_aware_predictor_results.csv").open(newline="", encoding="utf-8") as handle:
            policies = list(csv.DictReader(handle))
        capacity = min(int(row["capacity"]) for row in policies)
        selected = {row["policy"]: row for row in policies if int(row["capacity"]) == capacity}
        lru = selected["LRU"]
        learned = selected["Learned3to17"]
        oracle = selected["Oracle"]
        gap = float(oracle["hit_rate"]) - float(lru["hit_rate"])
        closed = 0.0 if gap <= 0 else (float(learned["hit_rate"]) - float(lru["hit_rate"])) / gap
        rows.append({
            "model": model,
            "valid_prompts": capture["valid_prompts"],
            "trace_rows": capture["trace_rows"],
            "layers": metadata["layers"],
            "experts": metadata["experts"],
            "test_capacity": capacity,
            "lru_hit_rate_pct": round(100 * float(lru["hit_rate"]), 2),
            "learned_hit_rate_pct": round(100 * float(learned["hit_rate"]), 2),
            "oracle_hit_rate_pct": round(100 * float(oracle["hit_rate"]), 2),
            "gap_closed_pct": round(100 * closed, 2),
            "selected_prefetch_budget": learned["prefetch_budget"],
            "deployment_status": "offline_policy_evaluation_only",
        })

    target = ROOT / "docs" / "tables" / "layer_aware_predictor_training.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Layer-aware predictor evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
