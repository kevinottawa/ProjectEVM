import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "results" / "pack_refinement" / "refinement_summary.json"
    decision = ROOT / "results" / "pack_refinement" / "decision_summary.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    decisions = json.loads(decision.read_text(encoding="utf-8"))
    rows = []
    for model, capacities in data["models"].items():
        for label, row in capacities.items():
            for method in ("frequency", "core_overlay"):
                score = row[method]
                rows.append({"model": model, "capacity_pct": float(label.replace("p", ".")), "method": method,
                             "baseline_passed": row["baseline_passed"], "total": row["total"], "passed": score["passed"],
                             "retained_baseline_passes": score["retained_baseline_passes"], "baseline_opportunities": score["baseline_opportunities"],
                             "decision": decisions["models"][model]["status"]})
    path = ROOT / "docs" / "tables" / "small_model_pack_refinement.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Pack refinement evidence: {len(rows)} rows | PASS")


if __name__ == "__main__":
    main()
