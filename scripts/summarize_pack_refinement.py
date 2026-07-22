import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Derive GO/NO-GO pack decisions from refinement evidence.")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "pack_refinement" / "refinement_summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pack_refinement" / "decision_summary.json")
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    decisions = {"format": "evm-pack-refinement-decision-v1", "models": {}}
    output = []
    for model, capacities in data["models"].items():
        candidates = []
        for label, row in capacities.items():
            baseline = row["baseline_passed"]
            for method in ("frequency", "core_overlay"):
                score = row[method]
                accepted = baseline > 0 and score["retained_baseline_passes"] == baseline and score["passed"] >= baseline
                candidates.append({"capacity": float(label.replace("p", ".")), "method": method, "accepted": accepted,
                                   "baseline_passed": baseline, "passed": score["passed"],
                                   "retained_baseline_passes": score["retained_baseline_passes"]})
        accepted = sorted((row for row in candidates if row["accepted"]), key=lambda row: (row["capacity"], row["method"]))
        decision = accepted[0] if accepted else None
        decisions["models"][model] = {"status": "GO" if decision else "NO-GO", "adopted": decision, "candidates": candidates}
        if decision:
            output.append(f"{model}: GO {decision['method']} {decision['capacity']:.1f}% ({decision['passed']}/{decision['baseline_passed']})")
        else:
            output.append(f"{model}: NO-GO")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
    print("Pack decisions: " + " | ".join(output))


if __name__ == "__main__":
    main()
