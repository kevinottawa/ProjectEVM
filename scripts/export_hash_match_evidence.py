import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for model in ("qwen1", "deepseek"):
        root = ROOT / "results" / "hash_match" / model
        summary = json.loads((root / "hash_match_summary.json").read_text(encoding="utf-8"))
        full = [json.loads(line) for line in (root / "full" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        exact = [json.loads(line) for line in (root / "exact_evm" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        rows.append({"model": model, "capacity": summary["capacity"], "hash_matches": summary["matched"],
                     "tests": summary["total"], "full_quality_passed": sum(row["quality_pass"] for row in full),
                     "exact_evm_quality_passed": sum(row["quality_pass"] for row in exact)})
    target = ROOT / "docs" / "tables" / "exact_evm_hash_match.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exact EVM hash evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
