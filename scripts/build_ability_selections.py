import argparse
import json
import re
from pathlib import Path


TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")


def main():
    parser = argparse.ArgumentParser(description="Build deterministic per-layer ability selections from EVM routing counts.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--sizes", default="8,16,24,32")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    counts = {}
    with args.profile.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            match = TENSOR_RE.match(row["tensor"])
            if not match:
                continue
            layer = int(match.group(1))
            values = [int(value) for value in row["counts"]]
            aggregate = counts.setdefault(layer, [0] * len(values))
            if len(aggregate) != len(values):
                raise ValueError(f"expert count changed at layer {layer}")
            for expert, value in enumerate(values):
                aggregate[expert] += value
    if not counts:
        raise SystemExit("profile contains no down-expert routing rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for size in [int(value) for value in args.sizes.split(",") if value.strip()]:
        selected = {}
        covered = total = 0
        for layer in sorted(counts):
            values = counts[layer]
            ranking = sorted(range(len(values)), key=lambda expert: (-values[expert], expert))
            chosen = sorted(ranking[:size])
            selected[str(layer)] = chosen
            covered += sum(values[expert] for expert in chosen)
            total += sum(values)
        payload = {
            "format": "evm-ability-selection-v1",
            "experts_per_layer": size,
            "profile": str(args.profile),
            "profile_accesses": total,
            "covered_accesses": covered,
            "profile_coverage_pct": round(100.0 * covered / total, 2) if total else 0.0,
            "selected": selected,
        }
        path = args.out_dir / f"selection_{size}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary.append({"size": size, "coverage_pct": payload["profile_coverage_pct"], "path": str(path)})
    (args.out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(f"pack_{row['size']}: profile coverage {row['coverage_pct']:.2f}% | PASS")


if __name__ == "__main__":
    main()
