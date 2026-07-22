import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOADS = ROOT / "config" / "mri_diagnostic_payloads.json"
sys.path.insert(0, str(ROOT / "scripts"))
from build_expert_vault import build_manifest  # noqa: E402


TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up|gate_up)_exps\.weight$")
ROLE_PRIORITY = {"down": 0, "gate_up": 1, "gate": 2, "up": 3}


def scan_model(model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(model, include_hash=False)
    scan = {
        "format": "moe-mri-scan-v1",
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "layers": manifest["layer_count"],
        "moe_layer_ids": manifest["layer_ids"],
        "experts_per_layer": manifest["expert_count"],
        "expert_tensor_count": manifest["expert_tensor_count"],
        "routed_expert_bytes": manifest["total_expert_bytes"],
        "estimated_spine_bytes": model.stat().st_size - manifest["total_expert_bytes"],
        "roles": sorted({row["role"] for row in manifest["tensors"]}),
        "slice_extractable": True,
    }
    (out_dir / "scan.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    (out_dir / "expert-vault-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return scan


def load_profile(path):
    tensors = defaultdict(lambda: None)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            match = TENSOR_RE.match(row["tensor"])
            if not match:
                continue
            layer = int(match.group(1))
            role = match.group(2)
            counts = [int(value) for value in row["counts"]]
            key = (layer, role)
            if tensors[key] is None:
                tensors[key] = [0] * len(counts)
            for expert, value in enumerate(counts):
                tensors[key][expert] += value
    by_layer = {}
    layers = sorted({layer for layer, _ in tensors})
    for layer in layers:
        choices = [(ROLE_PRIORITY[role], counts) for (candidate, role), counts in tensors.items() if candidate == layer]
        if choices:
            by_layer[layer] = min(choices, key=lambda item: item[0])[1]
    return by_layer


def parse_profiles(values):
    result = {}
    for value in values:
        category, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"profile must be CATEGORY=PATH: {value}")
        result[category] = load_profile(Path(path))
    return result


def coverage(selection, profile):
    covered = total = 0
    for layer, counts in profile.items():
        chosen = selection.get(layer, [])
        covered += sum(counts[expert] for expert in chosen)
        total += sum(counts)
    return round(100.0 * covered / total, 2) if total else 0.0


def select(profile, size, fallback_experts):
    selected = {}
    for layer in sorted(profile):
        counts = profile[layer]
        ranking = sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))
        selected[layer] = sorted(ranking[:size])
    return selected


def analyze(scan, profiles, pack_specs, out_dir, payload_suite=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scan.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    categories = sorted(profiles)
    combined = {}
    for layer in scan.get("moe_layer_ids", range(scan["layers"])):
        combined[layer] = [0] * scan["experts_per_layer"]
        for profile in profiles.values():
            for expert, value in enumerate(profile.get(layer, [])):
                combined[layer][expert] += value

    atlas_rows = []
    for layer, counts in combined.items():
        for expert, total_count in enumerate(counts):
            category_counts = {category: profiles[category].get(layer, [0] * len(counts))[expert] for category in categories}
            category_totals = {category: sum(profiles[category].get(layer, [])) for category in categories}
            shares = {category: category_counts[category] / category_totals[category] if category_totals[category] else 0 for category in categories}
            baseline = total_count / sum(counts) if sum(counts) else 0
            lifts = {category: shares[category] / baseline if baseline else 0 for category in categories}
            ranked = sorted(categories, key=lambda category: (lifts[category], category_counts[category]), reverse=True)
            candidate = ranked[0] if ranked else "unlabeled"
            lift = lifts.get(candidate, 0)
            runner_up_lift = lifts.get(ranked[1], 0) if len(ranked) > 1 else 0
            contrast_margin = lift - runner_up_lift
            if total_count >= 100 and lift >= 1.5 and contrast_margin >= 0.25:
                association, confidence = candidate, "high"
            elif total_count >= 25 and lift >= 1.2 and contrast_margin >= 0.10:
                association, confidence = candidate, "medium"
            elif total_count >= 25 and max(lifts.values(), default=0) < 1.2:
                association, confidence = "shared_cross_domain", "medium"
            else:
                association, confidence = "inconclusive", "low"
            domain_description = payload_suite.get("domains", {}).get(association, {}).get("description", association) if payload_suite else association
            atlas_rows.append({
                "layer": layer, "expert": expert, "total_accesses": total_count,
                "association": association, "association_lift": round(lift, 3),
                "contrast_margin": round(contrast_margin, 3), "confidence": confidence,
                "description": f"Layer-{layer} expert routing is associated with {domain_description} ({lift:.2f}x lift, {contrast_margin:.2f} contrast; {confidence} confidence)",
                **{f"{category}_accesses": category_counts[category] for category in categories},
            })
    with (out_dir / "expert_atlas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(atlas_rows[0]))
        writer.writeheader()
        writer.writerows(atlas_rows)

    pack_cards = []
    profile_sets = {"universal": combined, **profiles}
    for category, profile in profile_sets.items():
        for residency_pct, size in pack_specs:
            chosen = select(profile, size, scan["experts_per_layer"])
            coverage_by_category = {name: coverage(chosen, candidate) for name, candidate in profiles.items()}
            pct_label = str(residency_pct).replace(".", "p")
            selection_payload = {
                "format": "evm-ability-selection-v1", "model": scan["model"],
                "name": f"{category}-p{pct_label}", "category": category, "experts_per_layer": size,
                "requested_residency_pct": residency_pct,
                "actual_residency_pct": round(100.0 * size / scan["experts_per_layer"], 2),
                "coverage_pct": coverage_by_category,
                "selected": {str(layer): experts for layer, experts in chosen.items()},
            }
            selection_path = out_dir / f"selection_{category}_p{pct_label}.json"
            selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
            estimated_bytes = round(scan["routed_expert_bytes"] * size / scan["experts_per_layer"])
            pack_cards.append({
                "name": f"{category}-p{pct_label}", "category": category, "experts_per_layer": size,
                "requested_residency_pct": residency_pct,
                "actual_residency_pct": round(100.0 * size / scan["experts_per_layer"], 2),
                "estimated_pack_bytes": estimated_bytes, "coverage_pct": coverage_by_category,
                "intended_use": "broad mixed workloads" if category == "universal" else f"{category} workloads",
                "confidence": "routing association measured by fixed positive/contrast payloads; quality ablation required",
                "selection": str(selection_path),
            })
    (out_dir / "pack_cards.json").write_text(json.dumps(pack_cards, indent=2) + "\n", encoding="utf-8")

    lines = [f"# MoE MRI Report: {Path(scan['model']).name}", "", "## Model", "",
             f"- Layers: {scan['layers']}", f"- Experts per layer: {scan['experts_per_layer']}",
             f"- Routed expert storage: {scan['routed_expert_bytes'] / 1024**3:.2f} GiB",
             f"- Estimated spine: {scan['estimated_spine_bytes'] / 1024**3:.2f} GiB", "", "## Pack Cards", ""]
    for card in pack_cards:
        coverage_text = ", ".join(f"{key} {value:.1f}%" for key, value in card["coverage_pct"].items())
        lines += [f"### {card['name']}", "", f"Intended use: {card['intended_use']}.",
                  f"Resident experts: {card['experts_per_layer']}/{scan['experts_per_layer']} ({card['actual_residency_pct']:.2f}%).",
                  f"Estimated expert storage: {card['estimated_pack_bytes'] / 1024**3:.2f} GiB.",
                  f"Observed routing coverage: {coverage_text}.",
                  "Interpretation: association-based candidate; validate task quality before deployment.", ""]
    lines += ["## Interpretation", "",
              "Expert descriptions are statistical associations observed on the supplied offline calibration suites. They do not prove that knowledge or policy is localized exclusively in an expert.", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    confidence_counts = {level: sum(row["confidence"] == level for row in atlas_rows) for level in ("high", "medium", "low")}
    summary = {"model": scan["model"], "categories": categories, "payload_format": payload_suite.get("format") if payload_suite else None,
               "profile_accesses": {name: sum(sum(row) for row in profile.values()) for name, profile in profiles.items()},
               "atlas_confidence": confidence_counts, "pack_count": len(pack_cards)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline, reusable MoE expert MRI and ability-pack builder.")
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--model", type=Path, required=True)
    scan_parser.add_argument("--out-dir", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--scan", type=Path, required=True)
    analyze_parser.add_argument("--profile", action="append", help="CATEGORY=profile.jsonl")
    analyze_parser.add_argument("--profile-summary", type=Path, help="profile_summary.json emitted by run_moe_mri_profiles.py")
    analyze_parser.add_argument("--sizes", help="legacy absolute expert counts; prefer --percentages")
    analyze_parser.add_argument("--percentages", default="12.5,25,37.5,50")
    analyze_parser.add_argument("--out-dir", type=Path, required=True)
    analyze_parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    all_parser = sub.add_parser("all")
    all_parser.add_argument("--model", type=Path, required=True)
    all_parser.add_argument("--profile", action="append", required=True)
    all_parser.add_argument("--sizes", help="legacy absolute expert counts; prefer --percentages")
    all_parser.add_argument("--percentages", default="12.5,25,37.5,50")
    all_parser.add_argument("--out-dir", type=Path, required=True)
    all_parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    args = parser.parse_args()
    if args.command == "scan":
        scan = scan_model(args.model, args.out_dir)
        print(f"scan: {scan['layers']} layers | {scan['experts_per_layer']} experts | PASS")
        return
    if args.command == "analyze":
        scan = json.loads(args.scan.read_text(encoding="utf-8"))
    else:
        scan = scan_model(args.model, args.out_dir)
    profile_values = args.profile
    if args.command == "analyze" and args.profile_summary:
        profile_summary = json.loads(args.profile_summary.read_text(encoding="utf-8"))
        profile_values = [f"{name}={path}" for name, path in profile_summary["profiles"].items()]
    if not profile_values:
        raise SystemExit("provide --profile or --profile-summary")
    profiles = parse_profiles(profile_values)
    if args.sizes:
        pack_specs = [(round(100.0 * int(value) / scan["experts_per_layer"], 2), int(value)) for value in args.sizes.split(",")]
    else:
        percentages = [float(value) for value in args.percentages.split(",")]
        pack_specs = [(percentage, min(scan["experts_per_layer"], math.ceil(scan["experts_per_layer"] * percentage / 100.0))) for percentage in percentages]
    payload_suite = json.loads(args.payloads.read_text(encoding="utf-8"))
    summary = analyze(scan, profiles, pack_specs, args.out_dir, payload_suite)
    print(f"MRI: {len(summary['categories'])} categories | {summary['pack_count']} pack cards | PASS")


if __name__ == "__main__":
    main()
