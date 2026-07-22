import argparse
import csv
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path


TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up|gate_up)_exps\.weight$")
ROLE_PRIORITY = {"down": 0, "gate_up": 1, "gate": 2, "up": 3}


def load_manifest(path):
    suite = json.loads(path.read_text(encoding="utf-8"))
    prompts = {}
    split = suite.get("split", "calibration")
    for domain, row in suite["domains"].items():
        for index, _ in enumerate(row["prompts"], 1):
            prompt_id = f"{domain}.{split}.{index}"
            prompts[prompt_id] = {"domain": domain, "group": row.get("group", "unclassified"), "split": split}
    return suite, prompts


def load_routing(path, valid_prompt_ids):
    candidates = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("prompt_id") not in valid_prompt_ids:
                continue
            match = TENSOR_RE.match(row.get("tensor", ""))
            if not match:
                continue
            layer, role = int(match.group(1)), match.group(2)
            key = (row["prompt_id"], row.get("phase", "combined"), layer)
            current = candidates[key].get("priority", 99)
            if ROLE_PRIORITY[role] < current:
                candidates[key] = {"priority": ROLE_PRIORITY[role], "counts": [int(value) for value in row["counts"]]}
    return {key: value["counts"] for key, value in candidates.items()}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def semantic_family(prompt_id):
    """Group deterministic variants by their underlying hand-authored seed."""
    parts = prompt_id.rsplit(".", 1)
    if len(parts) != 2:
        return prompt_id
    try:
        return f"{parts[0]}.seed.{(int(parts[1]) - 1) % 4 + 1}"
    except ValueError:
        return prompt_id


def analyze(manifest_path, rows_path, routing_path, out_dir, top_per_layer=8, min_pair_support=2,
            min_pmi=0.5, min_family_support=4):
    suite, prompt_meta = load_manifest(manifest_path)
    result_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {row["prompt_id"]: row for row in result_rows}
    valid_ids = {prompt_id for prompt_id, row in latest.items() if row.get("valid") and prompt_id in prompt_meta}
    routing = load_routing(routing_path, valid_ids)
    prompt_vectors = defaultdict(lambda: defaultdict(list))
    for (prompt_id, phase, layer), counts in routing.items():
        total = sum(counts)
        if total:
            prompt_vectors[prompt_id][phase].append((layer, [value / total for value in counts], counts))

    domain_sums = defaultdict(lambda: defaultdict(float))
    domain_prompt_counts = defaultdict(set)
    global_sums = defaultdict(float)
    inverse_prompts = []
    coactivation = defaultdict(int)
    coactivation_families = defaultdict(set)
    expert_prompt_support = defaultdict(int)
    phase_domain_sums = defaultdict(lambda: defaultdict(float))
    for prompt_id, phases in prompt_vectors.items():
        meta = prompt_meta[prompt_id]
        domain_prompt_counts[meta["domain"]].add(prompt_id)
        leaf = {"prompt_id": prompt_id, **meta, "phases": {}}
        combined_layer = defaultdict(list)
        for phase, layers in phases.items():
            phase_leaf = []
            for layer, shares, counts in layers:
                ranked = sorted(range(len(shares)), key=lambda expert: (-shares[expert], expert))[:top_per_layer]
                phase_leaf.append({"layer": layer, "experts": [{"expert": expert, "share": round(shares[expert], 6)} for expert in ranked if shares[expert] > 0]})
                for expert, share in enumerate(shares):
                    key = (layer, expert)
                    phase_domain_sums[(phase, meta["domain"])][key] += share
                    combined_layer[layer].append((expert, share))
            leaf["phases"][phase] = phase_leaf
        for layer, values in combined_layer.items():
            merged = defaultdict(float)
            for expert, share in values:
                merged[expert] += share
            ranked = sorted(merged, key=lambda expert: (-merged[expert], expert))[:top_per_layer]
            for expert in ranked:
                key = (layer, expert)
                domain_sums[meta["domain"]][key] += merged[expert]
                global_sums[key] += merged[expert]
                expert_prompt_support[key] += 1
            for left, right in combinations(sorted(ranked), 2):
                coactivation[(layer, left, right)] += 1
                coactivation_families[(layer, left, right)].add(semantic_family(prompt_id))
        inverse_prompts.append(leaf)

    prompt_total = max(1, len(valid_ids))
    membership_rows = []
    for domain, sums in domain_sums.items():
        domain_n = max(1, len(domain_prompt_counts[domain]))
        raw = []
        for key, value in sums.items():
            domain_mean = value / domain_n
            global_mean = global_sums[key] / prompt_total
            lift = domain_mean / global_mean if global_mean else 0
            if lift >= 1.05:
                raw.append((key, domain_mean, lift))
        excess_total = sum(max(0, lift - 1) * mean for _, mean, lift in raw) or 1
        for (layer, expert), mean, lift in raw:
            membership_rows.append({"domain": domain, "layer": layer, "expert": expert,
                                    "mean_prompt_share": round(mean, 8), "lift": round(lift, 4),
                                    "soft_membership": round(max(0, lift - 1) * mean / excess_total, 8),
                                    "prompt_count": domain_n})
    membership_rows.sort(key=lambda row: (row["domain"], row["layer"], -row["soft_membership"], row["expert"]))
    coactivation_rows = [{"layer": layer, "expert_a": left, "expert_b": right, "prompt_support": support,
                          "support_pct": round(100 * support / prompt_total, 4),
                          "semantic_family_support": len(coactivation_families[(layer, left, right)])}
                         for (layer, left, right), support in coactivation.items() if support >= min_pair_support]
    coactivation_rows.sort(key=lambda row: (row["layer"], -row["prompt_support"], row["expert_a"], row["expert_b"]))
    refined_rows = []
    for (layer, left, right), support in coactivation.items():
        left_support = expert_prompt_support[(layer, left)]
        right_support = expert_prompt_support[(layer, right)]
        family_support = len(coactivation_families[(layer, left, right)])
        if not left_support or not right_support:
            continue
        pmi = math.log2((support * prompt_total) / (left_support * right_support))
        if support >= min_pair_support and family_support >= min_family_support and pmi >= min_pmi:
            refined_rows.append({"layer": layer, "expert_a": left, "expert_b": right, "prompt_support": support,
                                 "support_pct": round(100 * support / prompt_total, 4), "semantic_family_support": family_support,
                                 "pmi_bits": round(pmi, 4), "lift": round((support * prompt_total) / (left_support * right_support), 4)})
    refined_rows.sort(key=lambda row: (row["layer"], -row["pmi_bits"], -row["prompt_support"], row["expert_a"], row["expert_b"]))

    groups = defaultdict(lambda: defaultdict(list))
    for leaf in inverse_prompts:
        groups[leaf["group"]][leaf["domain"]].append(leaf)
    inverse_tree = {"format": "evm-mri-inverse-tree-v1", "root": {group: dict(domains) for group, domains in groups.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inverse_tree.json").write_text(json.dumps(inverse_tree, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "expert_domain_memberships.csv", membership_rows)
    write_csv(out_dir / "expert_coactivation_cloud.csv", coactivation_rows)
    write_csv(out_dir / "expert_coactivation_cloud_refined.csv", refined_rows)
    summary = {"valid_prompts": len(valid_ids), "domains": len(domain_prompt_counts), "groups": len(groups),
               "routing_prompt_phase_layers": len(routing), "membership_edges": len(membership_rows),
               "coactivation_edges": len(coactivation_rows), "refined_coactivation_edges": len(refined_rows),
               "top_experts_per_prompt_layer": top_per_layer, "min_pair_support": min_pair_support,
               "min_pmi_bits": min_pmi, "min_semantic_family_support": min_family_support}
    (out_dir / "cloud_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"MRI cloud: {summary['valid_prompts']} prompts | {summary['domains']} domains | {summary['membership_edges']} memberships | {summary['coactivation_edges']} coactivations | PASS")
    return summary


def build_pack(memberships_path, coactivation_path, domains, experts_per_layer, percentage, out_path, method, core_fraction):
    targets = {value.strip() for value in domains.split(",") if value.strip()}
    score = defaultdict(float)
    core_score = defaultdict(float)
    with memberships_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["expert"]))
            core_score[key] += float(row["mean_prompt_share"])
            if row["domain"] in targets:
                score[key] += float(row["mean_prompt_share"]) * float(row["lift"])
    centrality = defaultdict(float)
    if method == "graph" and coactivation_path.exists() and coactivation_path.stat().st_size:
        with coactivation_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                layer, left, right = int(row["layer"]), int(row["expert_a"]), int(row["expert_b"])
                support = float(row["support_pct"]) / 100
                centrality[(layer, left)] += support
                centrality[(layer, right)] += support
    layers = sorted({layer for layer, _ in score})
    count = max(1, min(experts_per_layer, math.ceil(experts_per_layer * percentage / 100)))
    selected = {}
    for layer in layers:
        if method == "core_overlay":
            core_count = max(1, min(count - 1, round(count * core_fraction)))
            core = sorted(range(experts_per_layer), key=lambda expert: (-core_score[(layer, expert)], expert))[:core_count]
            overlay = sorted((expert for expert in range(experts_per_layer) if expert not in core),
                             key=lambda expert: (-(score[(layer, expert)] + 0.10 * centrality[(layer, expert)]), expert))[:count - core_count]
            selected[str(layer)] = sorted(core + overlay)
        else:
            ranking = sorted(range(experts_per_layer), key=lambda expert: (-(score[(layer, expert)] + (0.15 * centrality[(layer, expert)] if method == "graph" else 0)), expert))
            selected[str(layer)] = sorted(ranking[:count])
    payload = {"format": "evm-ability-selection-v1", "name": "cloud-" + "-".join(sorted(targets)),
               "category": "workflow_cloud", "target_domains": sorted(targets), "core_fraction": core_fraction if method == "core_overlay" else None,
               "selection_method": "membership-plus-refined-coactivation-v1" if method == "graph" else "core-plus-workflow-overlay-v1" if method == "core_overlay" else "membership-frequency-v1",
               "requested_residency_pct": percentage, "experts_per_layer": count, "selected": selected}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"MRI cloud pack: {len(layers)} layers | {count}/{experts_per_layer} experts | {len(targets)} domains | PASS")


def main():
    parser = argparse.ArgumentParser(description="Build inverse prompt trees, expert clouds, and graph-informed pack selections.")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, required=True)
    analyze_parser.add_argument("--rows", type=Path, required=True)
    analyze_parser.add_argument("--routing", type=Path, required=True)
    analyze_parser.add_argument("--out-dir", type=Path, required=True)
    analyze_parser.add_argument("--top-per-layer", type=int, default=8)
    analyze_parser.add_argument("--min-pair-support", type=int, default=2)
    analyze_parser.add_argument("--min-pmi", type=float, default=0.5)
    analyze_parser.add_argument("--min-family-support", type=int, default=4)
    pack_parser = sub.add_parser("build-pack")
    pack_parser.add_argument("--analysis-dir", type=Path, required=True)
    pack_parser.add_argument("--domains", required=True)
    pack_parser.add_argument("--experts-per-layer", type=int, required=True)
    pack_parser.add_argument("--percentage", type=float, default=37.5)
    pack_parser.add_argument("--method", choices=("frequency", "graph", "core_overlay"), default="graph")
    pack_parser.add_argument("--core-fraction", type=float, default=0.65)
    pack_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args.manifest, args.rows, args.routing, args.out_dir, args.top_per_layer, args.min_pair_support,
                args.min_pmi, args.min_family_support)
    else:
        build_pack(args.analysis_dir / "expert_domain_memberships.csv", args.analysis_dir / "expert_coactivation_cloud_refined.csv",
                   args.domains, args.experts_per_layer, args.percentage, args.out, args.method, args.core_fraction)


if __name__ == "__main__":
    main()
