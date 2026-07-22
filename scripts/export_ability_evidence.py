import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "docs" / "tables"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    runtime = load("results/ability_packs/workflow_universal_runs/ability_workflow_summary.json")
    for row in runtime:
        row.setdefault("resident_expert_pct", round(100.0 * row["pack_size"] / 64.0, 2))
    quality = load("results/ability_packs/quality_universal_summary.json")
    write_csv(TABLES / "qwen2_ability_runtime.csv", runtime)
    quality_rows = []
    for pack in quality:
        for task in pack["tasks"]:
            quality_rows.append({"pack_size": pack["pack_size"], **task})
    write_csv(TABLES / "qwen2_ability_quality.csv", quality_rows)

    cross_runtime = []
    for model, path in (
        ("Qwen1.5-MoE", "results/ability_packs/qwen1_workflow_runs/ability_workflow_summary.json"),
        ("DeepSeek-Coder-V2-Lite", "results/ability_packs/deepseek_workflow_runs/ability_workflow_summary.json"),
    ):
        for row in load(path):
            cross_runtime.append({**row, "model": model})
    for row in runtime:
        cross_runtime.append({
            "model": "Qwen2-57B-A14B", "resident_expert_pct": row["resident_expert_pct"],
            "experts_per_layer": row["pack_size"], "mode": "pack_only" if row["pack_only"] else "exact_fallback",
            "trials": row["trials"], "valid_trials": row["valid_trials"],
            "mean_generation_tps": row["mean_generation_tps"], "stddev_generation_tps": row["stddev_generation_tps"],
            "mean_peak_vram_mb": row["mean_peak_vram_mb"], "mean_peak_rss_mb": row["mean_peak_rss_mb"],
            "mean_hit_rate_pct": row["mean_hit_rate_pct"], "mean_substitutions": row["mean_substitutions"], "pass": row["pass"],
        })
    write_csv(TABLES / "cross_model_ability_runtime.csv", cross_runtime)

    cross_quality = []
    for model, path in (
        ("Qwen1.5-MoE", "results/ability_packs/qwen1_quality.json"),
        ("DeepSeek-Coder-V2-Lite", "results/ability_packs/deepseek_quality.json"),
    ):
        data = load(path)
        cross_quality.append({"model": model, "mode": "baseline", "resident_expert_pct": 100, "passed": data["baseline_passed"], "total": 5})
        for percentage, row in data["pack_summary"].items():
            cross_quality.append({"model": model, "mode": "pack_only", "resident_expert_pct": percentage,
                                  "passed": row["passed"], "total": row["total"],
                                  "retained_baseline_passes": row["retained_baseline_passes"],
                                  "baseline_pass_opportunities": row["baseline_pass_opportunities"]})
    for row in quality:
        cross_quality.append({"model": "Qwen2-57B-A14B", "mode": "pack_only", "resident_expert_pct": 100.0 * row["pack_size"] / 64,
                              "passed": row["passed"], "total": row["total"], "retained_baseline_passes": "",
                              "baseline_pass_opportunities": ""})
    write_csv(TABLES / "cross_model_ability_quality.csv", cross_quality)

    mri_rows = []
    for model, folder in (("Qwen1.5-MoE", "qwen1_5"), ("DeepSeek-Coder-V2-Lite", "deepseek"), ("Qwen2-57B-A14B", "qwen2")):
        scan = load(f"results/moe_mri/{folder}/scan.json")
        cards = load(f"results/moe_mri_v2/{folder}/pack_cards.json")
        for card in cards:
            if card["category"] != "universal":
                continue
            mri_rows.append({
                "model": model, "layers": scan["layers"], "experts_per_layer": scan["experts_per_layer"],
                "requested_residency_pct": card["requested_residency_pct"],
                "actual_residency_pct": card["actual_residency_pct"],
                "pack_size": card["experts_per_layer"], "estimated_pack_bytes": card["estimated_pack_bytes"],
                **{f"coverage_{key}_pct": value for key, value in card["coverage_pct"].items()},
            })
    write_csv(TABLES / "cross_model_mri_summary.csv", mri_rows)
    gpu_only_rows = []
    for model, path in (
        ("Qwen1.5-MoE", "results/gpu_only_final/v2_qwen1/qwen1_p37p5_only_trial_1.json"),
        ("DeepSeek-Coder-V2-Lite", "results/gpu_only_final/v2_deepseek/deepseek_p37p5_only_trial_1.json"),
        ("Qwen2-57B-A14B", "results/gpu_only_final/v2_qwen2/pack24_only_trial_1.json"),
    ):
        row = load(path)
        gpu_only_rows.append({"model": model, "resident_expert_pct": 37.5, "generation_tps": row["generation_tps"],
                              "peak_vram_mb": row["peak_gpu_memory_mb"], "peak_rss_mb": row["peak_process_rss_mb"],
                              "end_rss_mb": row["end_process_rss_mb"], "pagefile_delta_mb": row["pagefile_used_delta_mb"],
                              "gpu_kv": row["kv"] == "gpu", "gpu_pack_only": row["evm_env"].get("EVM_GPU_PACK_ONLY") == "1",
                              "cpu_backing": "EVM_CPU_BACKING" in row["evm_env"], "valid": row["returncode"] == 0 and row["has_evm_counters"]})
    write_csv(TABLES / "gpu_only_pack_runtime.csv", gpu_only_rows)
    extended_mri = []
    for model, folder in (("Qwen1.5-MoE", "qwen1_5"), ("DeepSeek-Coder-V2-Lite", "deepseek"), ("Qwen2-57B-A14B", "qwen2")):
        profile = load(f"results/moe_mri_extended/{folder}/profile_summary.json")
        analysis = load(f"results/moe_mri_extended/{folder}/summary.json")
        extended_mri.append({"model": model, "scope": "all domains" if folder == "qwen1_5" else "stratified portability sample",
                             "domains": len(profile["domains"]), "runs": profile["runs"], "valid_runs": profile["valid_runs"],
                             "split": profile["split"], "source_format": profile["source_format"],
                             "pack_cards": analysis["pack_count"], **{f"confidence_{key}": value for key, value in analysis["atlas_confidence"].items()}})
    write_csv(TABLES / "extended_mri_verification.csv", extended_mri)
    isolated_rows = [json.loads(path.read_text(encoding="utf-8")) for path in (ROOT / "results" / "moe_mri_extended" / "qwen1_5" / "runs").glob("*.json")]
    batch_dir = ROOT / "results" / "mri_batch" / "qwen1" / "final_verification"
    batch_run = json.loads((batch_dir / "batch_run.json").read_text(encoding="utf-8"))
    batch_summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    cloud_summary = json.loads((batch_dir / "analysis" / "cloud_summary.json").read_text(encoding="utf-8"))
    isolated_time = round(sum(row["elapsed_s"] for row in isolated_rows), 2)
    persistent_evidence = [{"model": "Qwen1.5-MoE", "prompts": 26, "isolated_process_wall_s": isolated_time,
                            "persistent_process_wall_s": batch_run["wall_time_s"],
                            "wall_speedup": round(isolated_time / batch_run["wall_time_s"], 2),
                            "valid_prompts": batch_summary["valid_prompts"], "routing_rows": batch_summary["profile_rows"],
                            "membership_edges": cloud_summary["membership_edges"], "coactivation_edges": cloud_summary["coactivation_edges"],
                            "prefill_generation_split": batch_summary["prefill_generation_split"], "cold_experts": batch_summary["cold_experts"]}]
    write_csv(TABLES / "persistent_mri_batch_verification.csv", persistent_evidence)
    evidence = {"runtime": runtime, "quality": quality, "cross_model_runtime": cross_runtime,
                "cross_model_quality": cross_quality, "cross_model_mri": mri_rows, "gpu_only_pack_runtime": gpu_only_rows,
                "extended_mri_verification": extended_mri}
    evidence["persistent_mri_batch_verification"] = persistent_evidence
    corpus_rows = []
    workflow_quality = []
    for model, label in (("qwen1", "Qwen1.5-MoE"), ("deepseek", "DeepSeek-Coder-V2-Lite"), ("qwen2", "Qwen2-57B-A14B")):
        root = ROOT / "results" / "mri_batch" / model / "calibration"
        batch = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        cloud = json.loads((root / "analysis" / "cloud_summary.json").read_text(encoding="utf-8"))
        corpus_rows.append({"model": label, "prompts": batch["prompts"], "valid_prompts": batch["valid_prompts"],
                            "wall_time_s": json.loads((root / "batch_run.json").read_text(encoding="utf-8"))["wall_time_s"],
                            "generation_tps": batch["mean_generation_tps"], "routing_rows": batch["profile_rows"],
                            "membership_edges": cloud["membership_edges"], "raw_coactivation_edges": cloud["coactivation_edges"],
                            "refined_coactivation_edges": cloud["refined_coactivation_edges"]})
        quality_doc = json.loads((root / "workflow_quality.json").read_text(encoding="utf-8"))
        workflow_quality.append({"model": label, "baseline_mode": quality_doc.get("baseline_mode", "native_full_model"), **quality_doc["summary"]})
    write_csv(TABLES / "full_mri_corpus_summary.csv", corpus_rows)
    write_csv(TABLES / "workflow_graph_quality.csv", workflow_quality)
    evidence["full_mri_corpus"] = corpus_rows
    evidence["workflow_graph_quality"] = workflow_quality
    (TABLES / "final_ability_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("ability evidence export: PASS")


if __name__ == "__main__":
    main()
