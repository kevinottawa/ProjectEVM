import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    failures = []
    runtime = load("results/ability_packs/workflow_universal_runs/ability_workflow_summary.json")
    runtime_by_name = {row["name"]: row for row in runtime}
    for name in ("pack16_exact", "pack24_exact", "pack16_only", "pack24_only"):
        row = runtime_by_name.get(name)
        if not row or not row["pass"] or row["valid_trials"] != 3:
            failures.append(f"invalid runtime row: {name}")
    if runtime_by_name["pack16_only"]["mean_generation_tps"] <= 10:
        failures.append("pack16-only throughput gate failed")
    if runtime_by_name["pack24_only"]["mean_generation_tps"] <= 10:
        failures.append("pack24-only throughput gate failed")
    if runtime_by_name["pack16_only"]["mean_substitutions"] <= 0:
        failures.append("pack-only substitution accounting missing")

    for model, path in (
        ("qwen1", "results/ability_packs/qwen1_workflow_runs/ability_workflow_summary.json"),
        ("deepseek", "results/ability_packs/deepseek_workflow_runs/ability_workflow_summary.json"),
    ):
        rows = load(path)
        if len(rows) != 4 or any(not row["pass"] or row["valid_trials"] != 3 for row in rows):
            failures.append(f"cross-model runtime matrix failed: {model}")

    quality = load("results/ability_packs/quality_universal_summary.json")
    quality_by_pack = {row["pack_size"]: row for row in quality}
    if quality_by_pack.get(24, {}).get("passed") != 5:
        failures.append("Qwen2 37.5% quality candidate gate failed")
    if quality_by_pack.get(16, {}).get("passed") != 4:
        failures.append("Qwen2 25% quality rejection was not reproduced")
    if any(len(row["tasks"]) != row["total"] for row in quality):
        failures.append("quality task set incomplete")
    for model, expected in (("qwen1", (5, 1, 3)), ("deepseek", (5, 0, 0))):
        data = load(f"results/ability_packs/{model}_quality.json")
        actual = (data["baseline_passed"], data["pack_summary"]["25.0"]["passed"], data["pack_summary"]["37.5"]["passed"])
        if actual != expected:
            failures.append(f"cross-model quality gate changed: {model} {actual}")

    for model in ("qwen1_5", "deepseek", "qwen2"):
        profile = load(f"results/moe_mri_v2/{model}/profile_summary.json")
        mri = load(f"results/moe_mri_v2/{model}/summary.json")
        if profile["valid_runs"] != 12 or profile["runs"] != 12 or profile.get("payload_format") != "evm-mri-diagnostic-suite-v1":
            failures.append(f"MRI profile failed: {model}")
        if mri["pack_count"] != 28 or len(mri["categories"]) != 6:
            failures.append(f"MRI pack-card count failed: {model}")
        for artifact in ("scan.json", "expert_atlas.csv", "pack_cards.json", "report.md"):
            path = ROOT / "results" / "moe_mri_v2" / model / artifact
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"missing MRI artifact: {model}/{artifact}")

    for model, expected_domains, expected_runs, expected_cards in (("qwen1_5", 26, 26, 108), ("deepseek", 6, 6, 28), ("qwen2", 6, 6, 28)):
        profile = load(f"results/moe_mri_extended/{model}/profile_summary.json")
        analysis = load(f"results/moe_mri_extended/{model}/summary.json")
        if profile.get("source_format") != "evm-mri-domain-library-v2" or len(profile.get("domains", {})) != expected_domains:
            failures.append(f"extended MRI taxonomy failed: {model}")
        if profile.get("runs") != expected_runs or profile.get("valid_runs") != expected_runs or analysis.get("pack_count") != expected_cards:
            failures.append(f"extended MRI execution failed: {model}")

    corpus = load("results/mri_batch/corpus/calibration_100.json")
    if len(corpus.get("domains", {})) != 26 or sum(len(row["prompts"]) for row in corpus["domains"].values()) != 2600:
        failures.append("persistent MRI corpus failed")
    batch_root = ROOT / "results" / "mri_batch" / "qwen1" / "final_verification"
    batch = json.loads((batch_root / "summary.json").read_text(encoding="utf-8"))
    cloud = json.loads((batch_root / "analysis" / "cloud_summary.json").read_text(encoding="utf-8"))
    pack = json.loads((batch_root / "packs" / "cloud_37p5.json").read_text(encoding="utf-8"))
    if batch.get("valid_prompts") != 26 or batch.get("profile_rows") != 3744 or not batch.get("prefill_generation_split"):
        failures.append("persistent MRI execution failed")
    if cloud.get("valid_prompts") != 26 or cloud.get("membership_edges", 0) <= 0 or cloud.get("coactivation_edges", 0) <= 0:
        failures.append("MRI cloud analysis failed")
    if len(pack.get("selected", {})) != 24 or pack.get("experts_per_layer") != 23:
        failures.append("MRI cloud pack failed")

    for pack in (16, 24):
        output = ROOT / "results" / "ability_packs" / f"qwen2_universal_{pack}_build.stdout"
        if not output.exists() or "Verification: PASS" not in output.read_text(encoding="utf-8"):
            failures.append(f"pack hash verification failed: {pack}")
    for model, names in (
        ("qwen1", ("qwen1_full_vault", "qwen1_universal_25", "qwen1_universal_37p5")),
        ("deepseek", ("deepseek_full_vault", "deepseek_universal_25", "deepseek_universal_37p5")),
    ):
        for name in names:
            output = ROOT / "results" / "ability_packs" / f"{name}_build.stdout"
            if not output.exists() or "Verification: PASS" not in output.read_text(encoding="utf-8"):
                failures.append(f"cross-model pack verification failed: {model}/{name}")

    for figure in ("qwen2_ability_pack_coverage.png", "qwen2_ability_pack_runtime.png", "cross_model_mri_pack_coverage.png",
                   "cross_model_ability_runtime.png", "cross_model_ability_quality.png", "gpu_only_pack_runtime.png",
                   "mri_v2_domain_library.png", "mri_persistent_batch_speedup.png"):
        path = ROOT / "docs" / "figures" / figure
        if not path.exists() or path.stat().st_size < 10_000:
            failures.append(f"figure missing or empty: {figure}")

    paper = (ROOT / "docs" / "EVM_Paper.md").read_text(encoding="utf-8")
    for required in (
        "Static ability packs",
        "24.97",
        "38.67",
        "offline MoE MRI",
        "Future Work: MRI-Guided Mixed-Precision Experts",
        "four established workflows",
        "not a completed quantization-sensitivity atlas",
        "36 cross-model runtime trials passed",
        "Qwen1.5 retained only 1/5",
        "DeepSeek retained 0/5",
        "Final GPU-resident pack proof",
        "zero page-file growth",
        "26-domain MRI library",
        "156 explicit prompts",
        "2,600-prompt calibration corpus",
        "persistent batch profiler",
    ):
        if required not in paper:
            failures.append(f"paper missing result: {required}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("Runtime: 3 models, 12 configurations, 36/36 trials | PASS")
    print("Quality: Qwen2 37.5% provisional PASS | Qwen1/DeepSeek pack-only NO-GO")
    gpu_rows = (
        "results/gpu_only_final/v2_qwen1/qwen1_p37p5_only_trial_1.json",
        "results/gpu_only_final/v2_deepseek/deepseek_p37p5_only_trial_1.json",
        "results/gpu_only_final/v2_qwen2/pack24_only_trial_1.json",
    )
    for path in gpu_rows:
        row = load(path)
        env = row.get("evm_env", {})
        if row.get("returncode") != 0 or not row.get("has_evm_counters") or row.get("pagefile_used_delta_mb") != 0:
            failures.append(f"GPU-only runtime failed: {path}")
        if env.get("EVM_GPU_PACK_ONLY") != "1" or "EVM_CPU_BACKING" in env or env.get("EVM_ABILITY_PACK_ONLY") != "1":
            failures.append(f"GPU-only mode contract failed: {path}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("GPU-only: 3/3 models, zero page-file growth, no CPU expert fallback | PASS")
    print("MRI: 3/3 models, 36/36 diagnostic runs, 84 pack cards | PASS")
    print("Extended MRI: 26-domain full sample + 2 stratified portability samples, 38/38 runs | PASS")
    print("Persistent MRI: 2,600-prompt corpus + 26/26 batch verification + expert cloud/pack | PASS")
    print("Packs/charts/paper consistency | PASS")


if __name__ == "__main__":
    main()
