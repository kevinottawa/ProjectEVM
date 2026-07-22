import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "results" / "ability_packs" / "workflow_universal_runs" / "ability_workflow_summary.json"
QWEN1_RUNTIME = ROOT / "results" / "ability_packs" / "qwen1_workflow_runs" / "ability_workflow_summary.json"
DEEPSEEK_RUNTIME = ROOT / "results" / "ability_packs" / "deepseek_workflow_runs" / "ability_workflow_summary.json"
FIGURES = ROOT / "docs" / "figures"
MRI_MODELS = {
    "Qwen1.5-MoE": ROOT / "results" / "moe_mri_v2" / "qwen1_5" / "pack_cards.json",
    "DeepSeek-Coder-V2-Lite": ROOT / "results" / "moe_mri_v2" / "deepseek" / "pack_cards.json",
    "Qwen2-57B-A14B": ROOT / "results" / "moe_mri_v2" / "qwen2" / "pack_cards.json",
}
GPU_ONLY_ROWS = {
    "Qwen1.5-MoE": ROOT / "results" / "gpu_only_final" / "v2_qwen1" / "qwen1_p37p5_only_trial_1.json",
    "DeepSeek-Coder-V2-Lite": ROOT / "results" / "gpu_only_final" / "v2_deepseek" / "deepseek_p37p5_only_trial_1.json",
    "Qwen2-57B-A14B": ROOT / "results" / "gpu_only_final" / "v2_qwen2" / "pack24_only_trial_1.json",
}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    qwen2_cards = json.loads(MRI_MODELS["Qwen2-57B-A14B"].read_text(encoding="utf-8"))
    universal = [row for row in qwen2_cards if row["category"] == "universal"]
    sizes = [row["requested_residency_pct"] for row in universal]
    coverage = [sum(row["coverage_pct"].values()) / len(row["coverage_pct"]) for row in universal]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(sizes, coverage, marker="o", linewidth=2, color="#16697a")
    ax.set(xlabel="Expert residency budget (%)", ylabel="Observed routing accesses covered (%)",
           title="Qwen2-57B Ability-Pack Coverage")
    ax.set_xticks(sizes)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "qwen2_ability_pack_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for label, path in MRI_MODELS.items():
        cards = json.loads(path.read_text(encoding="utf-8"))
        cards = [row for row in cards if row["category"] == "universal"]
        x = [row["requested_residency_pct"] for row in cards]
        y = [sum(row["coverage_pct"].values()) / len(row["coverage_pct"]) for row in cards]
        ax.plot(x, y, marker="o", linewidth=2, label=label)
    ax.set(xlabel="Expert residency budget (%)", ylabel="Mean observed routing coverage (%)",
           title="Cross-Model Offline MRI Pack Coverage")
    ax.set_xticks([12.5, 25, 37.5, 50])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_model_mri_pack_coverage.png", dpi=180)
    plt.close(fig)

    if RUNTIME.exists():
        rows = json.loads(RUNTIME.read_text(encoding="utf-8"))
        labels = [row["name"] for row in rows]
        speeds = [row["mean_generation_tps"] for row in rows]
        vram = [row["mean_peak_vram_mb"] / 1024 for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
        axes[0].bar(labels, speeds, color="#16697a")
        axes[0].set_ylabel("Generation tokens/s")
        axes[1].bar(labels, vram, color="#489fb5")
        axes[1].set_ylabel("Peak VRAM (GiB)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Qwen2-57B Ability-Pack Runtime")
        fig.tight_layout()
        fig.savefig(FIGURES / "qwen2_ability_pack_runtime.png", dpi=180)
        plt.close(fig)

    cross_runtime = {
        "Qwen1.5-MoE": json.loads(QWEN1_RUNTIME.read_text(encoding="utf-8")),
        "DeepSeek-Coder-V2-Lite": json.loads(DEEPSEEK_RUNTIME.read_text(encoding="utf-8")),
        "Qwen2-57B-A14B": json.loads(RUNTIME.read_text(encoding="utf-8")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, rows in cross_runtime.items():
        normalized = []
        for row in rows:
            if "resident_expert_pct" in row:
                percentage = row["resident_expert_pct"]
                mode = row.get("mode", "pack_only" if row.get("pack_only") else "exact_fallback")
            else:
                percentage = 100.0 * row["pack_size"] / 64.0
                mode = "pack_only" if row["pack_only"] else "exact_fallback"
            normalized.append((percentage, mode, row))
        pack_rows = sorted((item for item in normalized if item[1] == "pack_only"), key=lambda item: item[0])
        exact_rows = sorted((item for item in normalized if item[1] == "exact_fallback"), key=lambda item: item[0])
        axes[0].plot([item[0] for item in pack_rows], [item[2]["mean_generation_tps"] for item in pack_rows], marker="o", label=label)
        axes[1].plot([item[0] for item in exact_rows], [item[2]["mean_generation_tps"] for item in exact_rows], marker="o", label=label)
    axes[0].set_title("Pack-Only Derived Runtime")
    axes[1].set_title("Exact-Fallback Runtime")
    for ax in axes:
        ax.set_xlabel("Expert residency budget (%)")
        ax.set_ylabel("Generation tokens/s")
        ax.set_xticks([25, 37.5])
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_model_ability_runtime.png", dpi=180)
    plt.close(fig)

    quality = {
        "Qwen1.5-MoE": json.loads((ROOT / "results" / "ability_packs" / "qwen1_quality.json").read_text(encoding="utf-8")),
        "DeepSeek-Coder-V2-Lite": json.loads((ROOT / "results" / "ability_packs" / "deepseek_quality.json").read_text(encoding="utf-8")),
    }
    qwen2_quality = json.loads((ROOT / "results" / "ability_packs" / "quality_universal_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = list(range(3))
    width = 0.22
    for offset, (label, data) in enumerate(quality.items()):
        values = [data["baseline_passed"], data["pack_summary"]["25.0"]["passed"], data["pack_summary"]["37.5"]["passed"]]
        ax.bar([value + (offset - 1) * width for value in x], values, width=width, label=label)
    qwen2_values = [float("nan"), next(row["passed"] for row in qwen2_quality if row["pack_size"] == 16), next(row["passed"] for row in qwen2_quality if row["pack_size"] == 24)]
    ax.bar([value + width for value in x], qwen2_values, width=width, label="Qwen2-57B-A14B")
    ax.set_xticks(x, ["Baseline", "25%", "37.5%"])
    ax.set_ylabel("Sanity tasks passed (of 5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Cross-Model Pack-Only Sanity Gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_model_ability_quality.png", dpi=180)
    plt.close(fig)

    labels = list(GPU_ONLY_ROWS)
    rows = [json.loads(GPU_ONLY_ROWS[label].read_text(encoding="utf-8")) for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].bar(labels, [row["generation_tps"] for row in rows], color="#16697a")
    axes[0].set_ylabel("Generation tokens/s")
    axes[1].bar(labels, [row["peak_gpu_memory_mb"] / 1024 for row in rows], color="#489fb5")
    axes[1].set_ylabel("Peak VRAM (GiB)")
    width = 0.36
    x = list(range(len(labels)))
    axes[2].bar([i - width / 2 for i in x], [row["peak_process_rss_mb"] / 1024 for row in rows], width, label="Transient peak", color="#82c0cc")
    axes[2].bar([i + width / 2 for i in x], [row["end_process_rss_mb"] / 1024 for row in rows], width, label="End of inference", color="#ffa62b")
    axes[2].set_ylabel("Process RSS (GiB)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("37.5% GPU-Resident Pack-Only Proof (No CPU Fallback, Zero Page-File Growth)")
    fig.tight_layout()
    fig.savefig(FIGURES / "gpu_only_pack_runtime.png", dpi=180)
    plt.close(fig)

    library = json.loads((ROOT / "config" / "mri" / "v2" / "domain_library.json").read_text(encoding="utf-8"))
    groups = list(library["groups"])
    domain_counts = [sum(row["group"] == group for row in library["domains"].values()) for group in groups]
    prompt_counts = [sum(sum(len(row[split]) for split in ("calibration", "validation", "held_out"))
                         for row in library["domains"].values() if row["group"] == group) for group in groups]
    labels = [group.replace("_", " ").title() for group in groups]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].bar(labels, domain_counts, color="#16697a")
    axes[0].set_ylabel("Domains")
    axes[1].bar(labels, prompt_counts, color="#ffa62b")
    axes[1].set_ylabel("Explicit payloads")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("EVM MRI v2 Domain Library: 26 Domains, 156 Payloads")
    fig.tight_layout()
    fig.savefig(FIGURES / "mri_v2_domain_library.png", dpi=180)
    plt.close(fig)

    isolated = [json.loads(path.read_text(encoding="utf-8")) for path in (ROOT / "results" / "moe_mri_extended" / "qwen1_5" / "runs").glob("*.json")]
    persistent = json.loads((ROOT / "results" / "mri_batch" / "qwen1" / "final_verification" / "batch_run.json").read_text(encoding="utf-8"))
    times = [sum(row["elapsed_s"] for row in isolated), persistent["wall_time_s"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(["Separate process\nper prompt", "Persistent model\nKV reset per prompt"], times, color=["#8d99ae", "#16697a"])
    ax.set_ylabel("Wall time for 26 prompts (seconds)")
    ax.set_title("Persistent MRI Removes Repeated Model Startup")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "mri_persistent_batch_speedup.png", dpi=180)
    plt.close(fig)

    corpus = []
    quality = []
    for model, label in (("qwen1", "Qwen1.5"), ("deepseek", "DeepSeek"), ("qwen2", "Qwen2-57B")):
        root = ROOT / "results" / "mri_batch" / model / "calibration"
        cloud = json.loads((root / "analysis" / "cloud_summary.json").read_text(encoding="utf-8"))
        corpus.append((label, cloud["coactivation_edges"], cloud["refined_coactivation_edges"]))
        quality_doc = json.loads((root / "workflow_quality.json").read_text(encoding="utf-8"))
        quality.append((label, quality_doc["summary"]["baseline_passed"], quality_doc["summary"]["frequency"]["passed"], quality_doc["summary"]["graph"]["passed"]))
    x = list(range(len(corpus)))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar([i - .18 for i in x], [row[1] for row in corpus], .36, label="Raw co-activation", color="#8d99ae")
    ax.bar([i + .18 for i in x], [row[2] for row in corpus], .36, label="PMI + family-stable", color="#16697a")
    ax.set_xticks(x, [row[0] for row in corpus]); ax.set_ylabel("Edges"); ax.set_title("Refined Expert Cloud Filtering"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "full_mri_cloud_refinement.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = .24
    for offset, (name, index, color) in enumerate((("Exact baseline", 1, "#8d99ae"), ("Frequency pack", 2, "#489fb5"), ("Graph pack", 3, "#ffa62b"))):
        ax.bar([i + (offset - 1) * width for i in x], [row[index] for row in quality], width, label=name, color=color)
    ax.set_xticks(x, [row[0] for row in quality]); ax.set_ylim(0, 8.5); ax.set_ylabel("Workflow checks passed (of 8)"); ax.set_title("37.5% Workflow Pack Quality Gate"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "workflow_graph_quality.png", dpi=180); plt.close(fig)
    print("ability charts: PASS")


if __name__ == "__main__":
    main()
