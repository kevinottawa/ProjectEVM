from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "controlled_final" / "controlled_runtime.csv"
FUSION_TRIALS_CSV = ROOT / "results" / "controlled_final" / "fusion_aware_trials.csv"
FIG_DIR = ROOT / "docs" / "figures" / "controlled_final"


MODEL_LABELS = {
    "qwen15_moe": "Qwen1.5-MoE",
    "deepseek_coder_v2_lite": "DeepSeek-Coder-V2-Lite",
    "qwen2_57b_a14b": "Qwen2-57B-A14B",
}

MODE_LABELS = {
    "native_gpu": "Native GPU",
    "cpu_backed_residency": "CPU-backed\nresidency",
    "gpu_backed_remap": "GPU-backed\nremap",
    "unified_streaming": "Unified\nstreaming",
    "unified_fusion_aware": "Unified\nfusion-aware",
}

COLORS = {
    "native_gpu": "#4472c4",
    "cpu_backed_residency": "#70ad47",
    "gpu_backed_remap": "#ed7d31",
    "unified_streaming": "#7030a0",
    "unified_fusion_aware": "#00a6a6",
}

MODE_ORDER = ["native_gpu", "cpu_backed_residency", "gpu_backed_remap", "unified_streaming", "unified_fusion_aware"]


def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    print(f"wrote {path}")
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df = df[df["success"] == True].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["mode_label"] = df["mode"].map(MODE_LABELS).fillna(df["mode"])

    for model, rows in df.groupby("model"):
        rows = rows.set_index("mode").reindex(MODE_ORDER).dropna(subset=["success"]).reset_index()
        labels = rows["mode"].map(MODE_LABELS)
        colors = [COLORS[m] for m in rows["mode"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
        bars = ax1.bar(labels, rows["peak_gpu_memory_mb"], color=colors)
        for bar, value in zip(bars, rows["peak_gpu_memory_mb"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)
        ax1.set_ylabel("Peak GPU memory (MB)")
        ax1.set_title("Dedicated VRAM")

        bars = ax2.bar(labels, rows["generation_tps"], color=colors)
        for bar, value in zip(bars, rows["generation_tps"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{float(value):.1f}", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("Generation tokens/s")
        ax2.set_title("Throughput")

        fig.suptitle(f"{MODEL_LABELS.get(model, model)} Controlled Runtime Modes")
        fig.text(
            0.5,
            0.01,
            "Unified streaming is CPU-backed expert storage plus CUDA EVM remapping/counters. GPU KV is explicit in all rows.",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.23, top=0.86)
        savefig(FIG_DIR / f"evm_{model}_controlled_modes.png")

    pivot = df.pivot(index="model", columns="mode", values="peak_gpu_memory_mb")
    summary = pivot.reindex(columns=["native_gpu", "cpu_backed_residency", "unified_streaming", "unified_fusion_aware", "gpu_backed_remap"]).copy()
    labels = [MODEL_LABELS.get(m, m) for m in summary.index]
    x = range(len(summary))
    width = 0.16
    plt.figure(figsize=(10, 5.2))
    for offset, mode in [(-2, "native_gpu"), (-1, "cpu_backed_residency"), (0, "unified_streaming"), (1, "unified_fusion_aware"), (2, "gpu_backed_remap")]:
        if mode not in summary.columns:
            continue
        plt.bar([i + offset * width for i in x], summary[mode], width=width, label=MODE_LABELS[mode].replace("\n", " "), color=COLORS[mode])
    plt.xticks(list(x), labels)
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Controlled Final VRAM by Mode")
    plt.legend()
    savefig(FIG_DIR / "evm_controlled_vram_by_mode.png")

    evm = df[df["has_evm_counters"] == True].copy()
    evm["label"] = evm["model_label"] + "\n" + evm["mode_label"].str.replace("\n", " ")
    plt.figure(figsize=(9.5, 4.8))
    bars = plt.bar(evm["label"], evm["hit_rate_pct"], color=[COLORS[m] for m in evm["mode"]])
    for bar, value in zip(bars, evm["hit_rate_pct"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{float(value):.1f}%", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, 100)
    plt.ylabel("EVM cache hit rate (%)")
    plt.title("Controlled Final Counter-Emitting EVM Rows")
    savefig(FIG_DIR / "evm_controlled_counter_hit_rates.png")

    if FUSION_TRIALS_CSV.exists():
        trials = pd.read_csv(FUSION_TRIALS_CSV)
        summary_rows = trials[trials["trial"].astype(str) == "summary"].copy()
        if not summary_rows.empty:
            summary_rows["model_label"] = summary_rows["model_id"].map(MODEL_LABELS).fillna(summary_rows["model_id"])
            summary_rows["generation_tps_mean"] = pd.to_numeric(summary_rows["generation_tps_mean"])
            summary_rows["generation_tps_std"] = pd.to_numeric(summary_rows["generation_tps_std"])
            summary_rows["peak_gpu_memory_mb_mean"] = pd.to_numeric(summary_rows["peak_gpu_memory_mb_mean"])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
            ax1.bar(summary_rows["model_label"], summary_rows["peak_gpu_memory_mb_mean"], color=COLORS["unified_fusion_aware"])
            ax1.set_ylabel("Mean peak GPU memory (MB)")
            ax1.set_title("VRAM")
            ax1.tick_params(axis="x", rotation=15)

            ax2.bar(
                summary_rows["model_label"],
                summary_rows["generation_tps_mean"],
                yerr=summary_rows["generation_tps_std"],
                capsize=4,
                color=COLORS["unified_fusion_aware"],
            )
            ax2.set_ylabel("Generation tokens/s")
            ax2.set_title("3-trial mean +/- std")
            ax2.tick_params(axis="x", rotation=15)

            fig.suptitle("Unified Fusion-Aware EVM Reproduction Trials")
            savefig(FIG_DIR / "evm_fusion_aware_reproduction_trials.png")


if __name__ == "__main__":
    main()
