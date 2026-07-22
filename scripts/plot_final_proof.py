from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "final_proof" / "runtime_proof.csv"
MEMORY_CSV_PATH = ROOT / "results" / "final_proof" / "memory_proof.csv"
REPRO_CSV_PATH = ROOT / "results" / "final_proof" / "reproduction_trials.csv"
CROSS_MODEL_CSV_PATH = ROOT / "results" / "final_proof" / "cross_model_memory.csv"
DEEPSEEK_GPU_TRIALS_CSV_PATH = ROOT / "results" / "final_proof" / "deepseek_gpu_backed_trials.csv"
SEGREGATION_CSV_PATH = ROOT / "results" / "final_proof" / "gpu_backed_segregation.csv"
MAX_CONTEXT_CSV_PATH = ROOT / "results" / "final_workflow" / "max_context.csv"
SPECULATIVE_WORKFLOW_CSV_PATH = ROOT / "results" / "final_workflow" / "speculative_workflow.csv"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures" / "final_proof"
WORKFLOW_FIG_DIR = DOCS_DIR / "figures" / "final_workflow"


def savefig(path, *args, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, *args, **kwargs)
    print(f"wrote {path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    WORKFLOW_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    small = df[df["name"].str.startswith("qwen15")].copy()
    small["label"] = small["name"].str.replace("qwen15_", "", regex=False).str.replace("_", " ")

    plt.figure(figsize=(9, 5))
    x = range(len(small))
    plt.bar([i - 0.18 for i in x], small["prompt_tps"], width=0.36, label="Prompt t/s")
    plt.bar([i + 0.18 for i in x], small["generation_tps"], width=0.36, label="Generation t/s")
    plt.xticks(list(x), small["label"], rotation=25, ha="right")
    plt.ylabel("Tokens per second")
    plt.title("Final Runtime Proof: Qwen1.5-MoE Throughput")
    plt.legend()
    plt.tight_layout()
    throughput_path = FIG_DIR / "evm_qwen15_moe_gpu_backed_capacity_throughput.png"
    savefig(throughput_path, dpi=160)
    plt.close()

    evm = small[small["cache_hits"] + small["cache_misses"] > 0].copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(evm["label"], evm["hit_rate_pct"], color="#4472c4")
    plt.ylim(0, 100)
    plt.ylabel("EVM cache hit rate (%)")
    plt.title("Final Runtime Proof: GPU-Backed EVM Hit Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    hit_rate_path = FIG_DIR / "evm_qwen15_moe_gpu_backed_capacity_hit_rate.png"
    savefig(hit_rate_path, dpi=160)
    plt.close()

    if MEMORY_CSV_PATH.exists():
        mem = pd.read_csv(MEMORY_CSV_PATH)
        ordered_memory_names = [
            "qwen15_cpu_only_baseline",
            "qwen15_gpu_native_baseline",
            "qwen15_cpu_backed_evm_33_memory",
            "qwen15_gpu_backed_evm_33_memory",
            "qwen15_gpu_backed_evm_66_memory",
        ]
        mem_savings = mem[mem["name"].isin(ordered_memory_names)].copy()
        mem_savings["order"] = mem_savings["name"].map({name: i for i, name in enumerate(ordered_memory_names)})
        mem_savings = mem_savings.sort_values("order")
        label_map = {
            "qwen15_cpu_only_baseline": "CPU-only\nbaseline",
            "qwen15_gpu_native_baseline": "Native GPU\nbaseline",
            "qwen15_cpu_backed_evm_33_memory": "CPU-backed\nEVM 33%\n(VRAM saving)",
            "qwen15_gpu_backed_evm_33_memory": "GPU-backed\nEVM 33%\n(remap test)",
            "qwen15_gpu_backed_evm_66_memory": "GPU-backed\nEVM 66%\n(remap test)",
        }
        mem_savings["label"] = mem_savings["name"].map(label_map).fillna(mem_savings["name"])
        colors = []
        for name in mem_savings["name"]:
            if "cpu_only" in name:
                colors.append("#a5a5a5")
            elif "gpu_native" in name:
                colors.append("#4472c4")
            elif "gpu_backed" in name:
                colors.append("#ed7d31")
            else:
                colors.append("#70ad47")
        plt.figure(figsize=(10, 5.6))
        bars = plt.bar(mem_savings["label"], mem_savings["peak_gpu_memory_mb"], color=colors)
        for bar, value in zip(bars, mem_savings["peak_gpu_memory_mb"]):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
        native = mem_savings[mem_savings["name"] == "qwen15_gpu_native_baseline"]["peak_gpu_memory_mb"]
        if not native.empty:
            plt.axhline(native.iloc[0], color="#4472c4", linestyle="--", linewidth=1, alpha=0.75)
            plt.text(len(mem_savings) - 0.55, native.iloc[0] + 220, "native GPU baseline", color="#4472c4", ha="right", fontsize=9)
        plt.ylabel("Peak GPU memory used (MB)")
        plt.title("Qwen1.5-MoE Peak GPU Memory: Baselines and EVM Modes")
        plt.figtext(
            0.5,
            0.01,
            "CPU-backed EVM is the VRAM-saving path. GPU-backed EVM intentionally duplicates experts in this prototype to validate CUDA remapping, so it uses more VRAM.",
            ha="center",
            fontsize=9,
        )
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.24)
        memory_path = FIG_DIR / "evm_qwen15_moe_vram_by_mode.png"
        savefig(memory_path, dpi=160)
        plt.close()

        if REPRO_CSV_PATH.exists():
            repro = pd.read_csv(REPRO_CSV_PATH)
            repro_map = {
                "qwen15_cpu_only_baseline": "cpu_only_baseline",
                "qwen15_gpu_native_baseline": "gpu_native_baseline",
                "qwen15_cpu_backed_evm_33_memory": "cpu_backed_evm_33",
                "qwen15_gpu_backed_evm_33_memory": "gpu_backed_evm_33",
                "qwen15_gpu_backed_evm_66_memory": "gpu_backed_evm_66",
            }
            trade = mem_savings.copy()
            trade["repro_name"] = trade["name"].map(repro_map)
            trade = trade.merge(
                repro[["name", "generation_tps_mean", "generation_tps_std"]],
                left_on="repro_name",
                right_on="name",
                how="left",
                suffixes=("", "_repro"),
            )
            plt.figure(figsize=(8.6, 5.4))
            plt.scatter(trade["peak_gpu_memory_mb"], trade["generation_tps_mean"], s=95, c=colors)
            for _, row in trade.iterrows():
                plt.text(
                    row["peak_gpu_memory_mb"] + 180,
                    row["generation_tps_mean"],
                    row["label"].replace("\n", " "),
                    fontsize=8,
                    va="center",
                )
            plt.xlabel("Peak GPU memory used (MB)")
            plt.ylabel("Generation t/s, mean over 5 trials")
            plt.title("Qwen1.5-MoE Memory/Throughput Tradeoff")
            plt.figtext(
                0.5,
                0.01,
                "CPU-backed EVM trades native-GPU speed for VRAM headroom. GPU-backed EVM validates remapping and is not a VRAM-saving configuration.",
                ha="center",
                fontsize=9,
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            trade_path = FIG_DIR / "evm_qwen15_moe_vram_throughput_tradeoff.png"
            savefig(trade_path, dpi=160)
            plt.close()

        remap = mem[mem["name"].str.contains("gpu_backed")].copy()
        if not remap.empty:
            remap["label"] = (
                remap["name"]
                .str.replace("qwen15_", "", regex=False)
                .str.replace("_memory", "", regex=False)
                .str.replace("_", " ")
            )
            plt.figure(figsize=(7.5, 4.8))
            bars = plt.bar(remap["label"], remap["peak_gpu_memory_mb"], color="#ed7d31")
            for bar, value in zip(bars, remap["peak_gpu_memory_mb"]):
                plt.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
            plt.ylabel("Peak GPU memory used (MB)")
            plt.title("GPU-Backed EVM Remapping Validation Memory")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            remap_path = FIG_DIR / "evm_qwen15_moe_gpu_backed_vram_overhead.png"
            savefig(remap_path, dpi=160)
            plt.close()

    if REPRO_CSV_PATH.exists():
        repro = pd.read_csv(REPRO_CSV_PATH)
        repro["label"] = repro["name"].str.replace("_", " ")
        x = range(len(repro))
        plt.figure(figsize=(10, 5.2))
        plt.bar(
            [i - 0.18 for i in x],
            repro["prompt_tps_mean"],
            yerr=repro["prompt_tps_std"],
            width=0.36,
            label="Prompt t/s",
            capsize=3,
        )
        plt.bar(
            [i + 0.18 for i in x],
            repro["generation_tps_mean"],
            yerr=repro["generation_tps_std"],
            width=0.36,
            label="Generation t/s",
            capsize=3,
        )
        plt.xticks(list(x), repro["label"], rotation=25, ha="right")
        plt.ylabel("Tokens per second, mean +/- std over 5 trials")
        plt.title("Reproduction Trials: Averaged Runtime Throughput")
        plt.legend()
        plt.tight_layout()
        repro_path = FIG_DIR / "evm_qwen15_moe_reproduction_throughput.png"
        savefig(repro_path, dpi=160)
        plt.close()

    if CROSS_MODEL_CSV_PATH.exists():
        cross = pd.read_csv(CROSS_MODEL_CSV_PATH)
        pivot = cross.pivot(index="model", columns="mode", values="peak_gpu_memory_mb").reset_index()
        pivot["saved_mb"] = pivot["native_gpu"] - pivot["cpu_backed_evm_33"]
        pivot["saved_pct"] = 100.0 * pivot["saved_mb"] / pivot["native_gpu"]
        labels = pivot["model"].str.replace("_", " ")
        x = range(len(pivot))
        plt.figure(figsize=(9, 5))
        plt.bar([i - 0.18 for i in x], pivot["native_gpu"], width=0.36, label="Native GPU")
        plt.bar([i + 0.18 for i in x], pivot["cpu_backed_evm_33"], width=0.36, label="CPU-backed EVM 33%")
        for i, row in pivot.iterrows():
            plt.text(i + 0.18, row["cpu_backed_evm_33"], f"-{row['saved_pct']:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.xticks(list(x), labels, rotation=20, ha="right")
        plt.ylabel("Peak GPU memory used (MB)")
        plt.title("Cross-Model VRAM Residency Reduction")
        plt.legend()
        plt.tight_layout()
        cross_path = FIG_DIR / "evm_cross_model_vram_residency.png"
        savefig(cross_path, dpi=160)
        plt.close()

        if SEGREGATION_CSV_PATH.exists():
            seg = pd.read_csv(SEGREGATION_CSV_PATH)
        else:
            seg = pd.DataFrame()

        model_names = {
            "qwen15_moe": "Qwen1.5-MoE",
            "deepseek_coder_v2_lite": "DeepSeek-Coder-V2-Lite",
            "qwen2_57b_a14b": "Qwen2-57B-A14B",
        }
        for model_id, model_label in model_names.items():
            model_rows = cross[cross["model"] == model_id].copy()
            if model_rows.empty:
                continue
            rows = []
            native = model_rows[model_rows["mode"] == "native_gpu"].iloc[0]
            rows.append({
                "mode": "Native GPU",
                "peak_gpu_memory_mb": native["peak_gpu_memory_mb"],
                "generation_tps": native["generation_tps"],
                "color": "#4472c4",
            })
            cpu = model_rows[model_rows["mode"] == "cpu_backed_evm_33"].iloc[0]
            rows.append({
                "mode": "CPU-backed\nEVM 33%",
                "peak_gpu_memory_mb": cpu["peak_gpu_memory_mb"],
                "generation_tps": cpu["generation_tps"],
                "color": "#70ad47",
            })
            if not seg.empty:
                gpu_rows = seg[(seg["model"] == model_id) & (seg["status"] == "completed")]
                if not gpu_rows.empty:
                    gpu = gpu_rows.iloc[0]
                    gen_tps = 0.0
                    if model_id == "deepseek_coder_v2_lite" and DEEPSEEK_GPU_TRIALS_CSV_PATH.exists():
                        deepseek_trials = pd.read_csv(DEEPSEEK_GPU_TRIALS_CSV_PATH)
                        summary = deepseek_trials[deepseek_trials["trial"] == "summary"].iloc[0]
                        gen_tps = float(str(summary["generation_tps"]).split("+/-")[0])
                    elif model_id == "qwen15_moe" and REPRO_CSV_PATH.exists():
                        repro = pd.read_csv(REPRO_CSV_PATH)
                        qwen_gpu = repro[repro["name"] == "gpu_backed_evm_33"].iloc[0]
                        gen_tps = qwen_gpu["generation_tps_mean"]
                    rows.append({
                        "mode": "GPU-backed\nEVM 33%",
                        "peak_gpu_memory_mb": gpu["peak_gpu_memory_mb"],
                        "generation_tps": gen_tps,
                        "color": "#ed7d31",
                    })

            model_plot = pd.DataFrame(rows)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
            bars = ax1.bar(model_plot["mode"], model_plot["peak_gpu_memory_mb"], color=model_plot["color"])
            for bar, value in zip(bars, model_plot["peak_gpu_memory_mb"]):
                ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)
            ax1.set_ylabel("Peak GPU memory used (MB)")
            ax1.set_title("VRAM")
            bars = ax2.bar(model_plot["mode"], model_plot["generation_tps"], color=model_plot["color"])
            for bar, value in zip(bars, model_plot["generation_tps"]):
                ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{float(value):.1f}", ha="center", va="bottom", fontsize=8)
            ax2.set_ylabel("Generation tokens/s")
            ax2.set_title("Throughput")
            fig.suptitle(f"{model_label}: Native vs CPU-Backed vs GPU-Backed EVM")
            fig.text(
                0.5,
                0.01,
                "CPU-backed EVM is the VRAM-saving path. GPU-backed EVM validates CUDA remapping and may use more VRAM.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.2, top=0.86)
            model_path = FIG_DIR / f"evm_{model_id}_mode_comparison.png"
            fig.savefig(model_path, dpi=160)
            print(f"wrote {model_path}")
            plt.close(fig)

        if not seg.empty:
            completed = seg[seg["status"] == "completed"].copy()
            if not completed.empty:
                completed["label"] = completed["model"].map(model_names).fillna(completed["model"]) + "\n" + completed["mode"].str.replace("_", " ")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
                bars = ax1.bar(completed["label"], completed["hit_rate_pct"], color="#70ad47")
                for bar, value in zip(bars, completed["hit_rate_pct"]):
                    ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{float(value):.1f}%", ha="center", va="bottom", fontsize=8)
                ax1.set_ylim(0, 100)
                ax1.set_ylabel("EVM hit rate (%)")
                ax1.set_title("Cache hit rate")
                bars = ax2.bar(completed["label"], completed["peak_gpu_memory_mb"], color="#ed7d31")
                for bar, value in zip(bars, completed["peak_gpu_memory_mb"]):
                    ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)
                ax2.set_ylabel("Peak GPU memory used (MB)")
                ax2.set_title("GPU-backed VRAM")
                for ax in (ax1, ax2):
                    ax.tick_params(axis="x", labelrotation=20)
                fig.suptitle("GPU-Backed EVM Remapping Validation Summary")
                fig.tight_layout()
                fig.subplots_adjust(top=0.84)
                summary_path = FIG_DIR / "evm_gpu_backed_remapping_summary.png"
                fig.savefig(summary_path, dpi=160)
                print(f"wrote {summary_path}")
                plt.close(fig)

    if DEEPSEEK_GPU_TRIALS_CSV_PATH.exists():
        deepseek = pd.read_csv(DEEPSEEK_GPU_TRIALS_CSV_PATH)
        trials = deepseek[deepseek["trial"] != "summary"].copy()
        trials["trial"] = trials["trial"].astype(int)
        trials["generation_tps"] = trials["generation_tps"].astype(float)
        trials["hit_rate_pct"] = trials["hit_rate_pct"].astype(float)
        x = range(len(trials))
        fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
        ax1.bar([i - 0.18 for i in x], trials["generation_tps"], width=0.36, color="#4472c4", label="Generation t/s")
        ax1.set_ylabel("Generation tokens/s")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([f"Trial {v}" for v in trials["trial"]])
        ax2 = ax1.twinx()
        ax2.plot(list(x), trials["hit_rate_pct"], color="#70ad47", marker="o", label="Hit rate")
        ax2.set_ylabel("EVM hit rate (%)")
        ax2.set_ylim(0, 100)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.title("DeepSeek-Coder-V2-Lite GPU-Backed EVM 33% Trials")
        plt.figtext(
            0.5,
            0.01,
            "GPU-backed EVM validates CUDA remapping with EVM_TARGET_EXPERT_COUNT=64; this is not the VRAM-saving path.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        deepseek_path = FIG_DIR / "evm_deepseek_coder_v2_lite_gpu_backed_trials.png"
        fig.savefig(deepseek_path, dpi=160)
        print(f"wrote {deepseek_path}")
        plt.close(fig)

    if MAX_CONTEXT_CSV_PATH.exists():
        max_ctx = pd.read_csv(MAX_CONTEXT_CSV_PATH)
        for model_id, model_label in {
            "qwen15_moe": "Qwen1.5-MoE",
            "deepseek_coder_v2_lite": "DeepSeek-Coder-V2-Lite",
        }.items():
            model = max_ctx[max_ctx["model"] == model_id].copy()
            if model.empty:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 5))
            for mode, label, color in [
                ("native_gpu", "Native GPU", "#4472c4"),
                ("cpu_backed_evm_33", "CPU-backed EVM 33%", "#70ad47"),
            ]:
                rows = model[model["mode"] == mode].sort_values("ctx_len")
                ax.plot(rows["ctx_len"], rows["peak_gpu_memory_mb"], marker="o", label=label, color=color)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Allocated context length")
            ax.set_ylabel("Peak GPU memory used (MB)")
            ax.set_title(f"{model_label}: VRAM Headroom vs Context Length")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            out = WORKFLOW_FIG_DIR / f"evm_{model_id}_max_context_vram.png"
            fig.savefig(out, dpi=160)
            print(f"wrote {out}")
            plt.close(fig)

        latest = (
            max_ctx.sort_values("ctx_len")
            .groupby(["model", "mode"], as_index=False)
            .tail(1)
            .copy()
        )
        latest["label"] = latest["model"].str.replace("_", " ") + "\n" + latest["mode"].str.replace("_", " ")
        fig, ax = plt.subplots(figsize=(9.5, 5))
        colors = ["#4472c4" if mode == "native_gpu" else "#70ad47" for mode in latest["mode"]]
        bars = ax.bar(latest["label"], latest["peak_gpu_memory_mb"], color=colors)
        for bar, value in zip(bars, latest["peak_gpu_memory_mb"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Peak GPU memory used (MB)")
        ax.set_title("Max Tested Context: Native GPU vs CPU-backed EVM")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        out = WORKFLOW_FIG_DIR / "evm_max_context_summary.png"
        fig.savefig(out, dpi=160)
        print(f"wrote {out}")
        plt.close(fig)

    if SPECULATIVE_WORKFLOW_CSV_PATH.exists():
        spec = pd.read_csv(SPECULATIVE_WORKFLOW_CSV_PATH)
        spec["label"] = spec["name"].str.replace("_", " ")
        spec["status_value"] = spec["success"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#70ad47" if value == 1 else "#c00000" for value in spec["status_value"]]
        bars = ax.bar(spec["label"], spec["status_value"], color=colors)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Completed inside smoke window")
        ax.set_title("Speculative Workflow Smoke Status")
        for bar, success in zip(bars, spec["status_value"]):
            ax.text(bar.get_x() + bar.get_width() / 2, 1.02 if success else 0.08, "pass" if success else "timeout", ha="center", fontsize=9)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        out = WORKFLOW_FIG_DIR / "evm_speculative_workflow_status.png"
        fig.savefig(out, dpi=160)
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
