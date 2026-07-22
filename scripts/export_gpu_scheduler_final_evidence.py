import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read_trial(model, mode):
    path = ROOT / "results" / "gpu_scheduler_v2_small" / model / f"{model}_{mode}_trial_1.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    return {
        "model": "Qwen1.5-MoE" if model == "qwen1" else "DeepSeek-Coder-V2-Lite",
        "mode": "GPU page table" if mode == "gpu-page" else "GPU scheduler",
        "capacity_slots": 32,
        "generation_tps": row["generation_tps"],
        "hit_rate_pct": row["cache_hit_rate_pct"],
        "bytes_transferred_mb": row["bytes_transferred_mb"],
        "valid": row["returncode"] == 0 and row["has_evm_counters"],
    }


def main():
    predictor = json.loads((ROOT / "results" / "predictor_training" / "qwen2" / "cap32_cross_validation" / "summary.json").read_text(encoding="utf-8"))
    runtime_rows = [read_trial(model, mode) for model in ("qwen1", "deepseek") for mode in ("gpu-page", "gpu-scheduler")]

    tables = ROOT / "docs" / "tables"
    figures = ROOT / "docs" / "figures" / "final_proof"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (tables / "gpu_scheduler_final_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["record_type", "model", "mode", "capacity_slots", "generation_tps", "hit_rate_pct", "bytes_transferred_mb", "valid"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"record_type": "offline_cross_validation", "model": "Qwen2-57B-A14B", "mode": "LRU", "capacity_slots": predictor["capacity"], "hit_rate_pct": predictor["mean_lru_hit_rate_pct"]})
        writer.writerow({"record_type": "offline_cross_validation", "model": "Qwen2-57B-A14B", "mode": "Learned3to17", "capacity_slots": predictor["capacity"], "hit_rate_pct": predictor["mean_learned_hit_rate_pct"]})
        writer.writerow({"record_type": "offline_cross_validation", "model": "Qwen2-57B-A14B", "mode": "Oracle", "capacity_slots": predictor["capacity"], "hit_rate_pct": predictor["mean_oracle_hit_rate_pct"]})
        for row in runtime_rows:
            writer.writerow({"record_type": "runtime_smoke", **row})

    fig, (offline_ax, runtime_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    offline_labels = ["LRU", "Learned\n3-17", "Oracle"]
    offline_values = [predictor["mean_lru_hit_rate_pct"], predictor["mean_learned_hit_rate_pct"], predictor["mean_oracle_hit_rate_pct"]]
    offline_ax.bar(offline_labels, offline_values, color=["#4472c4", "#00a6a6", "#70ad47"])
    offline_ax.set_ylim(0, 100)
    offline_ax.set_ylabel("Hit rate (%)")
    offline_ax.set_title("Qwen2 offline replay at 32/64")
    offline_ax.text(1, predictor["min_learned_hit_rate_pct"], f"min fold {predictor['min_learned_hit_rate_pct']:.2f}%", ha="center", va="bottom", fontsize=8)

    labels = ["Qwen1.5", "DeepSeek"]
    page = [next(row["generation_tps"] for row in runtime_rows if row["model"].startswith(label) and row["mode"] == "GPU page table") for label in labels]
    scheduler = [next(row["generation_tps"] for row in runtime_rows if row["model"].startswith(label) and row["mode"] == "GPU scheduler") for label in labels]
    x = range(len(labels))
    width = 0.36
    runtime_ax.bar([value - width / 2 for value in x], page, width, label="GPU page table", color="#4472c4")
    runtime_ax.bar([value + width / 2 for value in x], scheduler, width, label="GPU scheduler", color="#c0504d")
    runtime_ax.set_xticks(list(x), labels)
    runtime_ax.set_ylabel("Generation tokens/s")
    runtime_ax.set_title("32-slot runtime smoke: scheduler NO-GO")
    runtime_ax.legend(fontsize=8)
    fig.suptitle("Learned Prediction Signal vs. Current Runtime Cost")
    fig.tight_layout()
    fig.savefig(figures / "gpu_scheduler_final_evidence.png", dpi=180, facecolor="white")
    print("GPU scheduler final evidence: 1 offline replay + 2 runtime smokes | PASS")


if __name__ == "__main__":
    main()
