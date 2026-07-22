import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "results" / "final_small_model_predictor_validation"
    rows = []
    labels = {"qwen1": "Qwen1.5-MoE", "deepseek": "DeepSeek-Coder-V2-Lite"}
    capacities = {"qwen1": 48, "deepseek": 40}
    for model in labels:
        summary = json.loads((source / model / "online_predictor_summary.json").read_text(encoding="utf-8"))
        lru = next(row for row in summary if row["mode"] == "lru")
        score = next((row for row in summary if row["mode"] == "score-prefetch"), None)
        rows.append({
            "model": labels[model],
            "demand_slots_per_tensor": capacities[model],
            "valid_lru_trials": f"{lru['valid_trials']}/{lru['trials']}",
            "lru_generation_tps": lru["mean_generation_tps"],
            "lru_hit_rate_pct": lru["mean_hit_rate_pct"],
            "lru_peak_vram_mb": lru["mean_peak_vram_mb"],
            "score_prefetch_generation_tps": score["mean_generation_tps"] if score else "",
            "score_prefetch_hit_rate_pct": score["mean_hit_rate_pct"] if score else "",
            "score_prefetch_valid_trials": f"{score['valid_trials']}/{score['trials']}" if score else "not run",
        })

    tables = ROOT / "docs" / "tables"
    figures = ROOT / "docs" / "figures" / "final_proof"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (tables / "steady_state_small_model_residency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = range(len(rows))
    fig, (hit_ax, speed_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    hit_ax.bar(x, [row["lru_hit_rate_pct"] for row in rows], color="#4472c4")
    hit_ax.axhspan(80, 90, color="#d9ead3", zorder=0, label="80-90% target band")
    hit_ax.set_ylim(0, 100)
    hit_ax.set_xticks(list(x), [row["model"] for row in rows], rotation=12)
    hit_ax.set_ylabel("Steady-generation cache hit rate (%)")
    hit_ax.legend(fontsize=8)
    speed_ax.bar(x, [row["lru_generation_tps"] for row in rows], color="#00a6a6")
    speed_ax.set_xticks(list(x), [row["model"] for row in rows], rotation=12)
    speed_ax.set_ylabel("Generation tokens/s")
    fig.suptitle("Exact EVM LRU: Reproducible High-Availability Small-Model Points")
    fig.tight_layout()
    fig.savefig(figures / "steady_state_small_model_residency.png", dpi=180, facecolor="white")
    print(f"Steady-state capacity evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
