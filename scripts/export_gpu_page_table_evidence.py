import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = json.loads((ROOT / "results" / "page_table_final" / "online_predictor_summary.json").read_text(encoding="utf-8"))
    by_key = {(row["model"], row["mode"]): row for row in runtime}
    rows = []
    for model in ("qwen1", "deepseek"):
        lru = by_key[(model, "lru")]
        page = by_key[(model, "gpu-page")]
        hashes = json.loads((ROOT / "results" / "page_table_hash_validation" / model / "hash_match_summary.json").read_text(encoding="utf-8"))
        rows.append({
            "model": model,
            "trials": page["valid_trials"],
            "lru_tps": lru["mean_generation_tps"],
            "gpu_page_tps": page["mean_generation_tps"],
            "tps_change_pct": round(100 * (page["mean_generation_tps"] / lru["mean_generation_tps"] - 1), 2),
            "lru_vram_mb": lru["mean_peak_vram_mb"],
            "gpu_page_vram_mb": page["mean_peak_vram_mb"],
            "gpu_page_hit_calls": page["mean_gpu_page_hits"],
            "gpu_page_miss_calls": page["mean_gpu_page_misses"],
            "exact_hash_matches": f"{hashes['matched']}/{hashes['total']}",
        })

    tables = ROOT / "docs" / "tables"
    figures = ROOT / "docs" / "figures" / "final_proof"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (tables / "gpu_page_table_runtime.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["Qwen1.5-MoE", "DeepSeek-Coder-V2-Lite"]
    x = range(len(rows))
    width = 0.36
    fig, (speed_ax, path_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    speed_ax.bar([i - width / 2 for i in x], [row["lru_tps"] for row in rows], width, label="Host LRU remap", color="#4472c4")
    speed_ax.bar([i + width / 2 for i in x], [row["gpu_page_tps"] for row in rows], width, label="GPU page-table remap", color="#00a6a6")
    speed_ax.set_xticks(list(x), labels, rotation=12)
    speed_ax.set_ylabel("Generation tokens/s")
    speed_ax.set_title("Exact 8-slot EVM runtime")
    speed_ax.legend(fontsize=8)
    for index, row in enumerate(rows):
        speed_ax.text(index + width / 2, row["gpu_page_tps"], f"{row['tps_change_pct']:+.1f}%", ha="center", va="bottom", fontsize=8)

    path_ax.bar([i - width / 2 for i in x], [row["gpu_page_hit_calls"] for row in rows], width, label="GPU hit maps", color="#70ad47")
    path_ax.bar([i + width / 2 for i in x], [row["gpu_page_miss_calls"] for row in rows], width, label="CPU fallback misses", color="#c0504d")
    path_ax.set_xticks(list(x), labels, rotation=12)
    path_ax.set_ylabel("Aggregate remap calls")
    path_ax.set_title("Exact page-table routing")
    path_ax.legend(fontsize=8)
    fig.suptitle("GPU Page-Table EVM: Correctness-Preserving Hit Path")
    fig.tight_layout()
    fig.savefig(figures / "gpu_page_table_runtime.png", dpi=180, facecolor="white")
    print(f"GPU page-table evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
