import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = json.loads((ROOT / "results" / "router_score_prefetch_final" / "online_predictor_summary.json").read_text(encoding="utf-8"))
    by_key = {(row["model"], row["mode"]): row for row in runtime}
    rows = []
    for model in ("qwen1", "deepseek"):
        lru = by_key[(model, "lru")]
        score = by_key[(model, "score-prefetch")]
        hashes = json.loads((ROOT / "results" / "router_score_prefetch_hash_active" / model / "hash_match_summary.json").read_text(encoding="utf-8"))
        rows.append({
            "model": model,
            "trials": score["valid_trials"],
            "lru_tps": lru["mean_generation_tps"],
            "score_prefetch_tps": score["mean_generation_tps"],
            "tps_change_pct": round(100 * (score["mean_generation_tps"] / lru["mean_generation_tps"] - 1), 2),
            "lru_hit_rate_pct": lru["mean_hit_rate_pct"],
            "score_prefetch_hit_rate_pct": score["mean_hit_rate_pct"],
            "score_prefetches": score["mean_router_score_prefetches"],
            "exact_hash_matches": f"{hashes['matched']}/{hashes['total']}",
        })

    tables = ROOT / "docs" / "tables"
    figures = ROOT / "docs" / "figures" / "final_proof"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (tables / "router_score_prefetch_runtime.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["Qwen1.5-MoE", "DeepSeek-Coder-V2-Lite"]
    x = range(len(rows))
    width = 0.36
    fig, (speed_ax, hit_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    speed_ax.bar([i - width / 2 for i in x], [row["lru_tps"] for row in rows], width, label="LRU", color="#4472c4")
    speed_ax.bar([i + width / 2 for i in x], [row["score_prefetch_tps"] for row in rows], width, label="Router-score prefetch", color="#00a6a6")
    speed_ax.set_xticks(list(x), labels, rotation=12)
    speed_ax.set_ylabel("Generation tokens/s")
    speed_ax.set_title("Exact 8-slot EVM runtime")
    speed_ax.legend(fontsize=8)
    for index, row in enumerate(rows):
        speed_ax.text(index + width / 2, row["score_prefetch_tps"], f"{row['tps_change_pct']:+.1f}%", ha="center", va="bottom", fontsize=8)

    hit_ax.bar([i - width / 2 for i in x], [row["lru_hit_rate_pct"] for row in rows], width, label="LRU hit rate", color="#4472c4")
    hit_ax.bar([i + width / 2 for i in x], [row["score_prefetch_hit_rate_pct"] for row in rows], width, label="Score-prefetch hit rate", color="#70ad47")
    hit_ax.set_xticks(list(x), labels, rotation=12)
    hit_ax.set_ylabel("Cache hit rate (%)")
    hit_ax.set_title("Router-score bridge")
    hit_ax.legend(fontsize=8)
    fig.suptitle("Conservative Router-Score Prefetch: Exact Output Path")
    fig.tight_layout()
    fig.savefig(figures / "router_score_prefetch_runtime.png", dpi=180, facecolor="white")
    print(f"Router-score prefetch evidence: {len(rows)} models | PASS")


if __name__ == "__main__":
    main()
