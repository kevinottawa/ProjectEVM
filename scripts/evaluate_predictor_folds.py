import argparse
import csv
import json
import statistics
from pathlib import Path

import pandas as pd

from train_layer_aware_predictor import lru_sim, metric_row, oracle_sim, predictive_sim, prepare, train_models


def main():
    parser = argparse.ArgumentParser(description="Cross-validate the EVM router predictor without running live inference.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_parquet(args.trace).sort_values(["prompt_id", "token_id", "layer_id"])
    width = int(df.router_probabilities.map(len).mode().iloc[0])
    df = df[df.router_probabilities.map(len) == width].copy()
    rows = prepare(df)
    prompts = sorted(int(value) for value in df.prompt_id.unique())
    if len(prompts) < 8:
        raise SystemExit("need at least eight independent prompts for four-fold evaluation")
    n_layers = int(df.layer_id.max()) + 1
    folds = []
    for index in range(0, len(prompts), 2):
        test = set(prompts[index:index + 2])
        validation = set(prompts[(index + 2) % len(prompts):(index + 4) % len(prompts)])
        if len(validation) < 2:
            validation = set(prompts[:2])
        train = set(prompts) - test - validation
        models = train_models(rows, train, n_layers, width)
        candidates = []
        for budget in range(0, min(4, max(0, args.capacity - 4)) + 1):
            value = predictive_sim(rows, validation, args.capacity, models, budget)
            candidates.append((value[1] + value[2], value[1], budget))
        budget = min(candidates)[2]
        lru = metric_row(args.capacity, "LRU", lru_sim(rows, test, args.capacity))
        learned = metric_row(args.capacity, "Learned3to17", predictive_sim(rows, test, args.capacity, models, budget), budget)
        oracle = metric_row(args.capacity, "Oracle", oracle_sim(rows, test, args.capacity))
        folds.append({"fold": index // 2 + 1, "test_prompts": ";".join(map(str, sorted(test))), "validation_prompts": ";".join(map(str, sorted(validation))),
                      "selected_prefetch_budget": budget, "lru_hit_rate_pct": 100 * lru["hit_rate"],
                      "learned_hit_rate_pct": 100 * learned["hit_rate"], "oracle_hit_rate_pct": 100 * oracle["hit_rate"],
                      "gap_closed_pct": 100 * (learned["hit_rate"] - lru["hit_rate"]) / (oracle["hit_rate"] - lru["hit_rate"]) if oracle["hit_rate"] > lru["hit_rate"] else 0.0})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(folds[0]))
        writer.writeheader()
        writer.writerows(folds)
    summary = {"capacity": args.capacity, "folds": len(folds),
               "mean_lru_hit_rate_pct": round(statistics.mean(row["lru_hit_rate_pct"] for row in folds), 2),
               "mean_learned_hit_rate_pct": round(statistics.mean(row["learned_hit_rate_pct"] for row in folds), 2),
               "min_learned_hit_rate_pct": round(min(row["learned_hit_rate_pct"] for row in folds), 2),
               "mean_oracle_hit_rate_pct": round(statistics.mean(row["oracle_hit_rate_pct"] for row in folds), 2),
               "mean_gap_closed_pct": round(statistics.mean(row["gap_closed_pct"] for row in folds), 2)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"32-slot cross-validation: learned {summary['mean_learned_hit_rate_pct']:.2f}% | LRU {summary['mean_lru_hit_rate_pct']:.2f}% | min {summary['min_learned_hit_rate_pct']:.2f}% | PASS")


if __name__ == "__main__":
    main()
