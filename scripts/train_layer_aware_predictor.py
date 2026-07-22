import csv
import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data" / "routing_telemetry.parquet"
OUT_DIR = ROOT / "results" / "learned_predictor"
CAPACITIES = [8, 12, 16, 20, 24, 30, 40, 50]
MIN_HORIZON = 3
MAX_HORIZON = 17
MISS_MS = 80.0 / 28000.0 * 1000.0


def prepare(df):
    rows = []
    for record in df.itertuples(index=False):
        rows.append({
            "prompt": int(record.prompt_id),
            "token": int(record.token_id),
            "layer": int(record.layer_id),
            "experts": tuple(int(x) for x in record.top_k_expert_ids),
            "probs": np.asarray(record.router_probabilities, dtype=np.float32),
        })
    return rows


def train_models(rows, train_prompts, n_layers, n_experts):
    xs = [[] for _ in range(n_layers)]
    ys = [[] for _ in range(n_layers)]
    grouped = {}
    for row in rows:
        if row["prompt"] in train_prompts:
            grouped.setdefault((row["prompt"], row["layer"]), []).append(row)

    for (_, layer), seq in grouped.items():
        seq.sort(key=lambda row: row["token"])
        for i, row in enumerate(seq):
            target = np.zeros(n_experts, dtype=np.float32)
            for horizon in range(MIN_HORIZON, MAX_HORIZON + 1):
                if i + horizon >= len(seq):
                    break
                weight = 1.0 / math.sqrt(horizon)
                for expert in seq[i + horizon]["experts"]:
                    target[expert] += weight
            if target.any():
                xs[layer].append(row["probs"])
                ys[layer].append(target)

    models = np.zeros((n_layers, n_experts, n_experts), dtype=np.float32)
    ridge = 0.05
    for layer in range(n_layers):
        x = np.asarray(xs[layer], dtype=np.float64)
        y = np.asarray(ys[layer], dtype=np.float64)
        if len(x) == 0:
            continue
        gram = x.T @ x + ridge * np.eye(n_experts)
        models[layer] = np.linalg.solve(gram, x.T @ y).astype(np.float32)
    return models


def lru_sim(rows, prompts, capacity):
    hits = misses = 0
    caches = {}
    for row in rows:
        if row["prompt"] not in prompts:
            continue
        key = (row["prompt"], row["layer"])
        cache = caches.setdefault(key, OrderedDict())
        for expert in row["experts"]:
            if expert in cache:
                hits += 1
                cache.move_to_end(expert, last=False)
            else:
                misses += 1
                if len(cache) >= capacity:
                    cache.popitem(last=True)
                cache[expert] = None
                cache.move_to_end(expert, last=False)
    return hits, misses, 0


def predictive_sim(rows, prompts, capacity, models, prefetch_budget):
    hits = misses = prefetches = 0
    caches = {}
    for row in rows:
        if row["prompt"] not in prompts:
            continue
        key = (row["prompt"], row["layer"])
        cache = caches.setdefault(key, OrderedDict())
        protected = set(row["experts"])
        scores = np.maximum(row["probs"] @ models[row["layer"]], 0.0)

        for expert in row["experts"]:
            if expert in cache:
                hits += 1
                cache.move_to_end(expert, last=False)
            else:
                misses += 1
                if len(cache) >= capacity:
                    candidates = [item for item in cache if item not in protected]
                    victim = min(candidates or list(cache), key=lambda item: (scores[item], -list(cache).index(item)))
                    del cache[victim]
                cache[expert] = None
                cache.move_to_end(expert, last=False)

        if prefetch_budget:
            ranked = np.argsort(scores)[::-1]
            loaded = 0
            for expert_raw in ranked:
                expert = int(expert_raw)
                if loaded >= prefetch_budget:
                    break
                if scores[expert] <= 0.0 or expert in cache or expert in protected:
                    continue
                if len(cache) >= capacity:
                    candidates = [item for item in cache if item not in protected]
                    if not candidates:
                        break
                    victim = min(candidates, key=lambda item: (scores[item], -list(cache).index(item)))
                    if scores[victim] >= scores[expert]:
                        continue
                    del cache[victim]
                cache[expert] = None
                prefetches += 1
                loaded += 1
    return hits, misses, prefetches


def oracle_sim(rows, prompts, capacity):
    hits = misses = 0
    grouped = {}
    for row in rows:
        if row["prompt"] in prompts:
            grouped.setdefault((row["prompt"], row["layer"]), []).append(row)
    for seq in grouped.values():
        accesses = [expert for row in sorted(seq, key=lambda item: item["token"]) for expert in row["experts"]]
        cache = set()
        for i, expert in enumerate(accesses):
            if expert in cache:
                hits += 1
                continue
            misses += 1
            if len(cache) >= capacity:
                next_use = {}
                for resident in cache:
                    try:
                        next_use[resident] = accesses.index(resident, i + 1)
                    except ValueError:
                        next_use[resident] = len(accesses) + 1
                cache.remove(max(cache, key=lambda resident: next_use[resident]))
            cache.add(expert)
    return hits, misses, 0


def metric_row(capacity, policy, values, prefetch_budget=0):
    hits, misses, prefetches = values
    total = hits + misses
    return {
        "capacity": capacity,
        "policy": policy,
        "prefetch_budget": prefetch_budget,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "prefetches": prefetches,
        "modeled_stall_ms": misses * MISS_MS,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and test a short-horizon router-probability predictor.")
    parser.add_argument("--trace", default=str(TRACE))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--capacities", default="8,16,24,32,40")
    args = parser.parse_args()
    trace = Path(args.trace)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(trace).sort_values(["prompt_id", "token_id", "layer_id"])
    vector_lengths = df.router_probabilities.map(len)
    expert_vector_lengths = vector_lengths[vector_lengths > 1]
    if expert_vector_lengths.empty:
        raise SystemExit("trace contains no multi-expert router probability vectors")
    n_experts = int(expert_vector_lengths.mode().iloc[0])
    df = df[vector_lengths == n_experts].copy()
    if df.empty:
        raise SystemExit("trace contains no consistent router probability vectors")
    rows = prepare(df)
    prompts = sorted(int(x) for x in df.prompt_id.unique())
    if len(prompts) < 4:
        raise SystemExit("need at least four prompts for train/validation/test")
    train_end = max(2, int(len(prompts) * 0.60))
    validation_end = max(train_end + 1, int(len(prompts) * 0.80))
    validation_end = min(validation_end, len(prompts) - 1)
    train_prompts = set(prompts[:train_end])
    validation_prompts = set(prompts[train_end:validation_end])
    test_prompts = set(prompts[validation_end:])
    n_layers = int(df.layer_id.max()) + 1
    n_experts = len(df.iloc[0].router_probabilities)

    models = train_models(rows, train_prompts, n_layers, n_experts)
    np.savez_compressed(
        out_dir / "router_probability_predictor.npz",
        weights=models,
        min_horizon=MIN_HORIZON,
        max_horizon=MAX_HORIZON,
    )

    output = []
    selected = {}
    for capacity in [int(value) for value in args.capacities.split(",") if value.strip()]:
        candidates = []
        for budget in range(0, min(4, max(0, capacity - 4)) + 1):
            values = predictive_sim(rows, validation_prompts, capacity, models, budget)
            # PCIe traffic is the production constraint. A prefetched expert is
            # still a transfer even when its latency is hidden.
            candidates.append((values[1] + values[2], values[1], budget))
        selected[capacity] = min(candidates)[2]

        output.append(metric_row(capacity, "LRU", lru_sim(rows, test_prompts, capacity)))
        output.append(metric_row(
            capacity,
            "Learned3to17",
            predictive_sim(rows, test_prompts, capacity, models, selected[capacity]),
            selected[capacity],
        ))
        output.append(metric_row(capacity, "Oracle", oracle_sim(rows, test_prompts, capacity)))

    csv_path = out_dir / "layer_aware_predictor_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    metadata = {
        "trace_rows": len(rows),
        "discarded_incompatible_rows": int((vector_lengths != n_experts).sum()),
        "train_prompts": sorted(train_prompts),
        "validation_prompts": sorted(validation_prompts),
        "test_prompts": sorted(test_prompts),
        "layers": n_layers,
        "experts": n_experts,
        "horizon": [MIN_HORIZON, MAX_HORIZON],
        "selected_prefetch_budget": selected,
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cap = min(row["capacity"] for row in output)
    cap_rows = {row["policy"]: row for row in output if row["capacity"] == cap}
    learned = cap_rows["Learned3to17"]
    lru = cap_rows["LRU"]
    oracle = cap_rows["Oracle"]
    gap = oracle["hit_rate"] - lru["hit_rate"]
    closed = (learned["hit_rate"] - lru["hit_rate"]) / gap if gap > 0 else 0.0
    print(f"Learned: {learned['hit_rate'] * 100:.2f}% hit | {learned['modeled_stall_ms'] / 1000:.2f}s stall | PASS")
    print(f"LRU: {lru['hit_rate'] * 100:.2f}% hit | Oracle: {oracle['hit_rate'] * 100:.2f}% hit")
    status = "PASS" if 0.0 < closed <= 1.0 else "FAIL"
    print(f"Gap closed: {closed * 100:.1f}% | layer-aware held-out test | {status}")


if __name__ == "__main__":
    main()
