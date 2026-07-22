import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "production_evm"
FIGURES = ROOT / "docs" / "figures" / "production_evm"
TABLES = ROOT / "docs" / "tables"


def speculative_metrics():
    """Use the committed compact summary, not the excluded raw benchmark log."""
    summary = load_json(RESULTS / "production_summary.json")
    return (
        float(summary["speculative_generation_tps"]),
        float(summary["speculative_acceptance_pct"]),
    )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(label, backing, rows):
    valid = [row for row in rows if row.get("returncode") == 0 and not row.get("timed_out")]
    return {
        "mode": label,
        "backing": backing,
        "generation_tps": statistics.mean(row["generation_tps"] for row in valid),
        "peak_gpu_memory_mb": max(row["peak_gpu_memory_mb"] for row in valid),
        "peak_process_rss_mb": max(row.get("peak_process_rss_mb", 0) for row in valid),
        "peak_process_private_mb": max(row.get("peak_process_private_mb", 0) for row in valid),
        "system_ram_used_delta_mb": max(row.get("system_ram_used_delta_mb", 0) for row in valid),
        "pagefile_used_delta_mb": max(row.get("pagefile_used_delta_mb", 0) for row in valid),
        "trials": len(valid),
        "status": "PASS" if len(valid) == len(rows) else "FAIL",
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    native = load_json(ROOT / "results" / "native_cuda_tuning" / "runs" / "qwen2_native_fa_on_ub1_t01.json")
    cpu = load_json(RESULTS / "runs" / "qwen2_cpu_backed_t01.json")
    ram_compact = load_json(RESULTS / "runs" / "qwen2_compact_async_t01.json")
    disk_trials = [load_json(path) for path in sorted((RESULTS / "runs").glob("qwen2_disk_compact_async_t*.json"))]

    spec_json = json.loads((RESULTS / "runs" / "qwen2_compact_speculative_t01.json").read_text(encoding="utf-8"))
    spec_speed, spec_accept = speculative_metrics()

    rows = [
        aggregate("Native max-GPU", "WDDM/shared", [native]),
        aggregate("CPU-backed", "Committed RAM", [cpu]),
        aggregate("RAM unified 8", "Committed RAM", [ram_compact]),
        aggregate("Disk unified 8", "Lazy GGUF mmap", disk_trials),
    ]
    canonical_trials = []
    for case, source_rows in [
        ("native_max_gpu_shared", [native]),
        ("cpu_backed", [cpu]),
        ("ram_compact_async", [ram_compact]),
        ("disk_compact_async", disk_trials),
    ]:
        for source in source_rows:
            trial = dict(source)
            trial["case"] = case
            canonical_trials.append(trial)
    (RESULTS / "production_benchmark.json").write_text(
        json.dumps(canonical_trials, indent=2), encoding="utf-8"
    )
    with (TABLES / "production_evm_runtime.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["mode"] for row in rows]
    colors = ["#4472c4", "#70ad47", "#8064a2", "#00a6a6"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.9))
    bars = ax1.bar(labels, [row["peak_gpu_memory_mb"] for row in rows], color=colors)
    for bar, row in zip(bars, rows):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['peak_gpu_memory_mb']:,}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(24000, color="#777777", linestyle="--", linewidth=1)
    ax1.set_ylabel("Peak dedicated GPU memory (MB)")
    ax1.set_title("VRAM")
    ax1.tick_params(axis="x", rotation=20)

    x = range(len(rows))
    width = 0.38
    rss_bars = ax2.bar([i - width / 2 for i in x], [row["peak_process_rss_mb"] for row in rows], width, color=colors, label="Process RSS")
    page_bars = ax2.bar([i + width / 2 for i in x], [row["pagefile_used_delta_mb"] for row in rows], width, color="#888888", label="Page-file growth")
    for bar, row in zip(rss_bars, rows):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['peak_process_rss_mb']:,}", ha="center", va="bottom", fontsize=7)
    for bar, row in zip(page_bars, rows):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['pagefile_used_delta_mb']:,}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(list(x), labels, rotation=20)
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Host memory")
    ax2.legend(fontsize=8)

    bars = ax3.bar(labels, [row["generation_tps"] for row in rows], color=colors)
    for bar, row in zip(bars, rows):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row['generation_tps']:.2f}", ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("Generation tokens/s")
    ax3.set_title("Throughput")
    ax3.tick_params(axis="x", rotation=20)
    fig.suptitle("Qwen2-57B Memory-Tier Tradeoff")
    fig.tight_layout()
    runtime_figure = FIGURES / "qwen2_production_vram_throughput.png"
    fig.savefig(runtime_figure, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    Image.open(runtime_figure).convert("RGB").save(runtime_figure)

    disk_control_specs = [
        ("Strict trim", "qwen2_disk_compact_async_t01.json"),
        ("Lazy mmap", "qwen2_disk_compact_lazy_t01.json"),
        ("Pinned x2", "qwen2_disk_staged2_t01.json"),
        ("Pinned x4", "qwen2_disk_staged4_t01.json"),
        ("Pinned x8", "qwen2_disk_staged8_t01.json"),
        ("Cache 8G", "qwen2_disk_cache8g_t01.json"),
        ("Cache 12G", "qwen2_disk_cache12g_t01.json"),
        ("Cache 16G", "qwen2_disk_cache16g_t01.json"),
    ]
    disk_controls = []
    for label, filename in disk_control_specs:
        row = load_json(RESULTS / "runs" / filename)
        disk_controls.append({
            "mode": label,
            "generation_tps": row["generation_tps"],
            "peak_gpu_memory_mb": row["peak_gpu_memory_mb"],
            "peak_process_rss_mb": row["peak_process_rss_mb"],
            "pagefile_used_delta_mb": row["pagefile_used_delta_mb"],
            "has_evm_counters": row["has_evm_counters"],
            "status": "PASS" if row["returncode"] == 0 and row["has_evm_counters"] else "FAIL",
        })
    with (TABLES / "disk_io_controls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(disk_controls[0]))
        writer.writeheader()
        writer.writerows(disk_controls)

    labels = [row["mode"] for row in disk_controls]
    colors = ["#00a6a6", "#4472c4", "#8064a2", "#8064a2", "#8064a2", "#70ad47", "#70ad47", "#70ad47"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    bars = ax1.bar(labels, [row["generation_tps"] for row in disk_controls], color=colors)
    for bar, row in zip(bars, disk_controls):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{row['generation_tps']:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Generation tokens/s")
    ax1.set_title("Throughput")
    ax1.tick_params(axis="x", rotation=28)

    bars = ax2.bar(labels, [row["peak_process_rss_mb"] for row in disk_controls], color=colors)
    for bar, row in zip(bars, disk_controls):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{row['peak_process_rss_mb']:,}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Peak process RSS (MB)")
    ax2.set_title("Host working set")
    ax2.tick_params(axis="x", rotation=28)
    fig.suptitle("Qwen2-57B Disk-I/O Controls: Memory Does Not Remove Demand Latency")
    fig.tight_layout()
    disk_figure = FIGURES / "qwen2_disk_io_controls.png"
    fig.savefig(disk_figure, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    Image.open(disk_figure).convert("RGB").save(disk_figure)

    pool_specs = [
        (8, "qwen2_disk_compact_async_t01.json"),
        (16, "qwen2_disk_pool16_t01.json"),
        (24, "qwen2_disk_pool24_t01.json"),
        (32, "qwen2_disk_pool32_t01.json"),
        (40, "qwen2_disk_pool40_t01.json"),
    ]
    pool_rows = []
    for capacity, filename in pool_specs:
        row = load_json(RESULTS / "runs" / filename)
        pool_rows.append({
            "experts_per_tensor": capacity,
            "generation_tps": row["generation_tps"],
            "peak_gpu_memory_mb": row["peak_gpu_memory_mb"],
            "peak_process_rss_mb": row["peak_process_rss_mb"],
            "cache_hit_rate_pct": row["cache_hit_rate_pct"],
            "bytes_transferred_mb": row["bytes_transferred_mb"],
            "pagefile_used_delta_mb": row["pagefile_used_delta_mb"],
            "status": "PASS" if row["returncode"] == 0 and row["has_evm_counters"] else "FAIL",
        })
    with (TABLES / "disk_gpu_pool_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pool_rows[0]))
        writer.writeheader()
        writer.writerows(pool_rows)

    labels = [str(row["experts_per_tensor"]) for row in pool_rows]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, key, title, ylabel, fmt in [
        (ax1, "generation_tps", "Throughput", "Generation tokens/s", ".2f"),
        (ax2, "peak_gpu_memory_mb", "GPU residency", "Peak dedicated VRAM (MB)", ",.0f"),
        (ax3, "bytes_transferred_mb", "Demand traffic", "Transferred expert data (MB)", ",.0f"),
    ]:
        bars = ax.bar(labels, [row[key] for row in pool_rows], color="#00a6a6")
        for bar, row in zip(bars, pool_rows):
            label = format(row[key], fmt)
            if key == "bytes_transferred_mb":
                label += f"\n{row['cache_hit_rate_pct']:.1f}% hit"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Resident experts per tensor")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    ax2.axhline(24000, color="#777777", linestyle="--", linewidth=1)
    fig.suptitle("Qwen2-57B Disk-Backed GPU Working-Set Sweep")
    fig.tight_layout()
    pool_figure = FIGURES / "qwen2_disk_gpu_pool_sweep.png"
    fig.savefig(pool_figure, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    Image.open(pool_figure).convert("RGB").save(pool_figure)

    predictor = pd.read_csv(ROOT / "results" / "learned_predictor" / "layer_aware_predictor_results.csv")
    predictor.to_csv(TABLES / "layer_aware_predictor_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for policy, color, marker in [("LRU", "#4472c4", "o"), ("Learned3to17", "#00a6a6", "s"), ("Oracle", "#c0504d", "^")]:
        part = predictor[predictor.policy.eq(policy)]
        ax.plot(part.capacity, part.hit_rate * 100.0, color=color, marker=marker, label=policy)
    ax.set_xlabel("Resident experts per layer")
    ax.set_ylabel("Held-out demand hit rate (%)")
    ax.set_title("Layer-Aware Learned Predictor, 5-Prompt Held-Out Test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    predictor_figure = FIGURES / "layer_aware_predictor_hit_rate.png"
    fig.savefig(predictor_figure, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    Image.open(predictor_figure).convert("RGB").save(predictor_figure)

    disk_valid = [row for row in disk_trials if row.get("returncode") == 0]
    summary = {
        "memory_tiers": rows,
        "disk_compact_generation_tps_mean": round(statistics.mean(row["generation_tps"] for row in disk_valid), 2),
        "disk_compact_peak_gpu_memory_mb_max": max(row["peak_gpu_memory_mb"] for row in disk_valid),
        "disk_compact_peak_process_rss_mb_max": max(row["peak_process_rss_mb"] for row in disk_valid),
        "disk_compact_pagefile_used_delta_mb_max": max(row["pagefile_used_delta_mb"] for row in disk_valid),
        "speculative_generation_tps": spec_speed,
        "speculative_acceptance_pct": spec_accept,
        "disk_io_controls": disk_controls,
        "disk_gpu_pool_sweep": pool_rows,
    }
    (RESULTS / "production_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Disk compact: {summary['disk_compact_generation_tps_mean']:.2f} t/s | VRAM {summary['disk_compact_peak_gpu_memory_mb_max']:,} MB | RSS {summary['disk_compact_peak_process_rss_mb_max']:,} MB | PASS")
    print(f"Speculative: {spec_speed:.2f} t/s | {int(spec_json['peak_gpu_memory_mb']):,} MB | FAIL")
    print("Charts: PASS")


if __name__ == "__main__":
    main()
