import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CASES = (
    ("qwen1", 75.0, ROOT / "results/pack_refinement/qwen1/router_mask_75_quality.json",
     ROOT / "results/router_mask_runtime/qwen1_router_mask_75_smoke.json"),
    ("deepseek", 75.0, ROOT / "results/pack_refinement/deepseek/router_mask_75_quality.json",
     ROOT / "results/router_mask_runtime/deepseek_router_mask_75_smoke.json"),
    ("qwen2", 37.5, ROOT / "results/pack_refinement/qwen2_router_mask_37p5_quality.json",
     ROOT / "results/router_mask_runtime/qwen2_router_mask_37p5_smoke.json"),
)


def main():
    rows = []
    for model, capacity, quality_path, runtime_path in CASES:
        if not quality_path.exists():
            continue
        quality = json.loads(quality_path.read_text(encoding="utf-8"))["summary"]
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path and runtime_path.exists() else {}
        for method in ("frequency", "graph"):
            score = quality[method]
            rows.append({
                "model": model,
                "capacity_pct": capacity,
                "method": method,
                "baseline_passed": quality["baseline_passed"],
                "total": quality["total"],
                "quality_passed": score["passed"],
                "retained_baseline_passes": score["retained_baseline_passes"],
                "baseline_opportunities": score["baseline_opportunities"],
                "runtime_valid": bool(runtime and runtime.get("returncode") == 0 and runtime.get("has_evm_counters")),
                "generation_tps": runtime.get("generation_tps", ""),
                "peak_vram_mb": runtime.get("peak_gpu_memory_mb", ""),
                "pack_substitutions": runtime.get("pack_substitutions", ""),
            })
    if not rows:
        raise SystemExit("Router-mask evidence: no completed result files")
    target = ROOT / "docs/tables/router_mask_quality.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Router-mask evidence: {len(rows)} rows | PASS")


if __name__ == "__main__":
    main()
