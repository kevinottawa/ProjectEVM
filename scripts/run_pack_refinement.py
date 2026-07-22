import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MRI_BATCH = ROOT / "scripts" / "mri_batch.py"
VAULT_BUILDER = ROOT / "scripts" / "build_expert_vault.py"
QUALITY = ROOT / "scripts" / "compare_workflow_pack_quality.py"
MODELS = {
    "qwen1": {"source": "models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf", "experts": 60},
    "deepseek": {"source": "models/DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf", "experts": 64},
}
WORKFLOW_DOMAINS = "code_generation,debugging,code_review,systems_programming,tool_calling,planning,structured_output,repository_navigation"


def invoke(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", errors="ignore")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode:
        raise RuntimeError(lines[-1] if lines else "subprocess failed")
    return lines[-1] if lines else "PASS"


def main():
    parser = argparse.ArgumentParser(description="Derive and evaluate small-model frequency versus core-overlay pack refinements.")
    parser.add_argument("--models", default="qwen1,deepseek")
    parser.add_argument("--capacities", default="50,75")
    parser.add_argument("--core-fraction", type=float, default=0.65)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "pack_refinement")
    args = parser.parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    capacities = [float(value) for value in args.capacities.split(",")]
    summary = {"format": "evm-pack-refinement-v1", "models": {}, "capacities": capacities, "core_fraction": args.core_fraction}
    for model in models:
        if model not in MODELS:
            raise SystemExit(f"unknown model: {model}")
        model_result = summary["models"].setdefault(model, {})
        calibration = ROOT / "results" / "mri_batch" / model / "calibration"
        for capacity in capacities:
            label = str(capacity).replace(".", "p")
            selections = {}
            packs = {}
            for method in ("frequency", "core_overlay"):
                selection = calibration / "packs" / f"{method}_{label}.json"
                invoke([sys.executable, str(MRI_BATCH), "build-packs", "--model", model, "--out-dir", str(calibration),
                        "--domains", WORKFLOW_DOMAINS, "--percentage", str(capacity), "--method", method,
                        "--core-fraction", str(args.core_fraction)], args.out_dir / "runner.log")
                generated = calibration / "packs" / f"{method}_{label}.json"
                if generated != selection:
                    generated.replace(selection)
                selections[method] = selection
                pack_dir = args.out_dir / model / f"{method}_{label}"
                invoke([sys.executable, str(VAULT_BUILDER), "--model", str(ROOT / MODELS[model]["source"]),
                        "--selection", str(selection), "--out", str(pack_dir), "--verify"], pack_dir / "build.log")
                packs[method] = pack_dir
            quality_path = args.out_dir / model / f"quality_{label}.json"
            resident_count = min(MODELS[model]["experts"], math.ceil(MODELS[model]["experts"] * capacity / 100.0))
            invoke([sys.executable, str(QUALITY), "--model", model, "--frequency-pack", str(packs["frequency"]),
                    "--graph-pack", str(packs["core_overlay"]), "--frequency-label", "frequency", "--graph-label", "core_overlay",
                    "--frequency-count", str(resident_count), "--graph-count", str(resident_count),
                    "--out", str(quality_path), "--tokens", str(args.tokens)], args.out_dir / "runner.log")
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            model_result[label] = {"quality_file": str(quality_path), "baseline_passed": quality["summary"]["baseline_passed"],
                                   "total": quality["summary"]["total"], "frequency": quality["summary"]["frequency"],
                                   "core_overlay": quality["summary"]["core_overlay"], "packs": {key: str(value) for key, value in packs.items()}}
        (args.out_dir / "refinement_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Pack refinement: " + " | ".join(f"{model} {len(capacities)} capacities" for model in models) + " | PASS")


if __name__ == "__main__":
    main()
