import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_budgeted_llama.py"
DEFAULT_PAYLOADS = ROOT / "config" / "mri_diagnostic_payloads.json"


def load_payloads(path, split, domain_filter, limit):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") == "evm-mri-domain-library-v2":
        from mri_domain_library import compile_suite, load_and_validate
        payload = compile_suite(load_and_validate(path), split, domain_filter, limit)
    elif payload.get("format") != "evm-mri-diagnostic-suite-v1" or not payload.get("domains"):
        raise ValueError("invalid MRI diagnostic payload suite")
    return payload


def add_env(command, key, value):
    if value is not None:
        command += ["--env", f"{key}={value}"]


def main():
    parser = argparse.ArgumentParser(description="Capture categorized offline EVM routing-count profiles.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-count", type=int, required=True)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--gpu-budget-mb", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--trials-per-category", type=int, help="legacy repeat count; defaults to every selected prompt")
    parser.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    parser.add_argument("--split", choices=("calibration", "validation", "held_out"), default="calibration")
    parser.add_argument("--domains", help="comma-separated v2 domain IDs")
    parser.add_argument("--max-prompts-per-domain", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="retain completed rows and append missing payloads")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vault-index", type=Path)
    parser.add_argument("--vault-pack", type=Path)
    parser.add_argument("--ability-index", type=Path)
    args = parser.parse_args()
    suite = load_payloads(args.payloads, args.split, args.domains, args.max_prompts_per_domain)
    prompts_by_domain = {name: row["prompts"] for name, row in suite["domains"].items()}
    profiles = args.out_dir / "profiles"
    runs = args.out_dir / "runs"
    profiles.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for category, prompts in prompts_by_domain.items():
        profile = profiles / f"{category}.jsonl"
        if not args.resume:
            profile.unlink(missing_ok=True)
        trial_count = args.trials_per_category or len(prompts)
        for trial in range(trial_count):
            name = f"{args.name}_{category}_{trial + 1}"
            row_path = runs / f"{name}.json"
            if args.resume and row_path.exists():
                row = json.loads(row_path.read_text(encoding="utf-8"))
                outcomes.append({"category": category, "trial": trial + 1, "prompt_id": f"{category}.{args.split}.{trial + 1}",
                                 "valid": row.get("returncode") == 0, "tps": row.get("generation_tps", 0), "vram_mb": row.get("peak_gpu_memory_mb", 0)})
                continue
            command = [
                "python", str(RUNNER), "--name", name, "--model", str(args.model),
                "--prompt", prompts[trial % len(prompts)], "--tokens", str(args.tokens),
                "--ctx", "256", "--ngl", "99", "--ub", "1", "--kv", "gpu",
                "--gpu-budget-mb", str(args.gpu_budget_mb), "--require-evm-counters",
                "--timeout-s", "600", "--out-dir", str(runs),
            ]
            add_env(command, "EVM_EXPERTS_PER_TENSOR", args.capacity)
            add_env(command, "EVM_TARGET_EXPERT_COUNT", args.expert_count)
            add_env(command, "EVM_CPU_BACKING", 1)
            add_env(command, "EVM_CUDA_STREAMING", 1)
            add_env(command, "EVM_STRICT_BUDGET", 1)
            add_env(command, "EVM_PREFILL_BATCH_THRESHOLD", 999)
            add_env(command, "EVM_ROUTING_PROFILE_PATH", profile.resolve())
            add_env(command, "EVM_EXPERT_VAULT_INDEX", args.vault_index.resolve() if args.vault_index else None)
            add_env(command, "EVM_EXPERT_VAULT_PACK", args.vault_pack.resolve() if args.vault_pack else None)
            add_env(command, "EVM_ABILITY_PACK_INDEX", args.ability_index.resolve() if args.ability_index else None)
            add_env(command, "EVM_ABILITY_PACK_ONLY", 0 if args.ability_index else None)
            command += ["--", "--mmap", "--ignore-eos"]
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            row = json.loads(row_path.read_text(encoding="utf-8")) if row_path.exists() else {}
            outcomes.append({"category": category, "trial": trial + 1, "prompt_id": f"{category}.{args.split}.{trial + 1}",
                             "valid": completed.returncode == 0 and row.get("returncode") == 0, "tps": row.get("generation_tps", 0), "vram_mb": row.get("peak_gpu_memory_mb", 0)})
    summary = {
        "model": str(args.model), "runs": len(outcomes),
        "valid_runs": sum(row["valid"] for row in outcomes), "outcomes": outcomes,
        "payload_suite": str(args.payloads.resolve()),
        "payload_format": suite["format"],
        "source_format": suite.get("source_format", suite["format"]),
        "split": suite.get("split", args.split),
        "domains": {name: {"description": row["description"], "prompt_count": len(row["prompts"])} for name, row in suite["domains"].items()},
        "profiles": {category: str(profiles / f"{category}.jsonl") for category in prompts_by_domain},
    }
    (args.out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.name}: {summary['valid_runs']}/{summary['runs']} categorized profiles | {'PASS' if summary['valid_runs'] == summary['runs'] else 'FAIL'}")
    raise SystemExit(0 if summary["valid_runs"] == summary["runs"] else 1)


if __name__ == "__main__":
    main()
