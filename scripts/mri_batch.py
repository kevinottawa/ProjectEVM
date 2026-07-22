import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from mri_domain_library import DEFAULT_LIBRARY, load_and_validate


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-evm-batch-profiler.exe"
DEFAULT_CORPUS = ROOT / "results" / "mri_batch" / "corpus" / "calibration_100.json"
MODELS = {
    "qwen1": {"model": "models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf", "experts": 60, "capacity": 8},
    "deepseek": {"model": "models/DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf", "experts": 64, "capacity": 8},
    "qwen2": {"model": "results/expert_vault/qwen2_full_vault/qwen2-spine.gguf", "experts": 64, "capacity": 8,
              "vault_index": "results/expert_vault/qwen2_full_vault/experts.pack.idx",
              "vault_pack": "results/expert_vault/qwen2_full_vault/experts.pack"},
}
VARIANTS = (
    "Complete the following task directly: {prompt}",
    "Give a concise, accurate response to this task: {prompt}",
    "Work through this carefully, then answer: {prompt}",
    "State any necessary assumption before answering: {prompt}",
    "Answer for a knowledgeable beginner: {prompt}",
    "Answer for an experienced practitioner: {prompt}",
    "Provide the result first, followed by a short explanation: {prompt}",
    "Use a short numbered sequence where appropriate: {prompt}",
    "Focus on correctness rather than breadth: {prompt}",
    "Include one relevant edge case in the response: {prompt}",
    "Verify the proposed answer before presenting it: {prompt}",
    "Explain the key decision involved in this task: {prompt}",
    "Avoid unnecessary background and complete this task: {prompt}",
    "Use precise terminology while answering: {prompt}",
    "Give a practical response to the following: {prompt}",
    "Identify the most likely failure mode while answering: {prompt}",
    "Include one contrast with a nearby but incorrect approach: {prompt}",
    "Respond in a compact professional style: {prompt}",
    "Treat this as an independent test item: {prompt}",
    "Solve this without relying on prior conversation: {prompt}",
    "Preserve every constraint in the following request: {prompt}",
    "Give an answer that can be checked objectively: {prompt}",
    "Separate the conclusion from supporting reasoning: {prompt}",
    "Consider ambiguity in the request, then answer: {prompt}",
    "Complete this task and mention one way to validate the result: {prompt}",
)


def generate_corpus(library_path, out_path, per_domain):
    library = load_and_validate(library_path)
    domains = {}
    for name, row in library["domains"].items():
        seeds = row["calibration"]
        prompts = []
        for index in range(per_domain):
            seed = seeds[index % len(seeds)]
            variant = VARIANTS[(index // len(seeds)) % len(VARIANTS)]
            cycle = index // (len(seeds) * len(VARIANTS))
            prompt = variant.format(prompt=seed)
            if cycle:
                prompt = f"Independent variation {cycle + 1}. {prompt}"
            prompts.append(prompt)
        domains[name] = {"description": row["description"], "group": row["group"], "excludes": row["excludes"],
                         "contrast_domains": row["contrast_domains"], "prompts": prompts}
    corpus = {"format": "evm-mri-diagnostic-suite-v1", "source_format": library["format"], "source_version": library["version"],
              "split": "calibration", "generation": {"method": "deterministic-seed-variant-v1", "per_domain": per_domain},
              "description": "Expanded local MRI calibration corpus", "domains": domains}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    return corpus


def validate_corpus(path, expected_per_domain=0):
    corpus = json.loads(path.read_text(encoding="utf-8"))
    failures, seen = [], set()
    if corpus.get("format") != "evm-mri-diagnostic-suite-v1" or corpus.get("split") != "calibration":
        failures.append("format or split")
    for name, row in corpus.get("domains", {}).items():
        prompts = row.get("prompts", [])
        if expected_per_domain and len(prompts) != expected_per_domain:
            failures.append(f"{name}: expected {expected_per_domain}, got {len(prompts)}")
        for prompt in prompts:
            normalized = " ".join(prompt.lower().split())
            if normalized in seen:
                failures.append(f"duplicate prompt in {name}")
            seen.add(normalized)
    if failures:
        raise ValueError("; ".join(failures))
    return len(corpus["domains"]), len(seen)


def clean_evm_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("EVM_")}


def run_batch(model_name, manifest, out_dir, tokens, context, cold_experts):
    config = MODELS[model_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = clean_evm_env()
    env.update({"EVM_EXPERTS_PER_TENSOR": str(config["capacity"]), "EVM_TARGET_EXPERT_COUNT": str(config["experts"]),
                "EVM_CUDA_STREAMING": "1", "EVM_STRICT_BUDGET": "1", "EVM_PREFILL_BATCH_THRESHOLD": "999"})
    if config.get("vault_index"):
        env.update({"EVM_CPU_BACKING": "1", "EVM_EXPERT_VAULT_INDEX": str(ROOT / config["vault_index"]),
                    "EVM_EXPERT_VAULT_PACK": str(ROOT / config["vault_pack"])})
    command = [str(BINARY), "-m", str(ROOT / config["model"]), "--manifest", str(manifest.resolve()),
               "--profile", str((out_dir / "routing.jsonl").resolve()), "--rows", str((out_dir / "rows.jsonl").resolve()),
               "--checkpoint", str((out_dir / "checkpoint.txt").resolve()), "-n", str(tokens), "-c", str(context), "-ngl", "99"]
    if cold_experts:
        command.append("--cold-experts")
    started = time.perf_counter()
    with (out_dir / "run.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", errors="ignore", env=env)
    wall_time = round(time.perf_counter() - started, 2)
    summary_line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "Batch MRI: no summary | FAIL"
    (out_dir / "batch_run.json").write_text(json.dumps({"model": model_name, "manifest": str(manifest), "wall_time_s": wall_time,
                                                        "cold_experts": cold_experts, "returncode": completed.returncode,
                                                        "summary": summary_line}, indent=2) + "\n", encoding="utf-8")
    print(f"{summary_line} | {wall_time:.2f} s wall")
    return completed.returncode


def summarize(out_dir):
    rows_path = out_dir / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()] if rows_path.exists() else []
    latest = {row["prompt_id"]: row for row in rows}
    valid = [row for row in latest.values() if row.get("valid")]
    profile_path = out_dir / "routing.jsonl"
    profile_rows = sum(1 for line in profile_path.open(encoding="utf-8") if line.strip()) if profile_path.exists() else 0
    speeds = [row["generation_tps"] for row in valid]
    summary = {"prompts": len(latest), "valid_prompts": len(valid), "failed_prompts": len(latest) - len(valid),
               "mean_generation_tps": round(statistics.mean(speeds), 2) if speeds else 0,
               "profile_rows": profile_rows, "prefill_generation_split": True,
               "cold_experts": valid[0]["cold_experts"] if valid else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"MRI summary: {summary['valid_prompts']}/{summary['prompts']} valid | {summary['mean_generation_tps']:.2f} t/s | {profile_rows} profile rows | {'PASS' if not summary['failed_prompts'] else 'FAIL'}")
    return 0 if not summary["failed_prompts"] else 1


def main():
    parser = argparse.ArgumentParser(description="Generate, run, resume, and summarize persistent local MRI calibration.")
    parser.add_argument("command", choices=("generate", "validate", "run", "summary", "analyze", "build-packs"))
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--per-domain", type=int, default=100)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--warm-experts", action="store_true")
    parser.add_argument("--domains", help="comma-separated target domains for build-packs")
    parser.add_argument("--percentage", type=float, default=37.5)
    parser.add_argument("--method", choices=("frequency", "graph", "core_overlay"), default="graph")
    parser.add_argument("--core-fraction", type=float, default=0.65)
    args = parser.parse_args()
    if args.command == "generate":
        corpus = generate_corpus(args.library, args.manifest, args.per_domain)
        print(f"MRI corpus: {len(corpus['domains'])} domains | {sum(len(row['prompts']) for row in corpus['domains'].values())} prompts | PASS")
        return
    if args.command == "validate":
        domains, prompts = validate_corpus(args.manifest, args.per_domain)
        print(f"MRI corpus validation: {domains} domains | {prompts} unique prompts | PASS")
        return
    if not args.out_dir:
        if not args.model:
            raise SystemExit("--model or --out-dir is required")
        args.out_dir = ROOT / "results" / "mri_batch" / args.model / "calibration"
    if args.command == "run":
        if not args.model:
            raise SystemExit("--model is required")
        if not BINARY.exists():
            raise SystemExit(f"batch profiler binary missing: {BINARY}")
        raise SystemExit(run_batch(args.model, args.manifest, args.out_dir, args.tokens, args.ctx, not args.warm_experts))
    if args.command == "analyze":
        command = ["python", str(ROOT / "scripts" / "mri_cloud_analyze.py"), "analyze", "--manifest", str(args.manifest),
                   "--rows", str(args.out_dir / "rows.jsonl"), "--routing", str(args.out_dir / "routing.jsonl"),
                   "--out-dir", str(args.out_dir / "analysis")]
        raise SystemExit(subprocess.run(command).returncode)
    if args.command == "build-packs":
        if not args.model or not args.domains:
            raise SystemExit("--model and --domains are required")
        command = ["python", str(ROOT / "scripts" / "mri_cloud_analyze.py"), "build-pack",
                   "--analysis-dir", str(args.out_dir / "analysis"), "--domains", args.domains,
                   "--experts-per-layer", str(MODELS[args.model]["experts"]), "--percentage", str(args.percentage),
                   "--method", args.method,
                   "--core-fraction", str(args.core_fraction),
                   "--out", str(args.out_dir / "packs" / (args.method + "_" + str(args.percentage).replace(".", "p") + ".json"))]
        raise SystemExit(subprocess.run(command).returncode)
    raise SystemExit(summarize(args.out_dir))


if __name__ == "__main__":
    main()
