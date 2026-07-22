import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-routing-observatory.exe"
PROMPTS = ROOT / "config" / "predictor" / "router_training_prompts.json"
MODELS = {
    "qwen1": "models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf",
    "deepseek": "models/DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf",
    "qwen2": "results/expert_vault/qwen2_full_vault/qwen2-spine.gguf",
}


def main():
    parser = argparse.ArgumentParser(description="Capture full router vectors for a model-specific EVM predictor.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--max-prompts", type=int, default=0, help="cap the deterministic prompt contract; 0 uses all prompts")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reuse-trace", action="store_true", help="convert an existing trace without rerunning inference")
    args = parser.parse_args()
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace = args.out_dir / "routing_trace.jsonl"
    parquet = args.out_dir / "router_vectors.parquet"
    summary_path = args.out_dir / "capture_summary.json"
    if trace.exists() and not args.reuse_trace:
        trace.unlink()
    env = os.environ.copy()
    env["EVM_ROUTING_TRACE_PATH"] = str(trace)
    if args.model == "qwen2":
        vault = ROOT / "results" / "expert_vault" / "qwen2_full_vault"
        env.update({
            "EVM_EXPERTS_PER_TENSOR": "8",
            "EVM_TARGET_EXPERT_COUNT": "64",
            "EVM_CPU_BACKING": "1",
            "EVM_CUDA_STREAMING": "1",
            "EVM_EXPERT_VAULT_INDEX": str(vault / "experts.pack.idx"),
            "EVM_EXPERT_VAULT_PACK": str(vault / "experts.pack"),
            "EVM_STRICT_BUDGET": "1",
            "EVM_PREFILL_BATCH_THRESHOLD": "999",
        })
    outcomes = []
    for prompt_id, prompt in ([] if args.reuse_trace else enumerate(prompts)):
        command = [str(EXE), "-m", str(ROOT / MODELS[args.model]), "-p", prompt, "-c", str(args.ctx),
                   "-n", str(args.tokens), "-ngl", "99", "-ub", "1", "-s", str(prompt_id), "--temp", "0.0"]
        try:
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=300)
            outcomes.append({"prompt_id": prompt_id, "returncode": completed.returncode, "timed_out": False})
        except subprocess.TimeoutExpired:
            outcomes.append({"prompt_id": prompt_id, "returncode": None, "timed_out": True})
    records = []
    malformed_rows = 0
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            probs = row["probs"]
            ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
            records.append({"prompt_id": int(row["prompt_id"]), "token_id": int(row["token_idx"]),
                            "layer_id": int(row["layer_id"]), "top_k_expert_ids": ranked[:8], "router_probabilities": probs})
    if records:
        pd.DataFrame(records).to_parquet(parquet, index=False)
    summary = {"model": args.model, "capture_mode": "exact_evm_vault" if args.model == "qwen2" else "native_gpu", "prompts": len(prompts), "valid_prompts": len(prompts) if args.reuse_trace else sum(row["returncode"] == 0 for row in outcomes),
               "timed_out_prompts": sum(row["timed_out"] for row in outcomes), "trace_rows": len(records),
               "malformed_trace_rows": malformed_rows,
               "layers": max((row["layer_id"] for row in records), default=-1) + 1,
               "experts": len(records[0]["router_probabilities"]) if records else 0, "trace": str(trace), "parquet": str(parquet)}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("model", "prompts", "valid_prompts", "trace_rows", "layers", "experts")}, separators=(",", ":")))
    raise SystemExit(0 if summary["valid_prompts"] == len(prompts) and records else 1)


if __name__ == "__main__":
    main()
