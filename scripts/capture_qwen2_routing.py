import json
import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-routing-observatory.exe"
MODEL = ROOT / "models" / "qwen2-57b-a14b-instruct-q4_k_m.gguf"
TRACE = ROOT / "results" / "routing" / "qwen2_generation_trace.jsonl"
OUT = ROOT / "data" / "qwen2_routing_telemetry.parquet"
SUMMARY = ROOT / "results" / "routing" / "qwen2_generation_summary.json"

PROMPTS = [
    "Explain virtual memory in two sentences.",
    "Write a compact Python function that reverses a string.",
    "What causes a cache miss?",
    "Summarize the purpose of mixture-of-experts models.",
    "Give three practical uses for a long context window.",
    "Solve: if 3x + 5 = 20, what is x?",
    "Describe a reliable software benchmark in one paragraph.",
    "Compare an LRU cache with perfect future knowledge.",
]


def main():
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if TRACE.exists():
        TRACE.unlink()
    env = os.environ.copy()
    env["EVM_ROUTING_TRACE_PATH"] = str(TRACE)
    outcomes = []
    for prompt_id, prompt in enumerate(PROMPTS):
        command = [
            str(EXE), "-m", str(MODEL), "-p", prompt,
            "-c", "256", "-n", "24", "-ngl", "99", "-ub", "1",
            "-s", str(prompt_id), "--temp", "0.0",
        ]
        try:
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, timeout=300)
            outcomes.append({"prompt_id": prompt_id, "returncode": completed.returncode, "timed_out": False})
        except subprocess.TimeoutExpired:
            outcomes.append({"prompt_id": prompt_id, "returncode": None, "timed_out": True})

    records = []
    if TRACE.exists():
        with TRACE.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                probs = row["probs"]
                ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
                records.append({
                    "prompt_id": row["prompt_id"],
                    "token_id": row["token_idx"],
                    "layer_id": row["layer_id"],
                    "top_k_expert_ids": ranked[:8],
                    "router_probabilities": probs,
                })
    if records:
        pd.DataFrame(records).to_parquet(OUT, index=False)
    summary = {
        "prompts": len(PROMPTS),
        "valid_prompts": sum(item["returncode"] == 0 for item in outcomes),
        "timed_out_prompts": sum(item["timed_out"] for item in outcomes),
        "trace_rows": len(records),
        "layers": max((item["layer_id"] for item in records), default=-1) + 1,
        "experts": len(records[0]["router_probabilities"]) if records else 0,
        "trace": str(TRACE),
        "parquet": str(OUT),
        "outcomes": outcomes,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("prompts", "valid_prompts", "timed_out_prompts", "trace_rows", "layers", "experts")}, indent=2))
    raise SystemExit(0 if summary["valid_prompts"] == len(PROMPTS) and records else 1)


if __name__ == "__main__":
    main()
