import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
TASKS = [
    ("arithmetic", "Answer only: 3 + 4 =", r"\b7\b"),
    ("geography", "Answer only: the capital of France is", r"paris"),
    ("chemistry", "Answer only: the chemical formula for water is", r"h\s*2\s*o"),
    ("complexity", "Answer only: binary search time complexity is", r"log"),
    ("antonym", "Answer only: the opposite of hot is", r"cold"),
]
CONFIGS = {
    "qwen1": {
        "original": "models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf", "spine": "results/ability_packs/qwen1_full_vault/spine.gguf",
        "expert_count": 60, "packs": {25.0: (15, "results/ability_packs/qwen1_universal_25"), 37.5: (23, "results/ability_packs/qwen1_universal_37p5")},
    },
    "deepseek": {
        "original": "models/DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf", "spine": "results/ability_packs/deepseek_full_vault/spine.gguf",
        "expert_count": 64, "packs": {25.0: (16, "results/ability_packs/deepseek_universal_25"), 37.5: (24, "results/ability_packs/deepseek_universal_37p5")},
    },
}


def clean_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("EVM_")}


def run(model, prompt, env, tokens):
    command = [str(EXE), "-m", str(model), "-p", prompt, "-n", str(tokens), "-c", "128", "-ngl", "99", "-ub", "1",
               "--temp", "0", "--ignore-eos", "--no-display-prompt", "--simple-io", "--log-disable"]
    completed = subprocess.run(command, input="/exit\n", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="ignore", env=env, timeout=300)
    output = completed.stdout.strip()
    return completed.returncode, output


def pack_env(config, percentage):
    count, folder = config["packs"][percentage]
    pack = ROOT / folder
    env = clean_env()
    env.update({
        "EVM_EXPERTS_PER_TENSOR": str(count), "EVM_TARGET_EXPERT_COUNT": str(config["expert_count"]),
        "EVM_GPU_PACK_ONLY": "1", "EVM_CUDA_STREAMING": "1",
        "EVM_EXPERT_VAULT_INDEX": str(pack / "experts.pack.idx"), "EVM_EXPERT_VAULT_PACK": str(pack / "experts.pack"),
        "EVM_ABILITY_PACK_INDEX": str(pack / "experts.pack.idx"), "EVM_ABILITY_PACK_ONLY": "1",
        "EVM_ROUTER_MASK_UNAVAILABLE": "1",
        "EVM_STRICT_BUDGET": "1", "EVM_PREFILL_BATCH_THRESHOLD": "999",
    })
    return env


def main():
    parser = argparse.ArgumentParser(description="Compare cross-model ability packs with their original model sanity baseline.")
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = CONFIGS[args.model]
    result = {"model": args.model, "tasks": [], "baseline_passed": 0, "packs": {"25.0": [], "37.5": []}}
    for task, prompt, pattern in TASKS:
        code, output = run(ROOT / config["original"], prompt, clean_env(), args.tokens)
        baseline_pass = code == 0 and re.search(pattern, output, re.I) is not None
        result["tasks"].append({"task": task, "pass": baseline_pass, "returncode": code,
                                "output_sha256": hashlib.sha256(output.encode()).hexdigest(), "output_bytes": len(output.encode())})
        result["baseline_passed"] = sum(row["pass"] for row in result["tasks"])
        for percentage in (25.0, 37.5):
            code, output = run(ROOT / config["spine"], prompt, pack_env(config, percentage), args.tokens)
            passed = code == 0 and re.search(pattern, output, re.I) is not None
            result["packs"][str(percentage)].append({"task": task, "pass": passed, "baseline_pass": baseline_pass,
                                                     "returncode": code, "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                                                     "output_bytes": len(output.encode())})
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["pack_summary"] = {}
    for percentage, rows in result["packs"].items():
        result["pack_summary"][percentage] = {
            "passed": sum(row["pass"] for row in rows), "total": len(rows),
            "retained_baseline_passes": sum(row["pass"] and row["baseline_pass"] for row in rows),
            "baseline_pass_opportunities": sum(row["baseline_pass"] for row in rows),
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.model}: baseline {result['baseline_passed']}/5 | 25% {result['pack_summary']['25.0']['passed']}/5 | 37.5% {result['pack_summary']['37.5']['passed']}/5")


if __name__ == "__main__":
    main()
