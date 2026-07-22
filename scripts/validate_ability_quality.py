import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"
SPINE = ROOT / "results" / "expert_vault" / "qwen2_full_vault" / "qwen2-spine.gguf"
TASKS = [
    ("arithmetic", "Answer only: 3 + 4 =", r"\b7\b"),
    ("geography", "Answer only: the capital of France is", r"paris"),
    ("chemistry", "Answer only: the chemical formula for water is", r"h\s*2\s*o"),
    ("complexity", "Answer only: binary search time complexity is", r"log"),
    ("antonym", "Answer only: the opposite of hot is", r"cold"),
]


def run_pack(size, tokens, checkpoint, all_results):
    pack = ROOT / "results" / "ability_packs" / f"qwen2_universal_{size}"
    env = os.environ.copy()
    env.update({
        "EVM_EXPERTS_PER_TENSOR": str(size),
        "EVM_TARGET_EXPERT_COUNT": "64",
        "EVM_GPU_PACK_ONLY": "1",
        "EVM_CUDA_STREAMING": "1",
        "EVM_EXPERT_VAULT_INDEX": str(pack / "experts.pack.idx"),
        "EVM_EXPERT_VAULT_PACK": str(pack / "experts.pack"),
        "EVM_ABILITY_PACK_INDEX": str(pack / "experts.pack.idx"),
        "EVM_ABILITY_PACK_ONLY": "1",
        "EVM_ROUTER_MASK_UNAVAILABLE": "1",
        "EVM_STRICT_BUDGET": "1",
        "EVM_PREFILL_BATCH_THRESHOLD": "999",
    })
    result = {"pack_size": size, "passed": 0, "total": len(TASKS), "tasks": []}
    all_results.append(result)
    for name, prompt, pattern in TASKS:
        command = [str(EXE), "-m", str(SPINE), "-p", prompt, "-n", str(tokens), "-c", "128",
                   "-ngl", "99", "-ub", "1", "--temp", "0", "--ignore-eos", "--no-display-prompt",
                   "--simple-io", "--log-disable"]
        completed = subprocess.run(command, input="/exit\n", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, encoding="utf-8", errors="ignore", env=env, timeout=300)
        output = completed.stdout.strip()
        result["tasks"].append({
            "task": name,
            "pass": completed.returncode == 0 and re.search(pattern, output, re.I) is not None,
            "returncode": completed.returncode,
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "output_bytes": len(output.encode("utf-8")),
        })
        result["passed"] = sum(row["pass"] for row in result["tasks"])
        checkpoint.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run compact deterministic sanity tasks against pack-only Qwen2.")
    parser.add_argument("--sizes", default="16,24")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ability_packs" / "quality_summary.json")
    args = parser.parse_args()
    results = []
    for size in args.sizes.split(","):
        run_pack(int(size), args.tokens, args.out, results)
    for row in results:
        status = "PASS" if row["passed"] == row["total"] else "FAIL"
        print(f"pack{row['pack_size']}: {row['passed']}/{row['total']} sanity tasks | {status}")


if __name__ == "__main__":
    main()
