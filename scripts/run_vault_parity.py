import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-evm-vault-parity.exe"


def parse_env(items):
    env = os.environ.copy()
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"invalid environment assignment: {item}")
        env[key] = value
    return env


def run(exe, model, env, args):
    command = [str(exe), "-m", str(model), "-p", args.prompt, "-n", str(args.tokens), "-c", str(args.ctx), "-ngl", str(args.ngl)]
    complete = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore", env=env, timeout=args.timeout_s)
    match = re.search(r'\{\"tokens_generated\":(\d+),\"token_fingerprint_fnv1a64\":\"([0-9a-f]+)\"\}', complete.stdout)
    counters = re.search(r"Cache Hits\s*:\s*(\d+).*?Cache Misses\s*:\s*(\d+)", complete.stderr, re.S)
    return {
        "returncode": complete.returncode,
        "tokens_generated": int(match.group(1)) if match else 0,
        "fingerprint": match.group(2) if match else None,
        "cache_hits": int(counters.group(1)) if counters else 0,
        "cache_misses": int(counters.group(2)) if counters else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare generated token IDs without printing model logs.")
    parser.add_argument("--full-model", required=True)
    parser.add_argument("--spine-model", required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--prompt", default="EVM exact vault parity probe.")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--out", help="write the compact parity summary to this JSON file")
    parser.add_argument("--full-env", action="append", default=[])
    parser.add_argument("--spine-env", action="append", default=[])
    args = parser.parse_args()
    full = run(Path(args.exe), Path(args.full_model), parse_env(args.full_env), args)
    spine = run(Path(args.exe), Path(args.spine_model), parse_env(args.spine_env), args)
    same = full["returncode"] == 0 and spine["returncode"] == 0 and full["tokens_generated"] == spine["tokens_generated"] and full["fingerprint"] == spine["fingerprint"]
    result = {"full": full, "spine_vault": spine, "token_parity": same}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
