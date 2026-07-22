import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MriBatch = ROOT / "scripts" / "mri_batch.py"
DEFAULT_MANIFEST = ROOT / "results" / "mri_batch" / "corpus" / "calibration_100.json"


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def invoke(arguments):
    completed = subprocess.run([sys.executable, str(MriBatch), *arguments], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="ignore")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return completed.returncode, lines[-1] if lines else "no compact summary"


def main():
    parser = argparse.ArgumentParser(description="Run the full local 2,600-prompt MRI corpus with checkpoints and compact status.")
    parser.add_argument("--models", default="qwen1,deepseek,qwen2", help="comma-separated mri_batch model keys")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--status", type=Path, default=ROOT / "results" / "mri_batch" / "full_corpus_status.json")
    args = parser.parse_args()

    if not args.manifest.exists():
        code, message = invoke(["generate", "--manifest", str(args.manifest)])
        if code:
            raise SystemExit(message)
    code, message = invoke(["validate", "--manifest", str(args.manifest)])
    if code:
        raise SystemExit(message)

    models = [value.strip() for value in args.models.split(",") if value.strip()]
    status = {"format": "evm-mri-full-corpus-status-v1", "manifest": str(args.manifest), "models": {}, "state": "running", "started_unix_s": time.time()}
    write_status(args.status, status)
    for model in models:
        out_dir = ROOT / "results" / "mri_batch" / model / "calibration"
        entry = status["models"].setdefault(model, {})
        entry.update({"state": "running", "started_unix_s": time.time(), "out_dir": str(out_dir)})
        write_status(args.status, status)
        code, message = invoke(["run", "--model", model, "--manifest", str(args.manifest), "--out-dir", str(out_dir),
                                "--tokens", str(args.tokens), "--ctx", str(args.ctx)])
        entry["run_summary"] = message
        if code:
            entry.update({"state": "failed", "finished_unix_s": time.time()})
            status["state"] = "failed"
            write_status(args.status, status)
            raise SystemExit(1)
        code, message = invoke(["summary", "--out-dir", str(out_dir)])
        entry["metrics_summary"] = message
        if code:
            entry.update({"state": "failed", "finished_unix_s": time.time()})
            status["state"] = "failed"
            write_status(args.status, status)
            raise SystemExit(1)
        code, message = invoke(["analyze", "--model", model, "--manifest", str(args.manifest), "--out-dir", str(out_dir)])
        entry["cloud_summary"] = message
        entry.update({"state": "complete" if code == 0 else "failed", "finished_unix_s": time.time()})
        if code:
            status["state"] = "failed"
            write_status(args.status, status)
            raise SystemExit(1)
        write_status(args.status, status)
    status.update({"state": "complete", "finished_unix_s": time.time()})
    write_status(args.status, status)
    print("Full MRI corpus: all requested models complete | PASS")


if __name__ == "__main__":
    main()
