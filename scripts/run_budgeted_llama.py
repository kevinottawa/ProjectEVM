import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"


def gpu_used_mb():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0
    return int(proc.stdout.strip().splitlines()[0].strip())


def mib(value):
    return int(round(value / (1024 * 1024)))


def process_memory_bytes(process):
    rss = private = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [process]
    for item in processes:
        try:
            info = item.memory_info()
            rss += info.rss
            private += getattr(info, "private", info.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, private


def parse_env(assignments):
    parsed = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"environment assignment must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def parse_evm_metrics(text):
    hits = sum(int(match.group(1)) for match in re.finditer(r"Cache Hits\s*:\s*(\d+)", text))
    misses = sum(int(match.group(1)) for match in re.finditer(r"Cache Misses\s*:\s*(\d+)", text))
    evictions = sum(int(match.group(1)) for match in re.finditer(r"Evictions\s*:\s*(\d+)", text))
    bytes_transferred = sum(int(match.group(1)) for match in re.finditer(r"Bytes Transferred:\s*(\d+)", text))
    total = hits + misses
    substitutions = sum(int(match.group(1)) for match in re.finditer(r"Pack Substitutions:\s*(\d+)", text))
    predictor_prefetches = sum(int(match.group(1)) for match in re.finditer(r"Predictor Prefetches:\s*(\d+)", text))
    predictor_hits = sum(int(match.group(1)) for match in re.finditer(r"Predictor Hits\s*:\s*(\d+)", text))
    gpu_page_hits = sum(int(match.group(1)) for match in re.finditer(r"GPU Page Hits\s*:\s*(\d+)", text))
    gpu_page_misses = sum(int(match.group(1)) for match in re.finditer(r"GPU Page Misses\s*:\s*(\d+)", text))
    router_score_prefetches = sum(int(match.group(1)) for match in re.finditer(r"Router Score Prefetches\s*:\s*(\d+)", text))
    budget_no_go = "EVM_BUDGET_NO_GO:" in text
    return {
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate_pct": round(100.0 * hits / total, 2) if total else 0.0,
        "cache_evictions": evictions,
        "bytes_transferred_mb": round(bytes_transferred / (1024 * 1024), 2),
        "has_evm_counters": total > 0,
        "pack_substitutions": substitutions,
        "predictor_prefetches": predictor_prefetches,
        "predictor_hits": predictor_hits,
        "gpu_page_hits": gpu_page_hits,
        "gpu_page_misses": gpu_page_misses,
        "router_score_prefetches": router_score_prefetches,
        "evm_budget_no_go": budget_no_go,
    }


def parse_throughput(text):
    match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", text)
    if match:
        return {
            "prompt_tps": round(float(match.group(1)), 2),
            "generation_tps": round(float(match.group(2)), 2),
        }
    encoded = re.search(r"encoded\s+\d+\s+tokens.*?speed:\s*([\d.]+)\s*t/s", text)
    decoded = re.search(r"decoded\s+\d+\s+tokens.*?speed:\s*([\d.]+)\s*t/s", text)
    return {
        "prompt_tps": round(float(encoded.group(1)), 2) if encoded else 0.0,
        "generation_tps": round(float(decoded.group(1)), 2) if decoded else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run llama-cli with explicit EVM/KV/budget controls and VRAM guardrails."
    )
    parser.add_argument("--name", required=True, help="run name used for output files")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--exe", default=str(DEFAULT_EXE), help="llama-cli executable")
    parser.add_argument("--prompt", default="Briefly define virtual memory.")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--ngl", default="99")
    parser.add_argument("--ub", default="4")
    parser.add_argument("--kv", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--fit-target-mb", type=int, default=0)
    parser.add_argument("--gpu-budget-mb", type=int, default=0)
    parser.add_argument("--require-evm-counters", action="store_true")
    parser.add_argument("--response-file", help="optional path that receives model stdout for a diagnostic run")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--env", action="append", default=[], help="extra env assignment, KEY=VALUE")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "budgeted_runs"))
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="extra llama-cli args after --")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env.update(parse_env(args.env))

    cmd = [
        args.exe,
        "-m",
        args.model,
        "-p",
        args.prompt,
        "-n",
        str(args.tokens),
        "-c",
        str(args.ctx),
        "-ngl",
        str(args.ngl),
        "-ub",
        str(args.ub),
        "--temp",
        "0.0",
        "--kv-offload" if args.kv == "gpu" else "--no-kv-offload",
    ]
    if args.fit_target_mb > 0:
        cmd.extend(["--fit", "on", "--fit-target", str(args.fit_target_mb)])
    if args.extra and args.extra[0] == "--":
        cmd.extend(args.extra[1:])
    else:
        cmd.extend(args.extra)

    start = time.time()
    baseline_vm = psutil.virtual_memory()
    baseline_swap = psutil.swap_memory()
    peak = {"value": gpu_used_mb()}
    host_peak = {
        "process_rss": 0,
        "process_private": 0,
        "system_used": baseline_vm.used,
        "swap_used": baseline_swap.used,
    }
    budget_exceeded = {"value": False}
    stop = {"value": False}

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env=env,
    )
    monitored_process = psutil.Process(proc.pid)

    def terminate_process_tree():
        """Make a failed probe leave no GPU-owning child behind on Windows."""
        try:
            children = monitored_process.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def poll():
        while not stop["value"]:
            used = gpu_used_mb()
            peak["value"] = max(peak["value"], used)
            rss, private = process_memory_bytes(monitored_process)
            host_peak["last_process_rss"] = rss
            host_peak["last_process_private"] = private
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            host_peak["process_rss"] = max(host_peak["process_rss"], rss)
            host_peak["process_private"] = max(host_peak["process_private"], private)
            host_peak["system_used"] = max(host_peak["system_used"], vm.used)
            host_peak["swap_used"] = max(host_peak["swap_used"], swap.used)
            if args.gpu_budget_mb > 0 and used > args.gpu_budget_mb and proc.poll() is None:
                budget_exceeded["value"] = True
                terminate_process_tree()
                return
            time.sleep(0.25)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()

    timed_out = False
    try:
        stdout, stderr = proc.communicate(input="/exit\n", timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_tree()
        stdout, stderr = proc.communicate()
    finally:
        stop["value"] = True
        thread.join(timeout=2)

    elapsed = time.time() - start
    text = (stdout or "") + "\n" + (stderr or "")
    log_path = out_dir / f"{args.name}.log"
    json_path = out_dir / f"{args.name}.json"
    log_path.write_text(text, encoding="utf-8", errors="ignore")
    if args.response_file:
        Path(args.response_file).write_text(stdout or "", encoding="utf-8", errors="ignore")
    metrics = parse_evm_metrics(text)
    throughput = parse_throughput(text)
    output_fingerprint = hashlib.sha256((stdout or "").replace("\r\n", "\n").encode("utf-8")).hexdigest()

    evm_env = {k: v for k, v in sorted(env.items()) if k.startswith("EVM_")}
    row = {
        "name": args.name,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "budget_exceeded": budget_exceeded["value"],
        "gpu_budget_mb": args.gpu_budget_mb,
        "peak_gpu_memory_mb": peak["value"],
        "peak_process_rss_mb": mib(host_peak["process_rss"]),
        "peak_process_private_mb": mib(host_peak["process_private"]),
        "end_process_rss_mb": mib(host_peak.get("last_process_rss", 0)),
        "end_process_private_mb": mib(host_peak.get("last_process_private", 0)),
        "peak_system_ram_used_mb": mib(host_peak["system_used"]),
        "system_ram_total_mb": mib(baseline_vm.total),
        "system_ram_used_delta_mb": mib(max(0, host_peak["system_used"] - baseline_vm.used)),
        "peak_pagefile_used_mb": mib(host_peak["swap_used"]),
        "pagefile_used_delta_mb": mib(max(0, host_peak["swap_used"] - baseline_swap.used)),
        "elapsed_s": round(elapsed, 2),
        "kv": args.kv,
        "fit_target_mb": args.fit_target_mb,
        "evm_env": evm_env,
        **throughput,
        **metrics,
        "stdout_sha256": output_fingerprint,
        "stdout_bytes": len((stdout or "").encode("utf-8")),
        "cmd": cmd,
        "log": str(log_path),
    }
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))

    if args.require_evm_counters and not metrics["has_evm_counters"]:
        raise SystemExit(1)
    if timed_out or budget_exceeded["value"] or proc.returncode not in (0, None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
