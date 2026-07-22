import subprocess
import re
import os
import csv
import sys
import statistics
import time
from pathlib import Path

CAPACITIES = [100, 80, 66, 50, 33, 25]
PROMPT = "Explain the concept of Virtual Memory in operating systems."
ROOT = Path(__file__).resolve().parents[1]
EXE = os.environ.get(
    "EVM_LLAMA_CLI",
    str(ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe"),
)
TRIALS = 5

def run_benchmark(model_path, capacity_pct, ctx_len=512, disable_evm=False):
    env = os.environ.copy()
    if disable_evm:
        env["EVM_DISABLE"] = "1"
    else:
        env["EVM_CAPACITY_PCT"] = str(capacity_pct)
        if "EVM_DISABLE" in env:
            del env["EVM_DISABLE"]
            
    cmd = [
        EXE,
        "-m", model_path,
        "-p", PROMPT,
        "-n", "128",
        "-c", str(ctx_len),
        "-ngl", "99",
        "-ub", "4"
    ]
    
    process = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    stdout, stderr = process.communicate(input="/exit\n")
    
    prompt_ts = 0.0
    gen_ts = 0.0
    cache_hits = 0
    cache_misses = 0
    bytes_transferred = 0
    
    for line in stderr.splitlines() + stdout.splitlines():
        match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", line)
        if match:
            prompt_ts = float(match.group(1))
            gen_ts = float(match.group(2))
        
        hit_match = re.search(r"Cache Hits\s*:\s*(\d+)", line)
        if hit_match: cache_hits += int(hit_match.group(1))
            
        miss_match = re.search(r"Cache Misses\s*:\s*(\d+)", line)
        if miss_match: cache_misses += int(miss_match.group(1))
            
        byte_match = re.search(r"Bytes Transferred:\s*(\d+)", line)
        if byte_match: bytes_transferred += int(byte_match.group(1))

    if cache_hits + cache_misses == 0:
        print("WARNING: no EVM cache counters were emitted")
            
    return prompt_ts, gen_ts, cache_hits, cache_misses, bytes_transferred

def main():
    if len(sys.argv) < 2:
        print("Usage: python measure_throughput.py <path_to_model.gguf> [ctx_len]")
        return
        
    model_path = sys.argv[1]
    ctx_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    model_name = os.path.basename(model_path)
    csv_file = f"benchmark_results_{model_name}_ctx{ctx_len}.csv"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Prompt t/s (Mean)", "Prompt t/s (Std)", "Gen t/s (Mean)", "Gen t/s (Std)", "Hit Rate %", "Bytes Transferred (MB)"])
    
    configs = []
    for cap in CAPACITIES:
        configs.append((f"EVM {cap}%", cap, False))
        
    for name, cap, disable in configs:
        print(f"\n--- Testing {name} for {model_name} ({TRIALS} trials) ---")
        p_list, g_list, h_list, m_list, b_list = [], [], [], [], []
        
        for i in range(TRIALS):
            t0 = time.time()
            try:
                p, g, h, m, b = run_benchmark(model_path, cap, ctx_len, disable_evm=disable)
                if g > 5000.0:  # Detect instant EOS crash
                    raise ValueError("Instant EOS (Memory Corruption)")
                p_list.append(p)
                g_list.append(g)
                h_list.append(h)
                m_list.append(m)
                b_list.append(b)
                dt = time.time() - t0
                print(f"  Trial {i+1}/{TRIALS} ({dt:.1f}s): Prompt {p} t/s, Gen {g} t/s")
            except Exception as e:
                print(f"  Trial {i+1}/{TRIALS} FAILED: {str(e)}")
                continue
            
        p_mean = statistics.mean(p_list) if p_list else 0.0
        p_std = statistics.stdev(p_list) if len(p_list) > 1 else 0.0
        g_mean = statistics.mean(g_list) if g_list else 0.0
        g_std = statistics.stdev(g_list) if len(g_list) > 1 else 0.0
        
        total_hits = sum(h_list)
        total_misses = sum(m_list)
        hit_rate = 100.0 * total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        avg_mb = (sum(b_list) / len(b_list)) / (1024*1024) if b_list else 0.0
        
        print(f"Result {name}: Gen {g_mean:.1f}±{g_std:.1f} t/s | Hit Rate: {hit_rate:.1f}% | Bandwidth: {avg_mb:.1f} MB")
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, f"{p_mean:.2f}", f"{p_std:.2f}", f"{g_mean:.2f}", f"{g_std:.2f}", f"{hit_rate:.2f}", f"{avg_mb:.2f}"])

if __name__ == "__main__":
    main()
