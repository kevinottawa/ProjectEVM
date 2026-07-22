import json
import subprocess
from pathlib import Path

import pandas as pd
from datasets import load_dataset
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = ROOT / "llama.cpp"

def main():
    print("Loading dataset...")
    # Using 'tatsu-lab/alpaca' for a mix of tasks
    dataset = load_dataset('tatsu-lab/alpaca', split='train')
    
    # Sample 50 diverse prompts for the initial proof of concept
    sampled = dataset.shuffle(seed=42).select(range(50))
    
    executable_path = LLAMA_DIR / "build" / "bin" / "Release" / "llama-routing-observatory.exe"
    model_path = ROOT / "models" / "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf"
    
    jsonl_output = LLAMA_DIR / "routing_trace.jsonl"
    
    # Clear previous trace if exists
    jsonl_output.unlink(missing_ok=True)
        
    print(f"Running inference on {len(sampled)} prompts...")
    
    for i, row in enumerate(sampled):
        prompt = f"<|im_start|>user\n{row['instruction']} {row['input']}<|im_end|>\n<|im_start|>assistant\n"
        # To avoid command line length limits, write prompt to a temporary file and use -f
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as f:
            f.write(prompt)
            temp_prompt_path = f.name
            
        cmd = [
            str(executable_path),
            "-m", str(model_path),
            "-f", temp_prompt_path,
            "-c", "4096",    # max context
            "-n", "128",     # limit generated tokens for speed during observatory phase
            "-s", str(i),    # use seed as prompt_id
            "--temp", "0.7", # default temperature
        ]
        
        try:
            # We don't need the stdout, it's just spamming
            subprocess.run(cmd, cwd=LLAMA_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"Prompt {i} timed out.")
        finally:
            Path(temp_prompt_path).unlink(missing_ok=True)
            
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/50 prompts...")

    print("Converting JSONL to Parquet...")
    records = []
    with open(jsonl_output, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                prompt_id = data["prompt_id"]
                layer_id = data["layer_id"]
                token_idx = data["token_idx"]
                probs = data["probs"]
                if len(probs) != 60:
                    continue
                
                # In Qwen1.5-MoE, there are 60 experts. Top 4 are selected.
                # Find top 4 indices
                indexed_probs = list(enumerate(probs))
                indexed_probs.sort(key=lambda x: x[1], reverse=True)
                top_k = [x[0] for x in indexed_probs[:4]]
                
                records.append({
                    "prompt_id": prompt_id,
                    "token_id": token_idx, 
                    "generated_token": "", # We didn't capture string token in callback, just index
                    "layer_id": layer_id,
                    "top_k_expert_ids": top_k,
                    "router_probabilities": probs,
                    "timestamp": time.time()
                })
            except Exception as e:
                pass

    df = pd.DataFrame(records)
    parquet_path = ROOT / "data" / "routing_telemetry.parquet"
    df.to_parquet(parquet_path, engine='pyarrow')
    print(f"Saved parquet to {parquet_path}")

if __name__ == "__main__":
    main()
