import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = ROOT / "llama.cpp" / "routing_trace.jsonl"
PARQUET_PATH = ROOT / "data" / "routing_telemetry.parquet"
EXPECTED_EXPERTS = 60
TOP_K = 4


def main():
    records = []

    with JSONL_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)
            probs = data.get("probs", [])
            if len(probs) != EXPECTED_EXPERTS:
                continue

            top_k = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:TOP_K]
            records.append({
                "prompt_id": data["prompt_id"],
                "token_id": data["token_idx"],
                "generated_token": "",
                "layer_id": data["layer_id"],
                "top_k_expert_ids": [expert_id for expert_id, _ in top_k],
                "router_probabilities": probs,
                "timestamp": time.time(),
            })

    df = pd.DataFrame(records)
    df.to_parquet(PARQUET_PATH, engine="pyarrow")
    print(f"wrote {len(df)} sanitized rows to {PARQUET_PATH}")


if __name__ == "__main__":
    main()
