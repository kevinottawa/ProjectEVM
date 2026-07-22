import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Export a runtime-readable layer-aware EVM eviction prior.")
    parser.add_argument("--model", choices=("deepseek", "qwen1", "qwen2"), required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    artifact = args.artifact or ROOT / "results" / "predictor_training" / args.model / "model" / "router_probability_predictor.npz"
    out = args.out or ROOT / "results" / "predictor_training" / args.model / "model" / "runtime_layer_prior.txt"
    weights = np.load(artifact)["weights"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# EVM_LAYER_PRIOR_V1 layer source_expert candidate_expert score\n")
        for layer in range(weights.shape[0]):
            for source in range(weights.shape[1]):
                for candidate in range(weights.shape[2]):
                    score = float(weights[layer, source, candidate])
                    if score != 0.0:
                        handle.write(f"{layer} {source} {candidate} {score:.9g}\n")
    print(f"Runtime learned prior: model={args.model} layers={weights.shape[0]} experts={weights.shape[1]} | PASS")


if __name__ == "__main__":
    main()
