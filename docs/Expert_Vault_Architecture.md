# Expert Vault: Spine, Vault, and Domain Packs

## The Attack Point

Split an MoE GGUF into two independently addressable parts:

1. **Dense spine:** embeddings, attention, norms, router weights, shared experts, output tensors, tokenizer metadata, and model hyperparameters.
2. **Expert vault:** raw quantized slices keyed by `(layer, expert, role)`, where role is `gate`, `up`, `down`, or merged `gate_up`.

For Qwen2-57B-A14B Q4_K_M, the routed-expert portion is 27.81 GB across 28 layers, 64 experts, and 84 expert tensors. Each expert slice is contiguous in its source tensor. The manifest created by `scripts/build_expert_vault.py` records exact source offsets and byte counts.

## Three Different Products

| Product | Expert availability | Exact model behavior | Intended value |
| --- | --- | --- | --- |
| External vault | All experts remain in a local/remote vault | Yes, after an expert is fetched | Remove expert bytes from the loaded spine and choose the cold tier independently. |
| Domain pack plus fallback | A predicted/core subset is packaged locally; remaining experts stay in vault | Yes, if fallback remains enabled | Faster startup and fewer cold misses for a known workload. |
| Pruned domain model | Only selected experts exist | No | Smaller specialized derivative; measure quality, refusal behavior, and router coverage. |

The first two are memory architecture. The third is model surgery.

## What “Categorize Experts” Means

Do not label expert 17 as globally “the coding expert.” In an MoE, expert identity is layer-local. The useful object is a `(layer, expert)` pair. Build a profile from Qwen2 routing traces:

- activation frequency by prompt family and generation position;
- router probability mass, not only top-k selection;
- co-activation and replacement relationships within each layer;
- quality impact when the pair is withheld;
- transfer cost and reuse distance.

Use those profiles to produce a core pack per workload. The pack should contain the selected gate/up/down slices for each chosen pair, while the dense router stays in the spine.

## Verified First Milestone

The smoke command below extracted expert 0 across all 28 Qwen2 layers and all routed roles into a 0.43 GB pack. Every slice was SHA-256 checked against the source GGUF. The binary smoke pack is reproducible and intentionally omitted from the retained paper artifacts.

```powershell
python scripts/build_expert_vault.py `
  --model models/qwen2-57b-a14b-instruct-q4_k_m.gguf `
  --out results/expert_vault/qwen2_smoke_pack `
  --uniform-expert 0 --verify
```

This proves physical separation. It does not create a runnable pruned model yet.

## Runtime Surgery

The next loader boundary is `llama_model_base::create_tensor`: when expert-vault mode is selected, it should not allocate or load the original expert tensor. Instead it creates a lightweight virtual expert source with the original tensor metadata plus a vault index. The CUDA EVM manager resolves `(tensor, logical expert)` into a pack offset, streams that slice into its physical pool slot, and retains the existing logical-to-physical remapping.

This keeps the dense spine normal and changes only expert storage. A full-vault fallback preserves exact behavior. A pack-only mode must renormalize router selection over its available experts and be evaluated as a new derived model.

## Completed Ability-Pack Workflow

The offline MoE MRI workflow now runs a versioned six-domain positive/contrast payload suite, captures aggregate routing counts, creates evidence-thresholded expert descriptions and pack cards, and emits percentage-based selections. It passed 12/12 isolated payloads on each of Qwen1.5-MoE, DeepSeek-Coder-V2-Lite, and Qwen2-57B-A14B. Labels that lack sufficient lift or contrast are explicitly marked shared or inconclusive.

Qwen2 25% (16/64) and 37.5% (24/64) artifacts were extracted and hash-verified. The runtime supports two distinct modes:

| Mode | Behavior | Result |
| --- | --- | --- |
| Exact fallback | Pin the selected pack and use eight temporary slots for original-router misses. | Exact behavior, but 0.83-0.93 t/s in the averaged matrix. |
| Pack-only | Load the spine and selected pack without the full expert vault; substitute unavailable requests. | 24.97 t/s at 25% residency and 38.67 t/s at 37.5%, but thousands of substitutions and therefore derived-model semantics. |

The finalized pack-only runtime uses `EVM_GPU_PACK_ONLY=1`. It copies the selected expert slices into fixed CUDA pools, keeps KV on GPU, and releases the selected pack mapping after preload. Fresh 37.5% rows passed on all three models with zero page-file growth. End-of-inference process RSS was 1,026 MB for Qwen1.5, 1,089 MB for DeepSeek, and 219 MB for Qwen2; transient load peaks remain separately reported.

The virtual-tensor loader now reserves all 84 Qwen2 expert metadata objects before creating GGML buffer contexts. This fixes the latent clean-build allocation overflow discovered during ability-pack integration.

The remaining quality limitation is explicit: pack-only substitution proves physical model reduction and high throughput, but router masking/renormalization and broader held-out quality evaluation are required before calling a pack a production-quality specialized model.

The Qwen2 37.5% universal pack passed 5/5 compact deterministic sanity tasks. Its 25% pack passed 4/5 and failed chemistry, so 37.5% is the adopted Qwen2 candidate. This threshold is not hard-coded for other models; residency percentage and quality gates determine their candidate independently.

Cross-model execution confirms that distinction. Qwen1.5 and DeepSeek both pass all 12 runtime trials across 25%/37.5% exact and pack-only modes, but their pack-only quality gates fail: Qwen1.5 retains 1/5 and 3/5 tasks, while DeepSeek retains 0/5 at both percentages against 5/5 baselines. Their exact-fallback modes remain valid; their pack-only artifacts are research derivatives, not adopted models.

## CPU-Spine / GPU-Expert Placement Control

Qwen1.5-MoE-A2.7B Q3_K_M provides a controlled case where the entire model and the complete expert pool both fit in the 24 GB GPU. We compared five repeated 64-token generation trials using the same `llama-bench` build:

| Placement | Generation t/s | Peak VRAM | Peak process RSS |
| --- | ---: | ---: | ---: |
| Full model on GPU | 57.91 +/- 10.58 | 10,850 MB | 7,354 MB |
| Spine layers on CPU, all `*_exps` tensors on GPU | 32.21 +/- 4.46 | 10,216 MB | 7,378 MB |
| All model layers on CPU | 31.05 +/- 3.55 | 1,294 MB | 7,081 MB |

The tensor override makes the middle row a real placement test: `n_gpu_layers=0` keeps the transformer spine on CPU while `.*_exps.*=CUDA0` places the complete routed-expert tensors on CUDA. It is 3.7% faster than the CPU-layer control, but 44.4% slower than full-GPU execution and saves only 634 MB of peak VRAM. The result rejects CPU-spine/GPU-experts as the main EVM deployment layout for this model. Its frequent per-layer CPU/GPU activation synchronization dominates, while moving the comparatively small spine off GPU returns little memory.

Reproduction data is in `results/tensor_placement/qwen1_5_moe/summary.json`; `scripts/benchmark_tensor_placement.py` runs all three placements and records throughput, VRAM, and host RSS without printing raw logs.

## Future Extension: Soft Mixed-Precision Packs

The current packs are byte-exact slices of the source GGUF and inherit its whole-tensor quantization. A future pack format may retain prime experts at Q4-Q6, robust secondary experts at Q2-Q3, and sensitive rare experts in an exact cold vault. This could reduce memory without the missing-expert substitutions used by current pack-only mode.

This is not part of the delivered implementation. Expert-level mixed-precision quantization is established research territory; the EVM-specific opportunity is joint MRI selection of residency percentage, precision, and fallback tier. It requires per-expert quantization sensitivity, router-shift analysis, heterogeneous-type pack metadata, compatible CUDA execution, and held-out quality evaluation.
