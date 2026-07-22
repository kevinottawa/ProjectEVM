# GPU-Resident EVM Pack Workflow

## Purpose

This workflow runs a derived MoE model from a dense spine plus an MRI-selected expert pack. The spine, selected experts, KV cache, and inference kernels reside on the GPU. There is no full cold expert vault, CPU expert fallback, or page-file-backed expert tier during generation.

Host memory is still used transiently to read the GGUF and transfer the selected pack into CUDA memory. `EVM_GPU_PACK_ONLY=1` releases the external pack mapping after all selected expert tensors are synchronized into their CUDA pools. The verifier reports both transient peak RSS and end-of-inference RSS.

## Build And Identify A Pack

1. Run the fixed payload suite in `config/mri_diagnostic_payloads.json`.
2. Build a contrast-scored atlas with `scripts/moe_mri.py`.
3. Select a percentage budget and extract it with `scripts/build_expert_vault.py --verify`.
4. Run the pack with `EVM_GPU_PACK_ONLY=1`, `EVM_ABILITY_PACK_ONLY=1`, GPU KV, and a strict GPU budget.
5. Require runtime validity, zero page-file growth, and a held-out quality pass before adoption.

MRI labels are measured routing associations. A domain label requires enough observations, positive lift, and separation from the next-best contrast domain. Experts that do not clear those gates are labeled `shared_cross_domain` or `inconclusive`; the tool does not guess their contents.

## Runtime Contract

| Requirement | Accepted value |
| --- | --- |
| `EVM_GPU_PACK_ONLY` | `1` |
| `EVM_CPU_BACKING` | absent |
| `EVM_ABILITY_PACK_ONLY` | `1` |
| KV placement | GPU |
| Full cold vault | absent |
| EVM counters | required |
| GPU budget | strict |
| Page-file growth | 0 MB |
| Quality | pack-specific held-out gate must pass |

Run the machine-checkable audit with:

```powershell
python scripts/verify_gpu_only_pack.py <run.json> [<run.json> ...]
```

## Three-Model Proof

Each row is a fresh 37.5% pack-only run after introducing the explicit GPU-pack mode and source-mapping release.

| Model | Generation | Peak VRAM | Transient peak RSS | End RSS | Page-file growth | Runtime gate | Quality gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen1.5-MoE | 65.70 t/s | 4,378 MB | 3,521 MB | 1,026 MB | 0 MB | PASS | NO-GO at 3/5 |
| DeepSeek-Coder-V2-Lite | 32.40 t/s | 5,336 MB | 4,451 MB | 1,089 MB | 0 MB | PASS | NO-GO at 0/5 |
| Qwen2-57B-A14B | 31.50 t/s | 16,412 MB | 15,516 MB | 219 MB | 0 MB | PASS | Provisional at 5/5 |

Runtime success and quality acceptance are separate. Qwen1.5 and DeepSeek prove that a stable, fast GPU-resident derived model can still be wrong. Qwen2 37.5% is the only current provisional candidate.

## Boundaries

This mode is not exact inference. Missing expert routes are deterministically substituted, so every pack is a derived model. Exact behavior requires a complete lower-tier vault and fallback path. Mixed-precision expert packs remain future work.
