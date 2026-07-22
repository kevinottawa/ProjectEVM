# EVM Final Validation Report

## Status

The memory-mechanism proof passes. The 10-15 tokens/s production-performance target does not.

| Gate | Result | Evidence |
| --- | --- | --- |
| Sanitized routing telemetry | PASS | 27,972 full router rows, 50 prompts, 24 layers, 60 experts |
| Layer-aware simulator | PASS | Separate cache per layer; prompt-reset state; held-out test split |
| Learned predictor over LRU | PASS | 8/60 hit rate 22.58% -> 28.12%; 30.4% of Oracle gap closed |
| CPU-backed VRAM reduction | PASS | 6.2-18.8 GB lower peak VRAM across three models |
| CUDA logical-to-physical remapping | PASS | Counter-emitting valid runs on all three models |
| Strict RAM-backed Qwen2 unified path | PASS | Exact-8 CUDA pool; 9,116 MB VRAM; 1.90 t/s |
| Strict disk-backed Qwen2 unified path | PASS | 3/3 valid; 9,146 MB VRAM; 4,762 MB RSS; zero page-file growth |
| Native CUDA tuning | PASS | Flash Attention on, `-ub 1`: 6.30 t/s |
| Bounded pinned disk staging | PASS / NO SPEEDUP | 2/4/8 slots valid; each 0.40 t/s |
| Adaptive file-page cache | PASS / NO SPEEDUP | 8/12/16 GiB targets valid; 0.40/0.40/0.30 t/s |
| Disk-backed GPU pool sweep | PASS / LIMITED SPEEDUP | 8-40 experts: hit rate 32.08% -> 70.42%, traffic 59.2 -> 25.8 GB, speed 0.40 -> 0.60 t/s |
| Qwen2 10-15 tokens/s | NOT PHYSICAL BASELINE | Fastest max-GPU/shared row is 6.30 t/s |
| Qwen1.5 draft workflow | FAIL | 0.54 tokens/s; 40.0% acceptance; 23,145 MB |

## Corrected Simulator

The previous simulator keyed cache entries only by expert ID. That incorrectly merged same-numbered experts from different layers. Its 73.7% LRU and 89.6% Oracle results at 40/60 are superseded.

The final simulator uses independent layer caches and a 40/5/5 prompt train/validation/test split. The learned predictor uses current router probabilities to predict weighted expert demand 3-17 tokens ahead. Validation charges every prefetch as PCIe traffic, so the selected policy uses learned eviction scores without extra prefetch transfers.

| Capacity | LRU hit | Learned hit | Oracle hit | Learned stall reduction | Gap closed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8/60 | 22.58% | 28.12% | 40.83% | 7.2% | 30.4% |
| 12/60 | 31.20% | 36.26% | 47.41% | 7.4% | 31.2% |
| 16/60 | 37.54% | 41.42% | 50.81% | 6.2% | 29.3% |
| 20/60 | 41.02% | 45.63% | 52.36% | 7.8% | 40.7% |
| 24/60 | 44.69% | 48.35% | 52.77% | 6.6% | 45.3% |
| 30/60 | 48.93% | 50.84% | 52.80% | 3.8% | 49.5% |
| 40/60 | 52.29% | 52.61% | 52.80% | 0.7% | 62.0% |
| 50/60 | 52.80% | 52.80% | 52.80% | 0.0% | 0.0% |

Canonical paper table: `docs/tables/layer_aware_predictor_results.csv`. Training output and model weights are under `results/learned_predictor/`.

## Production Qwen2 Memory Tiers

The strict disk-backed configuration is:

- `EVM_EXPERTS_PER_TENSOR=8`
- `EVM_TARGET_EXPERT_COUNT=64`
- `EVM_CPU_BACKING=1`
- `EVM_CUDA_STREAMING=1`
- `EVM_DISK_BACKING=1`
- `EVM_DISK_TRIM=1`
- `EVM_DISK_TRIM_INTERVAL=1`
- `EVM_STRICT_BUDGET=1`
- `EVM_PREFILL_BATCH_THRESHOLD=999`
- `--kv-offload --mmap -ub 1`

| Trial | Generation t/s | Peak VRAM | Process RSS | System-RAM delta | Page-file delta | Valid |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.40 | 9,146 MB | 4,762 MB | 4,698 MB | 0 MB | yes |
| 2 | 0.40 | 9,115 MB | 4,752 MB | 4,780 MB | 0 MB | yes |
| 3 | 0.40 | 9,115 MB | 4,751 MB | 4,807 MB | 0 MB | yes |
| Mean/max | 0.40 | 9,146 MB | 4,762 MB | 4,807 MB | 0 MB | 3/3 |

This is the active-experts-only proof. The full 34.9 GB GGUF is mapped but not prefetched or committed to the page file. Clean expert pages are trimmed after each graph. The process retains only its active file pages and runtime allocations.

| Mode | Gen t/s | VRAM | Process RSS | Private commit | Page-file growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native max-GPU/shared | 6.30 | 23,951 MB | 24,025 MB | 34,079 MB | 5 MB |
| CPU-backed residency | 3.90 | 5,485 MB | 23,719 MB | 34,327 MB | 4,192 MB |
| RAM unified exact-8 | 1.90 | 9,116 MB | 24,115 MB | 37,838 MB | 5,092 MB |
| Disk unified exact-8 | 0.40 | 9,146 MB | 4,762 MB | 9,089 MB | 0 MB |

Canonical data: `results/production_evm/production_benchmark.json` and `docs/tables/production_evm_runtime.csv`.

### Disk-I/O Policy Sweep

| Policy | Gen t/s | Peak VRAM | Process RSS | Page-file growth | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Strict trim | 0.40 | 9,146 MB | 4,762 MB | 0 MB | adopted minimum-memory path |
| Lazy mmap | 0.60 | 9,095 MB | 24,160 MB | 49 MB | rejects host-memory goal |
| Pinned staging, 2 slots | 0.40 | 9,386 MB | 4,786 MB | 0 MB | valid, no speedup |
| Pinned staging, 4 slots | 0.40 | 9,329 MB | 4,755 MB | 70 MB | valid, no speedup |
| Pinned staging, 8 slots | 0.40 | 9,265 MB | 4,742 MB | 0 MB | valid, no speedup |
| 8 GiB cache trigger | 0.40 | 9,193 MB | 9,449 MB | 0 MB | valid, no speedup |
| 12 GiB cache trigger | 0.40 | 9,156 MB | 13,907 MB | 0 MB | valid, no speedup |
| 16 GiB cache trigger | 0.30 | 9,153 MB | 16,525 MB | 86 MB | valid, slower |

The pinned ring proves that bounded asynchronous host staging can be added without loading the model into RAM. It does not solve late demand discovery. The adaptive cache proves that retaining substantially more file pages also does not recover speed in this short run. `docs/tables/disk_io_controls.csv` is the canonical table.

### GPU Working-Set Sweep

| Experts/tensor | Gen t/s | Peak VRAM | Process RSS | Hit rate | Transferred |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.40 | 9,146 MB | 4,762 MB | 32.08% | 59,167 MB |
| 16 | 0.50 | 12,896 MB | 4,787 MB | 48.89% | 44,609 MB |
| 24 | 0.50 | 16,459 MB | 4,742 MB | 58.80% | 35,961 MB |
| 32 | 0.60 | 19,894 MB | 4,742 MB | 64.84% | 30,707 MB |
| 40 | 0.60 | 23,530 MB | 4,739 MB | 70.42% | 25,835 MB |

This is the adopted disk-backed tuning axis. More persistent GPU experts produce the expected monotonic hit-rate and traffic improvements while host RSS remains bounded. Throughput plateaus before the VRAM ceiling, proving that a predictor must turn future misses into overlapped reads rather than merely increasing capacity. Canonical data: `docs/tables/disk_gpu_pool_sweep.csv`.

`scripts/check_evm_storage_backend.py` provides the deployment gate. On the tested Windows host it selects `bounded_mmap`, reports no cuFile header/library, and passes without claiming GDS support.

## Runtime Changes

- Exact per-tensor capacity through `EVM_EXPERTS_PER_TENSOR`.
- Strict capacity also limits correctness-driven resize; the pool cannot silently grow.
- Pool reuse ordered with CUDA events instead of unconditional host-side compute-stream synchronization.
- Physical expert IDs uploaded asynchronously from double-buffered pinned host staging.
- Dedicated summary-only benchmark wrapper records speed, peak dedicated VRAM, counters, and pass/fail.
- Host monitor records process RSS, private commit, system-RAM delta, and page-file delta.
- `EVM_DISK_BACKING=1` disables whole-model mmap prefetch.
- `EVM_DISK_TRIM` bounds Windows file-page residency with configurable cadence.
- `EVM_DISK_CACHE_MB` adds an explicit mapped-page retention trigger.
- `EVM_DISK_STAGING` adds a bounded global pinned ring for asynchronous H2D uploads.
- Dedicated speculative binary path and parser support.

## Speculative Result

Qwen2-57B target plus Qwen1.5-MoE draft completes but fails performance: 0.54 tokens/s, 40.0% acceptance, and 23,145 MB peak VRAM. Multi-token verification requests more simultaneous experts than an 8-expert target pool. Strict mode now aborts instead of expanding beyond the declared capacity. This draft pair is not an adopted workflow.

## Adopted Workflows

| Scenario | Adopted path |
| --- | --- |
| Maximum measured Qwen2 speed | Native max-GPU/shared: 6.30 tokens/s, 23,951 MB VRAM, 24,025 MB RSS |
| Lowest VRAM with usable speed | CPU-backed residency: 3.90 tokens/s, 5,485 MB VRAM; requires about 24 GB RSS and page file |
| Strict counter-emitting RAM pool | RAM exact-8: 1.90 tokens/s, 9,116 MB VRAM; still full host residency |
| Strict VRAM and host-RAM bounds | Disk exact-8: 0.40 tokens/s, 9,146 MB VRAM, under 4.8 GB RSS, no page-file growth |
| Adjustable disk-backed speed/memory | Choose 16/24/32 experts for 12.9/16.5/19.9 GB VRAM; best measured speed is 0.60 t/s at 32 experts |
| Traditional constrained control | llama.cpp fit/offload: 1.00 token/s, 4,584 MB |
| Optional speculative path | Disabled for this model pair; measured result fails |

## Remaining Production Gate

The original targeted-loading goal is complete as a mechanism: only demanded expert file pages and the exact CUDA pool must be resident. The remaining production gap is speed. Larger CPU file-page windows and pinned staging were tested and rejected as solutions. Closing the gap requires Qwen2-specific early prediction, asynchronous prefetch before demand, and fewer total expert transfers. A custom CUDA kernel alone cannot solve NVMe and PCIe latency.
