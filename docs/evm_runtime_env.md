# EVM Runtime Environment Variables

These switches apply to the patched `llama.cpp` runtime in `llama.cpp/`.

Run `python scripts/check_evm_storage_backend.py` before deployment. It reports whether the host exposes cuFile/kernel GDS prerequisites and selects the currently implemented bounded-mmap backend. A GDS-capable host is reported as an integration candidate, not silently treated as supported.

| Variable | Meaning | Current proof status |
| --- | --- | --- |
| `EVM_DISABLE=1` | Disables CUDA EVM interception and runs the native path. | Used for baseline proof rows. |
| `EVM_CAPACITY_PCT=<n>` | Enables EVM and requests an expert-pool capacity of `<n>` percent of the model's expert count. | Validated for GPU-backed Qwen1.5-MoE and DeepSeek-Coder-V2-Lite remapping. |
| `EVM_EXPERTS_PER_TENSOR=<n>` | Sets an exact physical expert-slot ceiling for every targeted tensor. This also caps correctness-driven resize. | Disk-backed Qwen2 validated at 8/16/24/32/40 slots. Hit rate rises monotonically, but speed plateaus at 0.60 t/s by 32 slots. |
| `EVM_TARGET_EXPERT_COUNT=<n>` | Intercepts only tensors whose expert dimension equals `<n>`. | Used to target Qwen1.5-MoE (`60`), DeepSeek-Coder-V2-Lite (`64`), or Qwen2-57B (`64`) and avoid accidental draft-model interception. |
| `EVM_IGNORE_EXPERT_COUNT=<n>` | Legacy exclusion switch for tensors whose expert dimension equals `<n>`. | Kept for compatibility; `EVM_TARGET_EXPERT_COUNT` is preferred. |
| `EVM_CPU_BACKING=1` | Loader hook places targeted expert tensors in CPU backing storage instead of permanent GPU residency. | Validated for peak-VRAM reduction across Qwen1.5-MoE, DeepSeek-Coder-V2-Lite, and Qwen2-57B. |
| `EVM_CUDA_STREAMING=1` | Scheduler hook for CPU-backed targeted `MUL_MAT_ID` experts: keep the expert tensor as host backing, avoid a conventional full-size CUDA copy, and let the CUDA EVM manager stream requested experts into the compact pool. Requires `EVM_CPU_BACKING=1`. | Validated on Qwen1.5-MoE, DeepSeek-Coder-V2-Lite, and Qwen2-57B with required EVM counters. |
| `EVM_DISK_BACKING=1` | With `--mmap`, disables whole-model mmap prefetch so cold expert pages remain in the GGUF until demanded. | Qwen2 exact-8 disk path passes 3/3 with zero page-file growth. |
| `EVM_DISK_TRIM=1` | On Windows, trims clean mapped pages from the process working set after CUDA graphs. | Reduces Qwen2 process RSS from about 24 GB to below 4.9 GB. |
| `EVM_DISK_TRIM_INTERVAL=<n>` | Trims every `<n>` CUDA graphs. Lower values use less RAM; higher values retain more file cache. | Intervals 1, 4, and 8 tested. Interval 1 is the strict-memory preset. |
| `EVM_DISK_CACHE_MB=<n>` | Uses `<n>` MiB process RSS as a high-water trigger for working-set trimming. It is a retention target, not a hard process-memory cap. | 8, 12, and 16 GiB targets pass, but do not improve the 0.40 t/s strict result. |
| `EVM_DISK_STAGING=1` | Copies demanded mmap ranges through a bounded pinned-host ring before `cudaMemcpyAsync`; it never pins or copies the full model. | Correct at 2, 4, and 8 slots, but no measured Qwen2 speedup. Optional only. |
| `EVM_DISK_STAGING_SLOTS=<n>` | Sets the number of reusable pinned slots per expert byte-size class. | Tested at 2/4/8; all generated at 0.40 t/s. |
| `EVM_FUSION_AWARE=1` | Enables EVM remapping inside fused MoE gate/up CUDA paths by coordinating one logical-to-physical expert mapping across the fused expert tensors. | Validated 3/3 trials each on Qwen1.5-MoE, DeepSeek-Coder-V2-Lite, and Qwen2-57B. Leave unset to use the conservative no-fusion EVM fallback. |
| `EVM_DISABLE_METRICS=1` | Disables EVM metric increments and final EVM counter printing while leaving remapping, streaming, and synchronization behavior active. | Used for counter-overhead A/B. On Qwen2 unified 33%, disabling metrics changed generation from 1.90 to 2.00 t/s, so counters are not the material bottleneck. |
| `EVM_PREFILL_BATCH_THRESHOLD=<n>` | Expands the pool to full expert count when the observed batch size is above `<n>`, then returns to requested capacity for smaller generation steps. | Used to avoid prefill thrash during runtime proof rows. |
| `EVM_PREFILL_CAPACITY_PCT=<n>` | Caps the prefill-phase expansion as a percentage of total experts instead of always expanding to 100%. | Phase-aware control for budgeted prefill tests. |
| `EVM_PREFILL_POOL_MB=<n>` | Caps the prefill-phase CUDA EVM pool in MiB. | Phase-aware control for strict VRAM-budget tests. |
| `EVM_EXPERT_POOL_MB=<n>` | Byte cap for each CUDA EVM expert pool. Use `EVM_EXPERTS_PER_TENSOR` when an exact slot count is required. | Secondary budget control. |
| `EVM_CUDA_FREE_RESERVE_MB=<n>` | Soft initial cap that sizes the CUDA EVM expert pool so `<n>` MiB of CUDA free memory remains at manager initialization. | Added to reduce accidental VRAM wall pressure during GPU-backed/remapping runs. |
| `EVM_STRICT_BUDGET=1` | Turns the EVM pool budget into a hard correctness guard. If one CUDA expert call needs more unique experts than the budgeted pool can hold, the run aborts instead of silently evicting a still-needed expert. | Use for proof-grade budget tests; may require smaller prefill micro-batches or larger prefill quota. |
| `EVM_ROUTER_SCORE_PREFETCH=1` | Opt-in exact router-score candidate prefetch after the current MoE operation is scheduled. | Correctness-validated on both small models; not an adopted performance policy. |
| `EVM_PREDICTOR_RESERVED_SLOTS=<n>` | Adds protected physical pool slots beyond the demand LRU capacity for score-prefetch candidates. | Tested at 2 slots. It preserves the exact path but did not deliver a throughput win. |
| `EVM_ROUTER_SCORE_CANDIDATES=<n>` | Limits CUDA-ranked router candidates returned to the host scheduler (1-8). | Defaults to 8; avoids copying the full router-score vector. |
| `EVM_ROUTER_SCORE_PREFETCH_MIN_PPM=<n>` | Minimum router probability, in parts per million, required before a candidate copy is queued. | Defaults to 20,000 in the benchmark harness. |
| `EVM_LEARNED_ROUTER_SCORE=1` | Rank router-score candidates with the exported 3-to-17-token layer predictor and preserve high-scoring residents on later misses. Requires `EVM_LEARNED_SCHEDULER_PATH`. | Experimental NO-GO: host synchronization loses more throughput than the current small hit-rate gain. |
| `EVM_GPU_SCHEDULER=1` | Keep learned future scores on CUDA and consult them only after a page-table miss. Requires `EVM_GPU_PAGE_TABLE=1` and `EVM_LEARNED_SCHEDULER_PATH`. | Experimental NO-GO: tested at 32 slots on Qwen1 and DeepSeek; mandatory miss synchronization remains the bottleneck. |
| `EVM_DEBUG=1` | Prints each intercepted `mul_mat_id` call and sampled expert IDs. | Used only for debug proof logs; too verbose for benchmark runs. |
| `EVM_ROUTING_PROFILE_PATH=<file>` | Writes aggregate per-tensor expert access counts at process exit. | Used by the offline MoE MRI pipeline; generated text and raw router vectors are not stored. |
| `EVM_ABILITY_PACK_INDEX=<file>` | Declares the selected expert IDs that should be pinned in physical CUDA slots. | Validated with Qwen2 pack-16 and pack-24. |
| `EVM_ABILITY_PACK_ONLY=1` | Uses only the selected pack and substitutes unavailable expert requests with installed experts. | Fast derived-model mode; not exact and requires quality evaluation. Leave `0` for exact full-vault fallback. |

Two runtime modes should be kept distinct in the paper:

1. **GPU-backed EVM:** experts are already GPU resident; EVM remaps logical IDs into a managed GPU pool and emits counters. This validates interception, remapping, synchronization, and metric accounting on Qwen1.5-MoE and DeepSeek-Coder-V2-Lite, but it does not prove VRAM reduction by itself. For already-fitting MoE models, the intended next benchmark is to cap resident experts and spend the returned VRAM on larger KV cache, longer context, draft models, adapters, or speculative/DSpark-style workflows.
2. **CPU-backed EVM:** targeted experts are loaded into CPU backing storage instead of permanent GPU residency. This validates VRAM-residency reduction across three models.
3. **RAM-backed unified streaming EVM:** targeted experts are CPU-backed and `EVM_CUDA_STREAMING=1` keeps the CUDA EVM remapping path active. The latest memory-instrumented Qwen2 exact-8 row ran at 1.90 tokens/s, used 9,116 MB peak VRAM, and committed 37,838 MB of host virtual memory. It proves compact CUDA remapping, but not bounded total residency.
4. **Fusion-aware unified EVM:** adds `EVM_FUSION_AWARE=1`, allowing fused MoE gate/up kernels to run through coordinated EVM physical IDs. This is the final correctness path for fused CUDA execution.
5. **Disk-backed unified EVM:** combines `--mmap`, `EVM_DISK_BACKING=1`, exact CUDA capacity, and working-set trimming. Cold experts remain file-backed instead of committed to RAM/page file. The final Qwen2 exact-8 row passes 3/3 at 0.40 tokens/s, 9,146 MB peak VRAM, at most 4,762 MB process RSS, and zero page-file growth.

Pinned staging and retained file pages are optional disk-I/O policies, not new residency modes. They keep the authoritative cold copy on NVMe and preserve the exact CUDA pool. Neither improved throughput on the tested Windows/RTX 3090 Ti system, so strict trimming remains the adopted minimum-memory preset.

For an adjustable speed/memory workflow, change `EVM_EXPERTS_PER_TENSOR`, not the CPU file cache. Measured Qwen2 points are 8/16/24/32/40 slots at 9.1/12.9/16.5/19.9/23.5 GB VRAM. The 32-slot row is the practical upper point because 40 slots consumes another 3.6 GB without improving the measured 0.60 t/s throughput.

Neither mode assumes a prompt permanently uses a tiny expert subset. The intended policy target is a moving short-horizon generation working set.

## Benchmark controls

Future proof-grade benchmark rows should record four independent placements:

| Control | Required setting in final rows | Why it matters |
| --- | --- | --- |
| KV cache placement | Pass `--kv-offload` for GPU KV or `--no-kv-offload` for CPU KV explicitly. | llama.cpp defaults KV offload to enabled, so earlier `-ngl 99` rows likely kept KV on the GPU, but the paper should not rely on implicit defaults. |
| llama.cpp fit margin | Use `--fit --fit-target <MiB>` when testing against a VRAM budget. | Prevents the native/offload baseline from filling the device so tightly that Windows shared GPU memory can blur the result. |
| EVM expert budget | Set `EVM_EXPERTS_PER_TENSOR`, `EVM_CUDA_FREE_RESERVE_MB`, phase controls, and `EVM_STRICT_BUDGET=1`. | Enforces the intended working set and prevents prefill or speculative verification from silently expanding residency. |
| Peak-memory guard | Use `scripts/run_budgeted_llama.py --gpu-budget-mb <MiB>`. | Kills and labels a run if dedicated GPU memory crosses the declared budget. |
| Host-memory accounting | Record process RSS/private commit, system-RAM delta, and page-file delta from `run_budgeted_llama.py`. | Separates true targeted disk loading from VRAM savings that merely move the model into committed RAM. |

The current CPU-backed EVM proof rows reduced dedicated GPU memory by moving cold expert storage to CPU RAM. That is a VRAM-tiering win, not a total-memory deletion win. The current GPU-backed EVM rows intentionally use more GPU memory because they keep the original experts resident and allocate an extra physical expert pool; they validate remapping, not savings.

The unified streaming target is stricter: CPU-backed expert storage plus one compact CUDA expert pool, with EVM counters emitted during the same run. In that mode, the expert working set lives in the CUDA pool and KV cache placement is explicit. The RAM-backed control uses `--no-mmap`; the strict total-residency workflow uses `--mmap`, `EVM_DISK_BACKING=1`, and `EVM_DISK_TRIM=1`. Both use GPU KV, `EVM_CPU_BACKING=1`, and `EVM_CUDA_STREAMING=1`.
