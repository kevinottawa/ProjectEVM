# Offline MoE MRI and Ability-Pack Workflow

## Purpose

`scripts/moe_mri.py` turns offline MoE routing counts into reusable, layer-specific expert atlases and pack candidates. It is model-layout independent for GGUF models whose routed tensors use the standard `blk.<layer>.ffn_<role>_exps.weight` naming used by the tested Qwen and DeepSeek models.

The workflow is offline. It does not need integration into a chat application and does not retain generated text. Calibration inference is still required because tensor weights alone do not reveal human-readable expert behavior.

## Pipeline

1. Scan the GGUF to identify layers, expert count, tensor roles, routed-expert bytes, and estimated spine bytes.
2. Run categorized calibration prompts through `scripts/run_moe_mri_profiles.py`.
3. Run the six-domain payload contract in `config/mri_diagnostic_payloads.json` with positive and contrast prompts, without retaining generated text.
4. Rank experts independently in each layer. Assign a domain label only when observation, lift, and contrast-margin thresholds pass; otherwise label the expert shared or inconclusive.
5. Convert requested residency percentages into model-specific expert counts and emit selections, an expert atlas, JSON pack cards, and a Markdown report.
6. Extract selected expert slices with `scripts/build_expert_vault.py`.
7. Validate runtime memory, speed, substitutions, and task quality before deploying a pack.

## Extended Domain Library

The production library is stored at `config/mri/v2/domain_library.json`; the frozen six-domain suite remains available only for reproducing the original paper result. The v2 library contains 26 domains and 156 explicit prompts organized into three groups:

| Group | Domains | Purpose |
| --- | ---: | --- |
| Computer and agent | 10 | Code generation, debugging, review, systems, web, data, tools, planning, structured output, and repository navigation. |
| Professional reasoning | 9 | Technical writing, operations, science, mathematics, logic, research, legal-style, medical-style, and financial reasoning. |
| Language and human interaction | 7 | Instruction following, safety, conversation, creative writing, multilingual work, factual recall, and long-context synthesis. |

Each domain has four calibration prompts, one validation prompt, and one held-out prompt. Calibration may rank experts. Validation may choose pack size and thresholds. Held-out prompts must not influence selection. Every domain also records its description, exclusions, and nearest contrast domains.

Validate and compile the library:

```powershell
python scripts/mri_domain_library.py validate
python scripts/mri_domain_library.py compile --split calibration --out results/mri_library/calibration.json
python scripts/mri_domain_library.py catalog --out docs/tables/mri_domain_catalog.csv
```

Run all calibration domains or a named subset with `--payloads config/mri/v2/domain_library.json`, `--split`, `--domains`, and `--max-prompts-per-domain`. `--resume` continues an interrupted isolated sweep.

The implementation verification ran one calibration entry from all 26 domains on Qwen1.5 (26/26 valid). A stratified portability sample covering code generation, tool calling, planning, scientific reasoning, instruction following, and long-context synthesis passed 6/6 on DeepSeek and 6/6 on Qwen2. This is 38/38 extended workflow runs. It verifies taxonomy execution and model-layout portability; it is not a full three-model evaluation of all 156 prompts.

## Persistent Batch Calibration

`scripts/mri_batch.py` expands the domain seeds into a deterministic 2,600-prompt calibration corpus, validates uniqueness, runs a persistent local model, resumes from checkpoints, summarizes compact telemetry, builds inverse expert trees and co-activation clouds, and emits graph-informed pack selections. Detailed commands and the isolation contract are in `docs/MRI_Batch_Automation.md`.

The persistent profiler clears KV/model memory and sampler state between prompts. Controlled MRI also resets expert residency; serving-locality experiments may retain warm experts. Prefill and generation routing are captured separately, and generated text is discarded.

The Qwen1.5 26-domain verification passed 26/26 in 15.91 seconds wall time versus 170.18 seconds for isolated model processes, a 10.70x improvement. It emitted 3,744 routing rows, 4,799 soft membership edges, and 3,378 co-activation edges. The generated 37.5% computer-agent workflow candidate contains 23/60 experts in each of 24 layers. This candidate still requires validation and held-out quality comparison against the frequency-only selector.

The full 2,600-prompt corpus has now completed on all three models. PMI and semantic-family filtering reduces raw co-activation graphs to compact candidate assemblies. The first matched workflow-pack gate is mixed: Qwen1.5 and DeepSeek are NO-GO for both frequency and graph packs, while Qwen2 frequency reaches 6/8 against a 5/8 exact external-vault baseline. Treat that Qwen2 row as a narrow signal only; graph selection is not yet a proven quality improvement.

The automated small-model refinement runner then tested 50% and 75% frequency versus core-overlay packs under a stricter 12-check all-baseline-retention policy. Neither Qwen1.5 nor DeepSeek passed. At 75%, frequency retained 8/9 Qwen1.5 baseline checks and 7/8 DeepSeek checks; core-overlay was lower. This is a NO-GO decision, emitted by `scripts/summarize_pack_refinement.py`, not a manually interpreted benchmark.

## Commands

Capture categorized profiles for a model that fits in VRAM:

```powershell
python scripts/run_moe_mri_profiles.py `
  --name qwen1_5 `
  --model models/Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf `
  --expert-count 60 --capacity 8 --gpu-budget-mb 18000 `
  --tokens 8 --trials-per-category 2 `
  --out-dir results/moe_mri/qwen1_5
```

For larger-than-VRAM Qwen2, use the exact spine/vault runtime. A generated ability index may be supplied later as an optional profiling seed:

```powershell
python scripts/run_moe_mri_profiles.py `
  --name qwen2 `
  --model results/expert_vault/qwen2_full_vault/qwen2-spine.gguf `
  --expert-count 64 --capacity 24 --gpu-budget-mb 18000 `
  --vault-index results/expert_vault/qwen2_full_vault/experts.pack.idx `
  --vault-pack results/expert_vault/qwen2_full_vault/experts.pack `
  --tokens 6 --trials-per-category 2 `
  --out-dir results/moe_mri/qwen2
```

Build the atlas and pack cards:

```powershell
python scripts/moe_mri.py all `
  --model models/qwen2-57b-a14b-instruct-q4_k_m.gguf `
  --profile general=results/moe_mri/qwen2/profiles/general.jsonl `
  --profile coding=results/moe_mri/qwen2/profiles/coding.jsonl `
  --profile reasoning=results/moe_mri/qwen2/profiles/reasoning.jsonl `
  --percentages 12.5,25,37.5,50 `
  --out-dir results/moe_mri/qwen2
```

Extract and verify one candidate:

```powershell
python scripts/build_expert_vault.py `
  --model models/qwen2-57b-a14b-instruct-q4_k_m.gguf `
  --selection results/moe_mri/qwen2/selection_universal_p37p5.json `
  --out results/ability_packs/qwen2_universal_24 `
  --verify
```

Large derived pack binaries are intentionally not retained as paper deliverables. Recreate an optional profiling seed with the command above, then pass its `experts.pack.idx` as `--ability-index` when needed.

## Outputs

| Artifact | Audience | Meaning |
| --- | --- | --- |
| `scan.json` | Runtime/tooling | Model dimensions and expert/spine storage. |
| `expert_atlas.csv` | Researchers | One row per `(layer, expert)` with category association, lift, accesses, confidence, and plain description. |
| `selection_<category>_<size>.json` | Pack builder | Exact expert IDs retained in each layer. |
| `pack_cards.json` | Applications | Intended use, estimated size, category coverage, confidence, and selection path. |
| `report.md` | Humans | Readable model and pack summary. |

Descriptions are statistical associations, not claims that an expert exclusively stores a subject. Labels are derived from the versioned diagnostic payload suite, not guessed from tensor position or generated prose. A domain label requires positive lift over that domain and separation from contrast domains. It does not prove that removing the expert deletes that capability.

## Three-Model Verification

| Model | Layers | Experts/layer | Categorized runs | Valid |
| --- | ---: | ---: | ---: | ---: |
| Qwen1.5-MoE-A2.7B | 24 | 60 | 12 | 12 |
| DeepSeek-Coder-V2-Lite | 26 | 64 | 12 | 12 |
| Qwen2-57B-A14B | 28 | 64 | 12 | 12 |

Mean universal-pack routing coverage across general, coding, and reasoning profiles:

| Model | 12.5% residency | 25% residency | 37.5% residency | 50% residency |
| --- | ---: | ---: | ---: | ---: |
| Qwen1.5-MoE | 27.51% (8/60) | 44.24% (15/60) | 60.13% (23/60) | 71.55% (30/60) |
| DeepSeek-Coder-V2-Lite | 31.22% (8/64) | 51.67% (16/64) | 67.05% (24/64) | 78.64% (32/64) |
| Qwen2-57B-A14B | 36.05% (8/64) | 54.95% (16/64) | 68.45% (24/64) | 78.35% (32/64) |

These are calibration-set routing coverage values, not quality scores. Every generated pack must pass held-out quality tests.

The final compact MRI, cross-model runtime, and quality evidence is committed under `docs/tables/`; regenerate it with `python scripts/export_ability_evidence.py`.

## Runtime Modes

| Mode | Full behavior | Full cold vault required | Intended use |
| --- | --- | --- | --- |
| Pack plus exact fallback | Yes | Yes | Exact inference with pinned hot experts and temporary fallback slots. |
| Pack-only router-masked | No | No | Derived model: route only to installed experts and renormalize retained weights. |

The finalized pack-only command uses `EVM_GPU_PACK_ONLY=1`, not `EVM_CPU_BACKING`. After CUDA preload completes, the source pack mapping is released. See `docs/GPU_Only_Pack_Workflow.md` and audit result rows with `scripts/verify_gpu_only_pack.py`.

The GPU-only pack runtime now applies its constraint before top-k routing. `EVM_ROUTER_MASK_UNAVAILABLE=1` loads the pack index into a layer-local availability mask, excludes unavailable experts, and normalizes the surviving routed weights. CUDA aborts if an unavailable ID reaches the pool while this mode is active; it never silently substitutes an expert. This proves logical correctness of the restricted router, not quality preservation. Every pack still needs held-out task, perplexity, and safety evaluation.

For exact EVM modes, use `scripts/verify_evm_hash_match.py` before tuning performance. It compares full-model and EVM-generated token fingerprints on a frozen five-question manifest. The current eight-slot EVM pool matched all 5/5 fingerprints on Qwen1.5 and DeepSeek. This is the correctness gate for CPU/disk-backed working-set policies; it is intentionally separate from GPU-only approximate pack evaluation.

The current performance policy is an exact online cache prior, not a static pack: `EVM_ONLINE_PREDICTOR=1` uses recent layer-local routing transitions to queue one asynchronous candidate prefetch, while `EVM_PREDICTOR_MIN_CONFIDENCE_PCT=35` skips weak guesses. Treat it as opt-in and model-specific. The small-model pilot improved Qwen1.5 but was neutral on DeepSeek; use the generated `docs/tables/online_predictor_small_models.csv` and preserve the hash gate before adopting it.

## GPU Page-Table Hit Path

`EVM_GPU_PAGE_TABLE=1` is an opt-in exact cache-hit path for MoE tensors with up to 64 experts. CUDA resolves logical expert IDs against a device page table; only a one-word activity mask and miss flag return to the host. A miss immediately uses the proven CPU-backed exact transfer path. This does not make the cold tier GPU-only and it does not change router decisions.

Run the matched small-model gate before using it elsewhere:

```powershell
python scripts/benchmark_online_predictor.py --models qwen1,deepseek --modes lru,gpu-page --capacity 8 --trials 3 --tokens 48 --out-dir results/page_table_final
python scripts/verify_evm_hash_match.py --model qwen1 --capacity 8 --tokens 24 --gpu-page-table --out-dir results/page_table_hash_validation/qwen1
python scripts/verify_evm_hash_match.py --model deepseek --capacity 8 --tokens 24 --gpu-page-table --out-dir results/page_table_hash_validation/deepseek
python scripts/export_gpu_page_table_evidence.py
```

The current evidence is intentionally mixed: Qwen1.5 improves 22.13% (12.47 to 15.23 t/s), while DeepSeek declines 6.04% (10.93 to 10.27 t/s), both at unchanged VRAM and 5/5 full-model token hashes. Treat this as a correct mechanism and a narrow workload win. High miss counts still dominate; router-score prefetch and a miss-safe asynchronous dispatcher remain the next performance work.

## Router-Score Prefetch

`EVM_ROUTER_SCORE_PREFETCH=1` ranks up to `EVM_ROUTER_SCORE_CANDIDATES` router candidates on CUDA, returns only those IDs and probabilities to the host scheduler, and asynchronously fills `EVM_PREDICTOR_RESERVED_SLOTS` protected predictive slots. `EVM_ROUTER_SCORE_PREFETCH_MIN_PPM` rejects low-probability candidates. Every routed expert and current CUDA read remains untouched, so the policy is exact: no candidate is substituted for a selected expert.

Run it as an opt-in per-model gate:

```powershell
python scripts/benchmark_online_predictor.py --models qwen1,deepseek --modes lru,score-prefetch --capacity 8 --trials 3 --tokens 48 --out-dir results/router_score_prefetch_final
python scripts/verify_evm_hash_match.py --model qwen1 --capacity 8 --tokens 24 --router-score-prefetch --out-dir results/router_score_prefetch_hash_active/qwen1
python scripts/verify_evm_hash_match.py --model deepseek --capacity 8 --tokens 24 --router-score-prefetch --out-dir results/router_score_prefetch_hash_active/deepseek
python scripts/export_router_score_prefetch_evidence.py
```

The original empty-slot evidence was mixed: Qwen1.5 improved while DeepSeek declined. The reserved-slot follow-up passes the five-prompt exact-token gate on both small models, but remains a NO-GO policy result: at the stable 48/60 Qwen1.5 point, it raised hit rate only from 81.71% to 83.63% while reducing speed from 37.10 to 25.93 t/s. Do not enable it by default. The reproducible 80% availability gate is ordinary LRU at 48/60 Qwen1.5 slots and 40/64 DeepSeek slots, each with three valid 96-token runs; see `docs/tables/steady_state_small_model_residency.csv`.

`scripts/evaluate_predictor_folds.py` is the offline promotion gate for the next predictor iteration. Qwen2 four-fold replay at 32/64 produced 80.53% mean learned hit rate versus 78.92% LRU, but the current runtime score bridge is a NO-GO on both small models because its host synchronization costs more than its hit-rate gain. Keep `EVM_LEARNED_ROUTER_SCORE` disabled outside controlled research runs.

`EVM_GPU_SCHEDULER=1` is the next prototype: CUDA retains the full learned future-score vector and the host reads it only after a page-table miss. Its first 32-slot small-model gate is also NO-GO (Qwen1 16.0 to 15.0 t/s; DeepSeek 17.2 to 14.1 t/s). The page-table miss-status synchronization and the cold-tier service call still serialize the critical path. Do not use this mode in production runs.

## Layer-Aware Predictor Training

`scripts/capture_predictor_trace.py` and `scripts/train_layer_aware_predictor.py` provide a reusable, model-agnostic training path for full router-probability vectors. The capture contract runs each of the 24 deterministic prompts in `config/predictor/router_training_prompts.json` in a fresh process, writes a compact Parquet trace, and records only summary health metrics. The trainer splits prompts chronologically into train, validation, and untouched test partitions; it learns a separate 3-to-17-token score model for each MoE layer.

```powershell
python scripts/capture_predictor_trace.py --model deepseek --out-dir results/predictor_training/deepseek/trace
python scripts/train_layer_aware_predictor.py --trace results/predictor_training/deepseek/trace/router_vectors.parquet --out-dir results/predictor_training/deepseek/model --capacities 8,10,12
python scripts/capture_predictor_trace.py --model qwen1 --out-dir results/predictor_training/qwen1/trace
python scripts/train_layer_aware_predictor.py --trace results/predictor_training/qwen1/trace/router_vectors.parquet --out-dir results/predictor_training/qwen1/model --capacities 8,10,12
python scripts/export_layer_aware_predictor_evidence.py
```

The exported `docs/tables/layer_aware_predictor_training.csv` is an offline simulator result. It currently validates learned replacement scores, not a deployed transfer prefetch: validation selected a zero speculative-prefetch budget on both small models. Any runtime integration must retain the five-token exact-hash gate and be benchmarked against matched LRU controls before it becomes the default EVM policy.

The opt-in runtime bridge is `EVM_LEARNED_SCHEDULER=1` plus `EVM_LEARNED_SCHEDULER_PATH=<runtime_layer_prior.txt>`. Export it with `scripts/export_learned_scheduler_prior.py --model <qwen1|deepseek|qwen2>`. It projects the learned layer matrix from the current selected expert IDs and changes only eviction order; the normal EVM copy/remap and exact output path remain unchanged. The reproducible matched benchmark is `python scripts/benchmark_online_predictor.py --models deepseek,qwen1 --modes lru,learned --capacity 8 --trials 3 --tokens 48 --out-dir results/learned_scheduler_runtime`, followed by `python scripts/export_learned_scheduler_evidence.py`. Current exact results are in `docs/tables/learned_scheduler_runtime.csv`.

Qwen2 uses the same scripts with its full external vault profile. On a 32 GB host, first capture a compact exact-EVM trace, then train and export its prior:

```powershell
python scripts/capture_predictor_trace.py --model qwen2 --max-prompts 8 --tokens 12 --out-dir results/predictor_training/qwen2/trace
python scripts/train_layer_aware_predictor.py --trace results/predictor_training/qwen2/trace/router_vectors.parquet --out-dir results/predictor_training/qwen2/model --capacities 8,12,16
python scripts/export_learned_scheduler_prior.py --model qwen2
python scripts/benchmark_online_predictor.py --models qwen2 --modes lru,learned --capacity 8 --trials 3 --tokens 16 --out-dir results/qwen2_learned_scheduler_reproduction
```

`scripts/benchmark_qwen2_headroom.py` provides the explicit `--reserves 4,6,8,12` workflow. A no-go result is valid evidence: it means the declared VRAM headroom cannot hold the non-expert CUDA core plus the minimum exact active-expert pool. Current 32 GB evidence is in `docs/tables/qwen2_32gb_runtime.csv`; only the no-reserve 8-slot configuration passed, at 0.93 t/s with no learned-scheduler improvement.

The three-model restricted-router reproduction confirms that distinction. Scripted GPU-only smokes completed with zero substitutions on Qwen1.5 (75%, 48.3 t/s, 6,439 MB), DeepSeek (75%, 23.5 t/s, 8,597 MB), and Qwen2 (37.5%, 36.7 t/s, 16,213 MB). Their 12-check quality gates were all NO-GO: Qwen1.5 frequency/overlay 2/12 and 0/12, DeepSeek 2/12 and 3/12, and Qwen2 0/12 and 0/12, each against a 9/12 baseline. The generated evidence is `docs/tables/router_mask_quality.csv`.

The MRI universal Qwen2 37.5% pack passes 5/5 deterministic sanity tasks through `scripts/validate_ability_quality.py`. Its 25% pack passes 4/5 and fails chemistry, so the workflow adopts 37.5% for Qwen2. Another model may pass at 25% or require 50%; the workflow chooses the smallest percentage that satisfies its quality policy. The scorer records hashes and pass/fail without retaining generated answers.

## Percentage Policy

Pack sizes are requested as percentages of each layer's expert pool. The tool rounds upward to an integer expert count and records both requested and actual percentages. For example, 25% means 15/60 experts on Qwen1.5-MoE and 16/64 on Qwen2 or DeepSeek. Absolute `--sizes` remains available only for reproducing an existing artifact.

The production selector does not encode “16 is bad” or “24 is good.” It chooses the smallest tested residency percentage that passes the configured quality suite, coverage target, VRAM budget, and throughput target for that model.

## Cross-Model Runtime and Quality Gate

Static ability packs were extracted and hash-verified for all three models. Each model ran 25% and 37.5% residency in exact-fallback and pack-only modes for three trials per configuration: 36/36 runtime trials passed.

| Model | 25% pack-only t/s | 37.5% pack-only t/s | Baseline sanity | 25% sanity | 37.5% sanity | Adopted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen1.5-MoE | 40.23 | 53.03 | 5/5 | 1/5 | 3/5 | No pack-only candidate |
| DeepSeek-Coder-V2-Lite | 54.60 | 57.03 | 5/5 | 0/5 | 0/5 | No pack-only candidate |
| Qwen2-57B-A14B | 24.97 | 38.67 | Not repeated | 4/5 | 5/5 | 37.5% provisional candidate |

The workflow treats runtime validity and model quality as independent gates. High tokens/second, zero cache misses, and valid text do not make a derived pack acceptable. Qwen1.5 and DeepSeek demonstrate this failure mode directly. The older rows in this table used deterministic CUDA-side substitution and are retained only as historical mechanism data; router-masked evidence is emitted separately in `docs/tables/router_mask_quality.csv`.

## Delivered Scope and Future Scope

The delivered MRI is complete for structural scanning, split-aware categorized routing capture, evidence-thresholded association labels, percentage-based pack selection, pack cards, and extraction manifests. The 26-domain library supports workload-specific computer, agent, professional, and language packs. It does not yet measure causal expert importance or quantization sensitivity.

Per-expert Q2/Q3/Q4/Q6 stress testing is reserved for future work. That extension must add full router probabilities, prefill/generation separation, reconstruction and perplexity deltas, task ablations, co-activation redundancy, and router shifts after quantization. Until then, the MRI must not recommend bit widths or claim that it can precisely remove a subject, knowledge set, or safety policy.
