# Persistent Local MRI Batch Automation

## Purpose

`scripts/mri_batch.py` runs thousands of local calibration prompts without reloading the model for every sample and without sending prompts or outputs to a cloud service. The default corpus contains 100 calibration prompts for each of 26 domains: 2,600 prompts total.

## Isolation Contract

The C++ profiler loads the model and context once. Before every prompt it clears the model memory/KV state and resets the sampler. In controlled mode it also cold-resets the EVM expert cache. After prompt prefill it snapshots and clears routing counters; generation is then recorded as a separate phase. Generated text is discarded.

Each compact result retains only prompt ID, domain, split, token counts, validity, elapsed time, throughput, token fingerprint, and cache policy. Routing JSONL contains prompt ID, phase, tensor, and aggregate expert counts. The checkpoint contains completed prompt IDs and makes reruns resumable.

## Commands

```powershell
python scripts/mri_batch.py generate
python scripts/mri_batch.py validate
python scripts/mri_batch.py run --model qwen1
python scripts/mri_batch.py summary --model qwen1
python scripts/mri_batch.py analyze --model qwen1
python scripts/mri_batch.py build-packs --model qwen1 --domains tool_calling,planning,code_generation
```

Repeat `run` after interruption; completed prompt IDs are skipped. Use `--warm-experts` only for a separate serving-locality experiment. Controlled MRI defaults to cold experts.

Available model keys are `qwen1`, `deepseek`, and `qwen2`. Qwen2 uses the exact external vault configuration while retaining one loaded model/context process.

## Corpus Construction

The v1 deterministic expander combines four hand-authored semantic seeds per domain with 25 controlled instruction variants, producing 100 unique prompts per domain. This provides wording, format, audience, verification, ambiguity, and edge-case variation without cloud generation. It is a reproducible initial corpus, not the final word on semantic breadth; future revisions can add more hand-authored task seeds while retaining stable IDs and sealed validation/held-out splits.

## Inverse Tree And Expert Clouds

`analyze` builds an inverse hierarchy from groups to domains to prompt leaves to their strongest layer-local experts. It also exports soft expert-domain memberships and bounded same-layer co-activation edges. Prompt contributions are normalized before domain aggregation so long prompts do not automatically dominate.

`build-packs` combines target-domain membership scores with co-activation centrality under a percentage budget. Its output is a standard `evm-ability-selection-v1` file accepted by the expert-vault builder. This is a candidate selector; validation and held-out quality gates remain mandatory before adopting the pack.

## Verified Smoke Test

The persistent binary completed two independent Qwen1.5 prompts after one model load. Both prompts produced separate prefill and generation snapshots, 288 compact tensor rows total, and two checkpoint entries. A second invocation reported zero executed and two skipped prompts, verifying resume behavior.
