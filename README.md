# EVM: Expert Virtual Memory

**Short-Horizon Working-Set Residency for Mixture-of-Experts Inference**

EVM is a research prototype for treating MoE expert weights as a managed working set rather than a fixed full-model GPU-residency cost. It asks a practical question: can sparse routing let an MoE model use a bounded GPU expert pool while cold experts live in a lower memory tier or remain in a GGUF mapping?

This repository is the publication companion for the study by **Kevin Price**. It is organized as a research landing page, not a model or raw-experiment dump.

## Read the Study

- [Open the rendered research portal](https://kevinottawa.github.io/ProjectEVM/)
- [Paper: EVM, short-horizon working-set residency](docs/EVM_Paper.md)
- [Experimental hardware, runtime, and model artifacts](docs/EXPERIMENTAL_SETUP.md)
- [Final validation report](docs/validation_report.md)
- [llama.cpp prototype integration and patch guide](docs/LLAMA_CPP_INTEGRATION.md)

## What This Work Establishes

- A sanitized `llama.cpp` routing observatory captured 27,972 full router rows across 50 prompts and 24 MoE layers.
- MoE routing has measurable short-horizon reuse: median reuse distance is 3 tokens and the 95th percentile is 17 tokens.
- Oracle residency policy materially exceeds LRU in simulation; the learned predictor closes part of that gap but is not production-ready.
- Exact logical-to-physical CUDA expert remapping works without custom MoE kernels.
- CPU-backed EVM reduced peak VRAM by 6.2-18.8 GB across the tested models.
- Disk-backed exact-eight EVM demonstrated true targeted loading without full-model RAM or page-file residency, but at 0.40 tokens/s. It is a memory proof, not a production-speed result.
- Static GPU expert packs can reduce VRAM without CPU-expert fallback, but the present quality gate shows that low-budget derived packs are not yet production-quality models.

The paper is deliberately precise about the boundary: EVM validates memory mechanisms and the working-set opportunity, not a finished high-throughput alternative to full GPU residency.

## Repository Map

| Location | Contents |
| --- | --- |
| [`docs/index.html`](docs/index.html) | Source entry point for the GitHub Pages research portal. |
| [`docs/EVM_Paper.md`](docs/EVM_Paper.md) | Canonical paper, results, caveats, and references to evidence. |
| [`docs/figures/`](docs/figures/) and [`docs/tables/`](docs/tables/) | Curated figures and compact source tables used by the paper. |
| [`data/`](data/) | Sanitized routing telemetry used for analysis. |
| [`config/`](config/) | Versioned diagnostic prompts and predictor/MRI configurations. |
| [`results/`](results/) | Compact, curated machine-readable evidence behind the reported results. |
| [`scripts/`](scripts/) | Reproduction, measurement, analysis, and figure-export tools. |
| [`patches/llama.cpp/`](patches/llama.cpp/) | The exact 26-commit EVM patch series against the recorded llama.cpp base. |
| [`simulator/`](simulator/) | Simulator documentation and supporting code. |
| [`tools/publication/`](tools/publication/) | Publication-only reference and chart-integrity checks. |

## Reproduce the Paper Checks

```powershell
python tools/publication/check_paper_references.py
python tools/publication/verify_chart_reproducibility.py
```

The llama.cpp patch series is documented in [the integration guide](docs/LLAMA_CPP_INTEGRATION.md). It applies cleanly to the precise upstream base recorded there. The repository includes sanitized routing telemetry and compact evidence needed to reproduce its figures. Model weights, unsanitized observatory traces, local build outputs, expert vaults, and raw benchmark logs are intentionally not published. New bulk outputs under `results/` are ignored; the small evidence files already tracked there are intentional publication inputs.

### Publish the Reader

The visual paper reader is a static site in `docs/`. In GitHub, open **Settings -> Pages**, select **Deploy from a branch**, then choose `main` and `/docs`. GitHub will publish it at `https://kevinottawa.github.io/ProjectEVM/`.

## License and Citation

- Repository code and scripts: [MIT License](LICENSE)
- Paper, original figures, and original tables: [CC BY 4.0](PAPER_LICENSE.md)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Third-party notices: [NOTICE.md](NOTICE.md)

## Scope

This is a paper-contained research prototype. It is not a fork of `llama.cpp`, not a production inference engine, and not a claim that EVM beats full-GPU model residency when full residency is available.
