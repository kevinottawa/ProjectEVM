# Production EVM Figures

| Figure | Source |
| --- | --- |
| `qwen2_production_vram_throughput.png` | `docs/tables/production_evm_runtime.csv`; peak VRAM, process RSS/page-file growth, and throughput across all four residency paths |
| `qwen2_disk_io_controls.png` | `docs/tables/disk_io_controls.csv`; strict/lazy mmap, bounded pinned staging, and file-page retention experiments |
| `qwen2_disk_gpu_pool_sweep.png` | `docs/tables/disk_gpu_pool_sweep.csv`; bounded GPU expert capacity versus throughput, VRAM, hit rate, and transfer volume |
| `layer_aware_predictor_hit_rate.png` | `docs/tables/layer_aware_predictor_results.csv` |

Regenerate with `python scripts/plot_production_evm.py`.
