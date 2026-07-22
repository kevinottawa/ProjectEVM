# Final Proof Figures

Canonical final-proof figures live in this folder. Root-level PNGs in `docs/` are retained only for backward compatibility with earlier drafts.

Naming convention:

`evm_<model-or-scope>_<measurement>.png`

Model IDs:

| ID | Meaning |
| --- | --- |
| `qwen15_moe` | Qwen1.5-MoE-A2.7B |
| `deepseek_coder_v2_lite` | DeepSeek-Coder-V2-Lite |
| `qwen2_57b_a14b` | Qwen2-57B-A14B |
| `cross_model` | Native GPU vs CPU-backed VRAM residency across models |
| `gpu_backed` | GPU-backed remapping validation across models |

Figure index:

| Figure | Purpose |
| --- | --- |
| `evm_cross_model_vram_residency.png` | Native GPU vs CPU-backed EVM 33% VRAM residency for all three models. |
| `evm_qwen15_moe_mode_comparison.png` | Qwen1.5 native GPU, CPU-backed EVM, and GPU-backed EVM comparison. |
| `evm_deepseek_coder_v2_lite_mode_comparison.png` | DeepSeek native GPU, CPU-backed EVM, and GPU-backed EVM comparison. |
| `evm_qwen2_57b_a14b_mode_comparison.png` | Qwen2 native GPU and CPU-backed EVM comparison; GPU-backed duplicate-pool path is not run. |
| `evm_gpu_backed_remapping_summary.png` | GPU-backed EVM hit rate and VRAM overhead for completed remapping-validation runs. |
| `evm_qwen15_moe_gpu_backed_capacity_hit_rate.png` | Qwen1.5 GPU-backed hit rate by capacity. |
| `evm_qwen15_moe_gpu_backed_capacity_throughput.png` | Qwen1.5 GPU-backed throughput by capacity. |
| `evm_qwen15_moe_gpu_backed_vram_overhead.png` | Qwen1.5 GPU-backed duplicate-pool VRAM overhead. |
| `evm_qwen15_moe_reproduction_throughput.png` | Qwen1.5 five-trial averaged throughput for CPU-only, native GPU, CPU-backed, and GPU-backed modes. |
| `evm_qwen15_moe_vram_by_mode.png` | Qwen1.5 peak VRAM across CPU-only, native GPU, CPU-backed EVM, and GPU-backed EVM modes. |
| `evm_qwen15_moe_vram_throughput_tradeoff.png` | Qwen1.5 VRAM/throughput tradeoff. |
| `evm_deepseek_coder_v2_lite_gpu_backed_trials.png` | DeepSeek GPU-backed EVM 33% five-trial remapping validation. |
