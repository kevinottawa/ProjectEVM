# Final Workflow Figures

These figures extend the proof artifacts into workflow-level evaluation.

| Figure | Source data | Purpose |
| --- | --- | --- |
| `evm_qwen15_moe_max_context_vram.png` | `results/final_workflow/max_context.csv` | Qwen1.5-MoE VRAM growth from 512 to 65,536 context for native GPU and CPU-backed EVM. |
| `evm_deepseek_coder_v2_lite_max_context_vram.png` | `results/final_workflow/max_context.csv` | DeepSeek-Coder-V2-Lite VRAM growth from 512 to 65,536 context for native GPU and CPU-backed EVM. |
| `evm_max_context_summary.png` | `results/final_workflow/max_context.csv` | Highest tested context comparison for native GPU and CPU-backed EVM. |

Interpretation:

- The max-context workflow is supported: CPU-backed EVM leaves substantially more VRAM headroom at the same tested context length.
- The old speculative timeout figure is archived. The completed negative benchmark is included in `../production_evm/qwen2_production_vram_throughput.png`.
