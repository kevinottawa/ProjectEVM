# Citation Audit

This audit records the precise purpose of each external reference in `EVM_Paper.md`. It prevents a citation from being read as support for a broader claim than its source establishes.

| Ref. | Source role in this paper | Explicit boundary |
| ---: | --- | --- |
| 1 | Sparse gated MoE foundation. | Does not establish EVM's runtime results. |
| 2 | Modern sparse-MoE scaling context. | Does not establish consumer-GPU offloading. |
| 3 | Oracle's farthest-next-use replacement upper bound. | Not a citation for Windows paging, page-file behavior, or EVM implementation. |
| 4 | KV-cache paging comparison. | PagedAttention manages KV cache, not EVM expert weights. |
| 5 | General GPU/CPU/disk limited-memory inference context. | Not an expert-level exact-remapping implementation. |
| 6 | General transformer inference systems context. | Not evidence for this prototype's measurements. |
| 7 | Optional speculative-decoding motivation. | Not used to claim an EVM speedup. |
| 8 | Qwen2 model-family context. | Does not cover the project's measured GGUF quantization or hardware results. |
| 9 | DeepSeek-Coder-V2 model-family context. | Does not cover the project's measured GGUF quantization or hardware results. |
| 10 | Upstream runtime provenance. | The exact reproducibility base is the pinned commit in `LLAMA_CPP_INTEGRATION.md`, not upstream `master`. |
| 11 | Direct prior art for GPU hot-expert buffering. | EVM does not claim first expert caching. |
| 12 | Direct prior art for virtual-expert mappings under a memory budget. | EVM differs in its llama.cpp/GGUF implementation and measurement scope. |
| 13 | Direct prior art for consumer MoE offloading. | EVM does not claim first consumer-hardware MoE offload. |
| 14 | Qwen1.5-MoE-A2.7B-Chat source-model identity. | Does not cover the project's local Q3_K_M GGUF conversion or measurements. |

The paper's original contribution claim is therefore limited to its own implementation, experimental evidence, artifact package, and the measured boundaries reported in the paper.
