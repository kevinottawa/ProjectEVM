# Experimental Setup

This record describes the single workstation and local GGUF artifacts used for the runtime figures in the EVM paper. It is not a claim that results will reproduce unchanged on other hardware.

## Workstation

| Component | Recorded value |
| --- | --- |
| CPU | AMD Ryzen 9 5900X, 12 cores / 24 logical processors |
| Physical RAM | 31.9 GiB |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GPU memory reported by `nvidia-smi` | 24,564 MiB |
| NVIDIA driver | 610.62 |
| Operating system | Windows 11 Pro, build 22631 |
| CUDA toolkit used for local build | CUDA 13.2 |
| GGUF storage observed during experiments | ADATA LEGEND 840, 954 GiB SSD |

## Runtime Revision

| Item | Value |
| --- | --- |
| Tested EVM patch head | `c5b04c42c71d553f28260d20c3238fce3c13bbd3` |
| Pinned upstream llama.cpp base | `6f8895feec96773574c7e10fcf7b56965d23550a` |
| Public patch series | [`patches/llama.cpp/`](../patches/llama.cpp/) |
| Build configuration | CUDA enabled; exact command in [LLAMA_CPP_INTEGRATION.md](LLAMA_CPP_INTEGRATION.md) |

## Tested Local GGUF Artifacts

| Local file | Size | Quantization label | EVM target count | Original model source |
| --- | ---: | --- | ---: | --- |
| `Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf` | 6.93 GiB | Q3_K_M | 60 | [Qwen model card](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat) |
| `DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf` | 9.65 GiB | Q4_K_M | 64 | [DeepSeek-Coder-V2 report](https://arxiv.org/abs/2406.11931) |
| `qwen2-57b-a14b-instruct-q4_k_m.gguf` | 32.46 GiB | Q4_K_M | 64 | [Qwen2 report](https://arxiv.org/abs/2407.10671) |

The local GGUF files are not distributed in this repository. They are inference artifacts derived from their source-model releases. Exact runtime command controls, EVM environment variables, memory accounting, and validity gates are documented in [evm_runtime_env.md](evm_runtime_env.md) and [EVM_Paper.md](EVM_Paper.md).

## Measurement Scope

- Results are from one workstation, a PCIe-attached consumer GPU, and the listed Windows build/driver/CUDA stack.
- Peak GPU memory came from `nvidia-smi` sampling; host RSS, private commit, system-RAM delta, and page-file delta were separately recorded by the benchmark harness.
- Throughput, memory, and quality results are tied to the local GGUF quantizations and should be replicated before comparison with another system or another llama.cpp revision.
