# llama.cpp Integration

EVM is reproduced by applying the bundled patch series to the exact upstream
llama.cpp base below. The repository deliberately ships patches rather than a
separate llama.cpp fork.

## Pinned Base

| Item | Value |
| --- | --- |
| Upstream | `https://github.com/ggml-org/llama.cpp` |
| Base commit | `6f8895feec96773574c7e10fcf7b56965d23550a` |
| EVM patch head | `c5b04c42c` |
| Patch count | 26 |
| Scope | Paper-contained research prototype; not an upstream pull request. |

## Apply the Patch Series

```powershell
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
git -C llama.cpp checkout 6f8895feec96773574c7e10fcf7b56965d23550a
$patches = Get-ChildItem patches\llama.cpp\*.patch | Sort-Object Name | Select-Object -ExpandProperty FullName
git -C llama.cpp am --3way $patches
```

If Git reports a conflict, stop and use the exact base commit above rather
than trying to apply the series to a newer upstream revision.

## Windows CUDA Build

The reported tests used Windows, Visual Studio 2022, CUDA, and CMake. From the
ProjectEVM repository root with a patched `llama.cpp` checkout:

```powershell
scripts\build_llama.ps1 -Jobs 2
```

The benchmark scripts require a CUDA-capable build and local GGUF model files.
They do not download or redistribute models.

## What the Patches Add

- MoE routing observatory and trace capture.
- CPU/disk-backed expert-vault hooks.
- Exact CUDA logical-to-physical expert remapping.
- Bounded expert pools, counters, and strict capacity guards.
- Experimental LRU, learned, router-score, page-table, and GPU-scheduler
  policies.

The paper distinguishes validated mechanisms from experimental no-go policies.
Read [EVM_Paper.md](EVM_Paper.md) before treating any runtime mode as a
deployment recommendation.
