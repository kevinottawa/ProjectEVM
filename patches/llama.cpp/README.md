# llama.cpp EVM Patch Series

This directory contains the 26-commit EVM research prototype patch series.
It is the complete llama.cpp implementation used by the paper, packaged here
instead of as a separate public fork.

## Tested base

- Upstream project: `ggml-org/llama.cpp`
- Exact base commit: `6f8895feec96773574c7e10fcf7b56965d23550a`
- Tested patch head: `c5b04c42c`

## Apply

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
Set-Location llama.cpp
git checkout 6f8895feec96773574c7e10fcf7b56965d23550a
git am --3way ..\ProjectEVM\patches\llama.cpp\*.patch
```

Build the patched `llama-cli` with the CMake command documented in
[`docs/LLAMA_CPP_INTEGRATION.md`](../../docs/LLAMA_CPP_INTEGRATION.md).

The patches are a paper-contained prototype. They are not represented as an
upstream llama.cpp contribution or production-ready runtime.
