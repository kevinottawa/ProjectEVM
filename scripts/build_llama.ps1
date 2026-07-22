param(
    [int]$Jobs = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LlamaDir = Join-Path $ProjectRoot "llama.cpp"

if (-not (Test-Path $LlamaDir)) {
    throw "llama.cpp checkout not found. Run scripts/setup.ps1 first."
}

cmake -S $LlamaDir -B (Join-Path $LlamaDir "build") -DGGML_CUDA=ON
cmake --build (Join-Path $LlamaDir "build") --config Release -j $Jobs
