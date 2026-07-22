$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LlamaDir = Join-Path $ProjectRoot "llama.cpp"
$VenvDir = Join-Path $ProjectRoot "venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$LlamaBase = "6f8895feec96773574c7e10fcf7b56965d23550a"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "data") | Out-Null

if (-not (Test-Path $LlamaDir)) {
    git clone https://github.com/ggml-org/llama.cpp.git $LlamaDir
    git -C $LlamaDir checkout $LlamaBase
}

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "Setup complete. Apply the EVM patches using docs/LLAMA_CPP_INTEGRATION.md, then run scripts/build_llama.ps1."
