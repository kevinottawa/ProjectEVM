param(
    [string]$Repository = "tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF",
    [string]$FileName = "Qwen1.5-MoE-A2.7B-Chat-Q3_K_M.gguf",
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$HuggingFace = Join-Path $ProjectRoot "venv\Scripts\hf.exe"

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $ProjectRoot "models"
}
if (-not (Test-Path $HuggingFace)) {
    throw "Hugging Face CLI not found. Run scripts/setup.ps1 first."
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
& $HuggingFace download $Repository $FileName --local-dir $OutputDirectory
