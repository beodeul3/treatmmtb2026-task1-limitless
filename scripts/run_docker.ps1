# Build the container and run inference on .\input -> .\output (Anaconda PowerShell Prompt, repository root).
# Requires Docker Desktop with GPU support (NVIDIA driver + WSL2) for --gpus all; drop that flag to run on CPU.
param(
    [string]$Input = ".\input",
    [string]$Output = ".\output",
    [string]$Image = "treatmmtb-task1-limitless",
    [switch]$Cpu
)
$ErrorActionPreference = "Stop"
if ((Get-ChildItem -Recurse -Filter *.pt .\weights).Count -ne 15) { throw "weights/ is incomplete: expected 15 .pt files (see weights/README.md)" }
docker build -t $Image .
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$in = (Resolve-Path $Input).Path; $out = (Resolve-Path $Output).Path
if ($Cpu) { docker run --rm -v "${in}:/input" -v "${out}:/output" $Image }
else      { docker run --rm --gpus all -v "${in}:/input" -v "${out}:/output" $Image }
Write-Host "done -> $out"
