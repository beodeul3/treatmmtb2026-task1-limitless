# Write SHA256SUMS.txt for the checkpoints. Run from the repository root (Anaconda PowerShell Prompt).
$ErrorActionPreference = "Stop"
$files = Get-ChildItem -Recurse -Filter *.pt .\weights | Sort-Object FullName
if ($files.Count -eq 0) { throw "No .pt files under .\weights" }
$root = (Resolve-Path ".").Path
$lines = foreach ($f in $files) {
    $h = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower()
    $rel = $f.FullName.Substring($root.Length + 1) -replace "\\", "/"
    "$h  $rel"
}
$lines | Set-Content -Encoding ascii .\SHA256SUMS.txt
Write-Host "wrote SHA256SUMS.txt ($($files.Count) files)"
Get-Content .\SHA256SUMS.txt
