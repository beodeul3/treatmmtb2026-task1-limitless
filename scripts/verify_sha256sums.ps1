# Verify downloaded checkpoints against SHA256SUMS.txt. Run from the repository root.
$ErrorActionPreference = "Stop"
$bad = 0
foreach ($line in Get-Content .\SHA256SUMS.txt) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split "\s+", 2
    $expected = $parts[0].ToLower(); $rel = $parts[1].Trim()
    if (-not (Test-Path $rel)) { Write-Host "MISSING  $rel"; $bad++; continue }
    $actual = (Get-FileHash $rel -Algorithm SHA256).Hash.ToLower()
    if ($actual -eq $expected) { Write-Host "OK       $rel" } else { Write-Host "MISMATCH $rel"; $bad++ }
}
if ($bad -eq 0) { Write-Host "All checksums OK." } else { Write-Warning "$bad file(s) missing or mismatched." }
