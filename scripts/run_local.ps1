# Run inference without Docker (conda env with requirements.txt + torch 2.1 installed). Repository root.
param([string]$Input = ".\input", [string]$Output = ".\output", [string]$Weights = ".\weights")
$ErrorActionPreference = "Stop"
python .\predict.py --input $Input --output $Output --weights $Weights
