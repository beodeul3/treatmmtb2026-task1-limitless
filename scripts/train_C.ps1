# Model C  [I, dark residual, h-basin]  --  package v19_dark_hbasin_sep (3-channel input built by the dataset wrapper via CAVITY_FORCE_V19_DARK_HBASIN_3CH)
# Anaconda PowerShell Prompt, repository root. Exact arguments: configs\train_args_C.json.
param(
    [string]$Images = ".\Data\train\CXR",
    [string]$Masks  = ".\Data\train\CXR_label",
    [int[]]$Folds   = @(0, 1, 2, 3, 4),
    [string]$Gpu    = "0"
)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONWARNINGS = "ignore"
Get-ChildItem Env: | Where-Object { $_.Name -like "CAVITY_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
$env:CAVITY_FORCE_V19_DARK_HBASIN_3CH = "1"
$env:CAVITY_V19_HBASIN_H = "0.08"
$env:CAVITY_RESIDUAL_SIGMA = "5.0"
$env:CAVITY_V19_FULL_DECOUPLE = "1"
New-Item -ItemType Directory -Force ".\runs", ".\logs" | Out-Null
foreach ($Fold in $Folds) {
    $out = ".\runs\C_fold$Fold"
    python -X utf8 ".\pkgs\v19_dark_hbasin_sep\cavity_mtl_pkg\train_cavity_mtl.py" `
      --images_dir $Images --masks_dir $Masks --out_dir $out `
      --model convnext_tiny.fb_in22k_ft_in1k --in_chans 3 --residual_mode dark --residual_sigma 5.0 `
      --img_size 512 --epochs 40 --batch_size 4 --num_workers 0 --amp `
      --fold $Fold --n_splits 5 --seed 42 --lr 2e-4 --weight_decay 1e-4 `
      --decoder_ch 160 --cls_dim 256 --dropout 0.2 2>&1 | Tee-Object ".\logs\C_fold$Fold.log"
}
