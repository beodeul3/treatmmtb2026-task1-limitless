# Model B  [I, h-dome]  --  package v18_hdome2ch_boostonly (second channel replaced by the h-dome via CAVITY_FORCE_HDOME2CH)
# Anaconda PowerShell Prompt, repository root. Exact arguments: configs\train_args_B.json.
param(
    [string]$Images = ".\Data\train\CXR",
    [string]$Masks  = ".\Data\train\CXR_label",
    [int[]]$Folds   = @(0, 1, 2, 3, 4),
    [string]$Gpu    = "0"
)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
Get-ChildItem Env: | Where-Object { $_.Name -like "CAVITY_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
$env:CAVITY_FORCE_HDOME2CH = "1"
$env:CAVITY_HDOME_H = "0.08"
New-Item -ItemType Directory -Force ".\runs", ".\logs" | Out-Null
foreach ($Fold in $Folds) {
    $out = ".\runs\B_fold$Fold"
    python -u ".\pkgs\v18_hdome2ch_boostonly\cavity_mtl_pkg\train_cavity_mtl.py" `
      --images_dir $Images --masks_dir $Masks --out_dir $out `
      --model convnext_tiny.fb_in22k_ft_in1k --in_chans 2 --residual_mode dark --residual_sigma 5.0 `
      --img_size 512 --epochs 40 --batch_size 8 --num_workers 4 --amp `
      --fold $Fold --n_splits 5 --seed 42 --lr 2e-4 --weight_decay 1e-4 `
      --decoder_ch 160 --cls_dim 256 --dropout 0.2 2>&1 | Tee-Object ".\logs\B_fold$Fold.log"
}
