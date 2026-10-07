# v8: 2-channel residual-map experiment

This package extends v7.1 official-score logging with a 2-channel input option.

## Channel definitions

- `--in_chans 1`: original min-max normalized CXR only. This reproduces the v7.1 baseline.
- `--in_chans 2`: `[original min-max CXR, residual map]`.
- `--in_chans 3`: optional baseline-style `[minmax, CLAHE, minmax]`.

For `--in_chans 2`, use:

- `--residual_mode abs`: `abs(X - GaussianBlur(X))`. Recommended first run.
- `--residual_mode dark`: `ReLU(GaussianBlur(X) - X)`. Cavity-lucency-targeted but higher FP risk.
- `--residual_mode bright`: `ReLU(X - GaussianBlur(X))`. Edge/bright-structure ablation.
- `--residual_sigma`: Gaussian sigma before resize/pad. Start with `5.0`.

## Recommended first v8 command

```powershell
python cavity_mtl_pkg\train_cavity_mtl.py `
  --images_dir "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\Data\train\CXR" `
  --masks_dir  "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\Data\train\CXR_label" `
  --out_dir runs\cavity_mtl_convnext_tiny_fold0_v8_absres_2ch `
  --model convnext_tiny.fb_in22k_ft_in1k `
  --in_chans 2 `
  --residual_mode abs `
  --residual_sigma 5.0 `
  --img_size 512 `
  --epochs 80 `
  --batch_size 8 `
  --fold 0 `
  --n_splits 5 `
  --amp `
  --ema `
  --num_workers 0 `
  --seg_warmup_epochs 12 `
  --seg_weight_warmup 2.0 `
  --cls_weight_warmup 0.3 `
  --neg_seg_beta 0.08 `
  --lr 2e-5 `
  --consistency_warmup_epochs 30 `
  --cls_tail_weight 0.15 `
  --cls_margin_weight 0.05 `
  --cls_margin 1.0 `
  --cls_tail_topk_frac 0.35 `
  --late_pos2_start 0.85 `
  --late_hard_ratio 0.28
```

Compare against v7.1 using `best_official_score`, `official_mean_dice`, `recall`, `specificity`, `fn`, `fp`, and `mask_max_pos_mean / mask_max_neg_mean`.

