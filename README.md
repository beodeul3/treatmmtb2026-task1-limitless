# TREAT-MMTB 2026 Task 1 — Team Limitless

Training code, trained weights and inference / post-processing code of our Task 1 entry (1st place).

Paper: Kang B., Son W., Kang M., Lee H. *Morphology-Informed Deep Ensemble Learning for Pulmonary Cavity Detection and Segmentation on Chest X-Rays.* TREAT-MMTB 2026. <!-- TODO: link -->

## Layout

```
Dockerfile, requirements.txt, SHA256SUMS.txt, LICENSE, NOTICE
predict.py                 inference: DICOM in -> prediction.csv + NIfTI masks out (ensemble + post-processing)
pkgs/v12_4_ropad_lite_gate_boostonly/cavity_mtl_pkg/   Model A  [I, dark residual]          + train_cavity_mtl.py
pkgs/v18_hdome2ch_boostonly/cavity_mtl_pkg/            Model B  [I, h-dome]                 + train_cavity_mtl.py
pkgs/v19_dark_hbasin_sep/cavity_mtl_pkg/               Model C  [I, dark residual, h-basin] + train_cavity_mtl.py
weights/{A,B,C}/fold{0..4}.pt                          not in git (see Weights)
configs/train_args_{A,B,C}.json                        exact training arguments, read from the checkpoints
scripts/train_{A,B,C}.ps1                              training launchers; run_docker.ps1, run_local.ps1, verify_sha256sums.ps1, make_synthetic_dicom.py
```

## Weights

15 checkpoints, 120.75 MB each (1.69 GB). Download: <!-- TODO: Hugging Face URL -->. Place under `weights/A`, `weights/B`, `weights/C`. Verify: `.\scripts\verify_sha256sums.ps1` or `sha256sum -c SHA256SUMS.txt`.

Checkpoint keys: `model` (state dict), `args`, `model_config`, `preprocess_config`, `thresholds`, `epoch`, `best_scores`, ... Only `model` is used at inference; `thresholds` are per-fold validation values, not the ensemble thresholds.

## Run

Input: `input/<case_id>/*.dcm`. Output: `output/<case_id>.nii.gz` (uint8 mask, input geometry), `output/prediction.csv` (`our_id,cavity`).

```
docker build -t treatmmtb-task1-limitless .
docker run --rm --gpus all -v "${PWD}\input:/input" -v "${PWD}\output:/output" treatmmtb-task1-limitless
```

Without Docker (Python 3.10, torch 2.1.0):

```
pip install -r requirements.txt
python predict.py --input .\input --output .\output --weights .\weights
```

CPU is used when no GPU is visible.

## Inference and post-processing (predict.py)

Constants at the top of `predict.py`, printed at start-up.

- DICOM: pydicom `pixel_array`, MONOCHROME1 inverted, min–max to [0, 1]. No rescale slope/intercept, no VOI LUT.
- 512 × 512, aspect-preserving resize, zero padding.
- Dark residual: `max(GaussianBlur(I, σ=5) − I, 0)`, 1st/99th-percentile rescale. h-dome / h-basin: grayscale reconstruction, h = 0.08.
- Ensemble: 5 folds averaged per model; A/B/C = 0.49/0.38/0.13 on both `s` and `p`.
- `s < 0.30` → empty mask, `cavity = 0`. Otherwise `p ≥ 0.12` on the 512 grid; 8-connected components; a component is dropped if area ≤ 200 and p95 < 0.20 and max < 0.25; the 3 highest-scoring components (`p95·sqrt(area)`) are kept.
- Mask resized (nearest) to the original grid; `cavity = 1` iff the final mask is non-empty.

## Training

`pkgs/<variant>/cavity_mtl_pkg/train_cavity_mtl.py`, challenge training set only (444 CXRs, no external data), 5-fold (`--n_splits 5 --seed 42`), `convnext_tiny.fb_in22k_ft_in1k`, AdamW lr 2e-4, wd 1e-4. Launchers: `.\scripts\train_A.ps1`, `train_B.ps1`, `train_C.ps1` (Anaconda PowerShell Prompt; `-Images`, `-Masks`, `-Folds`, `-Gpu`). Every argument of every run is in `configs/train_args_{A,B,C}.json` (= `torch.load(ckpt)["args"]`); the checkpoint of each fold is the epoch with the best official validation score.

| | package | in_chans | env | epochs / batch / AMP |
|---|---|---|---|---|
| A | v12_4_ropad_lite_gate_boostonly | 2 | – | 80 / 4 / off |
| B | v18_hdome2ch_boostonly | 2 | `CAVITY_FORCE_HDOME2CH=1 CAVITY_HDOME_H=0.08` | 40 / 8 / on |
| C | v19_dark_hbasin_sep | 3 | `CAVITY_FORCE_V19_DARK_HBASIN_3CH=1 CAVITY_V19_HBASIN_H=0.08 CAVITY_RESIDUAL_SIGMA=5.0 CAVITY_V19_FULL_DECOUPLE=1` | 40 / 4 / on |

Augmentation (all variants): horizontal flip p = 0.5; shift 3 % / scale 5 % / rotate 5°, p = 0.45; brightness-contrast ±8 %, p = 0.35; Gaussian noise p = 0.15. No test-time augmentation. Ensemble weights, thresholds and the pruning rule were selected on the pooled out-of-fold predictions.

## License

Apache License 2.0 (code and weights) — see `LICENSE`, `NOTICE`.

## Citation

```bibtex
@inproceedings{kang2026morphology,
  title     = {Morphology-Informed Deep Ensemble Learning for Pulmonary Cavity Detection and Segmentation on Chest X-Rays},
  author    = {Kang, Beodeul and Son, Wonjun and Kang, Minwoo and Lee, Hyunyeol},
  booktitle = {TREAT-MMTB 2026 Challenge, MICCAI 2026},
  year      = {2026}
}
```
