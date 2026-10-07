# TREAT-MMTB 2026 Task 1 — Team Limitless

Training code, weights and inference / post-processing code of our Task 1 entry.

```
predict.py                                  inference: DICOM in -> prediction.csv + NIfTI masks (ensemble + post-processing)
pkgs/<variant>/cavity_mtl_pkg/              model code + train_cavity_mtl.py  (A: v12_4_ropad_lite_gate_boostonly, B: v18_hdome2ch_boostonly, C: v19_dark_hbasin_sep)
scripts/train_{A,B,C}.ps1                   training launchers; configs/train_args_{A,B,C}.json = exact arguments
weights/{A,B,C}/fold{0..4}.pt               https://huggingface.co/beodeul/treatmmtb2026-task1-limitless  (SHA256SUMS.txt)
Dockerfile, requirements.txt
```

## Run

```
pip install -r requirements.txt          # Python 3.10, torch 2.1.0
python predict.py --input .\input --output .\output --weights .\weights
```

Input `input/<case_id>/*.dcm`; output `output/<case_id>.nii.gz`, `output/prediction.csv` (`our_id,cavity`). Docker: `docker build -t limitless-task1 .` then `docker run --rm --gpus all -v <input>:/input -v <output>:/output limitless-task1`.

## Train

```
.\scripts\train_A.ps1 -Images <CXR dir> -Masks <label dir>
.\scripts\train_B.ps1 ...
.\scripts\train_C.ps1 ...
```

## License

Apache-2.0
