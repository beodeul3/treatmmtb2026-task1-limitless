# CavityMTLNet: Task1 공동 유무 + 공동 세그멘테이션 단일 모델

이 패키지는 TREAT-MMTB Task1용으로 설계한 **single-pass bidirectional multi-task cavity detection/segmentation** 코드입니다.

핵심 설계:

- **ConvNeXt backbone**: BatchNorm 의존 회피. ConvNeXt는 LayerNorm 기반이라 small batch와 curriculum sampling에 안전합니다.
- **GroupNorm decoder**: batch size 8~12에서도 안정적인 FPN/UNet식 decoder.
- **Classifier → Segmentation**: residual FiLM modulation.
  - `F_dec' = F_dec * (1 + alpha * A_cls) + alpha * beta_cls`
  - `alpha`는 0에서 시작하므로 초기 classifier 오류가 segmentation을 죽이지 않습니다.
- **Segmentation → Classifier**: mask-aware pooling.
  - segmentation branch가 찾은 localized evidence를 classifier가 직접 참조합니다.
- **Loss**:
  - positive: Dice/Tversky + Focal BCE
  - negative: Focal BCE 기반 false-positive suppression
  - classification: Focal BCE
  - uncertainty weighting: `s_seg`, `s_cls` 학습
  - consistency loss: warm-up 이후, stop-gradient 방식으로 약하게 사용
- **Sampler**:
  - batch-level class-balanced sampler
  - curriculum pos:neg schedule
  - online EMA hard example cache
  - trimmed hard mining으로 noise/outlier 상위 extreme sample 배제
  - 저빈도 full audit pass로 stale cache 보정

---

## 설치

```powershell
cd "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1"
pip install -r cavity_mtl_pkg\requirements.txt
```

또는 직접:

```powershell
pip install torch timm opencv-python albumentations numpy pandas scikit-learn tqdm
```

---

## 폴더 가정

이미지와 mask는 같은 stem 이름이어야 합니다.

```text
data/train/images/xxx.png
data/train/masks/xxx.png
```

mask가 비어 있으면 label 0, mask area가 1 pixel 이상이면 label 1로 자동 생성합니다.

---

## 1차 학습 명령어

VRAM이 넉넉하지 않으면 `--batch_size 8`부터 시작하세요.

```powershell
python cavity_mtl_pkg\train_cavity_mtl.py `
  --images_dir "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\data\train\images" `
  --masks_dir  "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\data\train\masks" `
  --out_dir runs\cavity_mtl_convnext_tiny_fold0 `
  --model convnext_tiny.fb_in22k_ft_in1k `
  --img_size 512 `
  --epochs 80 `
  --batch_size 8 `
  --fold 0 `
  --n_splits 5 `
  --amp `
  --ema
```

권장 후속 실험:

```powershell
python cavity_mtl_pkg\train_cavity_mtl.py `
  --images_dir "..." `
  --masks_dir "..." `
  --out_dir runs\cavity_mtl_convnext_small_fold0 `
  --model convnext_small.fb_in22k_ft_in1k `
  --img_size 512 `
  --epochs 80 `
  --batch_size 6 `
  --fold 0 `
  --n_splits 5 `
  --amp `
  --ema
```

---

## 학습 중 저장물

```text
runs/cavity_mtl_convnext_tiny_fold0/
├─ best_cavity_mtl.pt
├─ best_metrics.json
├─ last_metrics.json
├─ val_predictions_best.csv
├─ val_predictions_latest.csv
├─ hard_pool_latest.json
└─ args.json
```

`best_metrics.json`에서 반드시 확인할 항목:

```text
best_thresholds.dsc_all
best_thresholds.dsc_pos
best_thresholds.acc
best_thresholds.recall
best_thresholds.specificity
best_thresholds.fn
best_thresholds.objective
```

`dsc_all`만 보면 empty case 착시가 생길 수 있으므로 `dsc_pos`, `recall`, `fn`을 같이 봐야 합니다.

---

## Threshold 재최적화

학습 후 validation fold에서 후처리 threshold를 다시 계산합니다.

```powershell
python cavity_mtl_pkg\optimize_thresholds.py `
  --ckpt runs\cavity_mtl_convnext_tiny_fold0\best_cavity_mtl.pt `
  --images_dir "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\data\train\images" `
  --masks_dir  "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\data\train\masks" `
  --recall_constraint 0.90 `
  --amp
```

기본 objective:

```text
0.4 * DSC_all + 0.3 * DSC_pos + 0.3 * Accuracy - 0.3 * FN_rate_pos
subject to Recall_pos >= 0.90
```

---

## Test inference

```powershell
python cavity_mtl_pkg\infer_cavity_mtl.py `
  --ckpt runs\cavity_mtl_convnext_tiny_fold0\best_cavity_mtl.pt `
  --images_dir "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\data\test\images" `
  --out_csv runs\cavity_mtl_convnext_tiny_fold0\test_predictions.csv `
  --out_mask_dir runs\cavity_mtl_convnext_tiny_fold0\test_masks `
  --amp
```

추론은 항상 segmentation branch를 실행합니다. 이후 후처리에서만:

```python
if p_cavity < t_cls:
    final_label = 0
    final_mask = all_zero
else:
    final_label = 1
    final_mask = postprocess(mask_prob)
```

---

## 중요한 튜닝 포인트

### 1. BatchNorm 회피

기본 권장 backbone은 ConvNeXt입니다. ResNet류를 쓰려면 BN freeze 또는 GroupNorm 치환이 필요합니다.

### 2. Consistency loss

초반 collapse 방지를 위해 기본값은 다음입니다.

```text
--consistency_warmup_epochs 30
--consistency_weight 0.03
```

학습이 불안정하면 먼저 consistency를 끄세요.

```powershell
--consistency_weight 0.0
```

### 3. Hard mining

초기 10 epoch은 hard mining을 끕니다.

```text
--hard_warmup_epochs 10
```

이후 online EMA cache로 hard pool을 만들고, 매 5 epoch마다 full audit pass를 합니다.

```text
--audit_every 5
```

속도가 너무 느리면:

```powershell
--audit_every 10
```

또는 full audit을 사실상 끕니다.

```powershell
--audit_every 0
```

### 4. Negative suppression

negative image의 segmentation 억제는 mean penalty가 아니라 Focal BCE입니다.

```text
--neg_seg_beta 0.10
```

false positive mask가 많으면 0.15까지 올리고, positive Dice가 떨어지면 0.05로 낮추세요.

---

## 추천 실험 순서

1. `convnext_tiny`, batch 8, consistency on, 80 epochs
2. 같은 설정으로 fold 1~4
3. `convnext_small`, batch 4~6
4. best fold들의 threshold stability 확인
5. 최종 Docker 환경 시간/VRAM에 맞춰 tiny 단일 모델 또는 tiny+small ensemble 결정


## DICOM/NIfTI 데이터 사용

이 버전은 다음 구조를 직접 지원합니다.

```text
Data/train/CXR        # .dcm images
Data/train/CXR_label  # .nii 또는 .nii.gz masks
```

파일명 stem이 같아야 자동 매칭됩니다.

```text
CXR/abc123.dcm             ↔ CXR_label/abc123.nii.gz
CXR/abc123.dcm             ↔ CXR_label/abc123.nii
```

필요 패키지:

```powershell
pip install pydicom nibabel
```

학습 예시:

```powershell
python cavity_mtl_pkg\train_cavity_mtl.py `
  --images_dir "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\Data\train\CXR" `
  --masks_dir  "C:\Users\PC00\Desktop\TREAT-MMTB 2026\Task1\Data\train\CXR_label" `
  --out_dir runs\cavity_mtl_convnext_tiny_fold0 `
  --model convnext_tiny.fb_in22k_ft_in1k `
  --img_size 512 `
  --epochs 80 `
  --batch_size 8 `
  --fold 0 `
  --n_splits 5 `
  --amp `
  --ema
```


## v6 변경: 단일 채널 입력 기본값

공식 baseline의 3채널 구성은 nnU-Net 예시용이므로 기본 학습은 `--in_chans 1`입니다. 참고하는 것은 DICOM/NIfTI mask 적용 방식입니다.

- DICOM: MONOCHROME1 보정 + Min-Max normalization
- Mask: SimpleITK 우선 읽기 + binary 변환
- Image/Mask: 같은 `resize_and_pad` 좌표계 적용, mask는 nearest interpolation
- `--in_chans 3`을 명시한 경우에만 optional baseline-style `[minmax, CLAHE, minmax]` 입력을 사용합니다.


## v6.3 patch
- AMP 환경에서 consistency BCE가 `binary_cross_entropy unsafe to autocast` 오류를 내는 문제를 fp32 consistency 계산으로 수정했습니다.
- `--resume_ckpt` 옵션을 추가했습니다. 저장된 `best_cavity_mtl.pt`에서 model/criterion weight를 불러오고 optimizer/scheduler/hard-cache는 새로 시작합니다.
