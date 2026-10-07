from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cavity_mtl.dataset import build_image_mask_table, CavityDataset


def to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min())
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def make_overlay(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    m = (mask > 0.5).astype(np.uint8)
    color = np.zeros_like(base)
    color[..., 2] = 255
    overlay = cv2.addWeighted(base, 0.75, color, 0.25, 0)
    return np.where(m[..., None] > 0, overlay, base)


def main():
    ap = argparse.ArgumentParser(description='Visualize preprocessed DICOM/NIfTI image-mask overlays.')
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--masks_dir', required=True)
    ap.add_argument('--out_dir', default='runs/debug_dicom_nii_samples')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--num_pos', type=int, default=20)
    ap.add_argument('--num_neg', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--in_chans', type=int, default=1, choices=[1, 3], help='Visualization input channels; default 1. Use 3 only for optional baseline-style channels.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_image_mask_table(args.images_dir, args.masks_dir)
    pos = [r for r in rows if int(r['label']) == 1]
    neg = [r for r in rows if int(r['label']) == 0]
    rng = np.random.default_rng(args.seed)
    if len(pos) > args.num_pos:
        pos = [pos[i] for i in rng.choice(len(pos), size=args.num_pos, replace=False)]
    if len(neg) > args.num_neg:
        neg = [neg[i] for i in rng.choice(len(neg), size=args.num_neg, replace=False)]
    sample_rows = pos + neg
    ds = CavityDataset(sample_rows, img_size=args.img_size, augment=None, in_chans=args.in_chans, has_masks=True)

    summary_path = out_dir / 'summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'image_id', 'label', 'mask_pixels_512', 'mask_area_frac_512', 'image_path', 'mask_path', 'overlay_path'])
        for i in range(len(ds)):
            item = ds[i]
            img = item['image'].numpy()[0]
            mask = item['mask'].numpy()[0]
            img_u8 = to_u8(img)
            overlay = make_overlay(img_u8, mask)
            mask_u8 = ((mask > 0.5) * 255).astype(np.uint8)
            canvas = np.concatenate([cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR), overlay], axis=1)
            label = int(float(item['label']))
            fn = out_dir / f"{i:03d}_y{label}_{item['image_id']}.png"
            cv2.imwrite(str(fn), canvas)
            r = sample_rows[i]
            w.writerow([i, item['image_id'], label, int(mask.sum()), float(mask.mean()), r['image_path'], r['mask_path'], str(fn)])
    print(f'Saved overlays to: {out_dir}')
    print(f'Summary CSV: {summary_path}')
    print(f'Total rows={len(rows)} positives={sum(int(r["label"]) for r in rows)} negatives={sum(1-int(r["label"]) for r in rows)}')


if __name__ == '__main__':
    main()
