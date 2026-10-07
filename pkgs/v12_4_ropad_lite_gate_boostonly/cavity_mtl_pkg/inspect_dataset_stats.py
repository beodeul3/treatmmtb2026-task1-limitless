from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cavity_mtl.dataset import build_image_mask_table, read_image, read_mask, align_mask_to_image, resize_and_pad


def main():
    ap = argparse.ArgumentParser(description='Inspect DICOM/NIfTI matching and mask areas before training.')
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--masks_dir', required=True)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--max_examples', type=int, default=999999)
    args = ap.parse_args()

    rows = build_image_mask_table(args.images_dir, args.masks_dir)
    print(f'Total rows: {len(rows)}')
    print(f'Positive: {sum(int(r["label"]) for r in rows)}  Negative: {sum(1-int(r["label"]) for r in rows)}')

    areas_raw, areas_512 = [], []
    shapes = {}
    align_counts = {}
    examples = []
    for r in rows[:args.max_examples]:
        img = read_image(r['image_path'])
        mask = read_mask(r['mask_path'])
        shapes[(img.shape, mask.shape)] = shapes.get((img.shape, mask.shape), 0) + 1
        mask2, action = align_mask_to_image(mask, img)
        align_counts[action] = align_counts.get(action, 0) + 1
        mask512 = resize_and_pad(mask2, args.img_size, is_mask=True)
        areas_raw.append(float(mask2.mean()))
        areas_512.append(float(mask512.mean()))
        if int(r['label']) == 1 and len(examples) < 10:
            examples.append((r['image_id'], img.shape, mask.shape, float(mask2.sum()), float(mask512.sum()), r['image_path'], r['mask_path']))

    a = np.asarray(areas_512)
    pos_a = a[a > 0]
    print('Unique image/mask shape pairs, first 20:')
    for k, v in list(sorted(shapes.items(), key=lambda x: -x[1]))[:20]:
        print(f'  count={v} image={k[0]} mask={k[1]}')
    print('Mask alignment actions:')
    for k, v in sorted(align_counts.items()):
        print(f'  {k}: {v}')
    if len(pos_a):
        print('Positive mask area fraction after 512 resize:')
        print(f'  min={pos_a.min():.8f} p10={np.percentile(pos_a,10):.8f} median={np.median(pos_a):.8f} mean={pos_a.mean():.8f} max={pos_a.max():.8f}')
        print('Positive examples:')
        for e in examples:
            print(' ', e)
    else:
        print('No positive masks found after loading. This indicates label reading is broken.')


if __name__ == '__main__':
    main()
