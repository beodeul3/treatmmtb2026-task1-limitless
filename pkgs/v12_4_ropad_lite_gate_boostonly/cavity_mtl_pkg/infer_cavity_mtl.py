from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cavity_mtl.dataset import CavityDataset, build_image_mask_table
from cavity_mtl.metrics import remove_small_components
from cavity_mtl.model import CavityMTLNet
from cavity_mtl.utils import autocast_context, move_to_device, save_json

try:
    import albumentations as A
except Exception:
    A = None


def make_aug(img_size: int):
    if A is None:
        return None
    return A.Compose([A.Resize(img_size, img_size)])


def main():
    parser = argparse.ArgumentParser(description='Infer cavity probability and mask with trained CavityMTLNet')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--out_mask_dir', required=True)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--t_cls', type=float, default=None)
    parser.add_argument('--t_mask', type=float, default=None)
    parser.add_argument('--min_area', type=int, default=None)
    parser.add_argument('--keep_largest', action='store_true')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['model_config']
    train_args = ckpt.get('args', {})
    img_size = args.img_size or int(train_args.get('img_size', 512))

    thresholds = ckpt.get('thresholds', {}) or {}
    t_cls = args.t_cls if args.t_cls is not None else float(thresholds.get('t_cls', 0.5))
    t_mask = args.t_mask if args.t_mask is not None else float(thresholds.get('t_mask', 0.5))
    min_area = args.min_area if args.min_area is not None else int(thresholds.get('min_area', 0))

    model = CavityMTLNet(**cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    rows = build_image_mask_table(args.images_dir, masks_dir=None)
    prep = ckpt.get('preprocess_config', {}) or {}
    residual_mode = prep.get('residual_mode', train_args.get('residual_mode', 'abs'))
    residual_sigma = float(prep.get('residual_sigma', train_args.get('residual_sigma', 5.0)))
    ds = CavityDataset(
        rows, img_size=img_size, augment=make_aug(img_size),
        in_chans=int(cfg.get('in_chans', 1)), has_masks=False,
        residual_mode=residual_mode, residual_sigma=residual_sigma,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    out_mask_dir = Path(args.out_mask_dir)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'cavity_prob', 'cavity_label', 'threshold_cls', 'threshold_mask', 'min_area', 'mask_file'])
        for batch in tqdm(loader, desc='infer'):
            batch_dev = move_to_device(batch, device)
            with torch.no_grad(), autocast_context(args.amp):
                out = model(batch_dev['image'])
            probs = torch.sigmoid(out['cls_logit']).detach().cpu().numpy()
            mask_probs = torch.sigmoid(out['mask_logit']).detach().cpu().numpy()[:, 0]
            for iid, p, mp in zip(batch['image_id'], probs, mask_probs):
                label = int(p >= t_cls)
                if label == 0:
                    mask = np.zeros_like(mp, dtype=np.uint8)
                else:
                    mask = (mp >= t_mask).astype(np.uint8)
                    mask = remove_small_components(mask, min_area=min_area, keep_largest=args.keep_largest)
                mask_u8 = (mask * 255).astype(np.uint8)
                mask_name = f'{iid}.png'
                cv2.imwrite(str(out_mask_dir / mask_name), mask_u8)
                writer.writerow([iid, float(p), label, t_cls, t_mask, min_area, mask_name])

    save_json({'t_cls': t_cls, 't_mask': t_mask, 'min_area': min_area, 'keep_largest': args.keep_largest}, out_csv.parent / 'inference_thresholds.json')
    print(f'Saved: {out_csv}')
    print(f'Masks: {out_mask_dir}')


if __name__ == '__main__':
    main()
