from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import albumentations as A
except Exception:
    A = None
try:
    from sklearn.model_selection import StratifiedKFold
except Exception:
    StratifiedKFold = None

from cavity_mtl.dataset import CavityDataset, build_image_mask_table
from cavity_mtl.metrics import threshold_sweep
from cavity_mtl.model import CavityMTLNet
from cavity_mtl.utils import autocast_context, move_to_device, save_json


def make_aug(img_size):
    if A is None:
        return None
    return A.Compose([A.Resize(img_size, img_size)])


def split_rows(rows, fold, n_splits, seed):
    y = np.array([int(r['label']) for r in rows])
    if StratifiedKFold is None:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(rows))
        train_idx, val_idx = [], []
        for lab in [0, 1]:
            lab_idx = idx[y == lab]
            rng.shuffle(lab_idx)
            chunks = np.array_split(lab_idx, n_splits)
            val_idx.extend(chunks[fold].tolist())
            train_idx.extend(np.setdiff1d(lab_idx, chunks[fold]).tolist())
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = list(skf.split(np.zeros(len(rows)), y))[fold]
    return [rows[i] for i in train_idx], [rows[i] for i in val_idx]


@torch.no_grad()
def collect_predictions(model, loader, device, amp):
    model.eval()
    probs, labels, gt_masks, pred_masks, image_ids = [], [], [], [], []
    for batch in tqdm(loader, desc='collect'):
        bd = move_to_device(batch, device)
        with autocast_context(amp):
            out = model(bd['image'])
        probs.extend(torch.sigmoid(out['cls_logit']).cpu().numpy().tolist())
        labels.extend(bd['label'].cpu().numpy().astype(int).tolist())
        gt_masks.extend(bd['mask'].cpu().numpy()[:, 0].astype(np.uint8))
        pred_masks.extend(torch.sigmoid(out['mask_logit']).cpu().numpy()[:, 0].astype(np.float32))
        image_ids.extend(batch['image_id'])
    return np.asarray(labels, dtype=int), gt_masks, np.asarray(probs, dtype=np.float32), pred_masks, image_ids


def main():
    ap = argparse.ArgumentParser(description='Re-optimize CavityMTL post-processing thresholds on validation fold')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--masks_dir', required=True)
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--n_splits', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--img_size', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--recall_constraint', type=float, default=0.90)
    ap.add_argument('--keep_largest', action='store_true')
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    train_args = ckpt.get('args', {})
    fold = args.fold if args.fold is not None else int(train_args.get('fold', 0))
    n_splits = args.n_splits if args.n_splits is not None else int(train_args.get('n_splits', 5))
    seed = args.seed if args.seed is not None else int(train_args.get('seed', 42))
    img_size = args.img_size if args.img_size is not None else int(train_args.get('img_size', 512))

    rows = build_image_mask_table(args.images_dir, args.masks_dir)
    _, val_rows = split_rows(rows, fold, n_splits, seed)
    ds = CavityDataset(val_rows, img_size=img_size, augment=make_aug(img_size), in_chans=int(ckpt.get('model_config', {}).get('in_chans', 1)), has_masks=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CavityMTLNet(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    y, gt_masks, probs, pred_masks, image_ids = collect_predictions(model, loader, device, args.amp)
    sweep = threshold_sweep(y, gt_masks, probs, pred_masks, recall_constraint=args.recall_constraint, keep_largest=args.keep_largest)

    out_json = Path(args.out_json) if args.out_json else Path(args.ckpt).with_name('reoptimized_thresholds.json')
    save_json({'best': sweep['best'], 'n_val': len(y), 'n_pos': int(y.sum()), 'n_neg': int((1-y).sum())}, out_json)
    print('Best thresholds:', sweep['best'])
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    main()
