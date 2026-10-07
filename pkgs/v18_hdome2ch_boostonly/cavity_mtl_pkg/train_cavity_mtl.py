from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import albumentations as A
except Exception:  # pragma: no cover
    A = None

try:
    from sklearn.model_selection import StratifiedKFold
except Exception:  # pragma: no cover
    StratifiedKFold = None

from cavity_mtl.dataset import CavityDataset, build_image_mask_table
from cavity_mtl.ema import ModelEMA
from cavity_mtl.hard_cache import HardExampleCache
from cavity_mtl.losses import HomoscedasticMTLLoss
from cavity_mtl.metrics import compute_binary_metrics, threshold_sweep
from cavity_mtl.model import CavityMTLNet
from cavity_mtl.sampler import BalancedCavityBatchSampler
from cavity_mtl.utils import autocast_context, move_to_device, save_json, seed_everything


def make_train_aug(img_size: int):
    if A is None:
        return None
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.45),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.35),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
    ])


def make_val_aug(img_size: int):
    if A is None:
        return None
    return A.Compose([A.Resize(img_size, img_size)])


def split_rows(rows: List[Dict], fold: int, n_splits: int, seed: int):
    y = np.array([int(r['label']) for r in rows])
    if StratifiedKFold is None:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(rows))
        train_idx, val_idx = [], []
        for lab in [0, 1]:
            lab_idx = idx[y == lab]
            rng.shuffle(lab_idx)
            chunks = np.array_split(lab_idx, n_splits)
            val_part = chunks[fold]
            train_part = np.setdiff1d(lab_idx, val_part)
            train_idx.extend(train_part.tolist())
            val_idx.extend(val_part.tolist())
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(rows)), y))
        train_idx, val_idx = splits[fold]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    return train_rows, val_rows


def stage_schedule(
    epoch: int,
    epochs: int,
    batch_size: int,
    hard_warmup: int,
    late_pos2_start: float = 0.85,
    mid_hard_ratio: float = 0.25,
    late_hard_ratio: float = 0.28,
):
    """Curriculum sampler schedule.

    v7 change: delay/soften the late pos/b=2 phase because the previous
    transition at 70% improved specificity but caused recall collapse.
    """
    progress = epoch / max(1, epochs)
    if progress < 0.30:
        pos_per_batch = max(1, batch_size // 2)          # 1:1 warm localization
    elif progress < late_pos2_start:
        pos_per_batch = max(1, round(batch_size / 3))    # balanced-ish 1:2
    else:
        pos_per_batch = max(1, round(batch_size / 4))    # short late specificity tuning

    if epoch < hard_warmup:
        hard_ratio = 0.0
    elif progress < 0.30:
        hard_ratio = 0.20
    elif progress < late_pos2_start:
        hard_ratio = mid_hard_ratio
    else:
        hard_ratio = late_hard_ratio
    return int(pos_per_batch), float(hard_ratio)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, recall_constraint: float, keep_largest: bool):
    model.eval()
    probs, labels = [], []
    gt_masks, pred_masks = [], []
    image_ids = []
    for batch in tqdm(loader, desc='valid', leave=False):
        batch = move_to_device(batch, device)
        with autocast_context(amp):
            out = model(batch['image'])
        p = torch.sigmoid(out['cls_logit']).detach().cpu().numpy()
        mp = torch.sigmoid(out['mask_logit']).detach().cpu().numpy()[:, 0]
        y = batch['label'].detach().cpu().numpy().astype(int)
        m = batch['mask'].detach().cpu().numpy()[:, 0].astype(np.uint8)
        probs.extend(p.tolist())
        labels.extend(y.tolist())
        pred_masks.extend([x.astype(np.float32) for x in mp])
        gt_masks.extend([x.astype(np.uint8) for x in m])
        image_ids.extend(batch['image_id'])

    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    raw_cls = compute_binary_metrics(labels, probs, threshold=0.5)
    sweep = threshold_sweep(
        labels, gt_masks, probs, pred_masks,
        recall_constraint=recall_constraint,
        keep_largest=keep_largest,
    )
    best = sweep['best']

    # Diagnostics: whether masks are merely low-confidence or truly collapsed.
    mp_max = np.asarray([float(np.max(x)) for x in pred_masks], dtype=np.float32)
    mp_mean = np.asarray([float(np.mean(x)) for x in pred_masks], dtype=np.float32)
    mp_top1 = []
    gt_area = []
    for x, g in zip(pred_masks, gt_masks):
        flat = x.reshape(-1)
        k = max(1, int(flat.size * 0.01))
        mp_top1.append(float(np.partition(flat, -k)[-k:].mean()))
        gt_area.append(float(g.mean()))
    mp_top1 = np.asarray(mp_top1, dtype=np.float32)
    gt_area = np.asarray(gt_area, dtype=np.float32)
    pos = labels == 1
    neg = labels == 0
    diagnostics = {
        'n_pos': int(pos.sum()),
        'n_neg': int(neg.sum()),
        'p_pos_mean': float(probs[pos].mean()) if pos.any() else None,
        'p_neg_mean': float(probs[neg].mean()) if neg.any() else None,
        'p_pos_minmax': [float(probs[pos].min()), float(probs[pos].max())] if pos.any() else None,
        'p_neg_minmax': [float(probs[neg].min()), float(probs[neg].max())] if neg.any() else None,
        'mask_max_pos_mean': float(mp_max[pos].mean()) if pos.any() else None,
        'mask_max_neg_mean': float(mp_max[neg].mean()) if neg.any() else None,
        'mask_top1_pos_mean': float(mp_top1[pos].mean()) if pos.any() else None,
        'mask_top1_neg_mean': float(mp_top1[neg].mean()) if neg.any() else None,
        'mask_mean_pos_mean': float(mp_mean[pos].mean()) if pos.any() else None,
        'mask_mean_neg_mean': float(mp_mean[neg].mean()) if neg.any() else None,
        'gt_area_pos_mean': float(gt_area[pos].mean()) if pos.any() else None,
        'gt_area_pos_minmax': [float(gt_area[pos].min()), float(gt_area[pos].max())] if pos.any() else None,
    }
    return {
        'raw_cls@0.5': raw_cls,
        'best': best,
        'best_official': sweep.get('best_official'),
        'best_official_recall90': sweep.get('best_official_recall90'),
        'best_official_recall95': sweep.get('best_official_recall95'),
        'best_legacy': sweep.get('best_legacy'),
        'threshold_rows': sweep.get('rows', []),
        'diagnostics': diagnostics,
        'probs': probs,
        'labels': labels,
        'image_ids': image_ids,
    }


@torch.no_grad()
def full_audit_refresh(model, loader, device, amp: bool, hard_cache: HardExampleCache):
    model.eval()
    for batch in tqdm(loader, desc='full-audit', leave=False):
        batch_dev = move_to_device(batch, device)
        with autocast_context(amp):
            out = model(batch_dev['image'])
        hard_cache.update_from_batch(batch['idx'], batch_dev['label'], batch_dev['mask'], out)


def save_val_predictions(path: Path, image_ids, labels, probs):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'label', 'cavity_prob'])
        for iid, y, p in zip(image_ids, labels, probs):
            w.writerow([iid, int(y), float(p)])


def save_threshold_rows(path: Path, rows, top_k: int = 200):
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r.get('official_score', -1e9), reverse=True)[:top_k]
    keys = list(rows_sorted[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)


def _candidate_value(candidate, key, default=float('nan')):
    if candidate is None:
        return default
    return candidate.get(key, default)


def _fmt_candidate(name: str, candidate) -> str:
    if candidate is None:
        return f'{name}: None'
    return (
        f"{name}: "
        f"score={candidate.get('official_score', float('nan')):.5f} "
        f"mean_dice={candidate.get('official_mean_dice', float('nan')):.5f} "
        f"acc={candidate.get('acc', float('nan')):.4f} "
        f"recall={candidate.get('recall', float('nan')):.4f} "
        f"spec={candidate.get('specificity', float('nan')):.4f} "
        f"fn={candidate.get('fn', 'NA')} fp={candidate.get('fp', 'NA')} "
        f"t_cls={candidate.get('t_cls', float('nan')):.3f} "
        f"t_mask={candidate.get('t_mask', float('nan')):.3f} "
        f"min_area={candidate.get('min_area', 'NA')} "
        f"dice_count={candidate.get('official_dice_count', 'NA')}"
    )


def _candidate_columns(prefix: str, candidate) -> Dict[str, object]:
    keys = [
        'official_score', 'official_mean_dice', 'official_dice_count',
        'acc', 'recall', 'specificity', 'precision', 'fn', 'fp',
        't_cls', 't_mask', 'min_area',
        'pred_nonempty', 'pred_nonempty_pos', 'pred_nonempty_neg',
        'dsc_pos', 'dsc_all_legacy', 'legacy_objective',
    ]
    out = {}
    for k in keys:
        out[f'{prefix}_{k}'] = _candidate_value(candidate, k, '')
    return out


def make_epoch_history_row(epoch: int, record: Dict) -> Dict[str, object]:
    diag = record.get('diagnostics', {}) or {}
    row = {
        'epoch': epoch + 1,
        'train_loss_mean': record.get('train_loss_mean'),
        'lr': record.get('lr'),
        'log_var_seg': record.get('log_var_seg'),
        'log_var_cls': record.get('log_var_cls'),
        'raw_acc@0.5': (record.get('raw_cls@0.5') or {}).get('acc', ''),
        'raw_recall@0.5': (record.get('raw_cls@0.5') or {}).get('recall', ''),
        'raw_specificity@0.5': (record.get('raw_cls@0.5') or {}).get('specificity', ''),
        'p_pos_mean': diag.get('p_pos_mean', ''),
        'p_neg_mean': diag.get('p_neg_mean', ''),
        'p_pos_min': (diag.get('p_pos_minmax') or ['', ''])[0],
        'p_pos_max': (diag.get('p_pos_minmax') or ['', ''])[1],
        'p_neg_min': (diag.get('p_neg_minmax') or ['', ''])[0],
        'p_neg_max': (diag.get('p_neg_minmax') or ['', ''])[1],
        'mask_max_pos_mean': diag.get('mask_max_pos_mean', ''),
        'mask_max_neg_mean': diag.get('mask_max_neg_mean', ''),
        'mask_top1_pos_mean': diag.get('mask_top1_pos_mean', ''),
        'mask_top1_neg_mean': diag.get('mask_top1_neg_mean', ''),
    }
    row.update(_candidate_columns('best_official', record.get('best_official')))
    row.update(_candidate_columns('best_official_recall90', record.get('best_official_recall90')))
    row.update(_candidate_columns('best_official_recall95', record.get('best_official_recall95')))
    row.update(_candidate_columns('best_legacy', record.get('best_legacy')))
    return row


def append_epoch_history(path: Path, row: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = list(row.keys())
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Train bidirectional multi-task cavity detection/segmentation model')
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--masks_dir', required=True)
    parser.add_argument('--out_dir', default='runs/cavity_mtl')
    parser.add_argument('--model', default='convnext_tiny.fb_in22k_ft_in1k')
    parser.add_argument('--in_chans', type=int, default=1, choices=[1, 2, 3], help='Input channels. 1=minmax CXR, 2=[minmax,residual], 3=optional baseline-style [minmax,CLAHE,minmax].')
    parser.add_argument('--residual_mode', default='abs', choices=['abs', 'dark', 'bright'], help='Used only when --in_chans 2. abs=|X-blur|, dark=ReLU(blur-X), bright=ReLU(X-blur).')
    parser.add_argument('--residual_sigma', type=float, default=5.0, help='Gaussian sigma for residual channel before resize/pad. Try 5.0 first; 3.0/7.0 are ablations.')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--decoder_ch', type=int, default=160)
    parser.add_argument('--cls_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--neg_seg_beta', type=float, default=0.05)
    parser.add_argument('--cls_pos_weight', type=float, default=1.0)
    parser.add_argument('--seg_warmup_epochs', type=int, default=8)
    parser.add_argument('--seg_weight_warmup', type=float, default=2.0)
    parser.add_argument('--cls_weight_warmup', type=float, default=0.3)
    parser.add_argument('--consistency_weight', type=float, default=0.03)
    parser.add_argument('--consistency_warmup_epochs', type=int, default=30)
    parser.add_argument('--hard_warmup_epochs', type=int, default=10)
    parser.add_argument('--hard_cache_momentum', type=float, default=0.90)
    parser.add_argument('--audit_every', type=int, default=5)
    parser.add_argument('--recall_constraint', type=float, default=0.90)
    parser.add_argument('--late_pos2_start', type=float, default=0.85, help='Epoch fraction to switch to pos/b≈2. v7 default delays this to protect recall.')
    parser.add_argument('--mid_hard_ratio', type=float, default=0.25)
    parser.add_argument('--late_hard_ratio', type=float, default=0.28)
    parser.add_argument('--cls_tail_weight', type=float, default=0.0, help='Extra BCE on low-logit positives and high-logit negatives.')
    parser.add_argument('--cls_margin_weight', type=float, default=0.0, help='Pairwise margin loss weight for hard positive/negative tails.')
    parser.add_argument('--cls_margin', type=float, default=1.0)
    parser.add_argument('--cls_tail_topk_frac', type=float, default=0.35)
    parser.add_argument('--keep_largest', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--resume_ckpt', default=None, help='Optional checkpoint path to resume model/criterion weights. Optimizer is restarted.')
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / 'args.json')

    rows = build_image_mask_table(args.images_dir, args.masks_dir)
    y_all = np.array([int(r['label']) for r in rows])
    print(f'Total samples={len(rows)} positive={int(y_all.sum())} negative={int((1-y_all).sum())}')
    train_rows, val_rows = split_rows(rows, args.fold, args.n_splits, args.seed)
    print(f'Train={len(train_rows)} Val={len(val_rows)}')

    train_ds = CavityDataset(train_rows, img_size=args.img_size, augment=make_train_aug(args.img_size), in_chans=args.in_chans, has_masks=True, residual_mode=args.residual_mode, residual_sigma=args.residual_sigma)
    train_audit_ds = CavityDataset(train_rows, img_size=args.img_size, augment=make_val_aug(args.img_size), in_chans=args.in_chans, has_masks=True, residual_mode=args.residual_mode, residual_sigma=args.residual_sigma)
    val_ds = CavityDataset(val_rows, img_size=args.img_size, augment=make_val_aug(args.img_size), in_chans=args.in_chans, has_masks=True, residual_mode=args.residual_mode, residual_sigma=args.residual_sigma)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    sampler = BalancedCavityBatchSampler(train_ds.labels, batch_size=args.batch_size, steps_per_epoch=steps_per_epoch, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    audit_loader = DataLoader(train_audit_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CavityMTLNet(
        backbone=args.model,
        pretrained=not args.no_pretrained,
        in_chans=args.in_chans,
        decoder_ch=args.decoder_ch,
        cls_dim=args.cls_dim,
        dropout=args.dropout,
        mask_to_cls_stopgrad=True,
    ).to(device)

    criterion = HomoscedasticMTLLoss(
        neg_seg_beta=args.neg_seg_beta,
        cls_pos_weight=args.cls_pos_weight,
        consistency_weight=args.consistency_weight,
        consistency_warmup_epochs=args.consistency_warmup_epochs,
        seg_warmup_epochs=args.seg_warmup_epochs,
        seg_weight_warmup=args.seg_weight_warmup,
        cls_weight_warmup=args.cls_weight_warmup,
        cls_tail_weight=args.cls_tail_weight,
        cls_margin_weight=args.cls_margin_weight,
        cls_margin=args.cls_margin,
        cls_tail_topk_frac=args.cls_tail_topk_frac,
    ).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    hard_cache = HardExampleCache(train_ds.labels, momentum=args.hard_cache_momentum)

    best_scores = {
        'official': -1e9,
        'official_recall90': -1e9,
        'official_recall95': -1e9,
        'legacy': -1e9,
    }
    best_record = None
    start_epoch = 0
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        print(f'Resuming model weights from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        if ema is not None:
            ema.module.load_state_dict(model.state_dict(), strict=False)
        if 'criterion' in ckpt:
            try:
                criterion.load_state_dict(ckpt['criterion'], strict=False)
            except TypeError:
                criterion.load_state_dict(ckpt['criterion'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        print(f'Resume start_epoch={start_epoch + 1}/{args.epochs}. Optimizer/scheduler/hard-cache are restarted. v7 best-score trackers start fresh.')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        pos_per_batch, hard_ratio = stage_schedule(
            epoch, args.epochs, args.batch_size, args.hard_warmup_epochs,
            late_pos2_start=args.late_pos2_start,
            mid_hard_ratio=args.mid_hard_ratio,
            late_hard_ratio=args.late_hard_ratio,
        )
        sampler.set_schedule(pos_per_batch=pos_per_batch, hard_ratio=hard_ratio)
        pbar = tqdm(train_loader, desc=f'epoch {epoch+1}/{args.epochs} pos/b={pos_per_batch} hard={hard_ratio:.2f}')
        running = []
        for batch in pbar:
            batch_dev = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.amp):
                out = model(batch_dev['image'])
                loss_dict = criterion(out, batch_dev['mask'], batch_dev['label'], epoch=epoch)
                loss = loss_dict['total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            with torch.no_grad():
                hard_cache.update_from_batch(batch['idx'], batch_dev['label'], batch_dev['mask'], out)
            running.append(float(loss.detach().cpu()))
            pbar.set_postfix(
                loss=np.mean(running[-20:]),
                seg=float(loss_dict['seg_loss']),
                cls=float(loss_dict['cls_loss']),
                tail=float(loss_dict.get('cls_tail_bce', 0.0)),
                margin=float(loss_dict.get('cls_margin', 0.0)),
            )

        scheduler.step()

        if epoch >= args.hard_warmup_epochs:
            pools = hard_cache.build_pools(trim_top_percent=3.0)
            sampler.update_hard_pools(pools.hard_pos, pools.hard_neg)
            save_json({
                'epoch': epoch,
                'hard_pos': len(pools.hard_pos),
                'hard_neg': len(pools.hard_neg),
                'quarantine_pos': len(pools.quarantine_pos),
                'quarantine_neg': len(pools.quarantine_neg),
            }, out_dir / 'hard_pool_latest.json')

        # Low-frequency full audit corrects stale online cache. It uses EMA model if available.
        if (epoch >= args.hard_warmup_epochs) and (args.audit_every > 0) and ((epoch + 1) % args.audit_every == 0):
            audit_model = ema.module if ema is not None else model
            full_audit_refresh(audit_model, audit_loader, device, args.amp, hard_cache)
            pools = hard_cache.build_pools(trim_top_percent=3.0)
            sampler.update_hard_pools(pools.hard_pos, pools.hard_neg)

        eval_model = ema.module if ema is not None else model
        val = evaluate(eval_model, val_loader, device, args.amp, args.recall_constraint, args.keep_largest)
        best = val['best']
        print('VAL official summary:')
        print('  ' + _fmt_candidate('best_official', val.get('best_official')))
        print('  ' + _fmt_candidate('recall>=0.90', val.get('best_official_recall90')))
        print('  ' + _fmt_candidate('recall>=0.95', val.get('best_official_recall95')))
        print('  legacy:', val.get('best_legacy'))
        print('VAL diag:', val.get('diagnostics', {}))
        save_val_predictions(out_dir / 'val_predictions_latest.csv', val['image_ids'], val['labels'], val['probs'])
        save_threshold_rows(out_dir / 'val_threshold_sweep_top200_latest.csv', val.get('threshold_rows', []), top_k=200)

        record = {
            'epoch': epoch,
            'train_loss_mean': float(np.mean(running)) if running else None,
            'raw_cls@0.5': val['raw_cls@0.5'],
            'diagnostics': val.get('diagnostics', {}),
            'best_thresholds': best,
            'best_official': val.get('best_official'),
            'best_official_recall90': val.get('best_official_recall90'),
            'best_official_recall95': val.get('best_official_recall95'),
            'best_legacy': val.get('best_legacy'),
            'lr': float(scheduler.get_last_lr()[0]),
            'log_var_seg': float(criterion.log_var_seg.detach().cpu()),
            'log_var_cls': float(criterion.log_var_cls.detach().cpu()),
        }
        save_json(record, out_dir / 'last_metrics.json')
        history_row = make_epoch_history_row(epoch, record)
        append_epoch_history(out_dir / 'metrics_history.csv', history_row)
        append_jsonl(out_dir / 'metrics_history.jsonl', record)

        def make_ckpt(candidate, tag, score_value):
            return {
                'model': eval_model.state_dict(),
                'criterion': criterion.state_dict(),
                'args': vars(args),
                'model_config': {
                    'backbone': args.model,
                    'pretrained': False,
                    'in_chans': args.in_chans,
                    'decoder_ch': args.decoder_ch,
                    'cls_dim': args.cls_dim,
                    'dropout': args.dropout,
                    'mask_to_cls_stopgrad': True,
                },
                'preprocess_config': {
                    'in_chans': args.in_chans,
                    'residual_mode': args.residual_mode,
                    'residual_sigma': args.residual_sigma,
                },
                'thresholds': candidate,
                'epoch': epoch,
                'score_tag': tag,
                'objective': float(score_value),
                'best_scores': best_scores,
            }

        def maybe_save_candidate(candidate, tag, score_key, ckpt_name, metrics_name, preds_name=None, alias_best=False):
            if candidate is None:
                return
            score_value = float(candidate.get(score_key, -1e9))
            if score_value > best_scores[tag]:
                best_scores[tag] = score_value
                ckpt = make_ckpt(candidate, tag, score_value)
                torch.save(ckpt, out_dir / ckpt_name)
                save_json(record, out_dir / metrics_name)
                if preds_name is not None:
                    save_val_predictions(out_dir / preds_name, val['image_ids'], val['labels'], val['probs'])
                if alias_best:
                    torch.save(ckpt, out_dir / 'best_cavity_mtl.pt')
                    save_json(record, out_dir / 'best_metrics.json')
                    save_val_predictions(out_dir / 'val_predictions_best.csv', val['image_ids'], val['labels'], val['probs'])
                print(f'New best {tag}: {score_value:.5f} saved to {ckpt_name}')

        maybe_save_candidate(
            val.get('best_official'), 'official', 'official_score',
            'best_official_score.pt', 'best_official_metrics.json',
            'val_predictions_best_official.csv', alias_best=True,
        )
        maybe_save_candidate(
            val.get('best_official_recall90'), 'official_recall90', 'official_score',
            'best_official_recall90.pt', 'best_official_recall90_metrics.json',
            'val_predictions_best_official_recall90.csv',
        )
        maybe_save_candidate(
            val.get('best_official_recall95'), 'official_recall95', 'official_score',
            'best_official_recall95.pt', 'best_official_recall95_metrics.json',
            'val_predictions_best_official_recall95.csv',
        )
        maybe_save_candidate(
            val.get('best_legacy'), 'legacy', 'legacy_objective',
            'best_legacy_objective.pt', 'best_legacy_metrics.json',
            'val_predictions_best_legacy.csv',
        )

    print('Training finished.')
    print('Best scores:', best_scores)
    print('Last best-style record:', best_record)


if __name__ == '__main__':
    main()
