from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .losses import soft_dice_score


@dataclass
class HardPools:
    hard_pos: list[int]
    hard_neg: list[int]
    quarantine_pos: list[int]
    quarantine_neg: list[int]


class HardExampleCache:
    """Online EMA cache for hard-example mining without every-epoch full reevaluation."""
    def __init__(self, labels: Sequence[int], momentum: float = 0.9):
        self.labels = np.asarray(labels).astype(int)
        n = len(self.labels)
        self.momentum = float(momentum)
        self.score = np.zeros(n, dtype=np.float32)
        self.cls_loss = np.zeros(n, dtype=np.float32)
        self.seg_loss = np.zeros(n, dtype=np.float32)
        self.p_cavity = np.zeros(n, dtype=np.float32)
        self.mask_area = np.zeros(n, dtype=np.float32)
        self.mask_topk = np.zeros(n, dtype=np.float32)
        self.dice_proxy = np.zeros(n, dtype=np.float32)
        self.seen = np.zeros(n, dtype=np.int32)

    @torch.no_grad()
    def update_from_batch(self, indices, labels, masks, outputs):
        idx = torch.as_tensor(indices).detach().cpu().numpy().astype(int)
        y = labels.detach().float().view(-1)
        logits = outputs['cls_logit'].detach().float().view(-1)
        mask_logit = outputs['mask_logit'].detach().float()
        masks = masks.detach().float()

        p = torch.sigmoid(logits)
        cls_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        mp = torch.sigmoid(mask_logit)
        flat = mp.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        topk = flat.topk(k, dim=1).values.mean(dim=1)
        area = flat.mean(dim=1)

        dice = soft_dice_score(mask_logit, masks)
        # Seg proxy: for positives 1-dice + BCE; for negatives top-k + area.
        bce_pix = F.binary_cross_entropy_with_logits(mask_logit, masks, reduction='none').flatten(1).mean(1)
        seg_proxy = torch.where(y > 0.5, (1 - dice) + bce_pix, topk + area)

        small_bonus = torch.zeros_like(y)
        if (y > 0.5).any():
            gt_area = masks.flatten(1).mean(1)
            small_bonus = torch.where(y > 0.5, (1.0 / (gt_area.clamp_min(1e-5))).log().clamp(max=6) / 6.0, small_bonus)

        h_pos = 0.30 * cls_loss + 0.35 * seg_proxy + 0.20 * (1 - dice) + 0.15 * (1 - p) + 0.05 * small_bonus
        h_neg = 0.35 * cls_loss + 0.30 * p + 0.20 * topk + 0.15 * area
        h = torch.where(y > 0.5, h_pos, h_neg)

        values = {
            'score': h.cpu().numpy(),
            'cls_loss': cls_loss.cpu().numpy(),
            'seg_loss': seg_proxy.cpu().numpy(),
            'p_cavity': p.cpu().numpy(),
            'mask_area': area.cpu().numpy(),
            'mask_topk': topk.cpu().numpy(),
            'dice_proxy': dice.cpu().numpy(),
        }
        m = self.momentum
        for j, sample_idx in enumerate(idx):
            if self.seen[sample_idx] == 0:
                for name, arr in values.items():
                    getattr(self, name)[sample_idx] = arr[j]
            else:
                for name, arr in values.items():
                    buf = getattr(self, name)
                    buf[sample_idx] = m * buf[sample_idx] + (1 - m) * arr[j]
            self.seen[sample_idx] += 1

    def build_pools(
        self,
        pos_low_q: float = 0.70,
        pos_high_q: float = 0.95,
        neg_low_q: float = 0.80,
        neg_high_q: float = 0.97,
        trim_top_percent: float = 3.0,
        min_seen: int = 1,
    ) -> HardPools:
        hard_pos, hard_neg, q_pos, q_neg = [], [], [], []
        for label, low_q, high_q in [(1, pos_low_q, pos_high_q), (0, neg_low_q, neg_high_q)]:
            candidates = np.where((self.labels == label) & (self.seen >= min_seen))[0]
            if len(candidates) == 0:
                continue
            scores = self.score[candidates]
            lo = np.quantile(scores, low_q)
            hi = np.quantile(scores, high_q)
            # independent trim for the most extreme noise-like examples
            trim_hi = np.quantile(scores, max(0.0, 1.0 - trim_top_percent / 100.0))
            hard = candidates[(scores >= lo) & (scores <= min(hi, trim_hi))].tolist()
            quarantine = candidates[scores > trim_hi].tolist()
            if label == 1:
                hard_pos, q_pos = hard, quarantine
            else:
                hard_neg, q_neg = hard, quarantine
        return HardPools(hard_pos=list(map(int, hard_pos)), hard_neg=list(map(int, hard_neg)), quarantine_pos=list(map(int, q_pos)), quarantine_neg=list(map(int, q_neg)))
