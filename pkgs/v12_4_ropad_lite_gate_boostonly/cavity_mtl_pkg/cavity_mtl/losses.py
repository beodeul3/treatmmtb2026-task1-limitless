from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | None = None,
    gamma: float = 2.0,
    reduction: str = 'mean',
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    loss = bce * (1 - pt).pow(gamma)
    if alpha is not None:
        at = alpha * targets + (1 - alpha) * (1 - targets)
        loss = at * loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def soft_dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    den = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (2 * inter + eps) / (den + eps)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 1.0 - soft_dice_score(logits, targets).mean()


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1 - targets)).sum(dim=dims)
    fn = ((1 - probs) * targets).sum(dim=dims)
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - t.mean()


def mask_existence_prob(mask_logit: torch.Tensor, topk_frac: float = 0.01) -> torch.Tensor:
    probs = torch.sigmoid(mask_logit).flatten(1)
    k = max(1, int(probs.shape[1] * topk_frac))
    topk_mean = probs.topk(k, dim=1).values.mean(dim=1)
    return topk_mean.clamp(1e-5, 1 - 1e-5)



def classification_tail_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk_frac: float = 0.35,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-aware classification losses for hard positive/negative samples.

    * hard positives: positive samples with the lowest logits
    * hard negatives: negative samples with the highest logits
    Returns: hard-tail BCE loss, pairwise margin ranking loss.
    """
    labels = labels.float().view(-1)
    logits = logits.view(-1)
    pos_logits = logits[labels > 0.5]
    neg_logits = logits[labels <= 0.5]
    zero = logits.sum() * 0.0
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return zero, zero

    kp = max(1, int(round(pos_logits.numel() * float(topk_frac))))
    kn = max(1, int(round(neg_logits.numel() * float(topk_frac))))
    hard_pos = torch.topk(pos_logits, k=min(kp, pos_logits.numel()), largest=False).values
    hard_neg = torch.topk(neg_logits, k=min(kn, neg_logits.numel()), largest=True).values

    hard_logits = torch.cat([hard_pos, hard_neg], dim=0)
    hard_targets = torch.cat([torch.ones_like(hard_pos), torch.zeros_like(hard_neg)], dim=0)
    hard_bce = F.binary_cross_entropy_with_logits(hard_logits, hard_targets)

    # Enforce z_pos > z_neg + margin for the hardest tail samples.
    margin_loss = F.relu(float(margin) - hard_pos[:, None] + hard_neg[None, :]).mean()
    return hard_bce, margin_loss


class HomoscedasticMTLLoss(nn.Module):
    """Uncertainty-weighted seg/cls loss with safe delayed consistency."""
    def __init__(
        self,
        neg_seg_beta: float = 0.10,
        cls_pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        use_tversky: bool = True,
        consistency_weight: float = 0.03,
        consistency_warmup_epochs: int = 20,
        seg_warmup_epochs: int = 8,
        seg_weight_warmup: float = 2.0,
        cls_weight_warmup: float = 0.3,
        cls_tail_weight: float = 0.0,
        cls_margin_weight: float = 0.0,
        cls_margin: float = 1.0,
        cls_tail_topk_frac: float = 0.35,
    ):
        super().__init__()
        self.neg_seg_beta = neg_seg_beta
        self.cls_pos_weight_value = cls_pos_weight
        self.focal_gamma = focal_gamma
        self.use_tversky = use_tversky
        self.consistency_weight = consistency_weight
        self.consistency_warmup_epochs = consistency_warmup_epochs
        self.seg_warmup_epochs = int(seg_warmup_epochs)
        self.seg_weight_warmup = float(seg_weight_warmup)
        self.cls_weight_warmup = float(cls_weight_warmup)
        self.cls_tail_weight = float(cls_tail_weight)
        self.cls_margin_weight = float(cls_margin_weight)
        self.cls_margin = float(cls_margin)
        self.cls_tail_topk_frac = float(cls_tail_topk_frac)
        self.log_var_seg = nn.Parameter(torch.zeros(()))
        self.log_var_cls = nn.Parameter(torch.zeros(()))

    def forward(self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor, labels: torch.Tensor, epoch: int = 0) -> Dict[str, torch.Tensor]:
        labels = labels.float().view(-1)
        mask_logit = outputs['mask_logit']
        cls_logit = outputs['cls_logit']
        cls_logit_global = outputs.get('cls_logit_global', None)

        pos = labels > 0.5
        neg = ~pos

        if pos.any():
            pos_logits = mask_logit[pos]
            pos_masks = masks[pos]
            seg_pos = dice_loss(pos_logits, pos_masks)
            if self.use_tversky:
                seg_pos = 0.5 * seg_pos + 0.5 * tversky_loss(pos_logits, pos_masks)
            seg_pos = seg_pos + 0.5 * focal_bce_with_logits(pos_logits, pos_masks, gamma=self.focal_gamma)
        else:
            seg_pos = mask_logit.sum() * 0.0

        if neg.any():
            neg_logits = mask_logit[neg]
            zeros = torch.zeros_like(neg_logits)
            seg_neg = self.neg_seg_beta * focal_bce_with_logits(neg_logits, zeros, gamma=self.focal_gamma)
        else:
            seg_neg = mask_logit.sum() * 0.0

        seg_loss = seg_pos + seg_neg

        pw = torch.tensor(self.cls_pos_weight_value, device=labels.device, dtype=labels.dtype)
        cls_loss = focal_bce_with_logits(cls_logit, labels, gamma=self.focal_gamma, pos_weight=pw)
        if cls_logit_global is not None:
            cls_loss = cls_loss + 0.3 * focal_bce_with_logits(cls_logit_global, labels, gamma=self.focal_gamma, pos_weight=pw)

        if self.cls_tail_weight > 0 or self.cls_margin_weight > 0:
            tail_bce, margin_loss = classification_tail_losses(
                cls_logit, labels, topk_frac=self.cls_tail_topk_frac, margin=self.cls_margin
            )
            cls_loss = cls_loss + self.cls_tail_weight * tail_bce + self.cls_margin_weight * margin_loss
        else:
            tail_bce = cls_logit.sum() * 0.0
            margin_loss = cls_logit.sum() * 0.0

        # Safe consistency: off during warm-up; one-way target with detach afterward.
        if epoch >= self.consistency_warmup_epochs and self.consistency_weight > 0:
            p_mask = mask_existence_prob(mask_logit, topk_frac=0.01)
            p_cls = torch.sigmoid(cls_logit).detach()
            # BCE on probabilities is not autocast-safe in PyTorch AMP.
            # Consistency is a tiny auxiliary term, so compute it explicitly in fp32.
            if p_mask.is_cuda:
                with torch.cuda.amp.autocast(enabled=False):
                    cons_gt = F.binary_cross_entropy(p_mask.float(), labels.float())
                    cons_branch = F.binary_cross_entropy(p_mask.float(), p_cls.float())
            else:
                cons_gt = F.binary_cross_entropy(p_mask.float(), labels.float())
                cons_branch = F.binary_cross_entropy(p_mask.float(), p_cls.float())
            cons_loss = cons_gt + 0.25 * cons_branch
        else:
            cons_loss = mask_logit.sum() * 0.0

        # During the first few epochs, do not let the uncertainty weights or the
        # classification branch dominate. The first goal is to make the decoder
        # produce non-empty masks on positive images. After warm-up, switch to
        # homoscedastic uncertainty weighting.
        if epoch < self.seg_warmup_epochs:
            total = self.seg_weight_warmup * seg_loss + self.cls_weight_warmup * cls_loss
        else:
            total = torch.exp(-self.log_var_seg) * seg_loss + self.log_var_seg
            total = total + torch.exp(-self.log_var_cls) * cls_loss + self.log_var_cls
        total = total + self.consistency_weight * cons_loss

        return {
            'total': total,
            'seg_loss': seg_loss.detach(),
            'seg_pos': seg_pos.detach(),
            'seg_neg': seg_neg.detach(),
            'cls_loss': cls_loss.detach(),
            'cls_tail_bce': tail_bce.detach(),
            'cls_margin': margin_loss.detach(),
            'cons_loss': cons_loss.detach(),
            'log_var_seg': self.log_var_seg.detach(),
            'log_var_cls': self.log_var_cls.detach(),
        }
