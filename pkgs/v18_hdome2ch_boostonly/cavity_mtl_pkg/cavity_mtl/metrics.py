from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Legacy diagnostic Dice: both-empty is counted as 1.

    This is useful for debugging but is NOT the official Task 1 mean Dice.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def official_dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Official-style Dice used for Task 1 model selection.

    Matching the provided evaluator behavior:
      * pred empty + gt empty -> NaN, excluded from mean Dice
      * pred non-empty + gt empty -> 0
      * pred empty + gt non-empty -> 0
      * otherwise normal Dice
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    ps = int(pred.sum())
    gs = int(gt.sum())
    if ps == 0 and gs == 0:
        return float('nan')
    if ps == 0 or gs == 0:
        return 0.0
    inter = int(np.logical_and(pred, gt).sum())
    return float((2 * inter + eps) / (ps + gs + eps))


def remove_small_components(mask: np.ndarray, min_area: int = 0, keep_largest: bool = False) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if min_area <= 0 and not keep_largest:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if keep_largest:
        lab = 1 + int(np.argmax(areas))
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 1
    else:
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                out[labels == lab] = 1
    return out


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    recall = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    out = {'acc': acc, 'recall': recall, 'specificity': spec, 'precision': precision, 'f1': f1, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(np.unique(y_true)) > 1:
            out['auroc'] = float(roc_auc_score(y_true, y_prob))
            out['auprc'] = float(average_precision_score(y_true, y_prob))
        else:
            out['auroc'] = float('nan')
            out['auprc'] = float('nan')
    except Exception:
        out['auroc'] = float('nan')
        out['auprc'] = float('nan')
    return out


def _nanmean_or_zero(xs: List[float]) -> tuple[float, int]:
    arr = np.asarray(xs, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return 0.0, 0
    return float(arr[valid].mean()), int(valid.sum())


def _better(row: Dict | None, best: Dict | None, score_key: str) -> bool:
    if row is None:
        return False
    if best is None:
        return True
    # Primary: selected score. Ties: accuracy, dsc_pos, specificity, recall.
    lhs = (row[score_key], row.get('acc', 0.0), row.get('dsc_pos', 0.0), row.get('specificity', 0.0), row.get('recall', 0.0))
    rhs = (best[score_key], best.get('acc', 0.0), best.get('dsc_pos', 0.0), best.get('specificity', 0.0), best.get('recall', 0.0))
    return lhs > rhs


def threshold_sweep(
    y_true: np.ndarray,
    gt_masks: List[np.ndarray],
    probs: np.ndarray,
    mask_probs: List[np.ndarray],
    cls_thresholds=None,
    mask_thresholds=None,
    min_areas=None,
    recall_constraint: float = 0.90,
    keep_largest: bool = False,
    lambda_fn: float = 0.30,
) -> Dict:
    """Threshold sweep with both legacy diagnostics and official-like score.

    Main model-selection score:
      official_score = 0.7 * detection_accuracy + 0.3 * official_mean_dice
    """
    if cls_thresholds is None:
        cls_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 3)
    if mask_thresholds is None:
        mask_thresholds = np.array(
            [0.01, 0.02, 0.03, 0.05, 0.075] + list(np.round(np.arange(0.10, 0.91, 0.05), 3)),
            dtype=np.float32,
        )
    if min_areas is None:
        min_areas = [0, 5, 10, 20, 50, 100, 150, 200]

    y_true = y_true.astype(int)
    n_pos = max(1, int(y_true.sum()))
    best_official = None
    best_official_recall90 = None
    best_official_recall95 = None
    best_recall_constraint = None
    best_legacy = None
    all_rows = []

    for tc in cls_thresholds:
        y_pred = (probs >= tc).astype(int)
        binm = compute_binary_metrics(y_true, probs, threshold=float(tc))
        for tm in mask_thresholds:
            raw_masks = [(mp >= tm).astype(np.uint8) for mp in mask_probs]
            for ma in min_areas:
                legacy_dices = []
                official_dices = []
                dices_pos = []
                pred_nonempty = 0
                pred_nonempty_neg = 0
                pred_nonempty_pos = 0
                for i, rm in enumerate(raw_masks):
                    if y_pred[i] == 0:
                        pm = np.zeros_like(rm, dtype=np.uint8)
                    else:
                        pm = remove_small_components(rm, min_area=int(ma), keep_largest=keep_largest)
                    if pm.sum() > 0:
                        pred_nonempty += 1
                        if y_true[i] == 0:
                            pred_nonempty_neg += 1
                        else:
                            pred_nonempty_pos += 1
                    d_legacy = dice_np(pm, gt_masks[i])
                    d_off = official_dice_np(pm, gt_masks[i])
                    legacy_dices.append(d_legacy)
                    official_dices.append(d_off)
                    if y_true[i] == 1:
                        dices_pos.append(d_legacy)

                dsc_all = float(np.mean(legacy_dices))
                dsc_pos = float(np.mean(dices_pos)) if dices_pos else 0.0
                official_mean_dice, official_dice_count = _nanmean_or_zero(official_dices)
                fn_rate = binm['fn'] / n_pos
                legacy_objective = 0.4 * dsc_all + 0.3 * dsc_pos + 0.3 * binm['acc'] - lambda_fn * fn_rate
                official_score = 0.7 * binm['acc'] + 0.3 * official_mean_dice

                row = {
                    't_cls': float(tc), 't_mask': float(tm), 'min_area': int(ma),
                    'dsc_all_legacy': dsc_all,
                    'dsc_all': dsc_all,  # backward-compatible alias
                    'dsc_pos': dsc_pos,
                    'official_mean_dice': float(official_mean_dice),
                    'official_dice_count': int(official_dice_count),
                    'official_score': float(official_score),
                    'acc': float(binm['acc']),
                    'recall': float(binm['recall']),
                    'specificity': float(binm['specificity']),
                    'precision': float(binm['precision']),
                    'fn': int(binm['fn']), 'fp': int(binm['fp']),
                    'pred_nonempty': int(pred_nonempty),
                    'pred_nonempty_pos': int(pred_nonempty_pos),
                    'pred_nonempty_neg': int(pred_nonempty_neg),
                    'legacy_objective': float(legacy_objective),
                    'objective': float(official_score),  # main objective is now official-like score
                }
                row['passed_recall_constraint'] = row['recall'] >= recall_constraint
                all_rows.append(row)

                if _better(row, best_official, 'official_score'):
                    best_official = dict(row)
                if row['recall'] >= 0.90 and _better(row, best_official_recall90, 'official_score'):
                    best_official_recall90 = dict(row)
                if row['recall'] >= 0.95 and _better(row, best_official_recall95, 'official_score'):
                    best_official_recall95 = dict(row)
                if row['recall'] >= recall_constraint and _better(row, best_recall_constraint, 'official_score'):
                    best_recall_constraint = dict(row)
                if _better(row, best_legacy, 'legacy_objective'):
                    best_legacy = dict(row)

    best = best_recall_constraint if best_recall_constraint is not None else best_official
    return {
        'best': best,
        'best_official': best_official,
        'best_official_recall90': best_official_recall90,
        'best_official_recall95': best_official_recall95,
        'best_legacy': best_legacy,
        'rows': all_rows,
    }
