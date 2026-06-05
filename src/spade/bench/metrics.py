"""Localization and detection metrics for the benchmark.

Localization is scored at the pixel level between a predicted boolean mask and
the ground-truth mask (both in tampered-image coordinates). Detection is scored
at the image level from a per-image score (e.g. confidence) via ROC-AUC.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def localization_scores(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Pixel-level localization metrics between two boolean masks.

    Returns IoU, F1 (Dice), precision, recall, and MCC. Defined to degrade
    gracefully on empty masks (e.g. no prediction and no ground truth -> perfect).
    """
    pred = np.asarray(pred_mask, dtype=bool).ravel()
    gt = np.asarray(gt_mask, dtype=bool).ravel()
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: {pred.shape} vs {gt.shape}")

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))

    union = tp + fp + fn
    iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    # Matthews correlation coefficient, with the standard 0-denominator guard.
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denom == 0 else (tp * tn - fp * fn) / denom

    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "mcc": mcc}


def detection_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Image-level ROC-AUC from per-image scores and binary labels.

    Returns 0.5 if only one class is present (AUC undefined).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def aggregate(per_sample: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each localization metric across samples."""
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    return {k: float(np.mean([s[k] for s in per_sample])) for k in keys}
