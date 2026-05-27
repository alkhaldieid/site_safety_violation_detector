"""Metric computation for binary classification.

All functions accept ``y_true`` and ``y_prob`` as 1-D numpy arrays
(or anything convertible to one) and return plain Python types so the
results are JSON-serialisable.

We accept *probabilities* (post-sigmoid) rather than logits because the
threshold-search helpers operate naturally in probability space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    """Common binary-classification metrics."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
        }


def _to_1d(x) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    return arr


def compute_binary_metrics(
    y_true,
    y_prob,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute binary classification metrics at a given threshold.

    ``y_true`` should contain 0/1 labels and ``y_prob`` should contain
    probabilities in [0, 1] (post-sigmoid).

    ROC-AUC and PR-AUC are ``None`` when only one class is present in
    ``y_true`` (where they're undefined) — this happens commonly on tiny
    test slices and we should not crash.
    """
    y_true = _to_1d(y_true).astype(int)
    y_prob = _to_1d(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob must have the same shape, got {y_true.shape} vs {y_prob.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array.")

    y_pred = (y_prob >= threshold).astype(int)

    # zero_division=0 → don't crash when there are no predicted positives
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc: float | None
    pr_auc: float | None
    if len(np.unique(y_true)) < 2:
        roc_auc = None
        pr_auc = None
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def find_best_threshold(
    y_true,
    y_prob,
    *,
    metric: str = "f1",
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid-search the threshold that maximises ``metric`` on (y_true, y_prob).

    Returns ``(best_threshold, best_metric_value)``.
    """
    y_true = _to_1d(y_true).astype(int)
    y_prob = _to_1d(y_prob).astype(float)

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    if metric != "f1":
        raise ValueError(f"Only metric='f1' is currently supported, got {metric!r}")

    best_t = 0.5
    best_v = -1.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        v = float(f1_score(y_true, pred, zero_division=0))
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t, best_v
