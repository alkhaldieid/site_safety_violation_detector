"""Tests for binary-classification metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from site_safety_violation_detector.metrics import (
    compute_binary_metrics,
    find_best_threshold,
)


def test_perfect_predictions_score_one() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.01, 0.99, 0.02, 0.95, 0.97])
    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0
    assert (m.tp, m.fp, m.tn, m.fn) == (3, 0, 2, 0)


def test_threshold_changes_decisions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.4, 0.6, 0.6, 0.9])
    low = compute_binary_metrics(y_true, y_prob, threshold=0.3)
    high = compute_binary_metrics(y_true, y_prob, threshold=0.8)
    # Low threshold predicts everything positive → 100% recall, lower precision.
    assert low.recall == 1.0
    # High threshold predicts only the most confident sample → recall < 1.
    assert high.recall < 1.0


def test_auc_undefined_with_single_class() -> None:
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.2, 0.6, 0.8, 0.9])
    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert m.roc_auc is None
    assert m.pr_auc is None
    # And f1 still computable.
    assert 0.0 <= m.f1 <= 1.0


def test_find_best_threshold_picks_higher_f1() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Probs of positives cluster around 0.7; negatives around 0.4.
    y_prob = np.array([0.3, 0.45, 0.4, 0.65, 0.72, 0.8])
    best_t, best_f1 = find_best_threshold(y_true, y_prob, metric="f1")
    assert 0.45 < best_t < 0.75
    # F1 at the chosen threshold should be at least as high as at 0.5.
    baseline = compute_binary_metrics(y_true, y_prob, threshold=0.5).f1
    assert best_f1 >= baseline


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1, 0], [0.1, 0.9], threshold=0.5)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics([], [], threshold=0.5)


def test_metrics_dict_is_json_serialisable() -> None:
    import json

    y_true = np.array([0, 1])
    y_prob = np.array([0.2, 0.8])
    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    blob = json.dumps(m.to_dict())
    parsed = json.loads(blob)
    assert parsed["confusion_matrix"]["tp"] == 1
    assert math.isclose(parsed["f1"], 1.0)
