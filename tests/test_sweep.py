"""Sweep tests, including multi-seed aggregation.

These exercise the internal helpers and a tiny single-arch sweep so we
don't pay the cost of training multiple architectures in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from site_safety_violation_detector.config import load_config
from site_safety_violation_detector.sweep import (
    _aggregate_per_arch,
    _is_arch_better,
    _mean_std,
    run_sweep,
)


def test_mean_std_handles_none_and_nan() -> None:
    mean, std, n = _mean_std([1.0, 3.0, None, float("nan"), 5.0])
    assert n == 3
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(2.0)  # sample std of {1,3,5}


def test_mean_std_single_value() -> None:
    mean, std, n = _mean_std([0.7])
    assert (mean, std, n) == (0.7, 0.0, 1)


def test_mean_std_all_missing() -> None:
    assert _mean_std([None, float("nan")]) == (None, None, 0)


def test_aggregate_per_arch_collects_seeds() -> None:
    per_run = [
        {"architecture": "a", "seed": 1, "val_f1": 0.80, "val_loss": 0.50,
         "val_accuracy": 0.8, "val_precision": 0.8, "val_recall": 0.8,
         "val_roc_auc": 0.9, "val_pr_auc": 0.9,
         "inference_latency_ms": 10.0, "wall_time_seconds": 1.0,
         "params_total": 100, "params_trainable": 100, "threshold": 0.5,
         "checkpoint": "ckpt-a-1"},
        {"architecture": "a", "seed": 2, "val_f1": 0.84, "val_loss": 0.40,
         "val_accuracy": 0.85, "val_precision": 0.85, "val_recall": 0.85,
         "val_roc_auc": 0.9, "val_pr_auc": 0.9,
         "inference_latency_ms": 10.0, "wall_time_seconds": 1.0,
         "params_total": 100, "params_trainable": 100, "threshold": 0.5,
         "checkpoint": "ckpt-a-2"},
        {"architecture": "b", "seed": 1, "val_f1": 0.60, "val_loss": 0.70,
         "val_accuracy": 0.6, "val_precision": 0.6, "val_recall": 0.6,
         "val_roc_auc": 0.8, "val_pr_auc": 0.8,
         "inference_latency_ms": 5.0, "wall_time_seconds": 1.0,
         "params_total": 50, "params_trainable": 50, "threshold": 0.5,
         "checkpoint": "ckpt-b-1"},
    ]
    agg = _aggregate_per_arch(per_run)
    by_arch = {r["architecture"]: r for r in agg}
    assert by_arch["a"]["num_seeds"] == 2
    assert by_arch["a"]["seeds"] == [1, 2]
    assert by_arch["a"]["mean_val_f1"] == pytest.approx(0.82)
    assert by_arch["b"]["num_seeds"] == 1
    assert by_arch["b"]["mean_val_f1"] == pytest.approx(0.60)
    # Std for a single seed is zero by convention.
    assert by_arch["b"]["std_val_f1"] == 0.0


def test_is_arch_better_uses_mean_val_loss_on_tie() -> None:
    a = {"mean_val_f1": 0.8, "mean_val_loss": 0.5}
    b = {"mean_val_f1": 0.8, "mean_val_loss": 0.3}
    # Same mean_val_f1 → tie-breaker val_loss (lower-is-better) prefers b.
    assert _is_arch_better(b, a, primary="val_f1", tie_breaker="val_loss") is True
    assert _is_arch_better(a, b, primary="val_f1", tie_breaker="val_loss") is False


def test_is_arch_better_picks_higher_primary() -> None:
    high = {"mean_val_f1": 0.9, "mean_val_loss": 0.6}
    low = {"mean_val_f1": 0.7, "mean_val_loss": 0.2}
    assert _is_arch_better(high, low, primary="val_f1", tie_breaker="val_loss") is True


def test_run_sweep_single_arch_multi_seed(synthetic_dataset: Path, tmp_path: Path) -> None:
    """End-to-end smoke: one arch, two seeds — verifies the multi-seed flow
    produces aggregated metrics, per-run rows, and a promoted checkpoint."""
    cfg = load_config(
        None,
        overrides=[
            f"data.root={synthetic_dataset}",
            "data.image_size=32",
            "data.batch_size=4",
            "data.num_workers=0",
            "model.pretrained=false",
            "model.hidden_dim=16",
            "train.epochs=1",
            "train.early_stopping_patience=1",
            "optim.scheduler=none",
            f"runtime.output_dir={tmp_path / 'reports'}",
            f"runtime.models_dir={tmp_path / 'models'}",
            "runtime.device=cpu",
        ],
    )
    cfg.sweep.architectures = ["resnet18"]
    summary = run_sweep(cfg, seeds=[1, 2])

    assert summary["multi_seed"] is True
    assert summary["seeds"] == [1, 2]
    assert len(summary["per_run"]) == 2
    assert len(summary["aggregated"]) == 1
    agg = summary["aggregated"][0]
    assert agg["architecture"] == "resnet18"
    assert agg["num_seeds"] == 2
    assert agg["seeds"] == [1, 2]
    # Std fields should be numeric (or zero), not None.
    assert agg["std_val_f1"] is not None

    # Best model + metadata exist.
    best_ckpt = tmp_path / "models" / cfg.runtime.checkpoint_name
    best_meta = tmp_path / "models" / cfg.runtime.metadata_name
    assert best_ckpt.is_file()
    assert best_meta.is_file()
    meta = json.loads(best_meta.read_text())
    assert meta["selected_architecture"] == "resnet18"
    assert meta["multi_seed"] is True
    assert meta["seeds_evaluated"] == [1, 2]
    assert "rationale" in meta and "± " in meta["rationale"]

    # Comparison CSVs exist.
    reports = tmp_path / "reports"
    assert (reports / "model_comparison.json").is_file()
    assert (reports / "model_comparison.csv").is_file()
    assert (reports / "model_comparison_per_run.csv").is_file()
