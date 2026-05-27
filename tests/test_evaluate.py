"""Tests for the evaluate module's per-sample diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from site_safety_violation_detector.config import load_config
from site_safety_violation_detector.evaluate import (
    _build_misclassified_lists,
    evaluate_checkpoint,
)
from site_safety_violation_detector.models import SafetyClassifier


def test_build_misclassified_lists_separates_fp_fn() -> None:
    paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    y_pred = (y_prob >= 0.5).astype(int)
    misc, fps, fns = _build_misclassified_lists(
        sample_paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=["no_violation", "violation"],
        threshold=0.5,
    )
    assert len(misc) == 2
    assert len(fps) == 1 and fps[0]["path"] == "b.jpg"
    assert len(fns) == 1 and fns[0]["path"] == "c.jpg"
    assert fps[0]["true_label"] == "no_violation"
    assert fps[0]["predicted_label"] == "violation"


def test_build_misclassified_lists_handles_missing_paths() -> None:
    """If sample_paths is shorter than y_true, missing entries get None — no crash."""
    y_true = np.array([1, 0])
    y_prob = np.array([0.1, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)
    misc, _, _ = _build_misclassified_lists(
        sample_paths=[],
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=["no_violation", "violation"],
        threshold=0.5,
    )
    assert len(misc) == 2
    assert all(m["path"] is None for m in misc)


def test_evaluate_checkpoint_writes_misclassified_files(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    # Save a tiny untrained checkpoint so we exercise the evaluation pipeline.
    ckpt_path = tmp_path / "model.pt"
    model = SafetyClassifier("resnet18", pretrained=False, dropout=0.0, hidden_dim=32)
    torch.save(
        {
            "architecture": "resnet18",
            "model_state_dict": model.state_dict(),
            "class_names": ["no_violation", "violation"],
            "image_size": 32,
            "threshold": 0.5,
            "epoch": 1,
        },
        ckpt_path,
    )
    (ckpt_path.with_suffix(".json")).write_text(
        json.dumps({
            "architecture": "resnet18",
            "class_names": ["no_violation", "violation"],
            "image_size": 32,
            "threshold": 0.5,
            "config": {"model": {"dropout": 0.0, "hidden_dim": 32}},
        })
    )

    cfg = load_config(
        None,
        overrides=[
            f"data.root={synthetic_dataset}",
            "data.image_size=32",
            "data.batch_size=2",
            "data.num_workers=0",
            "eval.search_threshold_on_val=false",
            f"runtime.output_dir={tmp_path / 'reports'}",
            "runtime.device=cpu",
        ],
    )
    summary = evaluate_checkpoint(cfg, checkpoint_path=ckpt_path)
    # The default output dir is reports/<arch>_test/.
    out_dir = tmp_path / "reports" / "resnet18_test"
    assert (out_dir / "test_metrics.json").is_file()
    assert (out_dir / "classification_report.txt").is_file()
    assert (out_dir / "confusion_matrix.json").is_file()
    assert (out_dir / "misclassified.json").is_file()
    assert (out_dir / "false_positives.json").is_file()
    assert (out_dir / "false_negatives.json").is_file()

    misc = json.loads((out_dir / "misclassified.json").read_text())
    # Every misclassified record should have an image path under the dataset root.
    for record in misc:
        assert record["path"] is None or str(synthetic_dataset) in record["path"]
    # Total misclassified count reported in the summary matches the file.
    assert summary["num_misclassified"] == len(misc)
