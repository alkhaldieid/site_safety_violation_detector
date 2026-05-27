"""Tests for the data module."""

from __future__ import annotations

from pathlib import Path

import pytest

from site_safety_violation_detector.config import DataConfig
from site_safety_violation_detector.data import (
    build_dataloaders,
    build_eval_transform,
    build_train_transform,
    compute_positive_class_weight,
    validate_dataset_layout,
)


def test_validate_layout_happy_path(synthetic_dataset: Path) -> None:
    cfg = DataConfig(root=str(synthetic_dataset), image_size=32, batch_size=4, num_workers=0)
    paths = validate_dataset_layout(cfg)
    assert set(paths) == {"train", "val", "test"}
    for p in paths.values():
        assert p.is_dir()


def test_validate_layout_missing_root(tmp_path: Path) -> None:
    cfg = DataConfig(root=str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        validate_dataset_layout(cfg)


def test_validate_layout_missing_split(tmp_path: Path) -> None:
    # Create only a train split.
    (tmp_path / "train" / "violation").mkdir(parents=True)
    cfg = DataConfig(root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        validate_dataset_layout(cfg)


def test_transforms_have_expected_output_shape() -> None:
    import torch
    from PIL import Image

    img = Image.new("RGB", (96, 60), color=(50, 100, 150))
    train_t = build_train_transform(64)
    eval_t = build_eval_transform(64)
    train_out = train_t(img)
    eval_out = eval_t(img)
    assert isinstance(train_out, torch.Tensor)
    assert train_out.shape == (3, 64, 64)
    assert eval_out.shape == (3, 64, 64)
    # ImageNet normalisation should produce values roughly in [-3, 3].
    assert -5.0 < float(eval_out.min()) and float(eval_out.max()) < 5.0


def test_build_dataloaders_yields_batches(synthetic_dataset: Path) -> None:
    cfg = DataConfig(
        root=str(synthetic_dataset),
        image_size=32,
        batch_size=4,
        num_workers=0,
    )
    bundle = build_dataloaders(cfg)
    assert bundle.num_classes == 2
    assert bundle.class_names == ["no_violation", "violation"]
    # Train has 2 classes * 6 images = 12 images → 3 batches of 4.
    batches = list(bundle.train_loader)
    assert len(batches) == 3
    images, labels = batches[0]
    assert images.shape[1:] == (3, 32, 32)
    assert labels.shape[0] == images.shape[0]
    # Test loader exists.
    assert bundle.test_loader is not None
    assert sum(bundle.train_class_counts.values()) == 12


def test_compute_positive_class_weight_balanced() -> None:
    weight = compute_positive_class_weight(
        {"no_violation": 100, "violation": 100},
        class_names=["no_violation", "violation"],
    )
    assert weight == pytest.approx(1.0)


def test_compute_positive_class_weight_imbalanced() -> None:
    weight = compute_positive_class_weight(
        {"no_violation": 300, "violation": 100},
        class_names=["no_violation", "violation"],
    )
    assert weight == pytest.approx(3.0)


def test_compute_positive_class_weight_zero_positives() -> None:
    with pytest.raises(ValueError):
        compute_positive_class_weight(
            {"no_violation": 10, "violation": 0},
            class_names=["no_violation", "violation"],
        )
