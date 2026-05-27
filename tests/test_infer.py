"""End-to-end smoke tests for inference preprocessing + checkpoint loading.

These avoid timm pretrained weight downloads (``pretrained=False``) and
write a checkpoint directly so we don't depend on the training loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from site_safety_violation_detector.evaluate import load_checkpoint
from site_safety_violation_detector.infer import iter_image_paths, predict
from site_safety_violation_detector.models import SafetyClassifier


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Persist a minimal checkpoint + metadata for a 64-dim resnet18 model."""
    ckpt_path = tmp_path / "model.pt"
    meta_path = ckpt_path.with_suffix(".json")

    model = SafetyClassifier("resnet18", pretrained=False, dropout=0.0, hidden_dim=64)
    torch.save(
        {
            "architecture": "resnet18",
            "model_state_dict": model.state_dict(),
            "class_names": ["no_violation", "violation"],
            "image_size": 64,
            "threshold": 0.5,
            "epoch": 1,
        },
        ckpt_path,
    )
    meta_path.write_text(
        json.dumps(
            {
                "architecture": "resnet18",
                "class_names": ["no_violation", "violation"],
                "image_size": 64,
                "threshold": 0.5,
                "config": {"model": {"dropout": 0.0, "hidden_dim": 64}},
            }
        )
    )
    return ckpt_path, meta_path


def test_load_checkpoint_round_trip(fake_checkpoint: tuple[Path, Path]) -> None:
    ckpt, _ = fake_checkpoint
    loaded = load_checkpoint(ckpt, device="cpu")
    assert loaded.architecture == "resnet18"
    assert loaded.class_names == ["no_violation", "violation"]
    assert loaded.image_size == 64
    assert loaded.threshold == 0.5


def test_predict_single_image(fake_checkpoint: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, _ = fake_checkpoint
    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (40, 40), color=(123, 200, 50)).save(img_path)
    results = predict(checkpoint=ckpt, input_path=img_path, device="cpu")
    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert r.label in {"no_violation", "violation"}
    assert 0.0 <= r.probability <= 1.0


def test_predict_folder_skips_non_images(fake_checkpoint: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, _ = fake_checkpoint
    folder = tmp_path / "imgs"
    folder.mkdir()
    Image.new("RGB", (40, 40), color=(10, 10, 10)).save(folder / "a.jpg")
    Image.new("RGB", (40, 40), color=(220, 220, 220)).save(folder / "b.png")
    # Non-image files must not be picked up.
    (folder / "README.txt").write_text("ignore me")
    results = predict(checkpoint=ckpt, input_path=folder, device="cpu")
    assert len(results) == 2


def test_predict_handles_corrupt_image(fake_checkpoint: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, _ = fake_checkpoint
    folder = tmp_path / "imgs"
    folder.mkdir()
    Image.new("RGB", (40, 40), color=(10, 10, 10)).save(folder / "a.jpg")
    bad = folder / "broken.jpg"
    bad.write_bytes(b"not a real jpeg")
    results = predict(checkpoint=ckpt, input_path=folder, device="cpu")
    assert len(results) == 2
    errors = [r for r in results if r.error is not None]
    assert len(errors) == 1, "corrupt image should be reported, not skipped silently"
    # And the good one still succeeded.
    good = [r for r in results if r.error is None]
    assert len(good) == 1


def test_predict_missing_input_raises(fake_checkpoint: tuple[Path, Path], tmp_path: Path) -> None:
    ckpt, _ = fake_checkpoint
    with pytest.raises(FileNotFoundError):
        predict(checkpoint=ckpt, input_path=tmp_path / "missing.jpg", device="cpu")


def test_iter_image_paths_extensions(tmp_path: Path) -> None:
    (tmp_path / "x.jpg").write_bytes(b"")
    (tmp_path / "x.TIFF").write_bytes(b"")
    (tmp_path / "x.txt").write_bytes(b"")
    found = sorted(p.name for p in iter_image_paths(tmp_path))
    assert "x.jpg" in found and "x.TIFF" in found
    assert "x.txt" not in found
