"""Tests for the model factory + forward pass.

All tests use ``pretrained=False`` so no network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from site_safety_violation_detector.config import ModelConfig
from site_safety_violation_detector.models import (
    SUPPORTED_ARCHITECTURES,
    SafetyClassifier,
    build_model,
)
from site_safety_violation_detector.utils import count_parameters


def _quick_model(arch: str = "resnet18") -> SafetyClassifier:
    return SafetyClassifier(arch, pretrained=False, dropout=0.0, hidden_dim=64)


def test_forward_pass_shape() -> None:
    model = _quick_model()
    model.eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)


def test_forward_pass_returns_logits_not_probabilities() -> None:
    """If sigmoid is applied internally the outputs can never be < 0 or > 1.

    We can't force a particular logit range, but we *can* verify the
    head exposes no Sigmoid layer.
    """
    model = _quick_model()
    has_sigmoid = any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())
    assert not has_sigmoid, "Model head must not contain Sigmoid; loss expects logits."


def test_build_model_uses_modelconfig() -> None:
    cfg = ModelConfig(architecture="resnet18", pretrained=False, dropout=0.1, hidden_dim=128)
    model = build_model(cfg)
    assert model.architecture == "resnet18"
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable == total  # everything trainable by default


def test_single_image_inference_works() -> None:
    """SafetyClassifier should handle batch_size=1 (we use LayerNorm in the head)."""
    model = _quick_model()
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_supported_arch_list_is_nonempty() -> None:
    assert "efficientnet_b0" in SUPPORTED_ARCHITECTURES
    assert "resnet50" in SUPPORTED_ARCHITECTURES


def test_checkpoint_metadata_serialises(tmp_path: Path) -> None:
    """The metadata payload we save with checkpoints must round-trip through JSON."""
    metadata = {
        "architecture": "resnet18",
        "class_names": ["no_violation", "violation"],
        "image_size": 224,
        "threshold": 0.42,
        "best_epoch": 3,
        "validation_metrics": {
            "threshold": 0.42,
            "accuracy": 0.9,
            "precision": 0.88,
            "recall": 0.92,
            "f1": 0.9,
            "roc_auc": 0.95,
            "pr_auc": 0.91,
            "confusion_matrix": {"tp": 18, "fp": 2, "tn": 18, "fn": 2},
        },
        "params_total": 11_000_000,
        "params_trainable": 11_000_000,
        "config": {"data": {"image_size": 224}},
        "git_commit": None,
        "created_at": "2024-01-01T00:00:00+00:00",
        "rationale": "unit test",
        "checkpoint": "models/best_model.pt",
    }
    path = tmp_path / "best_model_metadata.json"
    path.write_text(json.dumps(metadata))
    reloaded = json.loads(path.read_text())
    assert reloaded["architecture"] == "resnet18"
    assert reloaded["validation_metrics"]["confusion_matrix"]["tp"] == 18
