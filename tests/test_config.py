"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from site_safety_violation_detector.config import (
    Config,
    config_to_dict,
    load_config,
    save_config,
)


def test_defaults_are_self_consistent() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    # Defaults shouldn't refer to anyone's filesystem.
    assert not cfg.data.root.startswith("/")
    assert cfg.train.epochs > 0
    assert cfg.data.image_size in (128, 160, 192, 224, 256, 320)
    assert cfg.runtime.checkpoint_name.endswith(".pt")
    assert "efficientnet_b0" in cfg.sweep.architectures
    assert cfg.sweep.selection_metric == "val_f1"


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = load_config(None)
    cfg.train.epochs = 7
    cfg.data.batch_size = 11
    cfg.sweep.architectures = ["resnet18"]
    out = save_config(cfg, tmp_path / "cfg.yaml")
    assert out.is_file()

    reloaded = load_config(out)
    assert reloaded.train.epochs == 7
    assert reloaded.data.batch_size == 11
    assert reloaded.sweep.architectures == ["resnet18"]


def test_yaml_partial_override(tmp_path: Path) -> None:
    yaml_path = tmp_path / "partial.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": {"epochs": 3},
                "model": {"architecture": "resnet18"},
            }
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.train.epochs == 3
    assert cfg.model.architecture == "resnet18"
    # Untouched fields fall back to defaults.
    assert cfg.optim.optimizer == "adam"
    assert cfg.data.image_size == 224


def test_cli_overrides_are_coerced() -> None:
    cfg = load_config(
        None,
        overrides=[
            "train.epochs=4",
            "optim.learning_rate=3e-4",
            "model.pretrained=false",
            "train.grad_clip_norm=none",
        ],
    )
    assert cfg.train.epochs == 4
    assert cfg.optim.learning_rate == pytest.approx(3e-4)
    assert cfg.model.pretrained is False
    assert cfg.train.grad_clip_norm is None


def test_unknown_field_raises() -> None:
    with pytest.raises(KeyError):
        load_config(None, overrides=["train.this_field_does_not_exist=1"])


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_config_to_dict_is_json_friendly() -> None:
    import json

    cfg = load_config(None)
    blob = json.dumps(config_to_dict(cfg))
    assert "sweep" in blob
