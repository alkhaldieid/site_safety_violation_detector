"""Configuration loading for the safety-violation pipeline.

All runtime knobs are described by dataclasses below and are loaded from
YAML files (typically under ``configs/``). CLI scripts may override
individual fields via ``--set key=value`` overrides.

This module deliberately contains no torch / timm imports so that
``test_config`` can run in a minimal environment.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Dataset locations and loader settings."""

    root: str = "data/Worksite-Safety-Monitoring-Dataset"
    train_subdir: str = "train"
    val_subdir: str = "valid"
    test_subdir: str = "test"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    use_weighted_sampler: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.5
    hidden_dim: int = 512


@dataclass
class OptimConfig:
    """Optimiser / scheduler configuration."""

    optimizer: str = "adam"           # adam | adamw | sgd
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9             # used by sgd
    scheduler: str = "cosine"         # cosine | step | plateau | none
    step_size: int = 10               # used by step
    gamma: float = 0.1                # used by step / plateau
    warmup_epochs: int = 0


@dataclass
class TrainConfig:
    """Training-loop configuration."""

    epochs: int = 25
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_f1"   # val_f1 | val_loss | val_pr_auc
    use_class_weights: bool = False
    grad_clip_norm: float | None = None
    log_every: int = 25


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    threshold: float = 0.5
    search_threshold_on_val: bool = True


@dataclass
class RuntimeConfig:
    """Misc runtime knobs."""

    seed: int = 42
    device: str = "auto"   # auto | cpu | cuda | mps
    output_dir: str = "reports"
    models_dir: str = "models"
    checkpoint_name: str = "best_model.pt"
    metadata_name: str = "best_model_metadata.json"


@dataclass
class SweepConfig:
    """Multi-model sweep configuration."""

    architectures: list[str] = field(
        default_factory=lambda: [
            "efficientnet_b0",
            "resnet50",
            "mobilenetv3_large_100",
            "convnext_tiny",
        ]
    )
    selection_metric: str = "val_f1"
    tie_breaker: str = "val_loss"   # lower is better when this is val_loss
    report_basename: str = "model_comparison"


@dataclass
class Config:
    """Top-level configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Convert a string to int/float/bool/None where possible (for CLI overrides)."""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"none", "null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _merge_into_dataclass(instance: Any, updates: Mapping[str, Any]) -> Any:
    """Recursively merge a mapping into a (possibly nested) dataclass."""
    if not dataclasses.is_dataclass(instance):
        raise TypeError(f"Cannot merge into non-dataclass: {type(instance)!r}")

    field_map = {f.name: f for f in dataclasses.fields(instance)}
    for key, value in updates.items():
        if key not in field_map:
            raise KeyError(f"Unknown config field: {key!r} (allowed: {sorted(field_map)})")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            _merge_into_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def _set_dotted(cfg: Config, dotted_key: str, value: Any) -> None:
    """Apply a single CLI override of the form ``section.field=value``."""
    parts = dotted_key.split(".")
    if not parts:
        raise ValueError(f"Empty override key: {dotted_key!r}")
    target: Any = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"Unknown override section: {part!r} in {dotted_key!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"Unknown override field: {leaf!r} in {dotted_key!r}")
    setattr(target, leaf, _coerce(value))


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> Config:
    """Load a :class:`Config` from a YAML file with optional CLI overrides.

    Parameters
    ----------
    path:
        Path to a YAML config file. If ``None``, defaults are returned.
    overrides:
        Iterable of ``section.field=value`` strings (e.g. ``"train.epochs=10"``).
    """
    cfg = Config()

    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r") as fh:
            payload = yaml.safe_load(fh) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"Top-level YAML in {p} must be a mapping, got {type(payload).__name__}")
        _merge_into_dataclass(cfg, payload)

    for override in overrides or ():
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override!r}")
        key, raw_value = override.split("=", 1)
        _set_dotted(cfg, key.strip(), raw_value.strip())

    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a :class:`Config` to a plain (JSON-serialisable) dict."""
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> Path:
    """Persist a :class:`Config` to disk as YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
    return p
