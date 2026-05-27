"""Shared utilities: logging, seeding, device selection, git/version helpers."""

from __future__ import annotations

import logging
import os
import random
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

_LOGGER_NAME = "site_safety_violation_detector"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger.

    The first call configures the root package logger with a single
    stream handler at INFO level; subsequent calls only return loggers.
    """
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(name or _LOGGER_NAME)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs deterministically.

    Heavy imports (numpy, torch) happen inside the function so that the
    rest of :mod:`utils` stays import-light.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic where feasible (slight perf hit).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device string, falling back gracefully.

    ``preference`` may be ``"auto" | "cpu" | "cuda" | "mps"``.
    """
    pref = (preference or "auto").lower()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        return out.decode("utf-8").strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def utc_now_iso() -> str:
    """Current UTC time in ISO-8601 (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model) -> tuple[int, int]:
    """Return ``(total_params, trainable_params)`` for a torch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def format_iterable(values: Iterable[str]) -> str:
    return ", ".join(str(v) for v in values)
