"""Shared pytest fixtures.

Synthesizes a tiny on-disk image dataset matching the expected
``train/valid/test`` layout, so tests can exercise the data pipeline
without needing the real Worksite-Safety-Monitoring-Dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path, format="JPEG")


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a tiny ImageFolder-compatible dataset and return its root."""
    root = tmp_path / "ds"
    classes = ("no_violation", "violation")
    colors = {"no_violation": (10, 200, 10), "violation": (200, 10, 10)}
    # 6 train, 2 val, 2 test per class — enough for batchnorm-free code paths.
    for split, n in (("train", 6), ("valid", 2), ("test", 2)):
        for cls in classes:
            for i in range(n):
                _write_image(root / split / cls / f"{cls}_{i}.jpg", colors[cls])
    return root
