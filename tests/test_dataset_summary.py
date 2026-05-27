"""Tests for dataset summary and duplicate / leakage detection."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from site_safety_violation_detector.config import DataConfig
from site_safety_violation_detector.dataset_summary import (
    find_exact_duplicates,
    find_near_duplicates,
    summarize_dataset,
)


def test_summarize_dataset_class_counts(synthetic_dataset: Path) -> None:
    cfg = DataConfig(root=str(synthetic_dataset), image_size=32, batch_size=4, num_workers=0)
    summary = summarize_dataset(cfg)
    assert summary["class_sets_consistent"] is True
    train = summary["splits"]["train"]
    assert train["classes"] == ["no_violation", "violation"]
    # conftest writes 6 train per class.
    assert train["per_class"]["no_violation"]["count"] == 6
    assert train["per_class"]["violation"]["count"] == 6
    assert train["total_images"] == 12
    # Balance values sum to 1.
    assert abs(sum(train["class_balance"].values()) - 1.0) < 1e-6


def test_find_exact_duplicates_clean(synthetic_dataset: Path) -> None:
    """Conftest writes the same colour into every image of a class.

    That's the worst case for exact-duplicate detection: every image
    within a class IS byte-identical. We still expect *no* cross-split
    duplicates (the splits write their own files) — wait, actually
    PIL JPEG encoding of identical RGB content is deterministic, so all
    images of a given class across all splits will share the same SHA1.
    This test verifies we *detect* those duplicates, not that they're
    absent (the synthetic fixture is deliberately uniform).
    """
    cfg = DataConfig(root=str(synthetic_dataset), image_size=32, batch_size=4, num_workers=0)
    dup = find_exact_duplicates(cfg)
    assert dup["num_files_scanned"] == 6 * 2 + 2 * 2 + 2 * 2  # train + valid + test counts
    # All images of one class are byte-identical → at least one cross-split
    # duplicate group per class.
    assert len(dup["cross_split"]) >= 2
    # And many within-split duplicates too.
    assert dup["num_exact_duplicate_groups"] >= 2


def _unique_colour(split: str, cls: str, i: int) -> tuple[int, int, int]:
    """Deterministic, per-file unique RGB colour for synthetic images.

    Mixes the split name, class name, and index so no two files in
    different splits / classes / positions collide.
    """
    base = (hash((split, cls, i)) & 0xFFFFFF)
    return (base & 0xFF, (base >> 8) & 0xFF, (base >> 16) & 0xFF)


def _build_unique_dataset(root: Path) -> None:
    for split, n in (("train", 3), ("valid", 2), ("test", 2)):
        for cls in ("no_violation", "violation"):
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.new("RGB", (16, 16), _unique_colour(split, cls, i)).save(
                    cls_dir / f"{split}_{cls}_{i}.png"
                )


def test_find_exact_duplicates_distinct_files(tmp_path: Path) -> None:
    """When images have unique content, no duplicate groups should fire."""
    root = tmp_path / "ds"
    _build_unique_dataset(root)
    cfg = DataConfig(root=str(root))
    dup = find_exact_duplicates(cfg)
    assert dup["cross_split"] == []
    assert dup["within_split"] == {}


def test_find_exact_duplicates_detects_cross_split_leakage(tmp_path: Path) -> None:
    """Plant the same file into train and test → must be flagged."""
    root = tmp_path / "ds"
    _build_unique_dataset(root)
    # Copy a train image into test (same class) → cross-split duplicate.
    leaked_src = root / "train" / "violation" / "train_violation_0.png"
    leaked_dst = root / "test" / "violation" / "leaked_from_train.png"
    shutil.copyfile(leaked_src, leaked_dst)

    cfg = DataConfig(root=str(root))
    dup = find_exact_duplicates(cfg)
    assert len(dup["cross_split"]) == 1
    splits = {occ["split"] for occ in dup["cross_split"][0]["occurrences"]}
    assert splits == {"train", "test"}


def test_find_near_duplicates_graceful_when_imagehash_absent(synthetic_dataset: Path) -> None:
    """The function must not crash when imagehash isn't installed.

    Either it returns a result (if the env happens to have imagehash) or
    it returns ``{"available": False, "reason": "..."}``. Both are
    acceptable.
    """
    cfg = DataConfig(root=str(synthetic_dataset))
    result = find_near_duplicates(cfg)
    assert "available" in result
    if not result["available"]:
        assert "imagehash" in result["reason"]
