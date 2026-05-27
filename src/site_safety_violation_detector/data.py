"""Data loading and transforms.

Expected on-disk layout (see README for details)::

    <data.root>/
      <train_subdir>/<class_a>/*.jpg
      <train_subdir>/<class_b>/*.jpg
      <val_subdir>/...
      <test_subdir>/...
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import DataConfig
from .utils import get_logger

LOGGER = get_logger(__name__)

# Standard ImageNet normalisation (matches pretrained backbones in timm).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_train_transform(image_size: int) -> transforms.Compose:
    """Augmenting transform used during training."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform used at validation / test / inference time."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def validate_dataset_layout(cfg: DataConfig, *, require_test: bool = True) -> dict[str, Path]:
    """Ensure the expected train/val/test split directories exist.

    Raises :class:`FileNotFoundError` with a helpful message if anything
    is missing. Returns a mapping ``{"train": ..., "val": ..., "test": ...}``
    of resolved paths (``test`` only present when ``require_test`` is True
    and the test directory exists).
    """
    root = Path(cfg.root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}\n"
            f"Expected layout:\n"
            f"  {root}/{cfg.train_subdir}/<class>/*.jpg\n"
            f"  {root}/{cfg.val_subdir}/<class>/*.jpg\n"
            f"  {root}/{cfg.test_subdir}/<class>/*.jpg"
        )

    paths: dict[str, Path] = {}
    for split, subdir, required in (
        ("train", cfg.train_subdir, True),
        ("val", cfg.val_subdir, True),
        ("test", cfg.test_subdir, require_test),
    ):
        candidate = root / subdir
        if candidate.is_dir():
            paths[split] = candidate
        elif required:
            raise FileNotFoundError(
                f"Missing dataset split directory: {candidate}\n"
                f"Expected to find class subfolders under it (e.g. 'violation/', 'no_violation/')."
            )
    return paths


# ---------------------------------------------------------------------------
# Dataloader assembly
# ---------------------------------------------------------------------------


@dataclass
class DataBundle:
    """Container for the loaders + class metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    class_names: list[str]
    class_to_idx: dict[str, int]
    train_class_counts: dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _class_counts(dataset: datasets.ImageFolder) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in dataset.classes}
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    for _, target in dataset.samples:
        counts[idx_to_class[int(target)]] += 1
    return counts


def _build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    counts = [0] * len(dataset.classes)
    for _, target in dataset.samples:
        counts[int(target)] += 1
    inv = [1.0 / c if c > 0 else 0.0 for c in counts]
    weights = [inv[int(target)] for _, target in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(
    cfg: DataConfig,
    *,
    require_test: bool = True,
    pin_memory: bool | None = None,
) -> DataBundle:
    """Construct train / val / (optional) test dataloaders.

    Parameters
    ----------
    cfg:
        :class:`DataConfig` describing dataset locations and loader knobs.
    require_test:
        If ``True``, the test directory must exist; otherwise the test
        loader is returned as ``None``.
    pin_memory:
        Override ``pin_memory``. Defaults to True iff CUDA is available.
    """
    paths = validate_dataset_layout(cfg, require_test=require_test)

    train_transform = build_train_transform(cfg.image_size)
    eval_transform = build_eval_transform(cfg.image_size)

    train_dataset = datasets.ImageFolder(str(paths["train"]), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(paths["val"]), transform=eval_transform)
    test_dataset: datasets.ImageFolder | None = None
    if "test" in paths:
        test_dataset = datasets.ImageFolder(str(paths["test"]), transform=eval_transform)

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training images found under {paths['train']}.")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation images found under {paths['val']}.")

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        sampler = _build_weighted_sampler(train_dataset)
        shuffle = False
        LOGGER.info("Using WeightedRandomSampler for class-imbalance correction.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader | None = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    class_names = list(train_dataset.classes)
    if val_dataset.classes != class_names or (
        test_dataset is not None and test_dataset.classes != class_names
    ):
        raise RuntimeError(
            f"Class set differs across splits.\n"
            f"  train: {train_dataset.classes}\n"
            f"  val:   {val_dataset.classes}\n"
            f"  test:  {None if test_dataset is None else test_dataset.classes}"
        )

    counts = _class_counts(train_dataset)
    LOGGER.info("Detected classes: %s", class_names)
    LOGGER.info("Train counts: %s", counts)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        class_to_idx=dict(train_dataset.class_to_idx),
        train_class_counts=counts,
    )


def compute_positive_class_weight(
    class_counts: dict[str, int],
    *,
    positive_class: str | None = None,
    class_names: Sequence[str] | None = None,
) -> float:
    """Compute ``pos_weight`` for :class:`~torch.nn.BCEWithLogitsLoss`.

    The positive class is taken to be the *second* class
    (index 1 in alphabetical ImageFolder ordering) unless explicitly
    specified — this matches the project's binary "violation vs.
    no_violation" framing where the alphabetical order puts
    ``no_violation`` (negative) before ``violation`` (positive).
    """
    if positive_class is None:
        names = list(class_names) if class_names is not None else sorted(class_counts)
        if len(names) != 2:
            raise ValueError(
                f"compute_positive_class_weight expects 2 classes, got {len(names)}: {names}"
            )
        positive_class = names[1]
    neg = sum(v for k, v in class_counts.items() if k != positive_class)
    pos = class_counts.get(positive_class, 0)
    if pos == 0:
        raise ValueError(f"Positive class {positive_class!r} has zero samples.")
    return float(neg) / float(pos)
