"""Dataset summary, class balance, and duplicate / leakage detection.

The CLI (``safety-dataset``) writes a JSON summary of the dataset
(per-split, per-class counts and sizes) and — with
``--check-duplicates`` — an exact-duplicate report based on SHA1
hashes of file bytes. Exact-duplicate detection across splits is
the canonical "data leakage" check: a test image that is byte-identical
to a training image inflates test metrics.

Perceptual / near-duplicate detection is optional; if the optional
``imagehash`` package is installed it is used, otherwise the check is
skipped with a clear message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import DataConfig, load_config
from .data import validate_dataset_layout
from .utils import ensure_dir, get_logger

LOGGER = get_logger(__name__)

# Match the suffix set used by inference for consistency.
VALID_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _list_image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )


def _summarize_split(split_path: Path) -> dict[str, Any]:
    """Per-split summary: per-class counts, sizes, balance."""
    classes = sorted(p.name for p in split_path.iterdir() if p.is_dir())
    per_class: dict[str, dict[str, Any]] = {}
    for cls in classes:
        files = _list_image_files(split_path / cls)
        total_bytes = sum(p.stat().st_size for p in files)
        per_class[cls] = {
            "count": len(files),
            "total_bytes": int(total_bytes),
            "mean_bytes": int(total_bytes // len(files)) if files else 0,
        }
    total = sum(c["count"] for c in per_class.values())
    balance: dict[str, float] = {}
    if total:
        balance = {cls: round(per_class[cls]["count"] / total, 6) for cls in classes}
    return {
        "classes": classes,
        "total_images": total,
        "per_class": per_class,
        "class_balance": balance,
    }


def summarize_dataset(cfg: DataConfig) -> dict[str, Any]:
    """Build a JSON-serialisable summary of the dataset described by ``cfg``."""
    paths = validate_dataset_layout(cfg, require_test=False)
    out: dict[str, Any] = {
        "root": str(Path(cfg.root).expanduser().resolve()),
        "splits": {},
    }
    for split_name, split_path in paths.items():
        out["splits"][split_name] = _summarize_split(split_path)

    # Cross-split sanity: do all splits share the same class set?
    class_sets = {name: tuple(info["classes"]) for name, info in out["splits"].items()}
    out["class_sets_consistent"] = len(set(class_sets.values())) == 1
    if not out["class_sets_consistent"]:
        out["class_sets_by_split"] = class_sets
    return out


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def _sha1(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def find_exact_duplicates(cfg: DataConfig) -> dict[str, Any]:
    """SHA1-based exact-duplicate report.

    Returns
    -------
    dict with:
      * ``within_split``  — duplicates fully inside one split (per-split, per-class)
      * ``cross_split``   — files appearing in more than one split (LEAKAGE risk)
      * ``num_files_scanned`` — for context
    """
    paths = validate_dataset_layout(cfg, require_test=False)

    # hash -> list of (split, class, path)
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    num_files = 0
    for split_name, split_path in paths.items():
        for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            cls = class_dir.name
            for img_path in _list_image_files(class_dir):
                digest = _sha1(img_path)
                hashes[digest].append((split_name, cls, str(img_path)))
                num_files += 1

    within_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_split: list[dict[str, Any]] = []

    for digest, entries in hashes.items():
        if len(entries) < 2:
            continue
        splits = {e[0] for e in entries}
        if len(splits) > 1:
            cross_split.append({
                "sha1": digest,
                "occurrences": [
                    {"split": s, "class": c, "path": p} for s, c, p in entries
                ],
            })
        else:
            split = next(iter(splits))
            within_split[split].append({
                "sha1": digest,
                "occurrences": [
                    {"class": c, "path": p} for _, c, p in entries
                ],
            })

    return {
        "num_files_scanned": num_files,
        "num_exact_duplicate_groups": sum(len(v) for v in within_split.values()) + len(cross_split),
        "cross_split": cross_split,                 # most important — possible leakage
        "within_split": dict(within_split),         # within-split duplicates (less severe)
    }


def find_near_duplicates(cfg: DataConfig, *, hash_size: int = 8) -> dict[str, Any]:
    """Perceptual-hash near-duplicate report (optional).

    Uses the ``imagehash`` package if available. Returns
    ``{"available": False, "reason": ...}`` if it isn't installed.
    """
    try:
        import imagehash  # type: ignore[import-not-found]
        from PIL import Image
    except ImportError:
        return {
            "available": False,
            "reason": (
                "Optional dependency 'imagehash' not installed. "
                "Install with 'pip install imagehash' to enable near-duplicate detection."
            ),
        }

    paths = validate_dataset_layout(cfg, require_test=False)
    buckets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    n_scanned = 0
    n_failed = 0
    for split_name, split_path in paths.items():
        for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            for img_path in _list_image_files(class_dir):
                try:
                    with Image.open(img_path) as img:
                        h = str(imagehash.phash(img.convert("RGB"), hash_size=hash_size))
                except Exception:
                    n_failed += 1
                    continue
                buckets[h].append((split_name, class_dir.name, str(img_path)))
                n_scanned += 1

    groups = [
        {
            "phash": h,
            "occurrences": [
                {"split": s, "class": c, "path": p} for s, c, p in entries
            ],
        }
        for h, entries in buckets.items()
        if len(entries) > 1
    ]
    cross_split = [
        g for g in groups if len({occ["split"] for occ in g["occurrences"]}) > 1
    ]
    return {
        "available": True,
        "hash_size": hash_size,
        "num_files_scanned": n_scanned,
        "num_files_failed": n_failed,
        "num_near_duplicate_groups": len(groups),
        "cross_split": cross_split,
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-dataset",
        description=(
            "Summarise the configured dataset and (optionally) check for "
            "duplicate or near-duplicate images across splits."
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config field, e.g. --set data.root=/path/to/data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output JSON files. Defaults to <runtime.output_dir>.",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Compute SHA1 hashes and report exact duplicates within and across splits.",
    )
    parser.add_argument(
        "--check-near-duplicates",
        action="store_true",
        help=(
            "Compute perceptual hashes and report near-duplicates. "
            "Requires the optional 'imagehash' package."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)

    out_dir = ensure_dir(args.output_dir or cfg.runtime.output_dir)
    summary = summarize_dataset(cfg.data)
    summary_path = out_dir / "dataset_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    LOGGER.info("Wrote dataset summary to %s", summary_path)
    for split, info in summary["splits"].items():
        LOGGER.info("  %s: %d images, classes=%s", split, info["total_images"], info["classes"])

    if args.check_duplicates:
        dup = find_exact_duplicates(cfg.data)
        dup_path = out_dir / "duplicates.json"
        with dup_path.open("w") as fh:
            json.dump(dup, fh, indent=2)
        LOGGER.info(
            "Scanned %d files; %d cross-split duplicate groups (LEAKAGE risk if > 0); %d within-split groups",
            dup["num_files_scanned"],
            len(dup["cross_split"]),
            sum(len(v) for v in dup["within_split"].values()),
        )
        LOGGER.info("Wrote duplicate report to %s", dup_path)

    if args.check_near_duplicates:
        near = find_near_duplicates(cfg.data)
        near_path = out_dir / "near_duplicates.json"
        with near_path.open("w") as fh:
            json.dump(near, fh, indent=2)
        if near.get("available"):
            LOGGER.info(
                "Scanned %d files; %d near-duplicate groups (%d cross-split)",
                near["num_files_scanned"],
                near["num_near_duplicate_groups"],
                len(near["cross_split"]),
            )
        else:
            LOGGER.info("Near-duplicate check skipped: %s", near.get("reason"))
        LOGGER.info("Wrote near-duplicate report to %s", near_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
