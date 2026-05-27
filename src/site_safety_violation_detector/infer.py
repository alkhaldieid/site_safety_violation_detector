"""Single-image and folder inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from .data import build_eval_transform
from .evaluate import load_checkpoint
from .utils import get_logger, resolve_device

LOGGER = get_logger(__name__)

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Prediction:
    image: str
    label_index: int
    label: str
    probability: float
    threshold: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def iter_image_paths(target: Path) -> Iterable[Path]:
    """Yield image paths under ``target`` (file or directory)."""
    if target.is_file():
        yield target
        return
    if target.is_dir():
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
                yield path
        return
    raise FileNotFoundError(f"Input path does not exist: {target}")


def _predict_one(
    model,
    image_path: Path,
    *,
    transform,
    device: str,
    class_names: list[str],
    threshold: float,
) -> Prediction:
    """Run a single-image prediction, catching common image errors."""
    try:
        with Image.open(image_path) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(tensor)).item())
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return Prediction(
            image=str(image_path),
            label_index=-1,
            label="error",
            probability=float("nan"),
            threshold=threshold,
            error=f"{type(exc).__name__}: {exc}",
        )
    label_idx = int(prob >= threshold)
    label_name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
    return Prediction(
        image=str(image_path),
        label_index=label_idx,
        label=label_name,
        probability=prob,
        threshold=threshold,
    )


def predict(
    *,
    checkpoint: str | Path,
    input_path: str | Path,
    metadata: str | Path | None = None,
    device: str = "auto",
    threshold: float | None = None,
) -> list[Prediction]:
    """Run inference on a single image or a folder.

    Returns a list of :class:`Prediction` (always a list, even for a single image).
    Corrupt images are reported per-prediction in ``error`` rather than crashing the batch.
    """
    resolved_device = resolve_device(device)
    loaded = load_checkpoint(checkpoint, device=resolved_device, metadata_path=metadata)
    transform = build_eval_transform(loaded.image_size)
    used_threshold = float(threshold if threshold is not None else loaded.threshold)

    results: list[Prediction] = []
    target = Path(input_path)
    for image_path in iter_image_paths(target):
        results.append(
            _predict_one(
                loaded.model,
                image_path,
                transform=transform,
                device=resolved_device,
                class_names=loaded.class_names,
                threshold=used_threshold,
            )
        )
    return results


def _write_predictions(predictions: list[Prediction], path: Path) -> None:
    """Write predictions to JSON or CSV based on file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("w") as fh:
            json.dump([p.to_dict() for p in predictions], fh, indent=2)
    elif suffix == ".csv":
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image", "label_index", "label", "probability", "threshold", "error"],
            )
            writer.writeheader()
            for p in predictions:
                writer.writerow(p.to_dict())
    else:
        raise ValueError(f"Unsupported output extension {suffix!r}. Use .json or .csv.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-infer",
        description=(
            "Run a trained safety-violation classifier on an image or folder of images. "
            "Examples:\n"
            "  safety-infer --checkpoint models/best_model.pt --input path/to/image.jpg\n"
            "  safety-infer --checkpoint models/best_model.pt --input path/to/folder --output preds.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt file.")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional sidecar metadata JSON. Defaults to <checkpoint>.json if present.",
    )
    parser.add_argument("--input", required=True, help="Single image path or folder path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file (.json or .csv). If omitted, predictions are printed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout (otherwise human-readable lines).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    predictions = predict(
        checkpoint=args.checkpoint,
        input_path=args.input,
        metadata=args.metadata,
        device=args.device,
        threshold=args.threshold,
    )

    if args.output:
        _write_predictions(predictions, Path(args.output))
        LOGGER.info("Wrote %d predictions to %s", len(predictions), args.output)
    elif args.json:
        print(json.dumps([p.to_dict() for p in predictions], indent=2))
    else:
        for p in predictions:
            if p.error:
                print(f"{p.image}\tERROR: {p.error}")
            else:
                print(
                    f"{p.image}\tlabel={p.label}\t"
                    f"prob={p.probability:.4f}\tthreshold={p.threshold:.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
