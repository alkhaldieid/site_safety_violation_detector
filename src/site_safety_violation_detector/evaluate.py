"""Evaluation against the held-out test split.

Loads a saved checkpoint (and its sidecar metadata when present) and:
  * Evaluates on the test set
  * Optionally tunes a probability threshold on the validation set first
  * Saves a classification report, confusion-matrix data, and the metrics JSON
    under ``reports/<architecture>_test/``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, load_config
from .data import build_dataloaders
from .metrics import compute_binary_metrics, find_best_threshold
from .models import SafetyClassifier
from .utils import ensure_dir, get_logger, resolve_device, set_seed, utc_now_iso

LOGGER = get_logger(__name__)


@dataclass
class LoadedModel:
    model: SafetyClassifier
    class_names: list[str]
    image_size: int
    threshold: float
    metadata: dict[str, Any]
    architecture: str


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
    metadata_path: str | Path | None = None,
) -> LoadedModel:
    """Load a checkpoint + sidecar metadata into a ready-to-evaluate model.

    Falls back to defaults from ``cfg`` when fields are missing in the
    checkpoint (this can happen with very old checkpoints).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} is not in the expected format "
            "(missing 'model_state_dict'). This package writes checkpoints with: "
            "{architecture, model_state_dict, class_names, image_size, threshold, epoch}."
        )

    # Resolve metadata.
    # We accept two sidecar conventions:
    #   * <checkpoint_stem>.json          (written by single-model training)
    #   * <checkpoint_stem>_metadata.json (written by the sweep when promoting
    #                                       the winning checkpoint as best_model.pt)
    meta: dict[str, Any] = {}
    if metadata_path is None:
        for candidate in (
            ckpt_path.with_suffix(".json"),
            ckpt_path.with_name(ckpt_path.stem + "_metadata.json"),
        ):
            if candidate.is_file():
                metadata_path = candidate
                break
    if metadata_path is not None:
        with Path(metadata_path).open("r") as fh:
            meta = json.load(fh)

    architecture = payload.get("architecture") or meta.get("architecture")
    if architecture is None:
        raise RuntimeError(
            f"Cannot determine architecture from checkpoint {ckpt_path} or metadata. "
            "Pass --metadata pointing at the matching .json or retrain with this package."
        )

    class_names = payload.get("class_names") or meta.get("class_names") or ["class_0", "class_1"]
    image_size = int(payload.get("image_size") or meta.get("image_size") or 224)
    threshold = float(payload.get("threshold") if payload.get("threshold") is not None else meta.get("threshold", 0.5))

    model_cfg = (meta.get("config") or {}).get("model", {}) if meta else {}
    model = SafetyClassifier(
        architecture=architecture,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.5)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    return LoadedModel(
        model=model,
        class_names=list(class_names),
        image_size=image_size,
        threshold=threshold,
        metadata=meta,
        architecture=architecture,
    )


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []
    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        y_prob_chunks.append(probs)
        y_true_chunks.append(labels.numpy().reshape(-1))
    return np.concatenate(y_true_chunks), np.concatenate(y_prob_chunks)


def evaluate_checkpoint(
    cfg: Config,
    *,
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on the test split.

    Returns a metrics dictionary and writes:
      * ``test_metrics.json``     — full metrics + tuned threshold
      * ``classification_report.txt``
      * ``confusion_matrix.json``
    """
    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)

    loaded = load_checkpoint(checkpoint_path, device=device, metadata_path=metadata_path)

    # Use the checkpoint's image_size if the config disagrees — the model was trained at that size.
    if cfg.data.image_size != loaded.image_size:
        LOGGER.warning(
            "Overriding cfg.data.image_size (%d) with checkpoint image_size (%d)",
            cfg.data.image_size,
            loaded.image_size,
        )
        cfg.data.image_size = loaded.image_size

    bundle = build_dataloaders(cfg.data, require_test=True)
    if bundle.test_loader is None:
        raise RuntimeError("Test loader unavailable — ensure the test split exists.")

    # Optional threshold search on validation set.
    threshold = loaded.threshold
    if cfg.eval.search_threshold_on_val:
        val_y, val_p = _collect_predictions(loaded.model, bundle.val_loader, device=device, desc="val [threshold search]")
        threshold, val_f1 = find_best_threshold(val_y, val_p, metric="f1")
        LOGGER.info("Threshold tuned on validation: %.3f (val_f1=%.4f)", threshold, val_f1)
    else:
        LOGGER.info("Using stored threshold: %.3f", threshold)

    test_y, test_p = _collect_predictions(loaded.model, bundle.test_loader, device=device, desc="test")
    test_metrics = compute_binary_metrics(test_y, test_p, threshold=threshold)

    out_dir = ensure_dir(
        output_dir
        or (Path(cfg.runtime.output_dir) / f"{loaded.architecture}_test")
    )

    y_pred = (test_p >= threshold).astype(int)
    report_text = classification_report(
        test_y.astype(int),
        y_pred,
        target_names=loaded.class_names,
        zero_division=0,
        digits=4,
    )

    summary = {
        "architecture": loaded.architecture,
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "test_metrics": test_metrics.to_dict(),
        "class_names": loaded.class_names,
        "evaluated_at": utc_now_iso(),
    }
    with (out_dir / "test_metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    (out_dir / "classification_report.txt").write_text(report_text)
    with (out_dir / "confusion_matrix.json").open("w") as fh:
        json.dump(
            {
                "labels": [0, 1],
                "class_names": loaded.class_names,
                "matrix": [
                    [test_metrics.tn, test_metrics.fp],
                    [test_metrics.fn, test_metrics.tp],
                ],
            },
            fh,
            indent=2,
        )

    LOGGER.info("Test metrics: %s", json.dumps(test_metrics.to_dict(), indent=2))
    LOGGER.info("Wrote evaluation artifacts to %s", out_dir)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-evaluate",
        description="Evaluate a saved checkpoint on the test split.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint produced by this package (e.g. models/best_model.pt).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional explicit path to the sidecar metadata JSON. "
             "Defaults to <checkpoint>.json if present.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts. Defaults to reports/<arch>_test/.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config field, e.g. --set data.batch_size=64.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    evaluate_checkpoint(
        cfg,
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
