"""Single-model training loop and CLI entrypoint.

Key design decisions:

* The model returns **logits**; the loss is :class:`torch.nn.BCEWithLogitsLoss`
  for numerical stability.
* Sigmoid is applied only when computing probabilities for metrics or
  inference.
* Early stopping watches a configurable validation metric
  (``val_f1`` by default; falls back to ``val_loss``).
* A checkpoint is saved whenever the watched metric improves;
  metadata (config, class names, metrics, git commit, timestamp) sits
  next to it.
* Training history is exported as JSON for downstream analysis or
  plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, OptimConfig, config_to_dict, load_config, save_config
from .data import DataBundle, build_dataloaders, compute_positive_class_weight
from .metrics import BinaryMetrics, compute_binary_metrics, find_best_threshold
from .models import build_model
from .utils import (
    count_parameters,
    ensure_dir,
    get_logger,
    git_commit,
    resolve_device,
    set_seed,
    utc_now_iso,
)

LOGGER = get_logger(__name__)

EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
    *,
    num_epochs: int,
):
    name = (cfg.scheduler or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    if name == "step":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=cfg.gamma, patience=2)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: BinaryMetrics
    learning_rate: float

    def to_history_row(self) -> dict[str, Any]:
        m = self.val_metrics
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": m.accuracy,
            "val_precision": m.precision,
            "val_recall": m.recall,
            "val_f1": m.f1,
            "val_roc_auc": m.roc_auc,
            "val_pr_auc": m.pr_auc,
            "learning_rate": self.learning_rate,
        }


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    grad_clip_norm: float | None,
    show_progress: bool,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch (training if optimizer is given, else evaluation).

    Returns ``(mean_loss, y_true, y_prob)``.
    """
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_samples = 0
    y_true_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []

    iterator = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for inputs, labels in iterator:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1)
            y_prob_chunks.append(probs)
            y_true_chunks.append(labels.detach().cpu().numpy().reshape(-1))

    mean_loss = total_loss / max(1, total_samples)
    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.empty(0)
    y_prob = np.concatenate(y_prob_chunks) if y_prob_chunks else np.empty(0)
    return mean_loss, y_true, y_prob


def _watched_value(metric_name: str, val_loss: float, val_metrics: BinaryMetrics) -> tuple[float, bool]:
    """Return ``(value, higher_is_better)`` for the early-stopping watcher."""
    name = metric_name.lower()
    if name == "val_loss":
        return val_loss, False
    if name == "val_f1":
        return val_metrics.f1, True
    if name == "val_pr_auc":
        # When PR-AUC is undefined (single-class val), fall back to F1.
        return (val_metrics.pr_auc if val_metrics.pr_auc is not None else val_metrics.f1), True
    raise ValueError(f"Unknown early_stopping_metric: {metric_name!r}")


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class TrainArtifacts:
    """Paths and summary produced by a training run."""

    checkpoint_path: Path
    metadata_path: Path
    history_path: Path
    final_metrics_path: Path
    best_epoch: int
    best_val_metrics: BinaryMetrics
    best_val_loss: float
    history: list[dict[str, Any]]


def train_one_model(
    cfg: Config,
    *,
    bundle: DataBundle | None = None,
    output_subdir: str | None = None,
    checkpoint_name: str | None = None,
    rationale: str | None = None,
) -> TrainArtifacts:
    """Train a single model end-to-end and persist artifacts.

    Parameters
    ----------
    cfg:
        Fully-populated configuration.
    bundle:
        Pre-built :class:`DataBundle`. If ``None``, one is constructed
        from ``cfg.data``. The sweep passes a single shared bundle so
        candidate models see identical splits.
    output_subdir:
        Subdirectory under ``cfg.runtime.output_dir`` to store the
        history / final-metrics files. Defaults to the architecture name.
    checkpoint_name:
        Override checkpoint filename (without directory). Defaults to
        ``f"{architecture}.pt"`` to avoid colliding sweep runs.
    rationale:
        Optional string saved alongside metadata explaining why this run
        exists (e.g. "selected by sweep on val_f1").
    """
    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    LOGGER.info(
        "Training %s (device=%s, seed=%d, epochs=%d)",
        cfg.model.architecture,
        device,
        cfg.runtime.seed,
        cfg.train.epochs,
    )

    # Data ------------------------------------------------------------
    if bundle is None:
        bundle = build_dataloaders(cfg.data, require_test=False)
    class_names = bundle.class_names
    if len(class_names) != 2:
        raise RuntimeError(
            f"Binary classifier expects 2 classes, found {len(class_names)}: {class_names}"
        )

    # Model -----------------------------------------------------------
    model = build_model(cfg.model).to(device)
    total_params, trainable_params = count_parameters(model)
    LOGGER.info(
        "Model %s — params: %d total, %d trainable",
        cfg.model.architecture,
        total_params,
        trainable_params,
    )

    # Loss / optimiser / scheduler ------------------------------------
    pos_weight: torch.Tensor | None = None
    if cfg.train.use_class_weights:
        weight = compute_positive_class_weight(
            bundle.train_class_counts, class_names=class_names
        )
        pos_weight = torch.tensor([weight], device=device, dtype=torch.float32)
        LOGGER.info("Applying BCEWithLogitsLoss pos_weight=%.3f", weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, num_epochs=cfg.train.epochs)

    # Output paths ----------------------------------------------------
    reports_root = ensure_dir(cfg.runtime.output_dir)
    sub = output_subdir or cfg.model.architecture
    run_dir = ensure_dir(reports_root / sub)
    models_dir = ensure_dir(cfg.runtime.models_dir)
    ckpt_name = checkpoint_name or f"{cfg.model.architecture}.pt"
    checkpoint_path = models_dir / ckpt_name
    metadata_path = checkpoint_path.with_suffix(".json")
    history_json_path = run_dir / "history.json"
    history_csv_path = run_dir / "history.csv"
    final_metrics_path = run_dir / "final_metrics.json"

    # Training loop ---------------------------------------------------
    history: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = 0
    best_val_metrics: BinaryMetrics | None = None
    best_val_loss: float = math.inf
    patience_left = cfg.train.early_stopping_patience

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=cfg.train.grad_clip_norm,
            show_progress=True,
            desc=f"epoch {epoch}/{cfg.train.epochs} [train]",
        )
        val_loss, val_y, val_p = _run_epoch(
            model,
            bundle.val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            grad_clip_norm=None,
            show_progress=True,
            desc=f"epoch {epoch}/{cfg.train.epochs} [val]",
        )
        val_metrics = compute_binary_metrics(val_y, val_p, threshold=cfg.eval.threshold)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            learning_rate=current_lr,
        )
        history.append(result.to_history_row())
        LOGGER.info(
            "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | val_acc=%.4f | lr=%.2e",
            epoch,
            cfg.train.epochs,
            train_loss,
            val_loss,
            val_metrics.f1,
            val_metrics.accuracy,
            current_lr,
        )

        value, higher_is_better = _watched_value(
            cfg.train.early_stopping_metric, val_loss, val_metrics
        )
        improved = (
            best_value is None
            or (higher_is_better and value > best_value + EPS)
            or (not higher_is_better and value < best_value - EPS)
        )
        if improved:
            best_value = value
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_loss = val_loss
            patience_left = cfg.train.early_stopping_patience
            torch.save(
                {
                    "architecture": cfg.model.architecture,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": cfg.data.image_size,
                    "threshold": cfg.eval.threshold,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            LOGGER.info("  -> new best (%s=%.4f), saved %s", cfg.train.early_stopping_metric, value, checkpoint_path)
        else:
            patience_left -= 1
            LOGGER.info("  -> no improvement (patience left: %d)", patience_left)
            if patience_left <= 0:
                LOGGER.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_val_metrics is None:
        raise RuntimeError("Training produced no epochs — was the train loader empty?")

    # Persist history -------------------------------------------------
    with history_json_path.open("w") as fh:
        json.dump(history, fh, indent=2)
    if history:
        with history_csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # Optional threshold tuning on validation set ---------------------
    tuned_threshold: float | None = None
    if cfg.eval.search_threshold_on_val:
        # Re-evaluate best checkpoint to get a fresh val probability vector.
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        _, vy, vp = _run_epoch(
            model,
            bundle.val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            grad_clip_norm=None,
            show_progress=False,
            desc="threshold search",
        )
        tuned_threshold, _ = find_best_threshold(vy, vp, metric="f1")
        best_val_metrics = compute_binary_metrics(vy, vp, threshold=tuned_threshold)
        LOGGER.info("Tuned validation threshold: %.3f (val_f1=%.4f)", tuned_threshold, best_val_metrics.f1)

    final_threshold = tuned_threshold if tuned_threshold is not None else cfg.eval.threshold
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics.to_dict(),
        "tuned_threshold": tuned_threshold,
    }
    with final_metrics_path.open("w") as fh:
        json.dump(final_metrics, fh, indent=2)

    # Metadata --------------------------------------------------------
    metadata = {
        "architecture": cfg.model.architecture,
        "class_names": class_names,
        "class_to_idx": bundle.class_to_idx,
        "image_size": cfg.data.image_size,
        "threshold": final_threshold,
        "best_epoch": best_epoch,
        "validation_metrics": best_val_metrics.to_dict(),
        "params_total": total_params,
        "params_trainable": trainable_params,
        "config": config_to_dict(cfg),
        "git_commit": git_commit(),
        "created_at": utc_now_iso(),
        "rationale": rationale,
        "checkpoint": str(checkpoint_path),
    }
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)

    # Persist the resolved config next to the run for reproducibility.
    save_config(cfg, run_dir / "config.yaml")

    return TrainArtifacts(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        history_path=history_json_path,
        final_metrics_path=final_metrics_path,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        best_val_loss=best_val_loss,
        history=history,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-train",
        description="Train a single safety-violation classifier.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="section.field=value",
        help="Override a config field. Repeatable, e.g. --set train.epochs=5 --set optim.learning_rate=3e-4.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Override the checkpoint filename under models/. Defaults to '<architecture>.pt'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    train_one_model(cfg, checkpoint_name=args.checkpoint_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
