"""Multi-model training sweep and best-model selection.

For each architecture listed in ``cfg.sweep.architectures``, this module:

  1. Trains the model on the *same* train/val split.
  2. Records per-model metrics, parameter count, model size on disk, and
     an inference latency estimate.
  3. Aggregates the results into ``reports/<basename>.{json,csv}``.
  4. Selects the winning checkpoint by ``cfg.sweep.selection_metric``
     (validation F1 by default), with ``cfg.sweep.tie_breaker``
     resolving ties (validation loss by default — lower is better).
  5. Copies the winning checkpoint to ``models/best_model.pt`` and
     writes ``models/best_model_metadata.json`` with full rationale.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import time
from typing import Any

import torch

from .config import Config, config_to_dict, load_config, save_config
from .data import build_dataloaders
from .models import estimate_inference_latency_ms
from .train import TrainArtifacts, train_one_model
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


# ---------------------------------------------------------------------------
# Per-architecture cfg derivation
# ---------------------------------------------------------------------------


def _cfg_for_architecture(base: Config, architecture: str) -> Config:
    """Clone the base config and pin the architecture."""
    cfg = copy.deepcopy(base)
    cfg.model.architecture = architecture
    return cfg


def _record_row(
    arch: str,
    artifacts: TrainArtifacts,
    *,
    params_total: int,
    params_trainable: int,
    checkpoint_size_bytes: int,
    latency_ms: float,
) -> dict[str, Any]:
    m = artifacts.best_val_metrics
    return {
        "architecture": arch,
        "best_epoch": artifacts.best_epoch,
        "val_loss": artifacts.best_val_loss,
        "val_accuracy": m.accuracy,
        "val_precision": m.precision,
        "val_recall": m.recall,
        "val_f1": m.f1,
        "val_roc_auc": m.roc_auc,
        "val_pr_auc": m.pr_auc,
        "threshold": m.threshold,
        "params_total": params_total,
        "params_trainable": params_trainable,
        "checkpoint_bytes": checkpoint_size_bytes,
        "inference_latency_ms": latency_ms,
        "checkpoint": str(artifacts.checkpoint_path),
        "metadata": str(artifacts.metadata_path),
    }


def _is_better(candidate: dict, current_best: dict, *, primary: str, tie_breaker: str) -> bool:
    """Return True if ``candidate`` should replace ``current_best``."""
    if current_best is None:
        return True
    cand_primary = candidate.get(primary)
    best_primary = current_best.get(primary)
    if cand_primary is None and best_primary is None:
        pass  # fall through to tie-breaker
    elif cand_primary is None:
        return False
    elif best_primary is None:
        return True
    elif cand_primary != best_primary:
        # Higher is better for f1 / accuracy / roc_auc / pr_auc.
        # We assume primary is one of those (val_loss is reserved for tie-breaking).
        return cand_primary > best_primary

    # Tie or both None — apply tie-breaker.
    cand_tb = candidate.get(tie_breaker)
    best_tb = current_best.get(tie_breaker)
    if cand_tb is None or best_tb is None:
        return False
    if tie_breaker == "val_loss":
        return cand_tb < best_tb
    return cand_tb > best_tb


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(cfg: Config) -> dict[str, Any]:
    """Train each architecture, select the best, and persist the comparison.

    Returns the comparison summary dict (also written to disk).
    """
    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)

    # Build the dataloaders once and reuse across architectures so that
    # every candidate sees exactly the same train/val split.
    bundle = build_dataloaders(cfg.data, require_test=False)

    reports_root = ensure_dir(cfg.runtime.output_dir)
    models_dir = ensure_dir(cfg.runtime.models_dir)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_artifacts: TrainArtifacts | None = None
    best_arch: str | None = None

    primary = cfg.sweep.selection_metric
    tie_breaker = cfg.sweep.tie_breaker

    for arch in cfg.sweep.architectures:
        LOGGER.info("=== Sweep: training %s ===", arch)
        t_start = time.perf_counter()
        sub_cfg = _cfg_for_architecture(cfg, arch)
        artifacts = train_one_model(
            sub_cfg,
            bundle=bundle,
            output_subdir=f"sweep/{arch}",
            checkpoint_name=f"{arch}.pt",
            rationale=f"Trained as part of model sweep on {utc_now_iso()}.",
        )

        # Latency / parameter / size measurements.
        from .models import build_model  # local import keeps top of file lean

        latency_ms = float("nan")
        model_for_measure = build_model(sub_cfg.model).to(device).eval()
        # Load best weights for a representative measurement.
        try:
            payload = torch.load(artifacts.checkpoint_path, map_location=device)
            model_for_measure.load_state_dict(payload["model_state_dict"])
            latency_ms = estimate_inference_latency_ms(
                model_for_measure,
                image_size=sub_cfg.data.image_size,
                device=device,
                batch_size=min(8, sub_cfg.data.batch_size),
            )
        except Exception as exc:  # pragma: no cover - informational only
            LOGGER.warning("Latency measurement failed for %s: %s", arch, exc)
        params_total, params_trainable = count_parameters(model_for_measure)
        del model_for_measure
        if device == "cuda":
            torch.cuda.empty_cache()

        try:
            ckpt_bytes = artifacts.checkpoint_path.stat().st_size
        except OSError:
            ckpt_bytes = 0

        row = _record_row(
            arch,
            artifacts,
            params_total=params_total,
            params_trainable=params_trainable,
            checkpoint_size_bytes=ckpt_bytes,
            latency_ms=latency_ms,
        )
        row["wall_time_seconds"] = round(time.perf_counter() - t_start, 2)
        rows.append(row)

        if _is_better(row, best_row, primary=primary, tie_breaker=tie_breaker):
            best_row = row
            best_artifacts = artifacts
            best_arch = arch
            LOGGER.info("New leader: %s (%s=%.4f)", arch, primary, row.get(primary) or 0.0)

    if best_row is None or best_artifacts is None or best_arch is None:
        raise RuntimeError("Sweep produced no candidates; check sweep.architectures.")

    # --- Write comparison report ----------------------------------------
    report_json = reports_root / f"{cfg.sweep.report_basename}.json"
    report_csv = reports_root / f"{cfg.sweep.report_basename}.csv"
    summary = {
        "candidates": rows,
        "selection_metric": primary,
        "tie_breaker": tie_breaker,
        "winner": best_arch,
        "winner_metrics": best_row,
        "data_class_names": bundle.class_names,
        "data_class_counts": bundle.train_class_counts,
        "config": config_to_dict(cfg),
        "created_at": utc_now_iso(),
        "git_commit": git_commit(),
    }
    with report_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    if rows:
        with report_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    LOGGER.info("Wrote comparison report to %s and %s", report_json, report_csv)

    # --- Promote the winning checkpoint ---------------------------------
    final_ckpt_path = models_dir / cfg.runtime.checkpoint_name
    final_meta_path = models_dir / cfg.runtime.metadata_name
    shutil.copyfile(best_artifacts.checkpoint_path, final_ckpt_path)

    rationale = (
        f"Selected {best_arch} from {len(rows)} candidates by {primary}="
        f"{best_row.get(primary):.4f} on the validation split, "
        f"with {tie_breaker}={best_row.get(tie_breaker)} resolving ties. "
        "All candidates trained on identical train/val splits."
    )
    final_metadata = {
        "selected_architecture": best_arch,
        "selection_metric": primary,
        "tie_breaker": tie_breaker,
        "winner_metrics": best_row,
        "all_candidates": rows,
        "class_names": bundle.class_names,
        "class_to_idx": bundle.class_to_idx,
        "image_size": cfg.data.image_size,
        "threshold": best_row["threshold"],
        "config": config_to_dict(cfg),
        "git_commit": git_commit(),
        "created_at": utc_now_iso(),
        "rationale": rationale,
        "source_checkpoint": str(best_artifacts.checkpoint_path),
        "checkpoint": str(final_ckpt_path),
    }
    with final_meta_path.open("w") as fh:
        json.dump(final_metadata, fh, indent=2)

    LOGGER.info("Best model: %s — copied to %s", best_arch, final_ckpt_path)
    LOGGER.info("Best metadata: %s", final_meta_path)

    save_config(cfg, reports_root / "sweep_config.yaml")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-sweep",
        description="Train multiple candidate architectures and pick the best.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_sweep.yaml",
        help="Path to a sweep YAML config (default: configs/model_sweep.yaml).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config field, e.g. --set train.epochs=3 (repeatable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    run_sweep(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
