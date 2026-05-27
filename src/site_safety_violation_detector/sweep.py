"""Multi-model training sweep, optionally repeated across multiple seeds.

For each architecture in ``cfg.sweep.architectures`` and each seed in
``--seeds`` (default: a single seed taken from ``cfg.runtime.seed``)::

  1. Trains one model on the *same* train/val split.
  2. Records per-run metrics, parameter count, on-disk size, and a rough
     inference-latency estimate.
  3. Aggregates per-architecture mean and std across seeds.

Selection picks the architecture with the best aggregated primary metric
(default ``mean_val_f1``) and promotes the **best individual seed's
checkpoint** of that architecture to ``models/best_model.pt``, with a
sidecar metadata file describing the selection.

When only one seed is used, the aggregated values equal the per-run
values and std is zero — but the report is still written in the same
format so downstream tools don't need a code path per case.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import time
from typing import Any

import torch

from .config import Config, config_to_dict, load_config, save_config
from .data import build_dataloaders
from .models import estimate_inference_latency_ms
from .provenance import collect_environment_info
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

# Metrics that are aggregated across seeds (per architecture). Order is
# stable so the report columns are predictable.
_AGGREGATED_METRICS: tuple[str, ...] = (
    "val_loss",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_roc_auc",
    "val_pr_auc",
    "inference_latency_ms",
    "wall_time_seconds",
)


# ---------------------------------------------------------------------------
# Per-run helpers
# ---------------------------------------------------------------------------


def _cfg_for_run(base: Config, *, architecture: str, seed: int) -> Config:
    cfg = copy.deepcopy(base)
    cfg.model.architecture = architecture
    cfg.runtime.seed = seed
    return cfg


def _per_run_row(
    *,
    architecture: str,
    seed: int,
    artifacts: TrainArtifacts,
    params_total: int,
    params_trainable: int,
    checkpoint_size_bytes: int,
    latency_ms: float,
    wall_time_s: float,
) -> dict[str, Any]:
    m = artifacts.best_val_metrics
    return {
        "architecture": architecture,
        "seed": seed,
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
        "wall_time_seconds": wall_time_s,
        "checkpoint": str(artifacts.checkpoint_path),
        "metadata": str(artifacts.metadata_path),
    }


def _mean_std(values: list[float]) -> tuple[float | None, float | None, int]:
    """Return (mean, std, n) skipping ``None``/NaN entries."""
    clean = [
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    n = len(clean)
    if n == 0:
        return None, None, 0
    mean = sum(clean) / n
    if n == 1:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)  # sample std
    return mean, math.sqrt(var), n


def _aggregate_per_arch(per_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group per-run rows by architecture and compute mean + std for each metric."""
    by_arch: dict[str, list[dict[str, Any]]] = {}
    for row in per_run_rows:
        by_arch.setdefault(row["architecture"], []).append(row)

    aggregated: list[dict[str, Any]] = []
    for arch, runs in by_arch.items():
        record: dict[str, Any] = {
            "architecture": arch,
            "num_seeds": len(runs),
            "seeds": sorted(int(r["seed"]) for r in runs),
            # Params should be identical across seeds; report the first as a static field.
            "params_total": int(runs[0]["params_total"]),
            "params_trainable": int(runs[0]["params_trainable"]),
        }
        for metric in _AGGREGATED_METRICS:
            mean, std, n = _mean_std([r.get(metric) for r in runs])
            record[f"mean_{metric}"] = mean
            record[f"std_{metric}"] = std
            record[f"n_{metric}"] = n
        # Convenience: best individual seed and value for the primary metric is added
        # by the caller (it knows what the primary metric is).
        aggregated.append(record)
    return aggregated


def _is_arch_better(
    candidate: dict, current_best: dict | None, *, primary: str, tie_breaker: str
) -> bool:
    """Compare aggregated records by ``mean_<primary>`` (higher better)
    with ``mean_<tie_breaker>`` resolving ties (``val_loss`` lower-is-better,
    everything else higher-is-better).
    """
    if current_best is None:
        return True
    p_key = f"mean_{primary}"
    t_key = f"mean_{tie_breaker}"
    cand_p, best_p = candidate.get(p_key), current_best.get(p_key)
    if cand_p is None and best_p is not None:
        return False
    if best_p is None and cand_p is not None:
        return True
    if cand_p is not None and best_p is not None and cand_p != best_p:
        return cand_p > best_p
    cand_t, best_t = candidate.get(t_key), current_best.get(t_key)
    if cand_t is None or best_t is None:
        return False
    if tie_breaker == "val_loss":
        return cand_t < best_t
    return cand_t > best_t


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(cfg: Config, *, seeds: list[int] | None = None) -> dict[str, Any]:
    """Run the sweep.

    Parameters
    ----------
    cfg:
        Fully-populated configuration.
    seeds:
        Optional list of integer seeds. When provided, each architecture
        is trained once per seed and per-arch metrics are aggregated as
        ``mean ± std`` across seeds. When ``None`` (or a single-element
        list), the sweep degenerates to one run per architecture.
    """
    seed_list = [int(s) for s in (seeds or [cfg.runtime.seed])]
    if not seed_list:
        raise ValueError("seeds list cannot be empty")
    set_seed(seed_list[0])
    device = resolve_device(cfg.runtime.device)

    # Build the dataloaders once and reuse across all (arch, seed) runs so
    # every candidate sees exactly the same train/val files.
    bundle = build_dataloaders(cfg.data, require_test=False)

    reports_root = ensure_dir(cfg.runtime.output_dir)
    models_dir = ensure_dir(cfg.runtime.models_dir)

    per_run_rows: list[dict[str, Any]] = []
    artifacts_by_key: dict[tuple[str, int], TrainArtifacts] = {}

    primary = cfg.sweep.selection_metric
    tie_breaker = cfg.sweep.tie_breaker

    for arch in cfg.sweep.architectures:
        for seed in seed_list:
            LOGGER.info("=== Sweep run: arch=%s seed=%d ===", arch, seed)
            t_start = time.perf_counter()
            sub_cfg = _cfg_for_run(cfg, architecture=arch, seed=seed)
            artifacts = train_one_model(
                sub_cfg,
                bundle=bundle,
                output_subdir=f"sweep/{arch}/seed{seed}",
                checkpoint_name=f"{arch}_seed{seed}.pt",
                rationale=(
                    f"Trained as part of model sweep on {utc_now_iso()} "
                    f"(arch={arch}, seed={seed})."
                ),
            )
            artifacts_by_key[(arch, seed)] = artifacts

            # Parameter / size / latency measurement on the best weights.
            from .models import build_model  # local import keeps top imports lean

            latency_ms = float("nan")
            measure_model = build_model(sub_cfg.model).to(device).eval()
            try:
                payload = torch.load(artifacts.checkpoint_path, map_location=device)
                measure_model.load_state_dict(payload["model_state_dict"])
                latency_ms = estimate_inference_latency_ms(
                    measure_model,
                    image_size=sub_cfg.data.image_size,
                    device=device,
                    batch_size=min(8, sub_cfg.data.batch_size),
                )
            except Exception as exc:  # pragma: no cover - informational
                LOGGER.warning("Latency measurement failed for %s seed=%d: %s", arch, seed, exc)
            params_total, params_trainable = count_parameters(measure_model)
            del measure_model
            if device == "cuda":
                torch.cuda.empty_cache()

            try:
                ckpt_bytes = artifacts.checkpoint_path.stat().st_size
            except OSError:
                ckpt_bytes = 0

            per_run_rows.append(
                _per_run_row(
                    architecture=arch,
                    seed=seed,
                    artifacts=artifacts,
                    params_total=params_total,
                    params_trainable=params_trainable,
                    checkpoint_size_bytes=ckpt_bytes,
                    latency_ms=latency_ms,
                    wall_time_s=round(time.perf_counter() - t_start, 2),
                )
            )

    if not per_run_rows:
        raise RuntimeError("Sweep produced no runs; check sweep.architectures / seeds.")

    # --- Aggregate per-architecture -------------------------------------
    aggregated = _aggregate_per_arch(per_run_rows)

    # Attach per-arch "best individual seed" stats for the primary metric.
    # This is the seed we'll promote to best_model.pt if its arch wins.
    for record in aggregated:
        arch = record["architecture"]
        candidates = [r for r in per_run_rows if r["architecture"] == arch]
        # Pick the seed with the largest val_<primary> for higher-is-better
        # metrics; the primary metric is f1/accuracy/auc here so higher is better.
        best_run = max(
            candidates,
            key=lambda r: (r.get(primary) is not None, r.get(primary) or float("-inf")),
        )
        record["best_seed"] = int(best_run["seed"])
        record[f"best_seed_{primary}"] = best_run.get(primary)
        record["best_seed_threshold"] = best_run.get("threshold")
        record["best_seed_checkpoint"] = best_run.get("checkpoint")

    # Pick winning arch by aggregated primary metric.
    best_arch_record: dict[str, Any] | None = None
    for record in aggregated:
        if _is_arch_better(record, best_arch_record, primary=primary, tie_breaker=tie_breaker):
            best_arch_record = record
    assert best_arch_record is not None  # per_run_rows non-empty

    best_arch = best_arch_record["architecture"]
    best_seed = best_arch_record["best_seed"]
    best_artifacts = artifacts_by_key[(best_arch, best_seed)]
    best_run_row = next(
        r for r in per_run_rows if r["architecture"] == best_arch and r["seed"] == best_seed
    )

    # --- Write comparison report ----------------------------------------
    report_json = reports_root / f"{cfg.sweep.report_basename}.json"
    report_csv = reports_root / f"{cfg.sweep.report_basename}.csv"
    report_per_run_csv = reports_root / f"{cfg.sweep.report_basename}_per_run.csv"

    multi_seed = len(seed_list) > 1
    rationale = _format_rationale(
        winner=best_arch,
        winner_seed=best_seed,
        record=best_arch_record,
        seeds=seed_list,
        primary=primary,
        tie_breaker=tie_breaker,
        multi_seed=multi_seed,
    )

    summary = {
        "seeds": seed_list,
        "multi_seed": multi_seed,
        "selection_metric": primary,
        "tie_breaker": tie_breaker,
        "winner": best_arch,
        "winner_seed": best_seed,
        "winner_aggregated": best_arch_record,
        "winner_best_run": best_run_row,
        "aggregated": aggregated,
        "per_run": per_run_rows,
        "data_class_names": bundle.class_names,
        "data_class_counts": bundle.train_class_counts,
        "config": config_to_dict(cfg),
        "created_at": utc_now_iso(),
        "git_commit": git_commit(),
        "environment": collect_environment_info(),
        "rationale": rationale,
    }
    with report_json.open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Aggregated CSV (one row per architecture).
    if aggregated:
        with report_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(aggregated[0].keys()))
            writer.writeheader()
            writer.writerows(aggregated)
    # Per-run CSV (one row per (arch, seed)).
    if per_run_rows:
        with report_per_run_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_run_rows)
    LOGGER.info(
        "Wrote comparison reports: %s, %s, %s", report_json, report_csv, report_per_run_csv
    )

    # --- Promote the best individual checkpoint of the winning arch -----
    final_ckpt_path = models_dir / cfg.runtime.checkpoint_name
    final_meta_path = models_dir / cfg.runtime.metadata_name
    shutil.copyfile(best_artifacts.checkpoint_path, final_ckpt_path)

    final_metadata = {
        "selected_architecture": best_arch,
        "selected_seed": best_seed,
        "selection_metric": primary,
        "tie_breaker": tie_breaker,
        "seeds_evaluated": seed_list,
        "multi_seed": multi_seed,
        "winner_aggregated_metrics": best_arch_record,
        "winner_best_run_metrics": best_run_row,
        "all_aggregated": aggregated,
        "all_per_run": per_run_rows,
        "class_names": bundle.class_names,
        "class_to_idx": bundle.class_to_idx,
        "train_class_counts": bundle.train_class_counts,
        "image_size": cfg.data.image_size,
        "threshold": best_run_row["threshold"],
        "config": config_to_dict(cfg),
        "git_commit": git_commit(),
        "created_at": utc_now_iso(),
        "rationale": rationale,
        "source_checkpoint": str(best_artifacts.checkpoint_path),
        "checkpoint": str(final_ckpt_path),
        "environment": collect_environment_info(),
    }
    with final_meta_path.open("w") as fh:
        json.dump(final_metadata, fh, indent=2)

    LOGGER.info(
        "Best: arch=%s seed=%d (%s=%.4f); copied to %s",
        best_arch,
        best_seed,
        primary,
        best_run_row.get(primary) or 0.0,
        final_ckpt_path,
    )
    LOGGER.info("Best metadata: %s", final_meta_path)

    save_config(cfg, reports_root / "sweep_config.yaml")
    return summary


def _format_rationale(
    *,
    winner: str,
    winner_seed: int,
    record: dict[str, Any],
    seeds: list[int],
    primary: str,
    tie_breaker: str,
    multi_seed: bool,
) -> str:
    """Compose a human-readable rationale string for the winning model."""
    mean_v = record.get(f"mean_{primary}")
    std_v = record.get(f"std_{primary}")
    parts: list[str] = []
    if multi_seed:
        parts.append(
            f"Selected {winner} from {len(record.get('seeds', []))} seed(s) per architecture by "
            f"mean_{primary}={mean_v:.4f} ± {std_v:.4f} across seeds={seeds}. "
            f"Tie-breaker: mean_{tie_breaker}={record.get(f'mean_{tie_breaker}'):.4f}. "
            f"Promoted checkpoint is the best individual seed (seed={winner_seed})."
        )
    else:
        parts.append(
            f"Selected {winner} (seed={winner_seed}) by {primary}={mean_v:.4f} "
            f"with tie-breaker {tie_breaker}={record.get(f'mean_{tie_breaker}')}. "
            f"This selection reflects a single training run; rerun with "
            f"`safety-sweep --seeds A B C` to estimate seed variance before "
            f"claiming general superiority."
        )
    parts.append("All candidates trained on identical train/val splits.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-sweep",
        description=(
            "Train candidate architectures and pick the best. "
            "Use --seeds to repeat each architecture under multiple random seeds "
            "and report mean ± std validation metrics."
        ),
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Repeated experiments: train each architecture once per seed and "
            "report mean ± std validation metrics. Example: --seeds 42 123 999. "
            "If omitted, a single run per architecture is performed with "
            "runtime.seed from the config."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    run_sweep(cfg, seeds=args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
