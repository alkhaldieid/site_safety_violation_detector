"""Model factory.

Binary safety-violation classification with a ``timm`` backbone and a
small MLP head. The head returns **logits** (no in-graph sigmoid) so
that training can use :class:`~torch.nn.BCEWithLogitsLoss` for numerical
stability.

Sigmoid is applied externally during metric computation and inference.
"""

from __future__ import annotations

from collections.abc import Iterable

import timm
import torch
import torch.nn as nn

from .config import ModelConfig

# Models that are *intended* to work via the standard timm `num_classes=0`
# feature-extractor interface in this project. Users may pass any timm
# model name; this list is just a convenience for the sweep.
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "efficientnet_b0",
    "efficientnet_b1",
    "resnet18",
    "resnet50",
    "mobilenetv3_large_100",
    "convnext_tiny",
    "convnext_small",
)


class SafetyClassifier(nn.Module):
    """Binary safety-violation classifier returning raw logits.

    The backbone is any ``timm`` model created with ``num_classes=0``,
    which exposes a feature vector of length ``backbone.num_features``.
    A small MLP head maps that feature vector to a single logit.

    BatchNorm in the head requires batch sizes >= 2. The head therefore
    falls back to LayerNorm when ``hidden_dim < 1`` is requested
    (i.e. user disables the hidden layer entirely) so single-image
    inference remains stable in either case.
    """

    def __init__(
        self,
        architecture: str,
        *,
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        # Some timm models (e.g. mobilenetv3, convnext) expose a `num_features`
        # value that does not match the actual feature vector emitted by the
        # backbone once the conv-head expansion runs. Determine the real
        # feature size via a dummy forward — this is robust across architectures
        # and timm versions.
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.backbone(dummy)
            if out.dim() != 2:
                raise RuntimeError(
                    f"Expected the backbone to emit a 2D (B, C) feature vector "
                    f"(set global_pool='avg' and num_classes=0); got shape {tuple(out.shape)}."
                )
            in_features = int(out.shape[-1])

        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                # LayerNorm is safe at batch_size == 1 (unlike BatchNorm1d),
                # which matters for single-image inference.
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits  # shape: (B, 1)


def build_model(cfg: ModelConfig) -> SafetyClassifier:
    """Build a :class:`SafetyClassifier` from a :class:`ModelConfig`."""
    return SafetyClassifier(
        architecture=cfg.architecture,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        hidden_dim=cfg.hidden_dim,
    )


def list_supported_architectures() -> Iterable[str]:
    return SUPPORTED_ARCHITECTURES


def estimate_inference_latency_ms(
    model: nn.Module,
    *,
    image_size: int,
    device: str,
    batch_size: int = 8,
    repeats: int = 5,
) -> float:
    """Time a few forward passes and return mean ms/batch.

    Returns ``float('nan')`` on any failure (the metric is informational
    and shouldn't break a sweep).
    """
    import time

    try:
        model.eval()
        dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)
        with torch.no_grad():
            for _ in range(2):  # warmup
                model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        return ((t1 - t0) / repeats) * 1000.0
    except Exception:  # pragma: no cover - informational only
        return float("nan")
