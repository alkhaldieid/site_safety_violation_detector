"""Reproducibility provenance: Python, package versions, hardware, CUDA.

A single :func:`collect_environment_info` call returns a JSON-serialisable
dict suitable for embedding in checkpoint metadata or run reports.

Everything here is best-effort: if a particular probe fails, the value is
recorded as ``None`` rather than raising — this metadata is informational
and should never break a training run.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

# Packages whose versions we care about for reproducibility. Add here when
# adding a new direct dependency.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "numpy",
    "Pillow",
    "PyYAML",
    "tqdm",
)


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _cuda_info() -> dict[str, object]:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return {"available": False}
    info: dict[str, object] = {"available": bool(torch.cuda.is_available())}
    if not info["available"]:
        return info
    try:
        info["device_count"] = int(torch.cuda.device_count())
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
    except Exception:  # pragma: no cover - best-effort
        pass
    return info


def _mps_info() -> dict[str, object]:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return {"available": False}
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return {"available": False}
    try:
        return {"available": bool(mps.is_available()), "built": bool(mps.is_built())}
    except Exception:  # pragma: no cover
        return {"available": False}


def collect_environment_info() -> dict[str, object]:
    """Capture a snapshot of the runtime environment.

    Designed to be embedded into JSON metadata: every value is either a
    JSON-native type or ``None``. Hostnames are included to help track
    which machine produced a checkpoint; remove this from outputs you
    intend to publish if it leaks identifying info.
    """
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "system": platform.system(),
        "hostname": platform.node() or None,
        "cpu_count": os.cpu_count(),
        "package_versions": _package_versions(),
        "cuda": _cuda_info(),
        "mps": _mps_info(),
    }
