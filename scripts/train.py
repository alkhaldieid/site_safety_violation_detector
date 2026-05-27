#!/usr/bin/env python
"""CLI wrapper: train a single safety-violation classifier.

Equivalent to running ``safety-train`` after ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a fresh checkout without installation.
_PKG_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from site_safety_violation_detector.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
