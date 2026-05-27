#!/usr/bin/env python
"""CLI wrapper: run inference with a saved checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from site_safety_violation_detector.infer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
