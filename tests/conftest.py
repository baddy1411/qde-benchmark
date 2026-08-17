"""Test bootstrap.

A few wiring tests import an experiment script directly, to check that the
script and the library agree rather than testing the library alone. The
experiment scripts live in experiments/, which is not an importable package, so
put the repository root and experiments/ on sys.path here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
