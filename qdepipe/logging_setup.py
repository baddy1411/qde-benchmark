"""Structured logging — stdlib ``logging``, no third-party dependency.

Replaces bare ``print(...)`` with leveled, timestamped, dual-sink (console + a
per-run file under ``logs/``) logging. Deliberately uses only the standard
library: at this project's scale, ``structlog``/``loguru`` would be a dependency
added for appearance, not need.

Usage::

    from qdepipe.logging_setup import configure, get_logger
    configure(run_name="scaling_proof")     # once, at script start
    log = get_logger(__name__)
    log.info("n=%d %s seed%d NRMSE=%.4e", n, system, seed, nrmse)

``configure`` is idempotent (safe to call repeatedly). Existing ``print``-based
scripts keep working unchanged; this is opt-in for new/updated scripts.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_CONFIGURED = False


class _Fmt(logging.Formatter):
    """ISO-ish timestamp, level, logger name, message — one line, greppable."""
    default_msec_format = "%s.%03d"

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")


def configure(run_name: Optional[str] = None, level: int = logging.INFO,
              to_file: bool = True, run_timestamp: Optional[str] = None) -> Optional[Path]:
    """Configure root logging once. Returns the log-file path (or None).

    ``run_timestamp`` lets a caller pass a deterministic stamp (the environment
    forbids ``Date.now``-style calls in some contexts); if omitted a real
    timestamp is used for the filename only (never for any recorded result).
    """
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level)
    # clear our own handlers on reconfigure (idempotent), leave others alone
    for h in list(root.handlers):
        if getattr(h, "_qde", False):
            root.removeHandler(h)

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(_Fmt()); ch._qde = True
    root.addHandler(ch)

    log_path = None
    if to_file and run_name:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if run_timestamp is None:
            import time
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{run_name}_{run_timestamp}.log"
        fh = logging.FileHandler(log_path)
        fh.setFormatter(_Fmt()); fh._qde = True
        root.addHandler(fh)

    _CONFIGURED = True
    return log_path


def get_logger(name: str = "qdepipe") -> logging.Logger:
    if not _CONFIGURED:
        configure()
    return logging.getLogger(name)
