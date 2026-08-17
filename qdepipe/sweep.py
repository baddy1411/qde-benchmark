"""``run_sweep`` — the one resume-safe sweep driver.

Consolidates the ``done``-set resume pattern that was independently reimplemented
in six ``run_*.py`` scripts (run_scaling_proof, run_leaky, run_ic_study,
run_lorenz96, run_entanglement, run_n12_attempt) into a single tested utility.

What it adds over the copy-pasted pattern:
* **Partial-failure isolation.** A cell that raises is logged (with its key and
  the exception) and recorded as a failed row, then the sweep CONTINUES to the
  next cell — one bad cell no longer kills a multi-hour unattended run (the gap
  named in DATA_ENGINEERING_AUDIT.md §15).
* **Optional retry.** ``retries>0`` re-attempts a failing cell before giving up.
* **Crash-safe resume.** Completed cells (identified by ``key_fn``) are read from
  the existing result file and skipped, exactly as before — but in one place.

It is storage-agnostic: ``append_fn(record)`` and the ``done`` set are supplied by
the caller, so a sweep can persist to CSV (legacy) or the Parquet registry without
``run_sweep`` knowing which.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional, Sequence

from .logging_setup import get_logger

log = get_logger("qdepipe.sweep")


class SweepResult:
    def __init__(self):
        self.executed = 0
        self.skipped = 0
        self.failed = 0
        self.failures: list[tuple[Any, str]] = []

    def __repr__(self):
        return (f"SweepResult(executed={self.executed}, skipped={self.skipped}, "
                f"failed={self.failed})")


def run_sweep(cells: Iterable[Any], run_one: Callable[[Any], dict], *,
              done: Optional[Sequence] = None,
              key_fn: Callable[[Any], Any] = lambda c: c,
              append_fn: Optional[Callable[[dict], None]] = None,
              on_fail: Optional[Callable[[Any, Exception], None]] = None,
              retries: int = 0, retry_delay: float = 0.0) -> SweepResult:
    """Run ``run_one(cell)`` over ``cells`` with resume + failure isolation.

    Parameters
    ----------
    cells      : iterable of work items (any hashable-via-key_fn value).
    run_one    : cell -> result dict. Raising is caught and isolated.
    done       : keys already completed (skipped). Compared via ``key_fn``.
    key_fn     : cell -> comparison key (default identity).
    append_fn  : called with each successful result dict (persist it). Optional.
    on_fail    : called with (cell, exception) after retries exhausted. Optional
                 (e.g. to append a status="failed" row to the run registry).
    retries    : extra attempts per cell before it is declared failed.
    retry_delay: seconds to wait between attempts.

    Returns a ``SweepResult`` with executed / skipped / failed counts.
    """
    done_set = set(done or [])
    res = SweepResult()
    for cell in cells:
        k = key_fn(cell)
        if k in done_set:
            res.skipped += 1
            continue
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                out = run_one(cell)
                if append_fn is not None and out is not None:
                    append_fn(out)
                res.executed += 1
                done_set.add(k)
                last_exc = None
                break
            except Exception as e:  # isolate: one cell must not kill the sweep
                last_exc = e
                if attempt < retries:
                    log.warning("cell %r failed (attempt %d/%d): %s — retrying",
                                k, attempt + 1, retries + 1, e)
                    if retry_delay:
                        time.sleep(retry_delay)
        if last_exc is not None:
            res.failed += 1
            res.failures.append((k, repr(last_exc)))
            log.error("cell %r FAILED after %d attempt(s): %s",
                      k, retries + 1, last_exc)
            if on_fail is not None:
                try:
                    on_fail(cell, last_exc)
                except Exception as e2:  # a failing failure-handler must not crash
                    log.error("on_fail handler itself raised for %r: %s", k, e2)
    log.info("sweep done: %s", res)
    return res
