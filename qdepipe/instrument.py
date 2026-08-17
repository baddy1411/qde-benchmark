"""Stage-level timing and peak-memory instrumentation (Contract D resource fields).

ADDITIVE and side-effect-free on results: wrapping a stage in ``stage_timer`` or
``StageProfiler`` measures it but does not touch the value the stage computes —
an integration test asserts NRMSE is byte-identical with and without profiling.

Timing uses ``time.perf_counter`` (stdlib, monotonic, no dependency). Peak memory
uses ``psutil`` — already an installed dependency (verified: psutil 7.2.2), so no
new package is introduced. On a platform without psutil the memory field degrades
to ``None`` rather than raising, so instrumentation can never break a run.

Two granularities:
* ``stage_timer(name)``  — a context manager timing one stage; ``.elapsed`` after.
* ``StageProfiler()``    — accumulates many named stages plus a whole-run peak-RSS
                           sampler, and emits the Contract-D ``*_seconds`` /
                           ``peak_memory_mb`` dict in one call.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # pragma: no cover - psutil is a declared dep, guard anyway
    psutil = None
    _PROC = None


def rss_mb() -> Optional[float]:
    """Current resident-set size in MB, or None if psutil is unavailable."""
    if _PROC is None:
        return None
    return _PROC.memory_info().rss / (1024 * 1024)


class _Timer:
    __slots__ = ("name", "_t0", "elapsed")

    def __init__(self, name: str):
        self.name = name
        self._t0 = None
        self.elapsed = 0.0


@contextmanager
def stage_timer(name: str = "stage"):
    """Time a single stage. Usage::

        with stage_timer("feature_generation") as t:
            feats = model.featurize(u)
        print(t.elapsed)   # seconds
    """
    t = _Timer(name)
    t._t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - t._t0


class StageProfiler:
    """Accumulate named stage timings + a background peak-RSS sampler for one run.

    The Contract-D field names map to stage names passed to ``stage``:
    ``data_generation``, ``validation``, ``transformation``, ``feature_generation``,
    ``training``, ``inference``. ``summary()`` emits ``{f"{name}_seconds": ...}``
    plus ``total_seconds`` and ``peak_memory_mb``.

    Peak memory: a lightweight daemon thread samples RSS every ``interval`` seconds
    between ``start()`` and ``stop()``, so the reported peak is a true high-water
    mark across the run, not just the start/end values (which would miss a
    transient spike during, e.g., a large statevector allocation). Sampling is
    best-effort: if psutil is absent the peak is ``None``.
    """

    def __init__(self, sample_interval: float = 0.05):
        self._stages: Dict[str, float] = {}
        self._interval = sample_interval
        self._peak = rss_mb() or 0.0
        self._sampling = False
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None

    # -- peak-memory sampler ------------------------------------------------
    def start(self) -> "StageProfiler":
        self._t0 = time.perf_counter()
        if _PROC is not None:
            self._sampling = True
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self):
        while self._sampling:
            m = rss_mb()
            if m is not None and m > self._peak:
                self._peak = m
            time.sleep(self._interval)

    def stop(self) -> "StageProfiler":
        self._sampling = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # one final reading in case the peak landed after the last sample
        m = rss_mb()
        if m is not None and m > self._peak:
            self._peak = m
        return self

    # -- stage timing -------------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = self._stages.get(name, 0.0) + (time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        """Record a timing measured elsewhere (e.g. via ``stage_timer``)."""
        self._stages[name] = self._stages.get(name, 0.0) + float(seconds)

    # -- output -------------------------------------------------------------
    def summary(self) -> dict:
        out = {f"{k}_seconds": round(v, 6) for k, v in self._stages.items()}
        wall = (time.perf_counter() - self._t0) if self._t0 is not None \
            else sum(self._stages.values())
        out["total_seconds"] = round(wall, 6)
        out["peak_memory_mb"] = (round(self._peak, 3) if _PROC is not None else None)
        return out

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
