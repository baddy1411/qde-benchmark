"""Gate (b) for the finite-shot wiring: the ENTIRE finite_shot_sweep output is
byte-identical with the cache OFF vs ON. Also documents, executably, the two
findings that shaped this wiring:

  * ~0 intra-run cache hits (shots are baked into featurize -> distinct keys), so
    the cache's honest value here is "re-runs are free", NOT within-run speedup.
  * a second run against a warm store is 100% hits and still byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from qdepipe import concentration as C
from qdepipe.feature_store import FeatureStore


@pytest.fixture
def small_npoints(monkeypatch):
    # keep the real circuit (4 qubits) but shrink the series so the test is quick;
    # byte-identity is independent of series length.
    monkeypatch.setitem(C.NPOINTS, 4, 200)


def _rows_equal(a, b):
    assert a == b, "cache changed the finite-shot output (gate b violation)"


def test_gate_b_finite_shot_byte_identical(tmp_path, small_npoints):
    quiet = lambda *a, **k: None
    args = dict(system="henon", seeds=(0, 1), n_list=(4,), log=quiet)

    off = C.finite_shot_sweep(store=None, **args)              # cache OFF (current behavior)

    store = FeatureStore(root=tmp_path, namespace="fs")
    on = C.finite_shot_sweep(store=store, **args)              # cache ON, cold -> all misses

    # GATE (b): byte-identical, cache on vs off
    _rows_equal(off, on)

    # finding 1: within one sweep, every (shots,seed) is a distinct key -> 0 hits.
    # 1 n x 3 shots x 2 seeds = 6 featurize calls, all unique.
    assert store.stats["hits"] == 0
    assert store.stats["misses"] == 6
    assert store.stats["entries"] == 6

    # finding 2: re-run against the warm store is all hits, and STILL byte-identical.
    on2 = C.finite_shot_sweep(store=store, **args)
    _rows_equal(off, on2)
    assert store.stats["hits"] == 6                            # re-run reuse: 100% hits


def test_default_off_is_unchanged(tmp_path, small_npoints):
    """Two cache-off runs are identical (the wiring added no nondeterminism)."""
    quiet = lambda *a, **k: None
    a = C.finite_shot_sweep(system="henon", seeds=(0,), n_list=(4,), log=quiet, store=None)
    b = C.finite_shot_sweep(system="henon", seeds=(0,), n_list=(4,), log=quiet, store=None)
    assert a == b
