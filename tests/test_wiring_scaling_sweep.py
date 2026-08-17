"""Gate (b) for path 1 (scaling_sweep): whole output byte-identical cache off vs on.

Also proves the cross-path reuse property: scaling_sweep's V=4 exact featurize and
finite_shot_sweep's exact case share a cache key, so a shared store hits across the
two passes.
"""
from __future__ import annotations

import pytest

from qdepipe import concentration as C
from qdepipe.feature_store import FeatureStore


@pytest.fixture
def small_npoints(monkeypatch):
    monkeypatch.setitem(C.NPOINTS, 4, 200)


def test_gate_b_scaling_sweep_byte_identical(tmp_path, small_npoints):
    quiet = lambda *a, **k: None
    args = dict(system="henon", seeds=(0, 1), n_list=(4,), log=quiet)

    off = C.scaling_sweep(store=None, **args)
    store = FeatureStore(root=tmp_path, namespace="ss")
    on = C.scaling_sweep(store=store, **args)

    assert off == on, "cache changed scaling_sweep output (gate b violation)"

    # after path 2, scaling_sweep caches BOTH featurizes per (n,seed): the V=4
    # (line 125) and the V=1 inside observable_variance. 1 n x 2 seeds x 2 = 4.
    assert store.stats["misses"] == 4 and store.stats["hits"] == 0

    on2 = C.scaling_sweep(store=store, **args)        # warm re-run = hits, still identical
    assert off == on2
    assert store.stats["hits"] == 4


def test_cross_path_reuse_with_finite_shot(tmp_path, small_npoints):
    """A store warmed by finite_shot_sweep's exact case serves scaling_sweep's
    V=4 featurize as hits — same matrix, same key, across two passes."""
    quiet = lambda *a, **k: None
    store = FeatureStore(root=tmp_path, namespace="shared")
    # warm with finite-shot (its exact case computes the V=4 exact matrices)
    C.finite_shot_sweep(system="henon", seeds=(0, 1), n_list=(4,), log=quiet, store=store)
    hits_before = store.stats["hits"]
    # scaling_sweep's line-125 featurize should now hit those exact matrices
    C.scaling_sweep(system="henon", seeds=(0, 1), n_list=(4,), log=quiet, store=store)
    assert store.stats["hits"] > hits_before          # cross-path cache hits are real
