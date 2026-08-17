"""Gate (b) for path 2 (observable_variance): its full output (the variance dict)
is byte-identical cache off vs on. Plus the cfg-required guard that prevents a
context-blind key (the inverted data-context check for a u-as-parameter function).
"""
from __future__ import annotations

import numpy as np
import pytest

from qdepipe import concentration as C
from qdepipe.experiment import ExperimentConfig
from qdepipe.feature_store import FeatureStore


def _cfg(npts):
    # mirror _scaled_series exactly (every production caller uses this recipe)
    return ExperimentConfig(system="henon", n_points=npts, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))


def test_gate_b_observable_variance_byte_identical(tmp_path):
    u, _ = C._scaled_series("henon", 200)
    off = C.observable_variance(u, 4, seed=0)                                  # cache OFF
    store = FeatureStore(root=tmp_path, namespace="ov")
    on = C.observable_variance(u, 4, seed=0, store=store, cfg=_cfg(200))       # miss
    on2 = C.observable_variance(u, 4, seed=0, store=store, cfg=_cfg(200))      # hit

    for k in off:
        a, b, c = off[k], on[k], on2[k]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b) and np.array_equal(a, c), k
        else:
            assert a == b == c, k
    assert store.stats["misses"] == 1 and store.stats["hits"] == 1


def test_caching_without_cfg_is_a_hard_error(tmp_path):
    store = FeatureStore(root=tmp_path, namespace="ov2")
    with pytest.raises(ValueError):
        C.observable_variance(np.zeros(60), 4, seed=0, store=store, cfg=None)


def test_distinct_npoints_do_not_alias(tmp_path):
    """The enumeration's safety condition, executable: callers differing only in
    n_points get DISTINCT keys (n_points is in the key), so no aliasing."""
    store = FeatureStore(root=tmp_path, namespace="ov3")
    u_hi, _ = C._scaled_series("henon", 300)
    u_lo, _ = C._scaled_series("henon", 200)
    C.observable_variance(u_hi, 4, seed=0, store=store, cfg=_cfg(300))
    C.observable_variance(u_lo, 4, seed=0, store=store, cfg=_cfg(200))
    assert store.stats["misses"] == 2 and store.stats["entries"] == 2   # not aliased
