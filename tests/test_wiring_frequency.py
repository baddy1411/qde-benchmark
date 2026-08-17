"""Gate (b) for path 3 (frequency.observable_timeseries): the (S, ZZ) output is
byte-identical cache off vs on. Plus the inverse-of-aliasing check: the V=1 probe
key is distinct from the V=4 paths, so it can't collide with them.
"""
from __future__ import annotations

import numpy as np

from qdepipe import frequency as F
from qdepipe import concentration as C
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.experiment import ExperimentConfig
from qdepipe.contracts import feature_key
from qdepipe.feature_store import FeatureStore


def test_gate_b_observable_timeseries_byte_identical(tmp_path):
    args = dict(system="henon", encode_scale=0.5, n_qubits=4, n_points=200, seed=0)
    S_off, ZZ_off = F.observable_timeseries(store=None, **args)            # cache OFF

    store = FeatureStore(root=tmp_path, namespace="freq")
    S_on, ZZ_on = F.observable_timeseries(store=store, **args)            # miss
    S_h, ZZ_h = F.observable_timeseries(store=store, **args)              # hit

    assert np.array_equal(S_off, S_on) and np.array_equal(ZZ_off, ZZ_on)  # gate (b)
    assert np.array_equal(S_off, S_h) and np.array_equal(ZZ_off, ZZ_h)    # hit == miss
    assert store.stats["misses"] == 1 and store.stats["hits"] == 1


def test_v1_key_distinct_from_v4_paths():
    ecfg = ExperimentConfig(system="henon", n_points=1500, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))
    k_v1 = feature_key(GateQRC(C._rich_cfg(6, V=1, encode_scale=1.0, seed=0)), ecfg)
    k_v4 = feature_key(GateQRC(C._rich_cfg(6, V=4, seed=0)), ecfg)        # scaling/finite-shot
    assert k_v1 != k_v4, "V=1 frequency probe must not collide with V=4 paths"
