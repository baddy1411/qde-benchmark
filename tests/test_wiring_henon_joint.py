"""Gate (b) for path 5 (henon_joint_mcs) — routine hook-reuse. The forecast
generation (the cacheable part) is byte-identical cache off vs on, and the wiring
caches only the quantum entries (classical battery left uncached, by scope).
"""
from __future__ import annotations

import numpy as np

import henon_joint_mcs as hjm
from qdepipe.experiment import ExperimentConfig
from qdepipe.feature_store import FeatureStore


def _small_cfg(seed, **ov):
    return ExperimentConfig(system="henon", n_points=250, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2),
                            washout=50, horizon=1, lookback=5, alpha=1e-6, seed=seed, **ov)


def test_gate_b_henon_joint_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(hjm, "_cfg", _small_cfg)
    quant = ("qrc_v4",)                                  # cached quantum subset
    comp_off = {k: hjm._build_comp(None)[k] for k in quant}
    store = FeatureStore(root=tmp_path, namespace="hj")
    comp_on = {k: hjm._build_comp(store)[k] for k in quant}

    off = hjm._run_comp(comp_off, seeds=(0, 1))
    on = hjm._run_comp(comp_on, seeds=(0, 1))

    assert len(off) == len(on) == 2
    for fc_off, fc_on in zip(off, on):
        for label in fc_off:
            assert np.array_equal(np.asarray(fc_off[label][1]), np.asarray(fc_on[label][1])), label  # y_pred
            assert np.array_equal(np.asarray(fc_off[label][0]), np.asarray(fc_on[label][0])), label  # y_true
    assert store.stats["misses"] >= 1                    # quantum featurize was cached


def test_only_quantum_entries_get_the_store(tmp_path):
    """Scope decision, executable: quantum comp entries carry the store; the
    classical battery does not."""
    store = FeatureStore(root=tmp_path, namespace="hj2")
    comp = hjm._build_comp(store)
    cfg = _small_cfg(0)
    assert getattr(comp["ESN"][0](0, cfg), "store", None) is None          # classical: uncached
    assert getattr(comp["RandomForest"][0](0, cfg), "store", None) is None  # classical: uncached
    assert comp["qrc_v4"][0](0, cfg).store is store                         # quantum: cached
    assert comp["qrc_rich"][0](0, cfg).store is store                       # quantum: cached
