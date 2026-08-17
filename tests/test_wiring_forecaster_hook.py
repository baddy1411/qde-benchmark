"""Gate (b) for path 4 — and the proving ground for the ReservoirForecaster.store
hook that paths 5-6 inherit. Tightened accordingly:

  * byte-identity at the run_experiment level on a REAL quantum forecaster
    (y_pred, nrmse, n_features), cache off vs on, miss and hit;
  * the rollout (onestep_predictor) provably never touches the store;
  * build_forecaster injects the store only into reservoir forecasters;
  * quantum_cross.encode_sweep byte-identical cache off vs on (script-level).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.registry import build_forecaster
from qdepipe.feature_store import FeatureStore
from qdepipe.forecasters import ReservoirForecaster


def _cfg(npts=300):
    return ExperimentConfig(system="lorenz", n_points=npts, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2),
                            washout=50, horizon=1, lookback=5, alpha=1e-6)


def test_gate_b_hook_run_experiment_byte_identical(tmp_path):
    """The proving ground: a real quantum forecaster through run_experiment,
    cache off vs on, byte-identical on every reported quantity."""
    cfg = _cfg()
    store = FeatureStore(root=tmp_path, namespace="hook")
    off = run_experiment(build_forecaster("qrc_v4", 0, cfg), cfg, keep_forecasts=True)
    on = run_experiment(build_forecaster("qrc_v4", 0, cfg, store=store), cfg, keep_forecasts=True)   # miss
    on2 = run_experiment(build_forecaster("qrc_v4", 0, cfg, store=store), cfg, keep_forecasts=True)  # hit

    assert np.array_equal(off.y_pred, on.y_pred), "cache changed y_pred (gate b violation)"
    assert np.array_equal(off.y_pred, on2.y_pred), "hit != miss"
    assert off.nrmse == on.nrmse == on2.nrmse
    assert off.n_features == on.n_features == on2.n_features
    assert np.array_equal(np.asarray(off.y_true), np.asarray(on.y_true))
    assert store.stats["misses"] == 1 and store.stats["hits"] == 1


def test_rollout_never_touches_the_store(tmp_path):
    """onestep_predictor must not use the store — its rollout featurizes
    cfg-independent history suffixes that would alias onto one key."""
    from qdepipe.models import NGRC, NGRCConfig
    from qdepipe.data import generate
    from qdepipe.pipeline import make_scaler, temporal_split
    store = FeatureStore(root=tmp_path, namespace="os")
    fc = ReservoirForecaster(NGRC(NGRCConfig(k=2, degree=2)), "NG-RC", store=store)
    cfg = ExperimentConfig(system="henon", n_points=300, washout=50, lookback=2)
    x = generate("henon", 300)
    split = temporal_split(len(x), (0.6, 0.2, 0.2), 50)
    sc = make_scaler("minmax"); sc.fit(x[split.train_fit])
    u = np.asarray(sc.transform(x)).ravel()
    predict = fc.onestep_predictor(u, split, cfg)
    for t in range(split.test.start, split.test.start + 5):
        predict(u[:t])
    assert store.stats == {**store.stats, "hits": 0, "misses": 0}, \
        "rollout touched the store — aliasing risk"


def test_build_forecaster_injects_store_only_when_given(tmp_path):
    cfg = _cfg()
    assert build_forecaster("qrc_v4", 0, cfg).store is None          # default off
    s = FeatureStore(root=tmp_path, namespace="inj")
    fc = build_forecaster("qrc_v4", 0, cfg, store=s)
    assert isinstance(fc, ReservoirForecaster) and fc.store is s     # injected


def test_gate_b_encode_sweep_script_level(tmp_path, monkeypatch):
    """quantum_cross.encode_sweep byte-identical cache off vs on (shrunk grids,
    RESULTS redirected so no real CSV is clobbered)."""
    import quantum_cross as qc
    monkeypatch.setattr(qc, "QUANTUM", {"qrc_v4": 16})
    monkeypatch.setattr(qc, "SCALERS", ["minmax"])
    monkeypatch.setattr(qc, "ENCODE_GRID", [1.0, 2.0])
    monkeypatch.setattr(qc, "NPTS", 250)
    monkeypatch.setattr(qc, "RESULTS", str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "cross"), exist_ok=True)

    off = qc.encode_sweep("lorenz", seeds=(0,), store=None)
    store = FeatureStore(root=tmp_path, namespace="es")
    on = qc.encode_sweep("lorenz", seeds=(0,), store=store)

    assert off == on, "cache changed encode_sweep output (gate b violation)"
    assert store.stats["misses"] >= 1          # the quantum featurize was cached
