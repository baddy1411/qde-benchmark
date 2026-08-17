"""sklearn adapter tests — the headline being that ReservoirRegressor computes
the SAME thing as the existing Forecaster path (it must, since it's additive)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from qdepipe.sklearn_api import ReservoirRegressor, reservoir_regressor
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig, ELM, ELMConfig
from qdepipe.experiment import ExperimentConfig
from qdepipe.data import generate as generate_system
from qdepipe.pipeline import make_scaler, temporal_split
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.readout import ridge_fit, ridge_predict
from qdepipe.pipeline.embedding import supervised_pairs


def _scaled(cfg):
    """Reproduce the runner's deterministic generate -> split -> train-only scale."""
    x = generate_system(cfg.system, cfg.n_points)
    split = temporal_split(len(x), cfg.split_fracs, cfg.washout)
    scaler = make_scaler(cfg.scaler)
    scaler.fit(x[split.train_fit])
    u = np.asarray(scaler.transform(x), dtype=float).ravel()
    return u, split


@pytest.mark.parametrize("make_model", [
    lambda s: ESN(ESNConfig(seed=s)),            # unbounded memory
    lambda s: NGRC(NGRCConfig(k=2, degree=2)),   # finite memory, deterministic
    lambda s: ELM(ELMConfig(units=80, lookback=2, seed=s)),
])
def test_matches_existing_forecaster_path(make_model):
    """Adapter (fit-on-prefix / predict-on-full) == ReservoirForecaster.run, to
    machine precision — proving it's the same computation, not a re-implementation."""
    cfg = ExperimentConfig(system="henon", n_points=400, washout=50, alpha=1e-6, horizon=1)
    u, split = _scaled(cfg)

    # --- existing path (what the thesis runs), inlined to expose y_pred on test ---
    res = make_model(0)
    feats = res.featurize(u)
    X, y = supervised_pairs(feats, u, cfg.horizon)
    tr = slice(cfg.washout, split.train.stop - cfg.horizon)
    te = slice(split.test.start, len(X))
    W = ridge_fit(X[tr], y[tr], alpha=cfg.alpha, bias=True)
    pred_existing = ridge_predict(X[te], W, bias=True).ravel()

    # --- adapter path: train on prefix u[:m], predict on full u, slice the tail ---
    m = split.train.stop
    est = ReservoirRegressor(make_model(0), alpha=cfg.alpha, fit_bias=True, washout=cfg.washout)
    est.fit(u[:m], u[cfg.horizon:m])
    pred_adapter = est.predict(u)[split.test.start: len(u) - cfg.horizon]

    assert pred_adapter.shape == pred_existing.shape
    np.testing.assert_allclose(pred_adapter, pred_existing, rtol=1e-9, atol=1e-12)


def test_fit_predict_shapes_and_factory():
    u = np.sin(np.linspace(0, 20, 300))
    est = reservoir_regressor(ESN(ESNConfig(seed=1)), alpha=1e-6, washout=20)
    est.fit(u[:200], u[1:200])
    p = est.predict(u)
    assert p.shape == (300,)
    assert est.n_features_in_ > 0 and est.n_outputs_ == 1
    assert est.feature_matrix(u).shape[0] == 300


def test_multi_output():
    rng = np.random.default_rng(0)
    u = np.cumsum(rng.normal(size=400)) * 0.01
    Y = np.column_stack([u, np.roll(u, -1)])[:399]   # 2-D target
    est = ReservoirRegressor(NGRC(NGRCConfig(k=2, degree=2)), washout=10)
    est.fit(u[:399], Y)
    assert est.n_outputs_ == 2
    assert est.predict(u[:399]).shape[1] == 2


def test_get_set_params_and_pipeline():
    est = ReservoirRegressor(ESN(ESNConfig(seed=2)), alpha=1e-6, washout=20)
    assert est.get_params()["alpha"] == 1e-6
    est.set_params(alpha=1e-3)
    assert est.get_params()["alpha"] == 1e-3
    # works as the final step of a sklearn Pipeline
    u = np.sin(np.linspace(0, 30, 400))
    pipe = Pipeline([("reservoir_ridge", ReservoirRegressor(ESN(ESNConfig(seed=3)), washout=20))])
    pipe.fit(u[:300], u[1:300])
    assert pipe.predict(u).shape == (400,)


def test_external_window_uses_delay_vectors():
    u = np.sin(np.linspace(0, 30, 300))
    est = ReservoirRegressor(ESN(ESNConfig(seed=4)), washout=20,
                             external_window=True, lookback=4)
    est.fit(u[:200], u[1:200])
    assert est.predict(u).shape == (300,)


def test_requires_reservoir():
    with pytest.raises(ValueError):
        ReservoirRegressor(None).fit(np.zeros(10), np.zeros(10))
