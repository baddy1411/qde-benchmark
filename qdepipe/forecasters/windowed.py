"""Windowed forecaster — any sklearn-style regressor on lag-window features.

Covers the standard tabular-ML benchmarks (linear/ridge, RandomForest, XGBoost,
LightGBM, SVR, kNN). The model sees the same causal delay window as the
feature-based reservoirs, so the *only* difference from a reservoir forecaster is
the regressor that sits on top of the features — isolating "what does a nonlinear
trained regressor buy over a linear readout on a fixed feature map?".

The estimator is supplied as a zero-arg factory so each run/seed gets a fresh,
unfitted model (no state leaks between runs).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..experiment import Forecaster, RunOutput, ExperimentConfig, windowed_xy, last_window


def _count_params(est):
    """Trainable-parameter count where it is well-defined (linear models);
    None for trees / kernel / instance-based models, where 'parameters' don't
    compare meaningfully to a linear readout. The per-model notebook says so."""
    if hasattr(est, "coef_"):
        n = int(np.size(est.coef_))
        if hasattr(est, "intercept_"):
            n += int(np.size(est.intercept_))
        return n
    return None


class WindowedForecaster(Forecaster):
    def __init__(self, estimator_factory: Callable[[], object], name: str,
                 standardize: bool = False):
        self.make = estimator_factory
        self.name = name
        self.standardize = standardize   # some models (SVR, kNN) need scaled feats

    def run(self, u: np.ndarray, split, cfg: ExperimentConfig) -> RunOutput:
        Xtr, ytr, Xte, yte = windowed_xy(u, split, cfg)

        mu = sd = None
        if self.standardize:
            mu, sd = Xtr.mean(0), np.where(Xtr.std(0) == 0, 1.0, Xtr.std(0))
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd

        est = self.make()
        est.fit(Xtr, ytr)
        y_pred = np.asarray(est.predict(Xte), dtype=float).ravel()

        return RunOutput(
            y_true=yte, y_pred=y_pred,
            n_features=Xtr.shape[1], n_params=_count_params(est),
            n_train=len(ytr), features=Xtr,
        )

    def onestep_predictor(self, u, split, cfg):
        c = ExperimentConfig(**{**cfg.__dict__, "horizon": 1})
        Xtr, ytr, _, _ = windowed_xy(u, split, c)
        mu = sd = None
        if self.standardize:
            mu, sd = Xtr.mean(0), np.where(Xtr.std(0) == 0, 1.0, Xtr.std(0))
            Xtr = (Xtr - mu) / sd
        est = self.make()
        est.fit(Xtr, ytr)
        L, st = cfg.lookback, cfg.stride

        def predict(history):
            x = last_window(history, L, st)
            if self.standardize:
                x = (x - mu) / sd
            return float(est.predict(x[None])[0])

        return predict
