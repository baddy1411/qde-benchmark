"""Statistical forecasters — the reference floor every TS paper reports.

Naive persistence and the classical linear stochastic models (AR, ARIMA, ETS).
On a *chaotic, deterministic* signal like the Hénon map these are expected to do
poorly — the map is quadratic, so no linear stochastic model can capture it — and
saying so cleanly, with numbers, is a legitimate baseline finding (it sets the
bar that any reservoir must clear).

Evaluation is a leakage-safe expanding-window one-step protocol: parameters are
fit on the TRAIN split only, then realized observations are appended (no refit)
so each forecast uses information only up to its own time t.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..experiment import Forecaster, RunOutput, ExperimentConfig


class NaiveForecaster(Forecaster):
    """Persistence: predict u[t+k] = u[t]. The zero-parameter reference."""
    name = "Naive"

    def run(self, u: np.ndarray, split, cfg: ExperimentConfig) -> RunOutput:
        k, T = cfg.horizon, len(u)
        t0, t1 = split.test.start, T - k
        return RunOutput(
            y_true=u[t0 + k:t1 + k], y_pred=u[t0:t1],
            n_features=1, n_params=0, n_train=0, features=None,
        )

    def onestep_predictor(self, u, split, cfg):
        # persistence fed back = a flat line at the last observed value
        def predict(history):
            return float(np.asarray(history, dtype=float)[-1])
        return predict


class StatsForecaster(Forecaster):
    """AR / ARIMA / ETS via statsmodels, expanding-window one-step (or k-step)."""

    def __init__(self, kind: str = "arima", order=(2, 0, 0),
                 trend: str | None = "add", name: str | None = None):
        self.kind = kind.lower()
        self.order = order
        self.trend = trend
        self.name = name or {"ar": "AR", "arima": "ARIMA", "ets": "ETS"}.get(self.kind, self.kind)

    def _build(self, history):
        from statsmodels.tsa.arima.model import ARIMA
        if self.kind in ("ar", "arima"):
            order = self.order if self.kind == "arima" else (self.order[0], 0, 0)
            return ARIMA(history, order=order)
        if self.kind == "ets":
            # statespace ES supports append() for the expanding-window protocol
            from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
            return ExponentialSmoothing(np.asarray(history, dtype=float),
                                        trend=bool(self.trend))
        raise ValueError(f"unknown stats kind {self.kind!r}")

    def run(self, u: np.ndarray, split, cfg: ExperimentConfig) -> RunOutput:
        k, T = cfg.horizon, len(u)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # fit params on TRAIN only, then extend (no refit) through val→test start
            res = self._build(u[split.train_fit]).fit()
            res = res.append(u[split.train_fit.stop:split.test.start + 1], refit=False)

            preds, trues = [], []
            for t in range(split.test.start, T - k):
                f = np.asarray(res.forecast(k))
                preds.append(float(f[-1]))
                trues.append(float(u[t + k]))
                res = res.append(u[t + 1:t + 2], refit=False)

        n_params = int(np.size(getattr(res, "params", [])))
        return RunOutput(
            y_true=np.array(trues), y_pred=np.array(preds),
            n_features=self.order[0], n_params=n_params,
            n_train=len(u[split.train_fit]), features=None,
        )

    def onestep_predictor(self, u, split, cfg):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._build(u[split.train_fit]).fit()
            res = res.append(u[split.train_fit.stop:split.test.start], refit=False)
        state = {"res": res}

        def predict(history):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                f = float(np.asarray(state["res"].forecast(1))[-1])
                state["res"] = state["res"].append([f], refit=False)  # feed back
            return f

        return predict


class ThetaForecaster(Forecaster):
    """The Theta method (Assimakopoulos & Nikolopoulos), via statsmodels.

    Expanding-window protocol matching StatsForecaster: parameters are
    effectively refit each step on the history seen so far (ThetaModel has no
    append(); its fit is a cheap SES + drift, so per-step refits are fast).
    Deterministic.
    """

    name = "Theta"

    def run(self, u: np.ndarray, split, cfg: ExperimentConfig) -> RunOutput:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        k, T = cfg.horizon, len(u)
        preds, trues = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(split.test.start, T - k):
                res = ThetaModel(np.asarray(u[:t + 1], dtype=float),
                                 period=1, deseasonalize=False).fit()
                preds.append(float(np.asarray(res.forecast(k))[-1]))
                trues.append(float(u[t + k]))
        return RunOutput(y_true=np.array(trues), y_pred=np.array(preds),
                         n_features=0, n_params=3,
                         n_train=int(split.train_fit.stop - split.train_fit.start),
                         extra={"protocol": "theta: expanding refit per step"})
