"""sklearn-style adapter: a reservoir + ridge readout as one estimator.

ADDITIVE. This wraps the *existing* `ReservoirModel.featurize` and the *existing*
`readout.ridge_fit/ridge_predict` behind the scikit-learn ``fit``/``predict``
surface, so ESN, NG-RC, ELM and GateQRC become drop-in, swappable estimators
usable in ``sklearn.pipeline.Pipeline`` and model-selection tooling. It changes
nothing about the `Forecaster` path used by the thesis experiments — both call the
same readout functions, so they are numerically identical (see
``tests/test_sklearn_api.py::test_matches_existing_forecaster_path``).

Sequential, not i.i.d.: a reservoir feature row is causal (row *t* depends on the
input history up to *t*), so this estimator is **not** shuffle-invariant and is not
meant for ``sklearn.utils.estimator_checks.check_estimator`` (which assumes
exchangeable rows). It is meant for the swap-a-model and time-series-CV use cases.
Use it with a temporal splitter (e.g. ``TimeSeriesSplit``), never a shuffled one.

Causal-prefix contract that makes it match the experiment runner exactly: because
``featurize`` is causal and ``delay_embedding`` is length-preserving,
``featurize(u[:m]) == featurize(u)[:m]``. So training on a prefix series and
predicting on the full series (slicing the tail) reproduces the runner's
split-based training bit-for-bit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .models.base import ReservoirModel
from .pipeline.embedding import delay_embedding
from .readout import ridge_fit, ridge_predict


class ReservoirRegressor(BaseEstimator, RegressorMixin):
    """Reservoir feature map + trained ridge readout, as a scikit-learn regressor.

    Parameters
    ----------
    reservoir : ReservoirModel
        Any model implementing ``featurize(u) -> (T, F)`` (ESN, NG-RC, ELM, GateQRC).
    alpha : float
        Ridge regularisation (the thesis's swept axis-13 knob).
    fit_bias : bool
        Append an intercept column, exactly as the existing readout does.
    washout : int
        Number of leading feature rows dropped from training (transient warm-up).
    external_window : bool
        If True, drive the reservoir with shared delay vectors
        ``delay_embedding(u, lookback, stride)`` instead of the scalar series —
        the shared-windowing axis. Internal-window models keep this False.
    lookback, stride : int
        Delay-embedding parameters, used only when ``external_window`` is True.

    Alignment
    ---------
    Feature row *t* is paired with target ``y[t]``; the longer of (features, y) is
    truncated from the front so both start at *t=0*, then ``washout`` rows are
    dropped. To forecast horizon *k*, pass ``y`` already shifted by *k*
    (``y[t] = series[t+k]``) — keeping this estimator horizon-agnostic and
    sklearn-pure; horizon/split orchestration stays in the experiment runner.
    """

    def __init__(self, reservoir: Optional[ReservoirModel] = None, alpha: float = 1e-6,
                 fit_bias: bool = True, washout: int = 0, external_window: bool = False,
                 lookback: int = 1, stride: int = 1):
        self.reservoir = reservoir
        self.alpha = alpha
        self.fit_bias = fit_bias
        self.washout = washout
        self.external_window = external_window
        self.lookback = lookback
        self.stride = stride

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _as_series(X) -> np.ndarray:
        u = np.asarray(X, dtype=float)
        if u.ndim == 2 and u.shape[1] == 1:
            u = u.ravel()
        return u

    def _featurize(self, X) -> np.ndarray:
        if self.reservoir is None:
            raise ValueError("ReservoirRegressor needs a `reservoir` (a ReservoirModel).")
        u = self._as_series(X)
        inp = delay_embedding(u, self.lookback, self.stride) if self.external_window else u
        return np.asarray(self.reservoir.featurize(inp), dtype=float)

    # -- sklearn API ---------------------------------------------------------
    def fit(self, X, y):
        feats = self._featurize(X)
        y = np.asarray(y, dtype=float)
        n = min(len(feats), len(y))
        F = feats[self.washout:n]
        t = y[self.washout:n]
        if F.shape[0] == 0:
            raise ValueError(f"no training rows after washout={self.washout} "
                             f"(features={len(feats)}, targets={len(y)}).")
        self.W_ = ridge_fit(F, t, alpha=self.alpha, bias=self.fit_bias)
        self.n_features_in_ = int(feats.shape[1])
        self.n_outputs_ = int(self.W_.shape[1])
        self.reservoir_ = self.reservoir   # fitted-context handle (stateless featurizer)
        return self

    def predict(self, X):
        check_is_fitted(self, "W_")
        feats = self._featurize(X)
        pred = ridge_predict(feats, self.W_, bias=self.fit_bias)
        return pred.ravel() if self.n_outputs_ == 1 else pred

    def feature_matrix(self, X) -> np.ndarray:
        """The raw reservoir feature matrix for X — the object the step-3 feature
        store will persist. Does not require the estimator to be fitted."""
        return self._featurize(X)


def reservoir_regressor(reservoir: ReservoirModel, **kw) -> ReservoirRegressor:
    """Convenience factory: ``reservoir_regressor(ESN(...), alpha=1e-6, washout=100)``."""
    return ReservoirRegressor(reservoir=reservoir, **kw)
