"""Forecaster families — one `run(u, split, cfg)` interface, four paradigms.

  ReservoirForecaster : random/explicit nonlinear features + ridge readout
                        (ESN, NG-RC, ELM, and later the QRC models)
  WindowedForecaster  : any sklearn-style regressor on lag-window features
                        (linear, RandomForest, XGBoost, LightGBM, SVR, kNN)
  StatsForecaster     : classical statistical models (AR, ARIMA, ETS) + Naive
  SequenceForecaster  : torch sequence nets (MLP, LSTM, GRU, 1D-CNN/TCN, Transformer)
"""
from __future__ import annotations

from .reservoir import ReservoirForecaster  # noqa: F401
from .windowed import WindowedForecaster  # noqa: F401
from .stats import NaiveForecaster, StatsForecaster, ThetaForecaster  # noqa: F401
from .sequence import SequenceForecaster  # noqa: F401

__all__ = [
    "ReservoirForecaster", "WindowedForecaster",
    "NaiveForecaster", "StatsForecaster", "ThetaForecaster", "SequenceForecaster",
]
