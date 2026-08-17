"""Config-driven experiment harness — one pipeline, any model family.

The methodological core of the thesis: the **data-engineering pipeline is applied
identically to every model**, and only the model's own training differs. The
runner owns the data engineering that must be shared to be fair —

    generate Hénon  ->  temporal split  ->  scale (fit on TRAIN ONLY)

— and then hands the scaled series to a `Forecaster`. Each forecaster family
implements `run(u, split, cfg)` in its own natural way (ridge-on-features for
reservoirs; native fit/predict for ARIMA, trees, nets), but every one is scored
by the same `metrics` on the same test target. That uniformity is what makes the
classical-vs-classical and later classical-vs-quantum comparisons honest.

Axis applicability is intentionally explicit (see THESIS_PLAN §2): not every axis
applies to every model (e.g. ridge `alpha` is meaningless for a Random Forest;
`lookback` is meaningless for naive persistence). A forecaster simply ignores the
cfg fields it cannot use, and the per-model notebook documents which axes apply.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics
from .data import generate as generate_system
from .pipeline import make_scaler, temporal_split
from .pipeline.embedding import delay_embedding
from .pipeline.split import SplitSpec


# ----------------------------------------------------------------------------
# Config & results
# ----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # data
    n_points: int = 4000
    seed: int = 42
    system: str = "henon"                  # 'henon' | 'lorenz' | 'mackeyglass'
    # pre-model data engineering (applies to ALL models)
    scaler: str = "minmax"                 # axis 1
    scaler_scope: str = "train"            # axis 2: 'train' (safe) | 'global'
    split_fracs: tuple = (0.6, 0.2, 0.2)   # axis 3
    washout: int = 100                     # axis 4
    horizon: int = 1                       # axis 6
    # windowing (axes 5 / 5b) — used by feature-based & sequence models
    lookback: int = 1                      # L
    stride: int = 1
    windowing: str = "internal"            # 'internal' | 'shared' (reservoirs)
    # post-model feature engineering (reservoir features; axes 10-12)
    postprocess: list = field(default_factory=list)
    # readout (reservoir / linear; axis 13)
    alpha: float = 1e-6
    # quantum encoding (axes 7-9; ignored by classical models)
    encode_scale: float = 1.0

    def summary(self) -> str:
        return (f"scaler={self.scaler}/{self.scaler_scope} split={self.split_fracs} "
                f"washout={self.washout} k={self.horizon} L={self.lookback} "
                f"win={self.windowing} alpha={self.alpha:g}")


@dataclass
class RunOutput:
    """What a forecaster returns; the runner turns it into scored metrics."""
    y_true: np.ndarray
    y_pred: np.ndarray
    n_features: int
    n_params: Optional[int]
    n_train: int
    features: Optional[np.ndarray] = None   # train feature matrix, for rank/MC
    extra: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    model: str
    nrmse: float
    metrics: dict
    n_features: int
    n_params: Optional[int]
    n_train: int
    n_test: int
    feature_rank: Optional[int]
    config: ExperimentConfig
    extra: dict = field(default_factory=dict)
    # raw aligned test-set forecasts, populated only when keep_forecasts=True;
    # needed by the significance machinery (Diebold–Mariano / MCS operate on the
    # per-step forecast errors, not on summary statistics).
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None

    def row(self) -> dict:
        """Flat dict for assembling result tables / DataFrames (all metrics)."""
        c = self.config
        base = {
            "model": self.model,
            "n_features": self.n_features, "n_params": self.n_params,
            "feature_rank": self.feature_rank,
            "scaler": c.scaler, "scope": c.scaler_scope, "washout": c.washout,
            "horizon": c.horizon, "lookback": c.lookback, "windowing": c.windowing,
            "alpha": c.alpha, "seed": c.seed,
        }
        return {**base, **self.metrics}

    def __repr__(self):
        rk = "-" if self.feature_rank is None else f"{self.feature_rank}/{self.n_features}"
        npar = "-" if self.n_params is None else str(self.n_params)
        return f"[{self.model}] NRMSE={self.nrmse:.4f} rank={rk} params={npar} | {self.config.summary()}"


# ----------------------------------------------------------------------------
# Forecaster interface
# ----------------------------------------------------------------------------
class Forecaster(ABC):
    """Every model family implements `run`; closed-loop is optional."""
    name: str = "forecaster"

    @abstractmethod
    def run(self, u: np.ndarray, split: SplitSpec, cfg: ExperimentConfig) -> RunOutput:
        """Map the scaled series `u` to aligned (y_true, y_pred) on the TEST split."""

    def onestep_predictor(self, u: np.ndarray, split: SplitSpec, cfg: ExperimentConfig):
        """Return a trained `predict(history)->float` for autonomous rollout, or
        None if this model does not support closed-loop forecasting. The model is
        trained at horizon 1 regardless of `cfg.horizon`."""
        return None


def last_window(history, L: int, stride: int = 1) -> np.ndarray:
    """The most recent length-L delay vector from a 1-D history (for rollout)."""
    return delay_embedding(np.asarray(history, dtype=float), L, stride)[-1]


# ----------------------------------------------------------------------------
# Shared windowing helper (feature-based & sequence models)
# ----------------------------------------------------------------------------
def windowed_xy(u: np.ndarray, split: SplitSpec, cfg: ExperimentConfig):
    """Build causal (X, y) for a horizon-`k` forecast from lag windows of length L.

    Row t holds the delay vector ending at time t and predicts u[t+k]. Training
    starts after the washout (and after the first full window) so no padded or
    transient window leaks in. Returns (Xtr, ytr, Xte, yte).
    """
    L, k, st = cfg.lookback, cfg.horizon, cfg.stride
    X = delay_embedding(u, L, st)
    T = len(u)
    start = max(split.washout, (L - 1) * st)
    tr0, tr1 = start, split.train.stop - k
    te0, te1 = split.test.start, T - k
    return (X[tr0:tr1], u[tr0 + k:tr1 + k], X[te0:te1], u[te0 + k:te1 + k])


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def run_experiment(fc: Forecaster, cfg: ExperimentConfig | None = None,
                   keep_forecasts: bool = False) -> ExperimentResult:
    cfg = cfg or ExperimentConfig()

    x = generate_system(cfg.system, cfg.n_points)
    split = temporal_split(len(x), cfg.split_fracs, cfg.washout)

    scaler = make_scaler(cfg.scaler)
    fit_slice = split.train_fit if cfg.scaler_scope == "train" else slice(0, len(x))
    scaler.fit(x[fit_slice])
    u = np.asarray(scaler.transform(x), dtype=float).ravel()

    out = fc.run(u, split, cfg)
    if not np.all(np.isfinite(out.y_pred)):
        # keep the run but flag it — a blown-up forecast is itself a finding
        out.extra["nonfinite_pred"] = int(np.sum(~np.isfinite(out.y_pred)))

    rank = metrics.feature_rank(out.features) if out.features is not None else None
    m = metrics.regression_metrics(out.y_true, out.y_pred)
    return ExperimentResult(
        model=fc.name,
        nrmse=m["nrmse"],
        metrics=m,
        n_features=out.n_features,
        n_params=out.n_params,
        n_train=out.n_train,
        n_test=len(np.asarray(out.y_true).ravel()),
        feature_rank=rank,
        config=cfg,
        extra=out.extra,
        y_true=(np.asarray(out.y_true).ravel() if keep_forecasts else None),
        y_pred=(np.asarray(out.y_pred).ravel() if keep_forecasts else None),
    )


def multiseed(make_forecaster, cfg: ExperimentConfig, seeds=(0, 1, 2, 3, 4)):
    """Run `make_forecaster(seed)` over seeds; return (mean, std, [results]).

    `make_forecaster` is a callable seed->Forecaster so each seed gets a freshly
    initialised model (reservoir draw, net init, etc.). The data/split/scaling are
    deterministic; only the model seed varies — isolating model-init variance.
    """
    results = []
    for s in seeds:
        c = ExperimentConfig(**{**cfg.__dict__, "seed": s})
        results.append(run_experiment(make_forecaster(s), c))
    vals = np.array([r.nrmse for r in results])
    return float(vals.mean()), float(vals.std()), results
