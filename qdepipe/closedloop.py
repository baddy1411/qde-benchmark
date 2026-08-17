"""Autonomous (closed-loop) forecasting + climate metrics.

One-step forecasting tests next-step accuracy; closed-loop tests whether a model
has learned the *dynamics* — it is fed its own predictions and must stay on the
attractor. We measure how long it tracks the true trajectory (valid prediction
time, in Lyapunov times) and whether its long-run statistics match the true
invariant density (log-spectral MSE, Wasserstein-1). A model can have excellent
1-step NRMSE yet diverge immediately in closed loop, so this is a distinct and
important axis of the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .data import generate as generate_system, lyapunov_step
from .experiment import ExperimentConfig, Forecaster
from .pipeline import make_scaler, temporal_split


@dataclass
class ClosedLoopResult:
    model: str
    n_steps: int
    true: np.ndarray
    pred: np.ndarray
    vpt_steps: int
    vpt_lyap: float          # valid prediction time in Lyapunov times
    spectral_mse: float
    wasserstein: float
    diverged: bool
    config: ExperimentConfig
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"model": self.model, "n_steps": self.n_steps,
                "vpt_steps": self.vpt_steps, "vpt_lyap": round(self.vpt_lyap, 3),
                "spectral_mse": self.spectral_mse, "wasserstein": self.wasserstein,
                "diverged": self.diverged, "scaler": self.config.scaler,
                "scope": self.config.scaler_scope, "lookback": self.config.lookback}


def run_closed_loop(fc: Forecaster, cfg: ExperimentConfig | None = None,
                    n_steps: int = 300, vpt_threshold: float = 0.4) -> ClosedLoopResult:
    cfg = cfg or ExperimentConfig()

    x = generate_system(cfg.system, cfg.n_points)
    le = lyapunov_step(cfg.system, x)
    lyap_time = 1.0 / le if le > 0 else np.nan

    split = temporal_split(len(x), cfg.split_fracs, cfg.washout)
    scaler = make_scaler(cfg.scaler)
    fit_slice = split.train_fit if cfg.scaler_scope == "train" else slice(0, len(x))
    scaler.fit(x[fit_slice])
    u = np.asarray(scaler.transform(x), dtype=float).ravel()

    pred_fn = fc.onestep_predictor(u, split, cfg)
    if pred_fn is None:
        raise NotImplementedError(f"{fc.name} does not support closed-loop forecasting")

    anchor = split.test.start
    n_steps = min(n_steps, len(u) - anchor)
    history = list(u[:anchor])
    true = np.asarray(u[anchor:anchor + n_steps], dtype=float)

    preds = []
    for _ in range(n_steps):
        p = pred_fn(np.asarray(history, dtype=float))
        if not np.isfinite(p):
            p = history[-1]               # freeze on blow-up rather than NaN-poison
        preds.append(p)
        history.append(p)
    preds = np.asarray(preds, dtype=float)

    vpt = metrics.valid_prediction_time(true, preds, vpt_threshold)
    diverged = bool(np.max(np.abs(preds)) > 10 * (np.max(np.abs(true)) + 1e-9))
    return ClosedLoopResult(
        model=fc.name, n_steps=n_steps, true=true, pred=preds,
        vpt_steps=vpt, vpt_lyap=float(vpt / lyap_time) if lyap_time else np.nan,
        spectral_mse=metrics.spectral_mse(true, preds),
        wasserstein=metrics.wasserstein1(true, preds),
        diverged=diverged, config=cfg,
        extra={"lyap_time": lyap_time},
    )
