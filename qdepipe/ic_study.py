"""Initial-condition robustness machinery.

The whole benchmark uses each system's default initial condition (henon x0=y0=0,
lorenz s0=(1,1,1), MG x0=1.2). The review standard demands evaluation "across
many random initial conditions". This module supplies:

  draw_ics(system, K)            — K reproducible on-attractor IC draws
  generate_with_ic(...)          — the target series from a custom IC
  run_onestep_on_series(...)     — run_experiment's exact body, custom series
  run_closedloop_on_series(...)  — run_closed_loop's exact body, custom series
  selfcheck()                    — default-IC equivalence gate vs the originals

CACHE HAZARD (why everything here is store=None): the feature-store key hashes
(system, n_points, scaler, scaler_scope, split_fracs) + model — it does NOT see
the initial condition, so two different-IC series would collide on one key and
alias each other's features. The IC study therefore NEVER passes a store; each
featurize is computed fresh. (The key's precondition "input is determined by
cfg" simply does not hold here.)
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .data import lyapunov_step
from .data.henon import generate_henon
from .data.lorenz import generate_lorenz
from .data.mackeyglass import generate_mackeyglass
from .closedloop import ClosedLoopResult
from .experiment import ExperimentConfig, ExperimentResult, Forecaster
from .pipeline import make_scaler, temporal_split

IC_RNG_SEED = 7          # unit-gated reproducible draws


def draw_ics(system: str, K: int = 20, rng_seed: int = IC_RNG_SEED) -> list[dict]:
    """K reproducible IC dicts for a system, rejecting diverged trajectories.

    Ranges: henon within the attractor basin; lorenz a Gaussian ball around the
    default (transient 5000 lands all on-attractor); MG the constant-history
    level within the physiological band used in the literature.
    """
    rng = np.random.default_rng(rng_seed)
    out: list[dict] = []
    guard = 0
    while len(out) < K:
        guard += 1
        if guard > 20 * K:
            raise RuntimeError("IC rejection loop not converging")
        if system == "henon":
            ic = {"x0": float(rng.uniform(-0.5, 0.5)),
                  "y0": float(rng.uniform(-0.3, 0.3))}
        elif system == "lorenz":
            ic = {"s0": tuple(np.array([1.0, 1.0, 1.0]) + rng.normal(0, 2.0, 3))}
        elif system == "mackeyglass":
            ic = {"x0": float(rng.uniform(0.8, 1.4))}
        else:
            raise ValueError(f"no IC scheme for {system!r}")
        try:
            x = generate_with_ic(system, 500, ic)
            if np.all(np.isfinite(x)) and np.std(x) > 1e-6:
                out.append(ic)
        except (FloatingPointError, OverflowError):
            continue
    return out


def generate_with_ic(system: str, n_points: int, ic: dict) -> np.ndarray:
    """The 1-D target series from a custom IC (mirrors data.generate's contract)."""
    with np.errstate(over="raise", invalid="raise"):
        if system == "henon":
            x = generate_henon(n_points, **ic)[0]
        elif system == "lorenz":
            x = generate_lorenz(n_points, **ic)[0]
        elif system == "mackeyglass":
            x = generate_mackeyglass(n_points, **ic)[0]
        else:
            raise ValueError(f"unknown system {system!r}")
    return np.asarray(x, dtype=float).ravel()


def _prepare(x: np.ndarray, cfg: ExperimentConfig):
    """generate -> split -> train-only scale, exactly as run_experiment does."""
    split = temporal_split(len(x), cfg.split_fracs, cfg.washout)
    scaler = make_scaler(cfg.scaler)
    fit_slice = split.train_fit if cfg.scaler_scope == "train" else slice(0, len(x))
    scaler.fit(x[fit_slice])
    u = np.asarray(scaler.transform(x), dtype=float).ravel()
    return u, split


def run_onestep_on_series(fc: Forecaster, x: np.ndarray,
                          cfg: ExperimentConfig) -> ExperimentResult:
    """run_experiment with the generation step replaced by a provided series.
    Body mirrors qdepipe.experiment.run_experiment line-for-line (verified by
    selfcheck() below on the default IC)."""
    u, split = _prepare(np.asarray(x, dtype=float).ravel(), cfg)
    out = fc.run(u, split, cfg)
    if not np.all(np.isfinite(out.y_pred)):
        out.extra["nonfinite_pred"] = int(np.sum(~np.isfinite(out.y_pred)))
    rank = metrics.feature_rank(out.features) if out.features is not None else None
    m = metrics.regression_metrics(out.y_true, out.y_pred)
    return ExperimentResult(
        model=fc.name, nrmse=m["nrmse"], metrics=m,
        n_features=out.n_features, n_params=out.n_params, n_train=out.n_train,
        n_test=len(np.asarray(out.y_true).ravel()), feature_rank=rank,
        config=cfg, extra=out.extra,
        y_true=np.asarray(out.y_true).ravel(), y_pred=np.asarray(out.y_pred).ravel())


def run_closedloop_on_series(fc: Forecaster, x: np.ndarray, cfg: ExperimentConfig,
                             n_steps: int = 300,
                             vpt_threshold: float = 0.4) -> ClosedLoopResult:
    """run_closed_loop with the generation step replaced by a provided series."""
    x = np.asarray(x, dtype=float).ravel()
    le = lyapunov_step(cfg.system, x)
    lyap_time = 1.0 / le if le > 0 else np.nan
    u, split = _prepare(x, cfg)
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
            p = history[-1]
        preds.append(p)
        history.append(p)
    preds = np.asarray(preds, dtype=float)
    vpt = metrics.valid_prediction_time(true, preds, vpt_threshold)
    diverged = bool(np.max(np.abs(preds)) > 10 * (np.max(np.abs(true)) + 1e-9))
    return ClosedLoopResult(
        model=fc.name, n_steps=n_steps, true=true, pred=preds, vpt_steps=vpt,
        vpt_lyap=float(vpt / lyap_time) if lyap_time else np.nan,
        spectral_mse=metrics.spectral_mse(true, preds),
        wasserstein=metrics.wasserstein1(true, preds), diverged=diverged,
        config=cfg, extra={"lyap_time": lyap_time})


DEFAULT_IC = {"henon": {"x0": 0.0, "y0": 0.0},
              "lorenz": {"s0": (1.0, 1.0, 1.0)},
              "mackeyglass": {"x0": 1.2}}


def selfcheck() -> None:
    """Equivalence gate: on the DEFAULT IC, the series-parameterized runners must
    reproduce the originals exactly (identical NRMSE / VPT). Run before any sweep."""
    from .experiment import run_experiment
    from .closedloop import run_closed_loop
    from .forecasters import ReservoirForecaster
    from .models import NGRC, NGRCConfig
    for system in ["henon", "lorenz", "mackeyglass"]:
        cfg = ExperimentConfig(system=system, n_points=800, lookback=3, washout=100)
        x = generate_with_ic(system, 800, DEFAULT_IC[system])
        a = run_experiment(ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC"), cfg)
        b = run_onestep_on_series(
            ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC"), x, cfg)
        assert a.nrmse == b.nrmse, f"{system}: one-step mismatch {a.nrmse} != {b.nrmse}"
        ca = run_closed_loop(ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC"),
                             cfg, n_steps=100)
        cb = run_closedloop_on_series(
            ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC"), x, cfg,
            n_steps=100)
        assert ca.vpt_steps == cb.vpt_steps, f"{system}: closed-loop VPT mismatch"
    print("ic_study selfcheck PASS: series-parameterized runners == originals on default ICs")
