"""Reusable experiment sweeps — the engine behind every notebook section.

A notebook cell calls one of these, gets a tidy DataFrame back, and renders a
table + plot + an auto-generated finding. Because every model's notebook calls
the *same* functions, the experiments are structurally identical across models
and the cross-model comparison is apples-to-apples.

All sweeps rebuild the forecaster per config via `registry.build_forecaster`, so
axes that are model-internal (NG-RC tap count, ELM/linear window, ridge alpha)
are swept correctly rather than silently ignored.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .experiment import run_experiment, ExperimentConfig
from .closedloop import run_closed_loop
from .registry import build_forecaster, MODELS
from .pipeline.postprocess import LeakyMemory, Polynomial, FeatureStandardize

METRICS = ["nrmse", "rmse", "mae", "r2", "smape"]


# ---- core sweep ------------------------------------------------------------
def sweep(key: str, levels, base_cfg: ExperimentConfig, seeds=(0,)) -> pd.DataFrame:
    """levels = list of (label, cfg_overrides_dict). Returns one row per (level, seed)."""
    rows = []
    for label, ov in levels:
        for s in seeds:
            cfg = replace(base_cfg, **ov, seed=s)
            r = run_experiment(build_forecaster(key, s, cfg), cfg)
            rows.append({"level": label, "seed": s, "n_params": r.n_params,
                         "n_features": r.n_features, "feature_rank": r.feature_rank,
                         **r.metrics})
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, metric="nrmse", by="level") -> pd.DataFrame:
    """Mean/std/min/max of a metric per level, preserving level order."""
    return df.groupby(by, sort=False)[metric].agg(["mean", "std", "min", "max"]).reset_index()


def summarize(df: pd.DataFrame, metric="nrmse", by="level", lower_better=True) -> str:
    a = df.groupby(by, sort=False)[metric].mean()
    best = a.idxmin() if lower_better else a.idxmax()
    worst = a.idxmax() if lower_better else a.idxmin()
    bv, wv = a[best], a[worst]
    spread = (wv / bv) if (lower_better and bv > 0) else (bv / wv if wv else float("inf"))
    return (f"FINDING [{metric}]: best = '{best}' ({bv:.4g}); worst = '{worst}' "
            f"({wv:.4g}); spread {spread:.2f}x across {len(a)} levels.")


def bar_plot(df: pd.DataFrame, metric="nrmse", by="level", title=None, logy=False):
    a = agg(df, metric, by)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(a[by].astype(str), a["mean"], yerr=a["std"].fillna(0.0), capsize=3)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel(by)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title or f"{metric.upper()} by {by}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return a


# ---- axis level presets ----------------------------------------------------
def scaler_levels():
    return [(k, {"scaler": k}) for k in ("minmax", "standard", "robust", "identity")]


def scope_levels():
    return [("train", {"scaler_scope": "train"}), ("global", {"scaler_scope": "global"})]


def split_levels():
    return [("60/20/20", {"split_fracs": (0.6, 0.2, 0.2)}),
            ("80/10/10", {"split_fracs": (0.8, 0.1, 0.1)}),
            ("50/25/25", {"split_fracs": (0.5, 0.25, 0.25)})]


def washout_levels():
    return [(str(w), {"washout": w}) for w in (0, 50, 100, 300)]


def lookback_levels():
    return [(str(L), {"lookback": L}) for L in (1, 2, 3, 5, 8, 12)]


def windowing_levels():
    return [("internal", {"windowing": "internal"}), ("shared", {"windowing": "shared"})]


def horizon_levels():
    return [(str(h), {"horizon": h}) for h in (1, 2, 3, 5, 10)]


def alpha_levels():
    return [(f"{a:g}", {"alpha": a}) for a in (1e-8, 1e-6, 1e-4, 1e-2, 1e0)]


def post_levels():
    return [("none", {"postprocess": []}),
            ("leaky0.1", {"postprocess": [LeakyMemory(0.1)]}),
            ("leaky0.3", {"postprocess": [LeakyMemory(0.3)]}),
            ("poly2", {"postprocess": [Polynomial()]}),
            ("zscore", {"postprocess": [FeatureStandardize()]})]


def encoding_levels():
    # encode_scale multiplies the rotation angle (axes 7-9: input scaling / range)
    return [(f"{e:g}", {"encode_scale": e}) for e in (0.5, 1.0, 1.5, 2.0, 3.0)]


AXIS_LEVELS = {
    "scaler": scaler_levels, "scope": scope_levels, "split": split_levels,
    "washout": washout_levels, "lookback": lookback_levels,
    "windowing": windowing_levels, "horizon": horizon_levels,
    "alpha": alpha_levels, "postprocess": post_levels, "encoding": encoding_levels,
}

AXIS_TITLE = {
    "scaler": "Scaler type (axis 1)", "scope": "Scaler fitting scope (axis 2 — leakage)",
    "split": "Train/val/test split (axis 3)", "washout": "Washout length (axis 4)",
    "lookback": "Lookback / window (axis 5)", "windowing": "Windowing location (axis 5b)",
    "horizon": "Forecast horizon (axis 6)", "postprocess": "Feature post-processing (axes 10-12)",
    "alpha": "Ridge alpha (axis 13)", "encoding": "Quantum encoding scale (axes 7-9)",
}

# horizon is expected to worsen with k — not a 'pipeline is worse' verdict
AXIS_LOWER_BETTER = {a: True for a in AXIS_LEVELS}


def axis_sweep(key, axis, base_cfg, seeds=(0,)):
    """Convenience: sweep a named axis using its preset levels."""
    return sweep(key, AXIS_LEVELS[axis](), base_cfg, seeds)


# ---- leakage interaction study (the headline) ------------------------------
def leakage_study(key, base_cfg, seeds=(0,)) -> pd.DataFrame:
    """Cross scaler-scope x split (x ridge-alpha where it applies). This is where
    a spurious 'advantage' hides: the same model can look very different under
    train-only vs global scaling combined with the split and the regulariser."""
    has_alpha = "alpha" in MODELS[key].axes
    alphas = (1e-8, 1e-6, 1e-4, 1e-2) if has_alpha else (base_cfg.alpha,)
    rows = []
    for scope in ("train", "global"):
        for slabel, sov in split_levels():
            for a in alphas:
                for s in seeds:
                    cfg = replace(base_cfg, scaler_scope=scope, alpha=a, seed=s,
                                  split_fracs=sov["split_fracs"])
                    r = run_experiment(build_forecaster(key, s, cfg), cfg)
                    rows.append({"scope": scope, "split": slabel, "alpha": a,
                                 "seed": s, **r.metrics})
    return pd.DataFrame(rows)


# ---- multi-seed robustness --------------------------------------------------
def multiseed(key, base_cfg, seeds=range(5)) -> pd.DataFrame:
    rows = []
    for s in seeds:
        cfg = replace(base_cfg, seed=s)
        r = run_experiment(build_forecaster(key, s, cfg), cfg)
        rows.append({"seed": s, "n_params": r.n_params, **r.metrics})
    return pd.DataFrame(rows)


# ---- closed-loop / climate --------------------------------------------------
def closed_loop_study(key, base_cfg, levels=None, n_steps=300, seed=0):
    levels = levels or [("baseline", {})]
    rows, traj = [], {}
    for label, ov in levels:
        cfg = replace(base_cfg, **ov, seed=seed)
        cl = run_closed_loop(build_forecaster(key, seed, cfg), cfg, n_steps=n_steps)
        rows.append({"level": label, **cl.row()})
        traj[label] = (cl.true, cl.pred)
    return pd.DataFrame(rows), traj


def plot_closed_loop(traj, label="baseline", n_show=150):
    true, pred = traj[label]
    n = min(n_show, len(true))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(true[:n], label="true", lw=1.2)
    ax.plot(pred[:n], label="closed-loop prediction", lw=1.2, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("x (scaled)")
    ax.set_title(f"Autonomous rollout — {label}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---- baseline one-row metric table -----------------------------------------
def baseline(key, base_cfg, seeds=(0,)) -> pd.DataFrame:
    return sweep(key, [("baseline", {})], base_cfg, seeds)
