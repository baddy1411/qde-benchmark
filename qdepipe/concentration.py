"""Exponential-concentration analysis — the mechanism behind the ZZ-ablation null.

The benchmarking showed adding ZZ (two-qubit) correlators to the QRC readout buys
almost nothing (Hénon ablation: Z→Z+X+Y = 59×, +ZZ = 1.03×). The hypothesis tested
here is **exponential concentration**: high-weight (2-local) observables' expectation
values concentrate toward an input-independent constant as qubit count grows, so they
carry vanishing *usable* signal — and finite sampling makes that fatal.

We measure it directly:
  1. observable_variance  — across-input variance of ⟨o⟩ per observable, by weight class.
  2. scaling_sweep        — that variance + forecasting NRMSE for Z / Z+X+Y / Z+X+Y+ZZ
                            readouts across n_qubits ∈ {4,6,8,10}.  →  results/concentration/scaling.csv
  3. finite_shot_sweep    — Z+X+Y vs Z+X+Y+ZZ NRMSE under exact/8192/1024 shots.
  4. matched_budget_shots — QRC-rich vs ESN(F) on Hénon under exact/8192/1024 shots.

Exact-expectation results are an IDEALIZED UPPER BOUND on quantum performance; the
finite-shot runs are precisely the demonstration that real sampling makes it worse.

Optimization: the three readout sets share the same state evolution (only the
measurement differs), so we featurize ONCE with the full (Z,X,Y,ZZ) readout and slice
columns — essential because n=10 featurization is ~30× slower than n=8.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from qdepipe.models.gate_qrc import GateQRC, GateQRCConfig
from qdepipe.data import generate
from qdepipe.pipeline import make_scaler, temporal_split
from qdepipe.pipeline.embedding import supervised_pairs
from qdepipe.readout import ridge_fit, ridge_predict
from qdepipe import metrics

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CDIR = os.path.join(RESULTS, "concentration")

# n_points per qubit count — reduced at n=10 (featurize ~30× slower); logged in CSV.
NPOINTS = {4: 1500, 6: 1500, 8: 1200, 10: 600}
N_QUBITS = [4, 6, 8, 10]
SHOTS = [None, 8192, 1024]


def _rich_cfg(n, **kw):
    """A QRC-rich-style reservoir at n qubits: IsingXX disorder, depth encoding,
    full Z+X+Y+ZZ readout. Construction fixed across n except qubit count."""
    base = dict(n_qubits=n, encoding="depth", r=1, coupling="isingxx", J_strength=1.0,
                channel="none", V=4, window=5, readout=("Z", "X", "Y", "ZZ"),
                encode_scale=1.0, seed=0)
    base.update(kw)
    return GateQRCConfig(**base)


def _scaled_series(system, n_points, washout=100, scaler="minmax"):
    """Generate + leakage-safe (train-only) scale, exactly as run_experiment does."""
    x = generate(system, n_points)
    split = temporal_split(len(x), (0.6, 0.2, 0.2), washout)
    sc = make_scaler(scaler)
    sc.fit(x[split.train_fit])
    u = np.asarray(sc.transform(x), dtype=float).ravel()
    return u, split


# ---------------------------------------------------------------------------
# 1. observable-variance probe
# ---------------------------------------------------------------------------
def observable_variance(u, n, seed=0, shots=None, store=None, cfg=None):
    """Across-input variance of each observable's expectation, grouped by weight.

    Uses V=1 so each column is exactly ⟨o⟩(t) for one observable; columns are laid
    out [Z_0..Z_{n-1}, X_.., Y_.., ZZ_..]. Returns dict with per-class arrays + means.
    A concentrating observable shows variance → 0.

    `store`/`cfg`: optional, DEFAULT-OFF cache. This function receives `u` rather
    than building it, so the caller MUST pass `cfg` (the ExperimentConfig that
    produced `u`) — the key needs system/n_points, which `u` alone doesn't carry.
    Every production caller builds `u` via _scaled_series (minmax/train/60-20-20),
    so one key per (system, n_points, model) is correct. Caching without `cfg` is a
    hard error so a context-blind key can never be formed.
    """
    from qdepipe.feature_store import maybe_cached_featurize
    if store is not None and cfg is None:
        raise ValueError("caching observable_variance requires cfg (the data "
                         "context that produced u), so the key includes system/n_points.")
    q = GateQRC(_rich_cfg(n, V=1, seed=seed, shots=shots))
    feats = maybe_cached_featurize(store, q, u, cfg)   # store=None -> exactly q.featurize(u)
    var = feats.var(axis=0)                      # across-input variance per observable
    one_local = var[:3 * n]                      # Z, X, Y
    two_local = var[3 * n:4 * n]                 # ZZ
    return {"var_1local": one_local, "var_2local": two_local,
            "mean_var_1local": float(one_local.mean()),
            "mean_var_2local": float(two_local.mean()),
            "std_var_1local": float(one_local.std()),
            "std_var_2local": float(two_local.std())}


# ---------------------------------------------------------------------------
# forecasting from a precomputed feature matrix (replicates ReservoirForecaster.run)
# ---------------------------------------------------------------------------
def _nrmse_from_feats(feats, u, split, alpha=1e-6, horizon=1):
    X, y = supervised_pairs(feats, u, horizon)
    a, b, w = split.train.stop, split.test.start, split.washout
    tr, te = slice(w, a - horizon), slice(b, len(X))
    W = ridge_fit(X[tr], y[tr], alpha=alpha, bias=True)
    yp = ridge_predict(X[te], W, bias=True).ravel()
    return metrics.nrmse(y[te], yp)


def _slice_readout(feats_full, n, V, which):
    """Slice the full (Z,X,Y,ZZ) feature matrix to a readout subset. Per virtual node
    the 4n block is [Z(n) X(n) Y(n) ZZ(n)]; keep the requested leading observables."""
    keep = {"Z": n, "ZXY": 3 * n, "ZXYZZ": 4 * n}[which]
    block = 4 * n
    cols = [v * block + j for v in range(V) for j in range(keep)]
    return feats_full[:, cols]


# ---------------------------------------------------------------------------
# 2 + 3. qubit-count scaling sweep (variance + NRMSE), exact expectation
# ---------------------------------------------------------------------------
def scaling_sweep(system, seeds=(0, 1, 2), n_list=None, log=print, store=None):
    # `store` is an optional, DEFAULT-OFF FeatureStore (store=None reproduces the
    # original computation byte-for-byte). Path 1 wires the V=4 featurize at line
    # below; the observable_variance V=1 featurize is wired separately (path 2).
    from qdepipe.experiment import ExperimentConfig
    from qdepipe.feature_store import maybe_cached_featurize
    n_list = n_list or N_QUBITS
    rows = []
    for n in n_list:
        npts = NPOINTS[n]
        u, split = _scaled_series(system, npts)
        # Key context mirrors _scaled_series (minmax/train/60-20-20). This config
        # matches finite_shot_sweep's exact case, so the two paths share entries.
        ecfg = ExperimentConfig(system=system, n_points=npts, scaler="minmax",
                                scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))
        # variance (mean over seeds) + NRMSE per readout (one featurize per seed)
        v1, v2 = [], []
        nr = {"Z": [], "ZXY": [], "ZXYZZ": []}
        for s in seeds:
            ov = observable_variance(u, n, seed=s, store=store, cfg=ecfg)
            v1.append(ov["mean_var_1local"]); v2.append(ov["mean_var_2local"])
            q = GateQRC(_rich_cfg(n, V=4, seed=s))
            feats_full = maybe_cached_featurize(store, q, u, ecfg)  # ONE featurize → slice
            for which in nr:
                f = _slice_readout(feats_full, n, 4, which)
                nr[which].append(_nrmse_from_feats(f, u, split))
        mv1, mv2 = float(np.mean(v1)), float(np.mean(v2))
        F = {"Z": 4 * n, "ZXY": 4 * 3 * n, "ZXYZZ": 4 * 4 * n}
        for which, label in (("Z", "Z"), ("ZXY", "Z+X+Y"), ("ZXYZZ", "Z+X+Y+ZZ")):
            rows.append({"system": system, "n_qubits": n, "n_points": npts,
                         "readout_set": label, "F": F[which],
                         "mean_var_1local": mv1, "mean_var_2local": mv2,
                         "NRMSE": float(np.mean(nr[which])),
                         "NRMSE_std": float(np.std(nr[which]))})
        log(f"    n={n:2d} (npts={npts}) var_1loc={mv1:.3g} var_2loc={mv2:.3g} "
            f"NRMSE[Z+X+Y={np.mean(nr['ZXY']):.3g}, +ZZ={np.mean(nr['ZXYZZ']):.3g}]")
    return rows


# ---------------------------------------------------------------------------
# 4. finite-shot: Z+X+Y vs Z+X+Y+ZZ degradation (n_qubits <= 8)
# ---------------------------------------------------------------------------
def finite_shot_sweep(system, seeds=(0, 1, 2), n_list=(4, 6, 8), log=print, store=None):
    # `store` is an optional, DEFAULT-OFF FeatureStore. store=None reproduces the
    # original computation byte-for-byte (maybe_cached_featurize falls through to
    # q.featurize(u)); a store makes re-runs free without changing any result.
    from qdepipe.experiment import ExperimentConfig
    from qdepipe.feature_store import maybe_cached_featurize
    rows = []
    for n in n_list:
        npts = NPOINTS[n]
        u, split = _scaled_series(system, npts)
        # Key context: must mirror how _scaled_series produced `u` (minmax,
        # train-scope, 60/20/20) so the cache key correctly identifies this matrix.
        ecfg = ExperimentConfig(system=system, n_points=npts, scaler="minmax",
                                scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))
        for shots in SHOTS:
            nr = {"ZXY": [], "ZXYZZ": []}
            for s in seeds:
                q = GateQRC(_rich_cfg(n, V=4, seed=s, shots=shots))
                feats_full = maybe_cached_featurize(store, q, u, ecfg)
                for which in nr:
                    f = _slice_readout(feats_full, n, 4, which)
                    nr[which].append(_nrmse_from_feats(f, u, split))
            for which, label in (("ZXY", "Z+X+Y"), ("ZXYZZ", "Z+X+Y+ZZ")):
                rows.append({"system": system, "n_qubits": n,
                             "readout_set": label, "shots": shots or "exact",
                             "NRMSE": float(np.mean(nr[which])),
                             "NRMSE_std": float(np.std(nr[which]))})
        log(f"    n={n}: shots done ({[s or 'exact' for s in SHOTS]})")
    return rows


# ---------------------------------------------------------------------------
# 5. headline matched-budget under shots: QRC-rich vs ESN(F) on Hénon
# ---------------------------------------------------------------------------
def matched_budget_shots(seeds=(0, 1, 2, 3, 4), log=print, store=None):
    # `store` (DEFAULT-OFF) caches the QRC-rich featurize only (the ESN(F)
    # reference is classical/cheap, left uncached). Pure one-step run_experiment,
    # no rollout — the proven run()-only hook applies directly.
    from qdepipe.experiment import run_experiment, ExperimentConfig
    from qdepipe.registry import build_forecaster
    from qdepipe.models import ESN, ESNConfig
    from qdepipe.forecasters import ReservoirForecaster
    system, n = "henon", 6
    F = 4 * 4 * n                                 # QRC-rich F at n=6 = 96
    be, bsc = 0.5, "minmax"                       # Hénon QRC-rich tuned best (encoding tune)
    rows = []
    # ESN(F) reference (shot-free; classical) under the same pipeline
    esn_vals = []
    for s in seeds:
        cfg = ExperimentConfig(system=system, n_points=1500, scaler=bsc, seed=s,
                               lookback=5, washout=100)
        esn = ReservoirForecaster(ESN(ESNConfig(units=F, seed=s)), f"ESN(F={F})",
                                  external_window=True)
        esn_vals.append(run_experiment(esn, cfg).nrmse)
    esn_nrmse = float(np.mean(esn_vals))
    for shots in SHOTS:
        qv = []
        for s in seeds:
            cfg = GateQRCConfig(n_qubits=n, encoding="depth", r=1, coupling="isingxx",
                                J_strength=1.0, channel="none", V=4, window=5,
                                readout=("Z", "X", "Y", "ZZ"), encode_scale=be,
                                seed=s, shots=shots)
            ecfg = ExperimentConfig(system=system, n_points=1500, scaler=bsc, seed=s,
                                    lookback=5, washout=100)
            qv.append(run_experiment(ReservoirForecaster(GateQRC(cfg), "QRC-rich", store=store), ecfg).nrmse)
        qn = float(np.mean(qv))
        rows.append({"system": system, "model": "QRC-rich", "F": F,
                     "shots": shots or "exact", "NRMSE_quantum": qn,
                     "NRMSE_ESN_F": esn_nrmse, "beats_ESN": bool(qn < esn_nrmse),
                     "margin_ratio_esn_over_qrc": esn_nrmse / qn})
        log(f"    shots={shots or 'exact':>5}: QRC-rich={qn:.4g} vs ESN(F)={esn_nrmse:.4g}"
            f"  -> {'QRC' if qn < esn_nrmse else 'ESN'} wins")
    return rows
