#!/usr/bin/env python3
"""Follow-up experiments (NOT in the thesis): can literature read-out tricks or
more virtual nodes rescue the quantum reservoir?

A. Polynomial read-out on quantum features (squares; and with pairwise
   products) — the trick that makes ELM+poly2 strong, applied identically to
   the cached-free, freshly recomputed QRC-rich features and to the matched
   ELM(F) control (fairness: both sides get the same trick).
B. Delay-stacked read-out (concat of the last d feature vectors, causal,
   train-fitted pipeline stage) — NG-RC's explicit-memory trick applied to
   both sides.
C. Virtual-node sweep V in {4, 8, 16} at n=6 — the one untouched capacity
   knob (temporal multiplexing); ELM(F) matched at the same measured F.

Conventions copied from run_scaling_proof.py: same systems, NPOINTS, lookback,
washout, NG-RC degrees, per-seed DM. EVERYTHING is computed fresh:
store=None -> no feature-cache reads or writes anywhere.
Outputs under results/followup/ ; committed artifacts untouched.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.concentration import _rich_cfg, NPOINTS
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.pipeline.base import Transformer
from qdepipe.pipeline.postprocess import Polynomial
from qdepipe.significance import diebold_mariano, errors

OUT = "results/followup"
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
SEEDS = [0, 1, 2, 3, 4]
LOOKBACK = 5
NGRC_DEGREE = {"henon": 2, "lorenz": 3, "mackeyglass": 2}


class DelayStack(Transformer):
    """Causal delay embedding of the FEATURE matrix: row t = concat of feature
    vectors at t, t-1, ..., t-d+1 (edge rows replicate the earliest available
    vector). Past-only, so leakage-safe; train-fitted column count."""

    def __init__(self, d: int = 3):
        self.d = d

    def fit(self, x):
        self.n = np.asarray(x).shape[1]
        self.fitted = True
        return self

    def transform(self, x):
        self._check_fitted()
        m = np.asarray(x, dtype=float)
        cols = []
        for lag in range(self.d):
            shifted = np.roll(m, lag, axis=0)
            shifted[:lag] = m[0]          # replicate earliest row: causal padding
            cols.append(shifted)
        return np.concatenate(cols, axis=1)


READOUTS = {
    "base":          None,
    "poly2sq":       [Polynomial(interactions=False)],
    "poly2int":      [Polynomial(interactions=True)],
    "delay3":        [DelayStack(3)],
    "delay3+poly2":  [DelayStack(3), Polynomial(interactions=False)],
}

# (experiment, system, n, V, model, readout, seed) cells
CELLS = []
# --- A+B: read-out tricks at V=4, n in {6, 8} ------------------------------
for system in SYSTEMS:
    for n in (6, 8):
        for ro in READOUTS:
            for mk in ("qrc", "elmF"):
                for s in SEEDS:
                    CELLS.append(("readout", system, n, 4, mk, ro, s))
        for s in SEEDS:
            CELLS.append(("readout", system, n, 4, "ngrc", "base", s))
# --- C: V sweep at n=6, base read-out --------------------------------------
for system in SYSTEMS:
    for V in (4, 8, 16):
        for mk in ("qrc", "elmF"):
            for s in SEEDS:
                CELLS.append(("vsweep", system, 6, V, mk, "base", s))
    for s in SEEDS:
        CELLS.append(("vsweep", system, 6, 4, "ngrc", "base", s))


def build(mk, system, n, V, seed):
    F = V * 4 * n
    if mk == "qrc":
        return ReservoirForecaster(GateQRC(_rich_cfg(n, V=V, seed=seed)),
                                   f"QRC-rich(n={n},V={V})", store=None)
    if mk == "elmF":
        return ReservoirForecaster(ELM(ELMConfig(units=F, lookback=LOOKBACK, seed=seed)),
                                   f"ELM(F={F})", store=None)
    if mk == "ngrc":
        return ReservoirForecaster(NGRC(NGRCConfig(k=LOOKBACK, degree=NGRC_DEGREE[system])),
                                   "NG-RC", store=None)
    raise KeyError(mk)


done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)
    done = {(r.experiment, r.system, int(r.n), int(r.V), r.model, r.readout, int(r.seed))
            for r in d.itertuples()}
    print(f"resuming: {len(done)} cells done", flush=True)

bank = {}
for cell in CELLS:
    exp, system, n, V, mk, ro, seed = cell
    key = (system, n, V, mk, ro, seed)          # forecast bank key (exp-agnostic)
    if cell in done:
        continue
    if key in bank and exp != "readout":        # vsweep may reuse a readout run's cell
        pass
    npts = NPOINTS.get(n, 600)
    cfg = ExperimentConfig(system=system, n_points=npts, seed=seed,
                           lookback=LOOKBACK, washout=100,
                           postprocess=READOUTS[ro])
    r = run_experiment(build(mk, system, n, V, seed), cfg, keep_forecasts=True)
    bank[key] = (r.y_true, r.y_pred)
    row = {"experiment": exp, "system": system, "n": n, "V": V, "model": mk,
           "readout": ro, "seed": seed, "nrmse": r.nrmse,
           "n_features": r.n_features, "n_points": npts}
    pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                               header=not os.path.exists(SCORES))
    print(f"[{exp}] {system:12s} n={n} V={V:2d} {mk:5s} {ro:12s} seed{seed} "
          f"NRMSE={r.nrmse:.4e} (F'={r.n_features})", flush=True)


def forecast(system, n, V, mk, ro, seed):
    key = (system, n, V, mk, ro, seed)
    if key not in bank:
        cfg = ExperimentConfig(system=system, n_points=NPOINTS.get(n, 600), seed=seed,
                               lookback=LOOKBACK, washout=100, postprocess=READOUTS[ro])
        r = run_experiment(build(mk, system, n, V, seed), cfg, keep_forecasts=True)
        bank[key] = (r.y_true, r.y_pred)
    return bank[key]


# ----------------------------- analysis -------------------------------------
sc = pd.read_csv(SCORES, keep_default_na=False)
sc["nrmse"] = sc["nrmse"].astype(float)
sc = sc.drop_duplicates(subset=["experiment", "system", "n", "V", "model",
                                "readout", "seed"], keep="last")
dm_rows = []
for system in SYSTEMS:
    # A+B: per (n, trick): qrc+trick vs qrc base / vs elmF+trick / vs ngrc
    for n in (6, 8):
        for ro in READOUTS:
            if ro == "base":
                continue
            for s in SEEDS:
                t1, p1 = forecast(system, n, 4, "qrc", ro, s)
                for label, rival in [("qrc_base", ("qrc", "base")),
                                     ("elmF_same", ("elmF", ro)),
                                     ("ngrc", ("ngrc", "base"))]:
                    t2, p2 = forecast(system, n, 4, rival[0], rival[1], s)
                    stat, p = diebold_mariano(errors(t1, p1), errors(t2, p2))
                    dm_rows.append({"experiment": "readout", "system": system,
                                    "n": n, "V": 4, "readout": ro, "vs": label,
                                    "seed": s, "dm_stat": stat, "p": p})
    # C: per V: qrc vs elmF(F) and qrc(V) vs qrc(V=4)
    for V in (4, 8, 16):
        for s in SEEDS:
            t1, p1 = forecast(system, 6, V, "qrc", "base", s)
            t2, p2 = forecast(system, 6, V, "elmF", "base", s)
            stat, p = diebold_mariano(errors(t1, p1), errors(t2, p2))
            dm_rows.append({"experiment": "vsweep", "system": system, "n": 6,
                            "V": V, "readout": "base", "vs": "elmF_sameF",
                            "seed": s, "dm_stat": stat, "p": p})
            if V != 4:
                t2, p2 = forecast(system, 6, 4, "qrc", "base", s)
                stat, p = diebold_mariano(errors(t1, p1), errors(t2, p2))
                dm_rows.append({"experiment": "vsweep", "system": system, "n": 6,
                                "V": V, "readout": "base", "vs": "qrc_V4",
                                "seed": s, "dm_stat": stat, "p": p})
pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "dm.csv"), index=False)

print("\n===== median NRMSE summaries =====")
med = sc.groupby(["experiment", "system", "n", "V", "model", "readout"]).nrmse.median()
print(med.to_string(float_format="%.4e"))
print("\nwrote", SCORES, "and dm.csv")
