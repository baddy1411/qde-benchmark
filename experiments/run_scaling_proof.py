#!/usr/bin/env python3
"""Qubit-scaling proof: is there ANY qubit count at which the quantum
reservoir gains on matched classical baselines?  ->  results/scaling_proof/

At each n in {4,6,8,10}, all three systems:
  QRC-rich(n)  F = 16n   (V=4 · m=4 observables/qubit · n qubits; _rich_cfg exactly,
                          so prior 5-seed cache entries HIT)
  ESN(F)       matched feature budget (quality-vs-quantity control)
  ELM(F)       matched random-projection control (separates "quantum" from
               "any nonlinear projection")
  NG-RC        the tuned yardstick (degree 2 henon/MG — their required degree;
               degree 3 lorenz per the threats-chapter probe)

Seeds: quantum 10 at n<=8, 5 at n=10 (cost); classical 20. NPOINTS follows the
concentration convention {4:1500, 6:1500, 8:1200, 10:600} and is written per row.

Analysis (written even on partial resume):
  scores.csv — every (system, n, model, seed) cell with the resource tuple
  dm.csv     — per-seed DM: QRC vs ESN(F), QRC vs ELM(F), QRC vs NG-RC
  trend.csv  — per (system, n): median/IQR NRMSE per model, gap ratios
               QRC/ESN(F) and QRC/NG-RC with paired bootstrap CIs
Crash-safe: per-cell CSV append; done cells skipped; featurize cached.
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
from qdepipe.feature_store import FeatureStore
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.significance import (diebold_mariano, errors, paired_bootstrap_ci,
                                  median_iqr)

OUT = "results/scaling_proof"
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
N_LIST = [4, 6, 8, 10, 12]   # 12 = guarded best-effort arm (2 quantum seeds)
Q_SEEDS = {4: list(range(10)), 6: list(range(10)), 8: list(range(10)), 10: [0, 1, 2, 3, 4], 12: [0, 1]}
C_SEEDS = list(range(20))
LOOKBACK = 5          # = _rich_cfg window
NGRC_DEGREE = {"henon": 2, "lorenz": 3, "mackeyglass": 2}   # required degrees (threats ch.)

store = FeatureStore()


def build(model_key, n, seed, system):
    F = 16 * n
    if model_key == "qrc":
        return ReservoirForecaster(GateQRC(_rich_cfg(n, V=4, seed=seed)),
                                   f"QRC-rich(n={n})", store=store)
    if model_key == "esnF":
        return ReservoirForecaster(ESN(ESNConfig(units=F, seed=seed)), f"ESN(F={F})",
                                   external_window=True, store=store)
    if model_key == "elmF":
        return ReservoirForecaster(ELM(ELMConfig(units=F, lookback=LOOKBACK, seed=seed)),
                                   f"ELM(F={F})", store=store)
    if model_key == "ngrc":
        return ReservoirForecaster(
            NGRC(NGRCConfig(k=LOOKBACK, degree=NGRC_DEGREE[system])), "NG-RC", store=store)
    raise KeyError(model_key)


SEEDS_FOR = lambda mk, n: Q_SEEDS[n] if mk == "qrc" else C_SEEDS

done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)
    done = {(r.system, int(r.n_qubits_eq), r.model, int(r.seed)) for r in d.itertuples()}
    print(f"resuming: {len(done)} cells done", flush=True)

forecast_bank = {}

for n in N_LIST:                      # cheap n first: early complete signal
    npts = NPOINTS.get(n, 600)
    for system in SYSTEMS:
        if n == 12 and system != "henon":
            continue                      # n=12 is the guarded Henon-only arm
        for model_key in ["ngrc", "elmF", "esnF", "qrc"]:     # classical first
            for seed in SEEDS_FOR(model_key, n):
                cell = (system, n, model_key, seed)
                if cell in done:
                    continue
                cfg = ExperimentConfig(system=system, n_points=npts, seed=seed,
                                       lookback=LOOKBACK, washout=100)
                r = run_experiment(build(model_key, n, seed, system), cfg,
                                   keep_forecasts=True)
                row = {"system": system, "n_qubits_eq": n, "model": model_key,
                       "seed": seed, "nrmse": r.nrmse, "n_features": r.n_features,
                       "n_points": npts,
                       # resource tuple
                       "n_qubits": n if model_key == "qrc" else "",
                       "V": 4 if model_key == "qrc" else "",
                       "window": LOOKBACK, "shots": "exact" if model_key == "qrc" else "",
                       "readout": "Z+X+Y+ZZ" if model_key == "qrc" else "",
                       "ngrc_degree": NGRC_DEGREE[system] if model_key == "ngrc" else ""}
                pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                                           header=not os.path.exists(SCORES))
                forecast_bank[cell] = (r.y_true, r.y_pred)
                print(f"n={n:2d} {system:12s} {model_key:5s} seed{seed:2d} "
                      f"NRMSE={r.nrmse:.4e}", flush=True)

# --------------------------- analysis ----------------------------------------
sc = pd.read_csv(SCORES, keep_default_na=False)
sc["nrmse"] = sc["nrmse"].astype(float)
sc = sc.drop_duplicates(subset=["system", "n_qubits_eq", "model", "seed"], keep="last")


def get_forecast(system, n, model_key, seed):
    cell = (system, n, model_key, seed)
    if cell not in forecast_bank:
        cfg = ExperimentConfig(system=system, n_points=NPOINTS.get(n, 600), seed=seed,
                               lookback=LOOKBACK, washout=100)
        r = run_experiment(build(model_key, n, seed, system), cfg, keep_forecasts=True)
        forecast_bank[cell] = (r.y_true, r.y_pred)
    return forecast_bank[cell]


dm_rows, trend_rows = [], []
for system in SYSTEMS:
    for n in N_LIST:
        if n == 12 and system != "henon":
            continue                      # n=12 exists for Henon only
        d = sc[(sc.system == system) & (sc.n_qubits_eq == n)]
        if not len(d):
            continue
        qseeds = sorted(d[d.model == "qrc"].seed.astype(int))
        per = {m: d[d.model == m].set_index("seed").nrmse for m in
               ["qrc", "esnF", "elmF", "ngrc"]}
        # per-seed paired DM on the quantum seed set
        for seed in qseeds:
            for rival in ["esnF", "elmF", "ngrc"]:
                t1, p1 = get_forecast(system, n, "qrc", seed)
                t2, p2 = get_forecast(system, n, rival, seed)
                stat, pval = diebold_mariano(errors(t1, p1), errors(t2, p2))
                dm_rows.append({"system": system, "n": n, "comparison": f"qrc_vs_{rival}",
                                "seed": seed, "dm_stat": stat, "p": pval})
        # medians, IQRs, paired gap ratios + bootstrap CI (quantum seed set)
        row = {"system": system, "n": n, "F": 16 * n, "n_points": NPOINTS.get(n, 600),
               "n_seeds_q": len(qseeds)}
        for m in per:
            mi = median_iqr(per[m].values)
            row[f"{m}_median"] = mi["median"]
            row[f"{m}_iqr"] = mi["iqr"]
        q = per["qrc"].loc[qseeds].values
        for rival in ["esnF", "elmF", "ngrc"]:
            r_ = per[rival].loc[qseeds].values
            row[f"gap_qrc_over_{rival}"] = float(np.median(q / r_))
            ci = paired_bootstrap_ci(np.log10(q), np.log10(r_))    # log-gap CI
            row[f"loggap_{rival}_lo"], row[f"loggap_{rival}_hi"] = ci["ci_lo"], ci["ci_hi"]
            row[f"gap_{rival}_excludes_zero"] = ci["excludes_zero"]
        trend_rows.append(row)

pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "dm.csv"), index=False)
pd.DataFrame(trend_rows).to_csv(os.path.join(OUT, "trend.csv"), index=False)

print("\n===== qubit-scaling summary (median NRMSE) =====")
t = pd.DataFrame(trend_rows)
for system in SYSTEMS:
    d = t[t.system == system]
    if len(d):
        cols = ["n", "qrc_median", "esnF_median", "elmF_median", "ngrc_median",
                "gap_qrc_over_esnF", "gap_qrc_over_ngrc"]
        print(f"--- {system} ---")
        print(d[cols].to_string(index=False))
print("\nwrote", SCORES, "dm.csv trend.csv")
