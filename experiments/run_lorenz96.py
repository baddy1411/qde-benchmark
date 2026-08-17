#!/usr/bin/env python3
"""Lorenz-96 (D=20, F=8) higher-dimensional stress test.
->  results/lorenz96/{ngrc_probe,scores,dm,mcs}.csv

The regime where the literature's quantum-advantage claims live (RF-QRC,
correlated-spin experiments on higher-dimensional chaos). Either outcome is a
finding: null extends, or the gap closes and we've located a regime boundary.

Protocol (mirrors the thesis conventions):
  0. Lyapunov data gate (calibrated 1.5449; PASS required).
  1. NG-RC fairness probe: (k, degree) grid so the yardstick isn't under-tuned
     — the threats-chapter error class, prevented by construction.
  2. Matrix at ONE common n_points=1200 (single shared test set so DM/MCS pair
     cleanly): NG-RC*, ESN(300), ELM(300), Linear, QRC-rich n∈{6,8} + matched
     ESN(F)/ELM(F) at each F=16n. Classical 10 seeds, quantum 5.
  3. Per-seed DM (QRC vs each classical) + per-seed MCS over the model set;
     report the fraction of seeds each model enters the best-set (thesis
     "per seed then aggregate" convention; p-values never averaged).
Crash-safe per-cell append; QRC featurize cached (fixed IC ⇒ key precondition
holds, unlike the initial-condition study).
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.concentration import _rich_cfg
from qdepipe.data.lorenz96 import lyapunov_gate
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster, WindowedForecaster
from qdepipe.feature_store import FeatureStore
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.significance import (diebold_mariano, errors, model_confidence_set,
                                  median_iqr)

OUT = "results/lorenz96"
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")
SYSTEM, N_POINTS, LOOKBACK = "lorenz96", 1200, 5
C_SEEDS, Q_SEEDS = list(range(10)), list(range(5))
N_QRC = [6, 8]

store = FeatureStore()

# ---- 0. data gate ------------------------------------------------------------
le, ok = lyapunov_gate()
print(f"[gate] lorenz96 lambda estimate {le:.4f} vs calibrated 1.5449: "
      f"{'PASS' if ok else 'FAIL'}", flush=True)
if not ok:
    raise SystemExit("Lyapunov gate FAILED — data generation wrong; not benchmarking.")

# ---- 1. NG-RC fairness probe ---------------------------------------------------
PROBE = os.path.join(OUT, "ngrc_probe.csv")
if not os.path.exists(PROBE):
    rows = []
    for k in [3, 5, 8, 12]:
        for deg in [2, 3]:
            cfg = ExperimentConfig(system=SYSTEM, n_points=N_POINTS,
                                   lookback=k, washout=100)
            r = run_experiment(ReservoirForecaster(
                NGRC(NGRCConfig(k=k, degree=deg)), "NG-RC", store=store), cfg)
            rows.append({"k": k, "degree": deg, "nrmse": r.nrmse,
                         "n_features": r.n_features})
            print(f"[probe] NG-RC k={k:2d} d={deg}: NRMSE={r.nrmse:.4e}", flush=True)
    pd.DataFrame(rows).to_csv(PROBE, index=False)
probe = pd.read_csv(PROBE)
best = probe.loc[probe.nrmse.idxmin()]
NG_K, NG_D = int(best.k), int(best.degree)
print(f"[probe] best NG-RC: k={NG_K} degree={NG_D} (NRMSE={best.nrmse:.4e})", flush=True)


# ---- 2. the matrix -------------------------------------------------------------
def build(model_key, seed):
    if model_key == "ngrc":
        return ReservoirForecaster(NGRC(NGRCConfig(k=NG_K, degree=NG_D)), "NG-RC",
                                   store=store)
    if model_key == "esn300":
        return ReservoirForecaster(ESN(ESNConfig(units=300, seed=seed)), "ESN",
                                   external_window=True, store=store)
    if model_key == "esn300t":
        # equal-budget tuned ESN (results/esn_budget_tune.csv: lorenz96 best
        # sr=0.7, leak=1.0 — the review's "optimized classical baseline")
        return ReservoirForecaster(
            ESN(ESNConfig(units=300, spectral_radius=0.7, leak_rate=1.0, seed=seed)),
            "ESN-tuned", external_window=True, store=store)
    if model_key == "elm300":
        return ReservoirForecaster(ELM(ELMConfig(units=300, lookback=LOOKBACK,
                                                 seed=seed)), "ELM", store=store)
    if model_key == "linear":
        from qdepipe.registry import _ridge
        return WindowedForecaster(lambda: _ridge(1e-6), "Linear-Ridge")
    if model_key.startswith("qrc"):
        n = int(model_key[3:])
        return ReservoirForecaster(GateQRC(_rich_cfg(n, V=4, seed=seed)),
                                   f"QRC-rich(n={n})", store=store)
    if model_key.startswith("esnF"):
        F = int(model_key[4:])
        return ReservoirForecaster(ESN(ESNConfig(units=F, seed=seed)), f"ESN(F={F})",
                                   external_window=True, store=store)
    if model_key.startswith("elmF"):
        F = int(model_key[4:])
        return ReservoirForecaster(ELM(ELMConfig(units=F, lookback=LOOKBACK,
                                                 seed=seed)), f"ELM(F={F})", store=store)
    raise KeyError(model_key)


MODEL_SEEDS = {"ngrc": [0], "linear": [0], "esn300": C_SEEDS,
               "esn300t": C_SEEDS, "elm300": C_SEEDS}
for n in N_QRC:
    MODEL_SEEDS[f"qrc{n}"] = Q_SEEDS
    MODEL_SEEDS[f"esnF{16 * n}"] = C_SEEDS
    MODEL_SEEDS[f"elmF{16 * n}"] = C_SEEDS

done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)
    done = {(r.model, int(r.seed)) for r in d.itertuples()}
    print(f"resuming: {len(done)} cells done", flush=True)

forecast_bank = {}
for model_key, seeds in MODEL_SEEDS.items():
    for seed in seeds:
        if (model_key, seed) in done:
            continue
        cfg = ExperimentConfig(system=SYSTEM, n_points=N_POINTS, seed=seed,
                               lookback=(NG_K if model_key == "ngrc" else LOOKBACK),
                               washout=100)
        r = run_experiment(build(model_key, seed), cfg, keep_forecasts=True)
        n_q = int(model_key[3:]) if model_key.startswith("qrc") else ""
        row = {"model": model_key, "seed": seed, "nrmse": r.nrmse,
               "n_features": r.n_features, "n_points": N_POINTS,
               "n_qubits": n_q, "V": 4 if n_q else "", "window": LOOKBACK,
               "shots": "exact" if n_q else "",
               "readout": "Z+X+Y+ZZ" if n_q else "",
               "ngrc_kd": f"k{NG_K}d{NG_D}" if model_key == "ngrc" else ""}
        pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                                   header=not os.path.exists(SCORES))
        forecast_bank[(model_key, seed)] = (r.y_true, r.y_pred)
        print(f"{model_key:9s} seed{seed}  NRMSE={r.nrmse:.4e}", flush=True)


def get_forecast(model_key, seed):
    if (model_key, seed) not in forecast_bank:
        cfg = ExperimentConfig(system=SYSTEM, n_points=N_POINTS, seed=seed,
                               lookback=(NG_K if model_key == "ngrc" else LOOKBACK),
                               washout=100)
        r = run_experiment(build(model_key, seed), cfg, keep_forecasts=True)
        forecast_bank[(model_key, seed)] = (r.y_true, r.y_pred)
    return forecast_bank[(model_key, seed)]


# ---- 3. DM + MCS per seed ------------------------------------------------------
# Test sets are aligned EXCEPT NG-RC runs at its tuned lookback NG_K, which can
# shift y-alignment; verified below and DM computed on the overlapping tail.
dm_rows, mcs_rows = [], []
JOINT = ["ngrc", "esn300", "esn300t", "elm300"] + [f"qrc{n}" for n in N_QRC] \
        + [f"esnF{16 * n}" for n in N_QRC]
for seed in Q_SEEDS:
    fc = {m: get_forecast(m, seed if len(MODEL_SEEDS[m]) > 1 else 0) for m in JOINT}
    L = min(len(v[0]) for v in fc.values())
    tails = {m: (v[0][-L:], v[1][-L:]) for m, v in fc.items()}
    y0 = tails[JOINT[0]][0]
    assert all(np.allclose(v[0], y0) for v in tails.values()), "test sets misaligned"
    for n in N_QRC:
        for rival in ["ngrc", "esn300", "elm300", f"esnF{16 * n}"]:
            e1 = errors(*tails[f"qrc{n}"])
            e2 = errors(*tails[rival])
            stat, pval = diebold_mariano(e1, e2)
            dm_rows.append({"seed": seed, "comparison": f"qrc{n}_vs_{rival}",
                            "dm_stat": stat, "p": pval})
    losses = np.stack([errors(*tails[m]) ** 2 for m in JOINT])
    mcs = model_confidence_set(losses, names=JOINT, seed=seed)
    for m in JOINT:
        mcs_rows.append({"seed": seed, "model": m, "mcs_p": mcs["mcs_pvalue"][m],
                         "in_75": m in mcs["in_set"][0.25],
                         "in_90": m in mcs["in_set"][0.10]})
pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "dm.csv"), index=False)
pd.DataFrame(mcs_rows).to_csv(os.path.join(OUT, "mcs.csv"), index=False)

# ---- summary --------------------------------------------------------------------
sc = pd.read_csv(SCORES, keep_default_na=False)
sc["nrmse"] = sc["nrmse"].astype(float)
sc = sc.drop_duplicates(subset=["model", "seed"], keep="last")
print("\n===== Lorenz-96 summary (median NRMSE / IQR over seeds) =====")
for m in MODEL_SEEDS:
    mi = median_iqr(sc[sc.model == m].nrmse.values)
    print(f"  {m:9s} median={mi['median']:.4e}  IQR={mi['iqr']:.1e}  n={mi['n']}")
mc = pd.DataFrame(mcs_rows)
print("\nMCS best-set membership (fraction of seeds, 90% level):")
print(mc.groupby("model").in_90.mean().round(2).to_string())
print("\nwrote", SCORES, "dm.csv mcs.csv")
