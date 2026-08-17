#!/usr/bin/env python3
"""RF-QRC-style leaky-memory variant (Ahmed/Tennie/Magri design)
inside our fair pipeline.  ->  results/leaky/{scores,dm}.csv

Question: does classical leaky integration on the feature matrix (the RF-QRC
recipe: our windowed recurrence-free QRC + r_t = (1-eps) r_{t-1} + eps m_t)
lift the QRC over its matched classical comparators — in the exact-expectation
regime and, critically, under finite shots (their claimed shot-robustness)?

Fairness: the identical leaky sweep is applied to the matched ESN(F), ELM(F)
and NG-RC feature matrices — if leaky helps everyone equally, it is a generic
post-processing win, not a quantum one.

eps=1.0 is literally the identity (unit-gated), so the no-leak baseline is one
point of the same sweep. Featurize is cached PRE-postprocess (key excludes
postprocess; verified), so all eps variants share one featurize per (model,
system, seed): the sweep is nearly free.

Crash-safe: appends per-cell rows to scores.csv; already-present cells skipped.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.feature_store import FeatureStore
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC, GateQRCConfig
from qdepipe.pipeline.postprocess import LeakyMemory
from qdepipe.significance import diebold_mariano, errors

OUT = "results/leaky"
SYSTEMS = ["henon", "lorenz", "mackeyglass"]
SEEDS = [0, 1, 2, 3, 4]
EPS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]          # 1.0 == identity == baseline
N_POINTS, N_QUBITS, LOOKBACK = 1500, 6, 5
F = 4 * 4 * N_QUBITS                            # QRC-rich F at n=6 = 96
SHOTS = [None, 8192, 1024]                      # shot arm runs on henon only

os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")
store = FeatureStore()


def qrc_cfg(seed, shots=None):
    return GateQRCConfig(n_qubits=N_QUBITS, encoding="depth", r=1,
                         coupling="isingxx", J_strength=1.0, channel="none",
                         V=4, window=LOOKBACK, readout=("Z", "X", "Y", "ZZ"),
                         encode_scale=1.0, seed=seed, shots=shots)


def build(model_key, seed, shots=None):
    if model_key == "qrc_rich":
        return ReservoirForecaster(GateQRC(qrc_cfg(seed, shots)), "QRC-rich", store=store)
    if model_key == "esn_matched":
        return ReservoirForecaster(ESN(ESNConfig(units=F, seed=seed)), f"ESN(F={F})",
                                   external_window=True, store=store)
    if model_key == "elm_matched":
        return ReservoirForecaster(ELM(ELMConfig(units=F, lookback=LOOKBACK, seed=seed)),
                                   f"ELM(F={F})", store=store)
    if model_key == "ngrc":
        return ReservoirForecaster(NGRC(NGRCConfig(k=LOOKBACK, degree=2)), "NG-RC", store=store)
    raise KeyError(model_key)


MODELS = ["qrc_rich", "esn_matched", "elm_matched", "ngrc"]

done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)   # keep the literal "None" string
    done = {(r.system, r.model, int(r.seed), float(r.eps), str(r.shots)) for r in d.itertuples()}
    print(f"resuming: {len(done)} cells already done")

forecast_bank = {}   # (system, model, seed, eps, shots) -> (y_true, y_pred), exact arm only

for system in SYSTEMS:
    for shots in SHOTS:
        if shots is not None and system != "henon":
            continue                                    # shot arm: henon only
        for model_key in MODELS:
            if shots is not None and model_key != "qrc_rich":
                continue                                # shots only apply to QRC
            for seed in SEEDS:
                for eps in EPS:
                    cell = (system, model_key, seed, eps, str(shots))
                    if cell in done:
                        continue
                    post = [] if eps == 1.0 else [LeakyMemory(eps=eps)]
                    # windowing left at default "internal" — the matched-ESN
                    # convention of matched_budget_shots (ESN keeps its own
                    # recurrence; external delay vectors are NOT engaged).
                    cfg = ExperimentConfig(system=system, n_points=N_POINTS,
                                           seed=seed, lookback=LOOKBACK,
                                           washout=100, postprocess=post)
                    fc = build(model_key, seed, shots)
                    r = run_experiment(fc, cfg, keep_forecasts=True)
                    row = {"system": system, "model": model_key, "seed": seed,
                           "eps": eps, "shots": str(shots), "nrmse": r.nrmse,
                           "n_features": r.n_features,
                           # resource tuple (the per-row convention adopted project-wide)
                           "n_qubits": N_QUBITS if model_key == "qrc_rich" else "",
                           "V": 4 if model_key == "qrc_rich" else "",
                           "window": LOOKBACK,
                           "readout": "Z+X+Y+ZZ" if model_key == "qrc_rich" else "",
                           "n_points": N_POINTS}
                    pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                                               header=not os.path.exists(SCORES))
                    forecast_bank[cell] = (r.y_true, r.y_pred)
                    print(f"{system:12s} {model_key:12s} seed{seed} eps={eps:<4} "
                          f"shots={str(shots):>5}  NRMSE={r.nrmse:.4e}", flush=True)

# ---------------- DM analysis (exact arm) -----------------------------------
# Per (system, seed): best-eps QRC vs baseline QRC (does leaky help at all?),
# and best-eps QRC vs best-eps matched ESN / NG-RC (does it change the verdict?).
sc = pd.read_csv(SCORES, keep_default_na=False)      # "None" stays a string
sc["eps"] = sc["eps"].astype(float)
sc["nrmse"] = sc["nrmse"].astype(float)
sc = sc.drop_duplicates(subset=["system", "model", "seed", "eps", "shots"], keep="last")


def get_forecast(system, model_key, seed, eps):
    """Bank hit, else recompute (features are cached, so this is cheap)."""
    cell = (system, model_key, seed, eps, "None")
    if cell not in forecast_bank:
        post = [] if eps == 1.0 else [LeakyMemory(eps=eps)]
        cfg = ExperimentConfig(system=system, n_points=N_POINTS, seed=seed,
                               lookback=LOOKBACK, washout=100, postprocess=post)
        r = run_experiment(build(model_key, seed), cfg, keep_forecasts=True)
        forecast_bank[cell] = (r.y_true, r.y_pred)
    return forecast_bank[cell]


dm_rows = []
ex = sc[sc.shots == "None"]
for system in SYSTEMS:
    d = ex[ex.system == system]
    # best eps per model by mean NRMSE across seeds
    best = {m: float(d[d.model == m].groupby("eps").nrmse.mean().idxmin())
            for m in MODELS}
    for seed in SEEDS:
        pairs = [
            ("qrc_leaky_vs_qrc_base", ("qrc_rich", best["qrc_rich"]), ("qrc_rich", 1.0)),
            ("qrcBest_vs_esnBest", ("qrc_rich", best["qrc_rich"]),
             ("esn_matched", best["esn_matched"])),
            ("qrcBest_vs_ngrcBest", ("qrc_rich", best["qrc_rich"]),
             ("ngrc", best["ngrc"])),
        ]
        for label, (m1, e1), (m2, e2) in pairs:
            t1, p1 = get_forecast(system, m1, seed, float(e1))
            t2, p2 = get_forecast(system, m2, seed, float(e2))
            stat, pval = diebold_mariano(errors(t1, p1), errors(t2, p2))
            dm_rows.append({"system": system, "comparison": label, "seed": seed,
                            "eps_1": e1, "eps_2": e2, "dm_stat": stat, "p": pval})
pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "dm.csv"), index=False)

# ---------------- summary ----------------------------------------------------
print("\n===== leaky-memory summary (mean NRMSE over seeds) =====")
for system in SYSTEMS:
    d = ex[ex.system == system]
    print(f"--- {system} ---")
    tab = d.groupby(["model", "eps"]).nrmse.mean().unstack()
    print(tab.round(6).to_string())
sh = sc[(sc.shots != "None") & (sc.system == "henon")]
if len(sh):
    print("--- henon, finite shots (QRC-rich) ---")
    print(sh.groupby(["shots", "eps"]).nrmse.mean().unstack().round(4).to_string())
print("\nwrote", SCORES, "and", os.path.join(OUT, "dm.csv"))
