#!/usr/bin/env python3
"""20-seed refresh of the headline (reservoir-family) leaderboard.
->  results/headline_20seed.csv

Declared seed policy (goes in thesis methods): stochastic classical
reservoir-family models 20 seeds; quantum 10 (n<=6) / 5 (n=8,10) / 2 (n=12,
best-effort); deterministic models (NG-RC, Linear) one run by construction;
tree/DL benchmark-table models remain 5 seeds (they carry no headline claim).
Reports median + IQR per the review standard.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.feature_store import FeatureStore
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.significance import median_iqr

OUT = "results/headline_20seed.csv"
SEEDS = list(range(20))
LOOKBACK = 5
NGRC_CFG = {"henon": (5, 2), "lorenz": (5, 3), "mackeyglass": (5, 2),
            "lorenz96": (12, 2)}
# equal-budget tuned ESN per system (results/esn_budget_tune.csv)
ESN_TUNED = {"henon": (0.7, 1.0), "lorenz": (0.9, 1.0), "mackeyglass": (1.1, 1.0),
             "lorenz96": (0.7, 1.0)}
NPTS = {"henon": 1500, "lorenz": 1500, "mackeyglass": 1500, "lorenz96": 1200}

store = FeatureStore()


def build(model_key, system, seed):
    if model_key == "ngrc":
        k, d = NGRC_CFG[system]
        return ReservoirForecaster(NGRC(NGRCConfig(k=k, degree=d)), "NG-RC", store=store)
    if model_key == "esn":
        return ReservoirForecaster(ESN(ESNConfig(units=300, seed=seed)), "ESN",
                                   external_window=True, store=store)
    if model_key == "esn_tuned":
        sr, lk = ESN_TUNED[system]
        return ReservoirForecaster(
            ESN(ESNConfig(units=300, spectral_radius=sr, leak_rate=lk, seed=seed)),
            "ESN-tuned", external_window=True, store=store)
    if model_key == "elm":
        return ReservoirForecaster(ELM(ELMConfig(units=300, lookback=LOOKBACK,
                                                 seed=seed)), "ELM", store=store)
    raise KeyError(model_key)


MODELS = {"ngrc": [0], "esn": SEEDS, "esn_tuned": SEEDS, "elm": SEEDS}

done = set()
if os.path.exists(OUT):
    d = pd.read_csv(OUT, keep_default_na=False)
    done = {(r.system, r.model, int(r.seed)) for r in d.itertuples()}

for system in NPTS:
    for model_key, seeds in MODELS.items():
        for seed in seeds:
            if (system, model_key, seed) in done:
                continue
            k = NGRC_CFG[system][0] if model_key == "ngrc" else LOOKBACK
            cfg = ExperimentConfig(system=system, n_points=NPTS[system], seed=seed,
                                   lookback=k, washout=100)
            r = run_experiment(build(model_key, system, seed), cfg)
            pd.DataFrame([{"system": system, "model": model_key, "seed": seed,
                           "nrmse": r.nrmse, "n_features": r.n_features,
                           "n_points": NPTS[system]}]).to_csv(
                OUT, mode="a", index=False, header=not os.path.exists(OUT))
            print(f"{system:12s} {model_key:9s} s{seed:2d} NRMSE={r.nrmse:.4e}",
                  flush=True)

d = pd.read_csv(OUT, keep_default_na=False)
d["nrmse"] = d["nrmse"].astype(float)
d = d.drop_duplicates(subset=["system", "model", "seed"], keep="last")
print("\n===== 20-seed headline (median / IQR) =====")
for system in NPTS:
    print(f"--- {system} ---")
    for m in MODELS:
        mi = median_iqr(d[(d.system == system) & (d.model == m)].nrmse.values)
        print(f"  {m:9s} median={mi['median']:.4e}  IQR={mi['iqr']:.2e}  n={mi['n']}")
print("wrote", OUT)
