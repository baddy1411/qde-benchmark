#!/usr/bin/env python3
"""Equal-budget classical ESN tune.  ->  results/esn_budget_tune.csv

The quantum encoding tune evaluated exactly 14 configurations per quantum model
(2 scalers x 7 encode_scales; results/adv_A_encoding.csv, 42 rows over 3 models). Fairness requires the classical flagship to receive the SAME search
budget, declared. This grants the ESN 14 configurations per system:

    spectral_radius in {0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9}  x  leak_rate in {0.3, 1.0}

(the two knobs the RC literature ranks highest for ESNs; Lukosevicius 2012),
3 seeds each, all four systems. The best config per system is reported so the
thesis can state whether any headline verdict changes under an equally-tuned ESN.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ESN, ESNConfig

OUT = "results/esn_budget_tune.csv"
SYSTEMS = ["henon", "lorenz", "mackeyglass", "lorenz96"]
SR = [0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
LEAK = [0.3, 1.0]
SEEDS = [0, 1, 2]
N_POINTS = 1500

done = set()
if os.path.exists(OUT):
    d = pd.read_csv(OUT)
    done = {(r.system, float(r.spectral_radius), float(r.leak_rate), int(r.seed))
            for r in d.itertuples()}

for system in SYSTEMS:
    npts = 1200 if system == "lorenz96" else N_POINTS   # the Lorenz-96 sample-size convention
    for sr in SR:
        for leak in LEAK:
            for seed in SEEDS:
                if (system, sr, leak, seed) in done:
                    continue
                cfg = ExperimentConfig(system=system, n_points=npts, seed=seed,
                                       lookback=5, washout=100)
                esn = ESN(ESNConfig(units=300, spectral_radius=sr,
                                    leak_rate=leak, seed=seed))
                r = run_experiment(ReservoirForecaster(esn, "ESN", external_window=True),
                                   cfg)
                pd.DataFrame([{"system": system, "spectral_radius": sr,
                               "leak_rate": leak, "seed": seed, "nrmse": r.nrmse,
                               "budget_note": "14 configs/system == the quantum encoding-tune budget"}]
                             ).to_csv(OUT, mode="a", index=False,
                                      header=not os.path.exists(OUT))
                print(f"{system:12s} sr={sr} leak={leak} s{seed} NRMSE={r.nrmse:.4e}",
                      flush=True)

d = pd.read_csv(OUT).drop_duplicates(
    subset=["system", "spectral_radius", "leak_rate", "seed"], keep="last")
print("\n===== equal-budget ESN tune: best per system =====")
g = d.groupby(["system", "spectral_radius", "leak_rate"]).nrmse.mean().reset_index()
for system in SYSTEMS:
    s = g[g.system == system]
    b = s.loc[s.nrmse.idxmin()]
    dflt = s[(s.spectral_radius == 0.9) & (s.leak_rate == 0.3)]
    print(f"  {system:12s} best sr={b.spectral_radius} leak={b.leak_rate} "
          f"NRMSE={b.nrmse:.4e}   (default-config NRMSE="
          f"{float(dflt.nrmse.iloc[0]):.4e})")
print("wrote", OUT)
