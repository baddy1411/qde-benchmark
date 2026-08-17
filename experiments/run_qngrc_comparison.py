#!/usr/bin/env python3
"""Reconstructed producer for results/cross/qngrc_comparison.csv.

The committed artifact (cited in the thesis's qNG-RC section) had no producer
script in the repository; the original was a one-off that was never
committed. This runner reconstructs it from the artifact's own schema and the
committed model implementations, and MUST reproduce the committed values
before being accepted (verified by the rebuild comparator).

Rows per system: NG-RC(k=3,d=2), qNG-RC(k=3,d=2), qNG-RC-big(k=8,d=3),
ESN(300 units), QRC-rich (n=6, V=4, F=96). Single deterministic frame:
n_points=1500, seed 0, washout 100, lookback 5 (concentration convention).
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
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.models.qngrc import QNGRC, QNGRCConfig

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
NPTS = 1500
SEED = 0

rows = []
for system in SYSTEMS:
    arms = [
        ("NG-RC", ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC",
                                      store=None), 3, 2),
        ("qNG-RC", ReservoirForecaster(QNGRC(QNGRCConfig(k=3, degree=2)),
                                       "qNG-RC", store=None), 3, 2),
        ("qNG-RC-big", ReservoirForecaster(QNGRC(QNGRCConfig(k=8, degree=3)),
                                           "qNG-RC-big", store=None), 8, 3),
        ("ESN", ReservoirForecaster(ESN(ESNConfig(units=300, seed=SEED)), "ESN",
                                    external_window=True, store=None),
         np.nan, np.nan),
        ("QRC-rich", ReservoirForecaster(GateQRC(_rich_cfg(6, V=4, seed=SEED)),
                                         "QRC-rich", store=None), np.nan, np.nan),
    ]
    for label, fc, k, degree in arms:
        cfg = ExperimentConfig(system=system, n_points=NPTS, seed=SEED,
                               lookback=5, washout=100)
        r = run_experiment(fc, cfg)
        rows.append({"system": system, "model": label, "k": k, "degree": degree,
                     "n_features": r.n_features, "nrmse": r.nrmse})
        print(f"{system:12s} {label:11s} F={r.n_features:4d} "
              f"NRMSE={r.nrmse:.6e}", flush=True)

os.makedirs("results/cross", exist_ok=True)
pd.DataFrame(rows).to_csv("results/cross/qngrc_comparison.csv", index=False)
print("wrote results/cross/qngrc_comparison.csv")
