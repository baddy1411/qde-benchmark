#!/usr/bin/env python3
"""Qubit-scaling best-effort n=12 arm (guarded).  Appends to results/scaling_proof/scores.csv.

n=12 statevector featurize measured at ~21 s/timestep (4096-dim, full-matrix
gate composition) -> ~3.5 h per seed at 600 points. Guarded: Henon only,
2 quantum seeds, MemoryError/KeyboardInterrupt-safe, per-cell append. The
matched classical arms (F=192) run at 20 seeds as elsewhere. NPOINTS=600
follows the n=10 convention (documented in-row).
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from qdepipe.concentration import _rich_cfg
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.feature_store import FeatureStore
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC

OUT = "results/scaling_proof"
SCORES = os.path.join(OUT, "scores.csv")
SYSTEM, N, NPTS, LOOKBACK, F = "henon", 12, 600, 5, 192
store = FeatureStore()

done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)
    done = {(r.system, int(r.n_qubits_eq), r.model, int(r.seed)) for r in d.itertuples()}


def cells():
    yield "ngrc", 0                    # deterministic — one seed
    for seed in range(20):             # classical arms, 20 seeds (cheap)
        yield "esnF", seed
        yield "elmF", seed
    for seed in [0, 1]:                # quantum arm, 2 seeds (~3.5 h each)
        yield "qrc", seed


def build(model_key, seed):
    if model_key == "qrc":
        return ReservoirForecaster(GateQRC(_rich_cfg(N, V=4, seed=seed)),
                                   f"QRC-rich(n={N})", store=store)
    if model_key == "esnF":
        return ReservoirForecaster(ESN(ESNConfig(units=F, seed=seed)), f"ESN(F={F})",
                                   external_window=True, store=store)
    if model_key == "elmF":
        return ReservoirForecaster(ELM(ELMConfig(units=F, lookback=LOOKBACK, seed=seed)),
                                   f"ELM(F={F})", store=store)
    if model_key == "ngrc":
        return ReservoirForecaster(NGRC(NGRCConfig(k=LOOKBACK, degree=2)), "NG-RC",
                                   store=store)
    raise KeyError(model_key)


seen = set()
for model_key, seed in cells():
    if (model_key, seed) in seen:
        continue
    seen.add((model_key, seed))
    cell = (SYSTEM, N, model_key, seed)
    if cell in done:
        continue
    cfg = ExperimentConfig(system=SYSTEM, n_points=NPTS, seed=seed,
                           lookback=LOOKBACK, washout=100)
    try:
        r = run_experiment(build(model_key, seed), cfg)
    except MemoryError:
        print(f"MEMORY-FAIL {model_key} seed{seed} — documented, aborting n=12 arm",
              flush=True)
        break
    row = {"system": SYSTEM, "n_qubits_eq": N, "model": model_key, "seed": seed,
           "nrmse": r.nrmse, "n_features": r.n_features, "n_points": NPTS,
           "n_qubits": N if model_key == "qrc" else "",
           "V": 4 if model_key == "qrc" else "", "window": LOOKBACK,
           "shots": "exact" if model_key == "qrc" else "",
           "readout": "Z+X+Y+ZZ" if model_key == "qrc" else "",
           "ngrc_degree": 2 if model_key == "ngrc" else ""}
    pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                               header=not os.path.exists(SCORES))
    print(f"n=12 {model_key:5s} seed{seed:2d} NRMSE={r.nrmse:.4e}", flush=True)
print("n=12 arm complete")
