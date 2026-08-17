#!/usr/bin/env python3
"""Dynamics-level entanglement ablation.  ->  results/entanglement/

The existing ablation is readout-level (drop ZZ features). This one removes
entanglement from the DYNAMICS: J_strength=0 makes the IsingXX fixed layer the
identity (unit-gated), so the circuit is a product of single-qubit operations —
zero entanglement by construction (verified: S(rho_A) < 1e-10).

Two arms, because encoding reach matters (discovered in unit-gating):
  depth arm  — production QRC-rich encoding injects input on qubit 0 ONLY, so
               J=0 leaves qubits 1..n-1 inert: this measures "remove coupling
               from the production config" (transport + entanglement).
  width arm  — width encoding with r=n injects input on EVERY qubit, so J=0 is
               n independent single-qubit reservoirs: the fair Pfeffer-style
               separability test (equal encoding reach, entanglement removed).

Plus a J-sweep at n=6 (regime scan: is default J=1 a strawman regime?) with the
half-chain entanglement entropy S(rho_A) measured for every config — so the
conclusion can be "S ~ x ebits was present and bought Delta-NRMSE ~ y".

J=0 note: seed only draws the couplings J_i ~ U(-J/2, +J/2), so J=0 models are
deterministic — one seed suffices and is recorded as such.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.concentration import NPOINTS, _scaled_series
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.feature_store import FeatureStore
from qdepipe.models.gate_qrc import GateQRC, GateQRCConfig
from qdepipe.entanglement_probe import mean_reservoir_entropy
from qdepipe.significance import diebold_mariano, errors, median_iqr

OUT = "results/entanglement"
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")
ENTROPY = os.path.join(OUT, "entropy.csv")

SYSTEMS = ["henon", "mackeyglass"]        # MG = where the ZZ benefit grew
N_ABL = [4, 6, 8]                          # ablation arms
J_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]  # regime scan at n=6
SEEDS = [0, 1, 2, 3, 4]
LOOKBACK = 5


def qcfg(n, J, encoding, seed):
    r = n if encoding == "width" else 1
    return GateQRCConfig(n_qubits=n, encoding=encoding, r=r, coupling="isingxx",
                         J_strength=J, channel="none", V=4, window=LOOKBACK,
                         readout=("Z", "X", "Y", "ZZ"), encode_scale=1.0, seed=seed)


store = FeatureStore()
done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)
    done = {(r.system, r.encoding, int(r.n), float(r.J), int(r.seed))
            for r in d.itertuples()}
    print(f"resuming: {len(done)} cells done", flush=True)

forecast_bank = {}

# cells: ablation arms (J in {0,1}, n in N_ABL) + J-sweep (n=6, all J)
CELLS = []
for system in SYSTEMS:
    for encoding in ["depth", "width"]:
        for n in N_ABL:
            for J in [0.0, 1.0]:
                CELLS.append((system, encoding, n, J))
        for J in J_SWEEP:
            if J not in (0.0, 1.0):                    # 0/1 covered above at n=6
                CELLS.append((system, encoding, 6, J))

for system, encoding, n, J in CELLS:
    npts = NPOINTS[n]
    seeds = [0] if J == 0.0 else SEEDS                 # J=0 is deterministic
    for seed in seeds:
        cell = (system, encoding, n, J, seed)
        if cell in done:
            continue
        cfg = ExperimentConfig(system=system, n_points=npts, seed=seed,
                               lookback=LOOKBACK, washout=100)
        fc = ReservoirForecaster(GateQRC(qcfg(n, J, encoding, seed)),
                                 f"QRC({encoding},n={n},J={J})", store=store)
        r = run_experiment(fc, cfg, keep_forecasts=True)
        row = {"system": system, "encoding": encoding, "n": n, "J": J,
               "seed": seed, "nrmse": r.nrmse, "n_features": r.n_features,
               "n_points": npts, "deterministic_J0": J == 0.0}
        pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                                   header=not os.path.exists(SCORES))
        forecast_bank[cell] = (r.y_true, r.y_pred)
        print(f"{system:12s} {encoding:5s} n={n} J={J:<4} seed{seed} "
              f"NRMSE={r.nrmse:.4e}", flush=True)

# --------------------------- entropy probe -----------------------------------
ent_done = set()
if os.path.exists(ENTROPY):
    d = pd.read_csv(ENTROPY, keep_default_na=False)
    ent_done = {(r.system, r.encoding, int(r.n), float(r.J)) for r in d.itertuples()}

for system, encoding, n, J in sorted({(s, e, n, J) for s, e, n, J in CELLS}):
    if (system, encoding, n, J) in ent_done:
        continue
    u, _ = _scaled_series(system, NPOINTS[n])
    q = GateQRC(qcfg(n, J, encoding, seed=0))          # seed-0 coupling draw
    ent = mean_reservoir_entropy(q, u, sample=200)
    row = {"system": system, "encoding": encoding, "n": n, "J": J, **ent}
    pd.DataFrame([row]).to_csv(ENTROPY, mode="a", index=False,
                               header=not os.path.exists(ENTROPY))
    print(f"entropy {system:12s} {encoding:5s} n={n} J={J:<4} "
          f"S={ent['S_mean']:.3f}±{ent['S_std']:.3f} bits "
          f"(max obs {ent['S_max_observed']:.3f}/{ent['S_max_possible']:.0f})",
          flush=True)

# --------------------------- DM: J=0 vs J=1 ----------------------------------
sc = pd.read_csv(SCORES, keep_default_na=False)
sc["nrmse"] = sc["nrmse"].astype(float)
sc["J"] = sc["J"].astype(float)
sc = sc.drop_duplicates(subset=["system", "encoding", "n", "J", "seed"], keep="last")


def get_forecast(system, encoding, n, J, seed):
    cell = (system, encoding, n, J, seed)
    if cell not in forecast_bank:
        cfg = ExperimentConfig(system=system, n_points=NPOINTS[n], seed=seed,
                               lookback=LOOKBACK, washout=100)
        fc = ReservoirForecaster(GateQRC(qcfg(n, J, encoding, seed)), "tmp", store=store)
        r = run_experiment(fc, cfg, keep_forecasts=True)
        forecast_bank[cell] = (r.y_true, r.y_pred)
    return forecast_bank[cell]


dm_rows = []
for system in SYSTEMS:
    for encoding in ["depth", "width"]:
        for n in N_ABL:
            for seed in SEEDS:
                t1, p1 = get_forecast(system, encoding, n, 1.0, seed)   # entangled
                t0, p0 = get_forecast(system, encoding, n, 0.0, 0)      # separable
                stat, pval = diebold_mariano(errors(t1, p1), errors(t0, p0))
                dm_rows.append({"system": system, "encoding": encoding, "n": n,
                                "seed": seed, "comparison": "J1_vs_J0",
                                "dm_stat": stat, "p": pval})
pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "separable_dm.csv"), index=False)

# --------------------------- summary ------------------------------------------
print("\n===== entanglement summary (median NRMSE over seeds) =====")
for system in SYSTEMS:
    for encoding in ["depth", "width"]:
        d = sc[(sc.system == system) & (sc.encoding == encoding)]
        tab = d.groupby(["n", "J"]).nrmse.median().unstack()
        print(f"--- {system} / {encoding} ---")
        print(tab.round(6).to_string())
print("\nwrote", SCORES, ENTROPY, "separable_dm.csv")
