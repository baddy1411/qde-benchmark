#!/usr/bin/env python3
"""Initial-condition robustness (K=20 ICs per system).
->  results/ic_robustness/{onestep,closedloop}.csv

Per the review standard: the headline ranking must hold across many random
initial conditions, not one default trajectory. Reports feed median/IQR +
bootstrap-CI + ranking-stability analysis.

store=None THROUGHOUT — the cache key cannot see the IC (documented in
qdepipe/ic_study.py); caching here would alias different trajectories.
Crash-safe per-cell CSV append.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.experiment import ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.concentration import _rich_cfg
from qdepipe.ic_study import (draw_ics, generate_with_ic, run_onestep_on_series,
                              run_closedloop_on_series, selfcheck)

OUT = "results/ic_robustness"
os.makedirs(OUT, exist_ok=True)
ONE = os.path.join(OUT, "onestep.csv")
CLO = os.path.join(OUT, "closedloop.csv")

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
K = 20
N_POINTS, LOOKBACK, N_QUBITS = 1500, 5, 6
NGRC_DEGREE = {"henon": 2, "lorenz": 3, "mackeyglass": 2}
CL_STEPS = 300

# (model_key, n_model_seeds, closed_loop?)
MODELS = [("ngrc", 1, True), ("esn300", 3, True), ("elm300", 3, True),
          ("qrc_rich", 3, True)]


def build(model_key, seed, system):
    if model_key == "ngrc":
        return ReservoirForecaster(
            NGRC(NGRCConfig(k=LOOKBACK, degree=NGRC_DEGREE[system])), "NG-RC")
    if model_key == "esn300":
        return ReservoirForecaster(ESN(ESNConfig(units=300, seed=seed)), "ESN",
                                   external_window=True)
    if model_key == "elm300":
        return ReservoirForecaster(ELM(ELMConfig(units=300, lookback=LOOKBACK,
                                                 seed=seed)), "ELM")
    if model_key == "qrc_rich":
        # store=None (IC-aliasing hazard) — every featurize computed fresh
        return ReservoirForecaster(GateQRC(_rich_cfg(N_QUBITS, V=4, seed=seed)),
                                   "QRC-rich", store=None)
    raise KeyError(model_key)


selfcheck()

done_one, done_clo = set(), set()
if os.path.exists(ONE):
    d = pd.read_csv(ONE, keep_default_na=False)
    done_one = {(r.system, int(r.ic_index), r.model, int(r.seed)) for r in d.itertuples()}
if os.path.exists(CLO):
    d = pd.read_csv(CLO, keep_default_na=False)
    done_clo = {(r.system, int(r.ic_index), r.model, int(r.seed)) for r in d.itertuples()}
if done_one or done_clo:
    print(f"resuming: {len(done_one)} one-step, {len(done_clo)} closed-loop cells done",
          flush=True)

for system in SYSTEMS:
    ics = draw_ics(system, K)
    for i, ic in enumerate(ics):
        x = generate_with_ic(system, N_POINTS, ic)
        for model_key, n_seeds, do_cl in MODELS:
            for seed in range(n_seeds):
                cfg = ExperimentConfig(system=system, n_points=N_POINTS, seed=seed,
                                       lookback=LOOKBACK, washout=100)
                cell = (system, i, model_key, seed)
                if cell not in done_one:
                    r = run_onestep_on_series(build(model_key, seed, system), x, cfg)
                    pd.DataFrame([{"system": system, "ic_index": i, "ic": str(ic),
                                   "model": model_key, "seed": seed,
                                   "nrmse": r.nrmse, "n_features": r.n_features}]
                                 ).to_csv(ONE, mode="a", index=False,
                                          header=not os.path.exists(ONE))
                    print(f"{system:12s} ic{i:02d} {model_key:9s} s{seed} "
                          f"1step NRMSE={r.nrmse:.4e}", flush=True)
                if do_cl and cell not in done_clo:
                    c = run_closedloop_on_series(build(model_key, seed, system), x,
                                                 cfg, n_steps=CL_STEPS)
                    pd.DataFrame([{"system": system, "ic_index": i, "ic": str(ic),
                                   "model": model_key, "seed": seed,
                                   "vpt_steps": c.vpt_steps, "vpt_lyap": c.vpt_lyap,
                                   "diverged": c.diverged,
                                   "wasserstein": c.wasserstein}]
                                 ).to_csv(CLO, mode="a", index=False,
                                          header=not os.path.exists(CLO))
                    print(f"{system:12s} ic{i:02d} {model_key:9s} s{seed} "
                          f"VPT={c.vpt_lyap:.2f}L div={c.diverged}", flush=True)

# --------------------------- analysis ----------------------------------------
from qdepipe.significance import paired_bootstrap_ci, median_iqr

one = pd.read_csv(ONE, keep_default_na=False)
one["nrmse"] = one["nrmse"].astype(float)
one = one.drop_duplicates(subset=["system", "ic_index", "model", "seed"], keep="last")
rows = []
for system in SYSTEMS:
    d = one[one.system == system]
    # per-IC score = median over model seeds (IC variance is the target)
    per_ic = d.groupby(["model", "ic_index"]).nrmse.median().unstack()  # model x IC
    row = {"system": system, "K": per_ic.shape[1]}
    for m in per_ic.index:
        mi = median_iqr(per_ic.loc[m].values)
        row[f"{m}_median"] = mi["median"]
        row[f"{m}_iqr"] = mi["iqr"]
    # ranking stability: fraction of ICs on which each model is the best
    best = per_ic.idxmin(axis=0)
    for m in per_ic.index:
        row[f"{m}_wins"] = int((best == m).sum())
    # paired CI: QRC - best classical, per IC
    classical = per_ic.drop("qrc_rich").min(axis=0).values
    ci = paired_bootstrap_ci(np.log10(per_ic.loc["qrc_rich"].values),
                             np.log10(classical))
    row["loggap_qrc_vs_bestclassical_lo"] = ci["ci_lo"]
    row["loggap_qrc_vs_bestclassical_hi"] = ci["ci_hi"]
    row["gap_excludes_zero"] = ci["excludes_zero"]
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)

print("\n===== initial-condition summary =====")
print(pd.DataFrame(rows).to_string(index=False))
print("\nwrote", ONE, CLO, os.path.join(OUT, "summary.csv"))
