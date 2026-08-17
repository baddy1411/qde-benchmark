#!/usr/bin/env python3
"""Regenerate the Hénon quantum--classical JOINT significance into distinctly-named
CSVs, closing the provenance gap (the original Hénon joint CSVs were
overwritten by the classical n=3000 pass via a filename collision).

Mirrors the Lorenz / Mackey-Glass joint analysis exactly: the same model set on the
aligned n_points=1500 pipeline, each quantum model at its Hénon best encoding
(from adv_A2_matched.csv: v4 scale 6, v6 scale 1, rich scale 0.5). Writes
forecasts to a SEPARATE dir (forecasts/henon_joint/) so nothing is clobbered.

Outputs:
  results/significance/henon_MCS_joint.csv
  results/significance/henon_DM_pairs_joint.csv
  results/significance/henon_quantum_DM.csv
  results/forecasts/henon_joint/{model}_seed{n}.csv
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.registry import build_forecaster
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.pipeline.postprocess import Polynomial
from qdepipe import significance as sig

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEEDS = tuple(range(5))
BEST_ENCODE = {"qrc_v4": 6.0, "qrc_v6": 1.0, "qrc_rich": 0.5}   # Hénon best (adv_A2)
NGRC_K, NGRC_D = 3, 2                                            # Hénon NG-RC best


def _cfg(seed, **ov):
    return ExperimentConfig(system="henon", n_points=1500, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2),
                            washout=100, horizon=1, lookback=5, alpha=1e-6,
                            seed=seed, **ov)


def _build_comp(store=None):
    # `store` (DEFAULT-OFF) threads into the QUANTUM lambdas only; the classical
    # battery stays uncached (cheap featurize). henon_joint_mcs is pure one-step
    # MCS — no closed-loop/rollout part, so the run()-only exclusion that
    # quantum_cross's climate needed does not apply here.
    poly = [Polynomial(interactions=False)]
    return {
        f"NG-RC(k{NGRC_K}d{NGRC_D})": (lambda s, c: ReservoirForecaster(
            NGRC(NGRCConfig(k=NGRC_K, degree=NGRC_D)), "NG-RC"), {}),
        "ESN": (lambda s, c: build_forecaster("esn", s, c), {}),
        "ELM": (lambda s, c: build_forecaster("elm", s, c), {}),
        "ELM+poly2": (lambda s, c: build_forecaster("elm", s, c), {"postprocess": poly}),
        "RandomForest": (lambda s, c: build_forecaster("random_forest", s, c), {}),
        "qrc_v4": (lambda s, c: build_forecaster("qrc_v4", s, c, store=store), {"encode_scale": BEST_ENCODE["qrc_v4"]}),
        "qrc_v6": (lambda s, c: build_forecaster("qrc_v6", s, c, store=store), {"encode_scale": BEST_ENCODE["qrc_v6"]}),
        "qrc_rich": (lambda s, c: build_forecaster("qrc_rich", s, c, store=store), {"encode_scale": BEST_ENCODE["qrc_rich"]}),
        "ESN(F=16)": (lambda s, c: ReservoirForecaster(ESN(ESNConfig(units=16, seed=s)), "ESN(F=16)", external_window=True), {"windowing": "internal"}),
        "ESN(F=24)": (lambda s, c: ReservoirForecaster(ESN(ESNConfig(units=24, seed=s)), "ESN(F=24)", external_window=True), {"windowing": "internal"}),
        "ESN(F=96)": (lambda s, c: ReservoirForecaster(ESN(ESNConfig(units=96, seed=s)), "ESN(F=96)", external_window=True), {"windowing": "internal"}),
    }


def _run_comp(comp, seeds):
    per_seed = []
    for s in seeds:
        fc = {}
        for label, (make, ov) in comp.items():
            r = run_experiment(make(s, _cfg(s, **ov)), _cfg(s, **ov), keep_forecasts=True)
            fc[label] = (r.y_true, r.y_pred)
        per_seed.append(fc)
        print(f"  seed {s}: {len(fc)} models", flush=True)
    return per_seed


def main(store=None):
    comp = _build_comp(store)
    fdir = os.path.join(RESULTS, "forecasts", "henon_joint")
    os.makedirs(fdir, exist_ok=True)
    per_seed = _run_comp(comp, SEEDS)
    L = min(len(v[0]) for fc in per_seed for v in fc.values())
    aligned = [{m: (np.asarray(yt)[-L:], np.asarray(yp)[-L:]) for m, (yt, yp) in fc.items()}
               for fc in per_seed]
    for si, s in enumerate(SEEDS):
        for label, (yt, yp) in aligned[si].items():
            safe = label.replace("(", "").replace(")", "").replace("=", "").replace("+", "p")
            pd.DataFrame({"t": np.arange(L), "y_true": yt, "y_pred": yp}).to_csv(
                os.path.join(fdir, f"{safe}_seed{s}.csv"), index=False)

    names = list(comp)
    sdir = os.path.join(RESULTS, "significance")

    # joint MCS
    in10 = {m: 0 for m in names}; pv = {m: [] for m in names}
    for fc in aligned:
        losses = np.stack([(fc[m][1] - fc[m][0]) ** 2 for m in names], 0)
        res = sig.model_confidence_set(losses, names=names, alpha=(0.10, 0.25),
                                       n_boot=1000, block_length=20, seed=0)
        for m in names:
            pv[m].append(res["mcs_pvalue"][m])
            if m in res["in_set"][0.10]:
                in10[m] += 1
    mcs = pd.DataFrame([{"model": m, "mean_mcs_pvalue": float(np.mean(pv[m])),
                         "frac_seeds_in_set_0.10": in10[m] / len(SEEDS),
                         "is_quantum": m.startswith("qrc_")} for m in names]
                       ).sort_values("mean_mcs_pvalue", ascending=False)
    mcs.to_csv(os.path.join(sdir, "henon_MCS_joint.csv"), index=False)

    # full DM pairs (joint)
    n = len(names)
    stats = np.full((len(aligned), n, n), np.nan); ps = np.full((len(aligned), n, n), np.nan)
    for si, fc in enumerate(aligned):
        st, pvm, nm = sig.dm_matrix(fc, h=1, loss="se")
        idx = [nm.index(x) for x in names]
        stats[si] = st[np.ix_(idx, idx)]; ps[si] = pvm[np.ix_(idx, idx)]
    mean_stat, median_p, frac = np.nanmean(stats, 0), np.nanmedian(ps, 0), np.mean(ps < 0.05, 0)
    rows = [{"model_i": names[i], "model_j": names[j], "mean_dm": mean_stat[i, j],
             "median_p": median_p[i, j], "frac_seeds_sig": frac[i, j]}
            for i in range(n) for j in range(n) if i != j]
    pd.DataFrame(rows).to_csv(os.path.join(sdir, "henon_DM_pairs_joint.csv"), index=False)

    # quantum headline DM pairs
    ng = f"NG-RC(k{NGRC_K}d{NGRC_D})"
    qpairs = [("qrc_rich", "ESN(F=96)"), ("qrc_rich", ng), ("qrc_v6", "ESN(F=24)")]
    qrows = []
    for a, b in qpairs:
        st, p = [], []
        for fc in aligned:
            s_, p_ = sig.diebold_mariano(fc[a][1] - fc[a][0], fc[b][1] - fc[b][0])
            st.append(s_); p.append(p_)
        st, p = np.array(st), np.array(p)
        qrows.append({"pair": f"{a} vs {b}", "frac_sig_0.10": float(np.mean(p < 0.10)),
                      "frac_sig_0.05": float(np.mean(p < 0.05)), "median_dm": float(np.median(st)),
                      "dm_min": float(st.min()), "dm_max": float(st.max()), "median_p": float(np.median(p))})
    pd.DataFrame(qrows).to_csv(os.path.join(sdir, "henon_quantum_DM.csv"), index=False)

    best_set = mcs[mcs["frac_seeds_in_set_0.10"] >= 0.5]["model"].tolist()
    q_in = [m for m in best_set if m.startswith("qrc_")]
    print(f"\nHénon joint best-set = {best_set}; quantum in set = {q_in or 'none'}")
    print("wrote henon_MCS_joint.csv, henon_DM_pairs_joint.csv, henon_quantum_DM.csv")


if __name__ == "__main__":
    main()
