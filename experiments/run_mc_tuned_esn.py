"""Memory capacity with a properly tuned classical control (review response).

The committed memory-capacity comparison (run_dissipative_qrc.py phase 3) used
the ESN at its forecasting defaults (spectral radius 0.9, leak 0.3, input
scaling 0.5) with a single seed, and truncated the delay range at d=15 while
the undamped dQRC still had r^2 ~ 0.22 there. Both choices are unfair, in
opposite directions. This script redoes the comparison properly:

  - identical input protocol to the committed run (iid uniform, rng seed 7,
    NPTS=1500, same temporal split, same ridge read-out alpha)
  - EQUAL model-selection budgets: 63 configurations per side
      dQRC : gamma (7) x tau (3) x Hamiltonian seed (3)
      ESN  : spectral radius (7) x leak rate (3) x input scaling (3)
  - selection on VALIDATION capacity, report TEST capacity (never selected on)
  - delay range d = 1..40 (past saturation for every model); the d<=15 sum is
    also reported for comparability with the committed table
  - winning configuration per side re-run over 10 seeds
  - bootstrap 95% CI on the per-seed capacity difference (tuned ESN - dQRC)

Outputs: results/dissipative/mc_tuned_grid.csv, mc_tuned_final.csv,
         mc_tuned_summary.txt
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from qdepipe.models import ELM, ELMConfig, ESN, ESNConfig
from qdepipe.pipeline import temporal_split
from qdepipe.readout import ridge_fit, ridge_predict
from run_dissipative_qrc import DissipativeQRC

OUT = "results/dissipative"
NPTS = 1500
WASHOUT = 100
DMAX = 40

rng = np.random.default_rng(7)
u = rng.random(NPTS)
split = temporal_split(NPTS, (0.6, 0.2, 0.2), WASHOUT)
a, b = split.train.stop, split.val.stop
tr = np.arange(WASHOUT, a)
va = np.arange(a, b)
te = np.arange(b, NPTS)


def mem_cap(X: np.ndarray, idx: np.ndarray, lo: int) -> tuple[float, float, list[float]]:
    """Sum of squared delay-reconstruction correlations on segment `idx`.

    Read-out is fitted on the training segment only, exactly as in the
    committed run. Returns (capacity d<=DMAX, capacity d<=15, per-delay r^2).
    """
    per = []
    for d in range(1, DMAX + 1):
        tr_d = tr[tr >= WASHOUT + d]
        ev_d = idx[idx >= lo + d]
        W = ridge_fit(X[tr_d], u[tr_d - d], alpha=1e-9)
        pred = np.asarray(ridge_predict(X[ev_d], W)).ravel()
        r = np.corrcoef(pred, u[ev_d - d])[0, 1]
        per.append(0.0 if not np.isfinite(r) else max(r, 0.0) ** 2)
    return float(sum(per)), float(sum(per[:15])), per


def esn_features(sr: float, leak: float, inp: float, seed: int) -> np.ndarray:
    cfg = ESNConfig(units=48, spectral_radius=sr, leak_rate=leak,
                    input_scaling=inp, seed=seed)
    return ESN(cfg).featurize(u)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    grid_rows = []

    # ---- dQRC grid: gamma x tau x Hamiltonian seed (63) -------------------
    for g in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for tau in [1.0, 4.0, 10.0]:
            for hs in [0, 1, 2]:
                X = DissipativeQRC(n=4, tau=tau, gamma=g, seed=hs).featurize(u)
                vc, _, _ = mem_cap(X, va, a)
                tc, tc15, _ = mem_cap(X, te, b)
                grid_rows.append({"family": "dqrc4", "sr": "", "leak": "",
                                  "inp": "", "gamma": g, "tau": tau, "seed": hs,
                                  "val_cap": vc, "test_cap": tc,
                                  "test_cap_d15": tc15})
        print(f"[grid] dqrc gamma={g} done", flush=True)

    # ---- ESN grid: spectral radius x leak x input scaling (63) ------------
    for sr in [0.7, 0.8, 0.9, 0.95, 0.99, 1.1, 1.3]:
        for leak in [0.3, 0.6, 1.0]:
            for inp in [0.05, 0.2, 0.5]:
                X = esn_features(sr, leak, inp, seed=0)
                vc, _, _ = mem_cap(X, va, a)
                tc, tc15, _ = mem_cap(X, te, b)
                grid_rows.append({"family": "esn48", "sr": sr, "leak": leak,
                                  "inp": inp, "gamma": "", "tau": "", "seed": 0,
                                  "val_cap": vc, "test_cap": tc,
                                  "test_cap_d15": tc15})
        print(f"[grid] esn sr={sr} done", flush=True)

    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(f"{OUT}/mc_tuned_grid.csv", index=False)

    # ---- select winners on validation capacity ----------------------------
    best_q = gdf[gdf.family == "dqrc4"].sort_values("val_cap").iloc[-1]
    best_e = gdf[gdf.family == "esn48"].sort_values("val_cap").iloc[-1]
    print(f"[select] dqrc: gamma={best_q.gamma} tau={best_q.tau} "
          f"(val {best_q.val_cap:.2f})", flush=True)
    print(f"[select] esn : sr={best_e.sr} leak={best_e.leak} inp={best_e.inp} "
          f"(val {best_e.val_cap:.2f})", flush=True)

    # ---- 10-seed final runs of the selected configurations ----------------
    final_rows = []
    for seed in range(10):
        X = DissipativeQRC(n=4, tau=float(best_q.tau),
                           gamma=float(best_q.gamma), seed=seed).featurize(u)
        tc, tc15, per = mem_cap(X, te, b)
        final_rows.append({"model": "dqrc4_tuned", "seed": seed,
                           "test_cap": tc, "test_cap_d15": tc15,
                           **{f"r2_d{d}": per[d - 1] for d in (1, 5, 10, 15, 20, 30, 40)}})
        X = esn_features(float(best_e.sr), float(best_e.leak),
                         float(best_e.inp), seed=seed)
        tc, tc15, per = mem_cap(X, te, b)
        final_rows.append({"model": "esn48_tuned", "seed": seed,
                           "test_cap": tc, "test_cap_d15": tc15,
                           **{f"r2_d{d}": per[d - 1] for d in (1, 5, 10, 15, 20, 30, 40)}})
        # forecasting-default ESN, for continuity with the committed table
        X = esn_features(0.9, 0.3, 0.5, seed=seed)
        tc, tc15, per = mem_cap(X, te, b)
        final_rows.append({"model": "esn48_default", "seed": seed,
                           "test_cap": tc, "test_cap_d15": tc15,
                           **{f"r2_d{d}": per[d - 1] for d in (1, 5, 10, 15, 20, 30, 40)}})
        print(f"[final] seed {seed} done", flush=True)
    # ELM anchor (deterministic given seed; 10 seeds anyway)
    for seed in range(10):
        X = ELM(ELMConfig(units=48, lookback=5, seed=seed)).featurize(u)
        tc, tc15, per = mem_cap(X, te, b)
        final_rows.append({"model": "elm48", "seed": seed,
                           "test_cap": tc, "test_cap_d15": tc15,
                           **{f"r2_d{d}": per[d - 1] for d in (1, 5, 10, 15, 20, 30, 40)}})

    fdf = pd.DataFrame(final_rows)
    fdf.to_csv(f"{OUT}/mc_tuned_final.csv", index=False)

    # ---- bootstrap CI on the difference (tuned ESN - tuned dQRC) ----------
    qs = fdf[fdf.model == "dqrc4_tuned"].test_cap.to_numpy()
    es = fdf[fdf.model == "esn48_tuned"].test_cap.to_numpy()
    brng = np.random.default_rng(0)
    diffs = [np.mean(brng.choice(es, 10)) - np.mean(brng.choice(qs, 10))
             for _ in range(10000)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    lines = [
        f"selected dqrc : gamma={best_q.gamma} tau={best_q.tau}",
        f"selected esn  : sr={best_e.sr} leak={best_e.leak} inp={best_e.inp}",
        f"test capacity d<=40 (mean +/- sd over 10 seeds):",
        f"  dqrc4_tuned  : {qs.mean():.2f} +/- {qs.std():.2f}",
        f"  esn48_tuned  : {es.mean():.2f} +/- {es.std():.2f}",
        f"  esn48_default: {fdf[fdf.model=='esn48_default'].test_cap.mean():.2f}",
        f"  elm48        : {fdf[fdf.model=='elm48'].test_cap.mean():.2f}",
        f"test capacity d<=15:",
        f"  dqrc4_tuned  : {fdf[fdf.model=='dqrc4_tuned'].test_cap_d15.mean():.2f}",
        f"  esn48_tuned  : {fdf[fdf.model=='esn48_tuned'].test_cap_d15.mean():.2f}",
        f"  esn48_default: {fdf[fdf.model=='esn48_default'].test_cap_d15.mean():.2f}",
        f"bootstrap 95% CI on (tuned ESN - tuned dQRC), d<=40: "
        f"[{lo:.2f}, {hi:.2f}]",
    ]
    txt = "\n".join(lines)
    with open(f"{OUT}/mc_tuned_summary.txt", "w") as f:
        f.write(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
