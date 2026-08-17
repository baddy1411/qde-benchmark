"""Recompute the FULL closed-loop metric battery for the classical models of the
cross-system climate comparison (Lorenz-63, Mackey-Glass).

The original cross_system.climate() ran run_closed_loop(), which returns
vpt_lyap, spectral_mse, wasserstein and diverged, but persisted only the VPT
mean. This script re-runs the identical model specs (same base_cfg, same tuned
NG-RC (k, degree) from the committed *_ngrc_tune.csv, same 3 seeds, same
n_steps=300) and persists all four metrics, mirroring the column layout of the
committed quantum climate files. Existing artifacts are NOT modified; output
goes to results/cross/{system}_climate_full.csv.
"""
import os
import numpy as np
import pandas as pd

from cross_system import base_cfg, _ngrc, RESULTS
from qdepipe.registry import build_forecaster
from qdepipe.closedloop import run_closed_loop

SEEDS = (0, 1, 2)
N_STEPS = 300

for system in ("lorenz", "mackeyglass"):
    tune = pd.read_csv(os.path.join(RESULTS, "cross", f"{system}_ngrc_tune.csv"))
    best = tune.sort_values("nrmse_mean").iloc[0]
    specs = {
        "NG-RC(best)": (_ngrc(int(best.k), int(best.degree)), int(best.k)),
        "ESN": (lambda s, c: build_forecaster("esn", s, c), 5),
        "ELM": (lambda s, c: build_forecaster("elm", s, c), 5),
        "RandomForest": (lambda s, c: build_forecaster("random_forest", s, c), 5),
    }
    rows = []
    for label, (make, lookback) in specs.items():
        vpt, spec, w1, div = [], [], [], 0
        for s in SEEDS:
            cfg = base_cfg(system, seed=s, lookback=lookback)
            cl = run_closed_loop(make(s, cfg), cfg, n_steps=N_STEPS)
            vpt.append(cl.vpt_lyap)
            spec.append(cl.spectral_mse)
            w1.append(cl.wasserstein)
            div += int(cl.diverged)
        rows.append({"model": label,
                     "vpt_lyap_mean": float(np.mean(vpt)),
                     "spectral_mse_mean": float(np.mean(spec)),
                     "wasserstein_mean": float(np.mean(w1)),
                     "diverged_runs": div})
        print(f"{system:12} {label:14} vpt={np.mean(vpt):.3f} "
              f"spec={np.mean(spec):.3f} w1={np.mean(w1):.4f} div={div}/3", flush=True)
    out = pd.DataFrame(rows).sort_values("vpt_lyap_mean", ascending=False)
    path = os.path.join(RESULTS, "cross", f"{system}_climate_full.csv")
    out.to_csv(path, index=False)
    print("wrote", path, flush=True)

# consistency check against the committed VPT-only tables
for system in ("lorenz", "mackeyglass"):
    old = pd.read_csv(os.path.join(RESULTS, "cross", f"{system}_climate.csv"))
    new = pd.read_csv(os.path.join(RESULTS, "cross", f"{system}_climate_full.csv"))
    m = old.merge(new, on="model", suffixes=("_old", "_new"))
    m["abs_diff"] = (m.vpt_lyap_mean_old - m.vpt_lyap_mean_new).abs()
    print(f"\n{system} VPT consistency vs committed artifact:")
    print(m[["model", "vpt_lyap_mean_old", "vpt_lyap_mean_new", "abs_diff"]].to_string(index=False))
