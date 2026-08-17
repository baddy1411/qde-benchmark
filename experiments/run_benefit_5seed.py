#!/usr/bin/env python3
"""Five-seed ZZ-benefit scan — checks the SECOND two-seed trend (fig7-bottom:
'ZZ benefit grows on Mackey-Glass', 1.035->1.174->1.215). Per-seed benefit ratio
NRMSE(Z+X+Y)/NRMSE(Z+X+Y+ZZ) at five seeds for n_qubits in {4,6,8}, both systems,
through the wired cache (V=4) on CPU. n=10 stays 2-seed (excluded). New CSV; does
NOT touch committed scaling.csv. Reports confirm-or-diverge vs the 2-seed numbers.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from qdepipe import concentration as C
from qdepipe.experiment import ExperimentConfig
from qdepipe.feature_store import FeatureStore, maybe_cached_featurize
from qdepipe.models.gate_qrc import GateQRC

CDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "concentration")
SEEDS = (0, 1, 2, 3, 4)
NS = (4, 6, 8)
SYSTEMS = ("henon", "mackeyglass")


def main():
    print("device:", str(GateQRC(C._rich_cfg(4, V=4, seed=0)).device), "| cuda:",
          torch.cuda.is_available(), "| mps:",
          bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()), flush=True)
    sc = pd.read_csv(os.path.join(CDIR, "scaling.csv"))
    store = FeatureStore()
    rows = []
    for system in SYSTEMS:
        d = sc[sc.system == system]
        for n in NS:
            npts = C.NPOINTS[n]
            u, split = C._scaled_series(system, npts)
            ecfg = ExperimentConfig(system=system, n_points=npts, scaler="minmax",
                                    scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))
            ratios = []
            for s in SEEDS:
                q = GateQRC(C._rich_cfg(n, V=4, seed=s))           # V=4 NRMSE config
                feats = maybe_cached_featurize(store, q, u, ecfg)  # wired cache
                zxy = C._nrmse_from_feats(C._slice_readout(feats, n, 4, "ZXY"), u, split)
                full = C._nrmse_from_feats(C._slice_readout(feats, n, 4, "ZXYZZ"), u, split)
                ratios.append(zxy / full)
            r = np.array(ratios)
            c_zxy = float(d[(d.n_qubits == n) & (d.readout_set == "Z+X+Y")]["NRMSE"].iloc[0])
            c_full = float(d[(d.n_qubits == n) & (d.readout_set == "Z+X+Y+ZZ")]["NRMSE"].iloc[0])
            rows.append({
                "system": system, "n_qubits": n, "n_points": npts, "n_seeds": 5,
                "benefit_5seed_mean": float(r.mean()), "benefit_5seed_std": float(r.std()),
                "benefit_5seed_min": float(r.min()), "benefit_5seed_max": float(r.max()),
                "benefit_2seed": c_zxy / c_full,
            })
            print(f"  {system:11s} n={n}: 5-seed benefit={r.mean():.3f} ± {r.std():.3f} "
                  f"(range {r.min():.3f}-{r.max():.3f})  vs 2-seed {c_zxy/c_full:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CDIR, "scaling_benefit_5seed.csv"), index=False)
    print("\nwrote scaling_benefit_5seed.csv | store:", store.stats)
    print("\nVERDICT — does benefit GROW with n (the fig7-bottom claim)?")
    for system in SYSTEMS:
        b5 = df.query("system == @system")["benefit_5seed_mean"].to_numpy()
        b2 = df.query("system == @system")["benefit_2seed"].to_numpy()
        g5 = all(b5[i] <= b5[i + 1] for i in range(len(b5) - 1))
        g2 = all(b2[i] <= b2[i + 1] for i in range(len(b2) - 1))
        print(f"  {system:11s}: 5-seed grows={g5}  2-seed grows={g2}  "
              f"5-seed={np.round(b5,3).tolist()}  2-seed={np.round(b2,3).tolist()}")


if __name__ == "__main__":
    main()
