#!/usr/bin/env python3
"""Five-seed concentration variance scan (closes the n=2 attack on Chapter 5).

Re-runs ONLY the variance side (two-local vs one-local observable variance) at
five seeds for n_qubits in {4,6,8}, BOTH systems (Hénon, Mackey-Glass), through
the wired `observable_variance` (V=1, cache-backed) on CPU. Excludes n=10. Writes
a NEW csv with per-seed MEAN and STD (the std is what fig7's error bars show);
does NOT touch the committed scaling.csv. Preserves the device exhibit.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from qdepipe import concentration as C
from qdepipe.device import get_device
from qdepipe.experiment import ExperimentConfig
from qdepipe.feature_store import FeatureStore
from qdepipe.models.gate_qrc import GateQRC

CDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "concentration")
SEEDS = (0, 1, 2, 3, 4)
NS = (4, 6, 8)
SYSTEMS = ("henon", "mackeyglass")


def main():
    dev = {
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "device_py_get_device": str(get_device()),
        "gate_qrc_instance_device": str(GateQRC(C._rich_cfg(4, V=1, seed=0)).device),
    }
    print("DEVICE EXHIBIT:", dev, flush=True)
    with open(os.path.join(CDIR, "scaling_variance_5seed_device.json"), "w") as f:
        json.dump(dev, f, indent=2)

    committed = pd.read_csv(os.path.join(CDIR, "scaling.csv"))
    store = FeatureStore()
    rows = []
    for system in SYSTEMS:
        comm = (committed.query("system == @system").drop_duplicates("n_qubits")
                .set_index("n_qubits")[["mean_var_1local", "mean_var_2local"]])
        for n in NS:
            npts = C.NPOINTS[n]
            u, _ = C._scaled_series(system, npts)
            cfg = ExperimentConfig(system=system, n_points=npts, scaler="minmax",
                                   scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))
            v1s, v2s, ratios = [], [], []
            for s in SEEDS:
                ov = C.observable_variance(u, n, seed=s, store=store, cfg=cfg)
                v1s.append(ov["mean_var_1local"]); v2s.append(ov["mean_var_2local"])
                ratios.append(ov["mean_var_2local"] / ov["mean_var_1local"])
            v1s, v2s, ratios = np.array(v1s), np.array(v2s), np.array(ratios)
            c1, c2 = float(comm.loc[n, "mean_var_1local"]), float(comm.loc[n, "mean_var_2local"])
            rows.append({
                "system": system, "n_qubits": n, "n_points": npts, "n_seeds": len(SEEDS),
                "var_1local_mean": float(v1s.mean()), "var_1local_std": float(v1s.std()),
                "var_2local_mean": float(v2s.mean()), "var_2local_std": float(v2s.std()),
                "ratio_2over1_meanofratios": float(ratios.mean()),
                "ratio_2over1_std": float(ratios.std()),
                "ratio_2over1_ofmeans_5seed": float(v2s.mean() / v1s.mean()),
                "ratio_2over1_ofmeans_2seed": c2 / c1,
            })
            print(f"  {system:11s} n={n}: ratio(2/1) 5-seed={v2s.mean()/v1s.mean():.3f} "
                  f"vs 2-seed={c2/c1:.3f} | per-seed ratio std={ratios.std():.3f} "
                  f"(range {ratios.min():.2f}-{ratios.max():.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CDIR, "scaling_variance_5seed.csv"), index=False)
    print("\nwrote scaling_variance_5seed.csv | store:", store.stats)

    print("\nVERDICT — monotonic decline of ratio with n (the 2-seed 'concentration' signature):")
    for system in SYSTEMS:
        s5 = df.query("system == @system")["ratio_2over1_ofmeans_5seed"].to_numpy()
        s2 = df.query("system == @system")["ratio_2over1_ofmeans_2seed"].to_numpy()
        m5 = all(s5[i] >= s5[i + 1] for i in range(len(s5) - 1))
        m2 = all(s2[i] >= s2[i + 1] for i in range(len(s2) - 1))
        print(f"  {system:11s}: 5-seed monotonic={m5}  2-seed monotonic={m2}  "
              f"5-seed ratios={np.round(s5,3).tolist()}")


if __name__ == "__main__":
    main()
