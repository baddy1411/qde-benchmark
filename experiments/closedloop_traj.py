#!/usr/bin/env python3
"""Generate closed-loop (free-running) trajectories for the phase-space attractor figure.

The committed forecasts are 1-step only; the attractor reconstruction needs the model
fed its own predictions over a long autonomous rollout. We save (step, true, pred) per
model per system so the figure regenerates from CSVs. One seed (illustrative); the
quantitative climate verdict lives in the VPT CSVs, this is the visual.

Output: results/trajectories/{system}_{model}.csv  (+ {system}_true.csv shared)
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd

from qdepipe.experiment import ExperimentConfig
from qdepipe.closedloop import run_closed_loop
from qdepipe.registry import build_forecaster
from qdepipe.models import NGRC, NGRCConfig
from qdepipe.forecasters import ReservoirForecaster

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TDIR = os.path.join(RESULTS, "trajectories")
NGRC_BEST = {"henon": (3, 2), "lorenz": (8, 3), "mackeyglass": (8, 3)}
N_STEPS = 1500


def _log(m): print(m, flush=True)


def _models(system):
    k, d = NGRC_BEST[system]
    return {
        f"NG-RC": lambda s, c: ReservoirForecaster(NGRC(NGRCConfig(k=k, degree=d)), "NG-RC"),
        "ESN": lambda s, c: build_forecaster("esn", s, c),
        "RandomForest": lambda s, c: build_forecaster("random_forest", s, c),
    }


def main():
    os.makedirs(TDIR, exist_ok=True)
    for system in ("henon", "lorenz", "mackeyglass"):
        cfg = ExperimentConfig(system=system, n_points=4000, scaler="minmax",
                               scaler_scope="train", split_fracs=(0.6, 0.2, 0.2),
                               washout=100, lookback=8, seed=0)
        true_saved = False
        for label, make in _models(system).items():
            cl = run_closed_loop(make(0, cfg), cfg, n_steps=N_STEPS)
            if not true_saved:
                pd.DataFrame({"step": np.arange(len(cl.true)), "true": cl.true}).to_csv(
                    os.path.join(TDIR, f"{system}_true.csv"), index=False)
                true_saved = True
            pd.DataFrame({"step": np.arange(len(cl.pred)), "pred": cl.pred}).to_csv(
                os.path.join(TDIR, f"{system}_{label}.csv"), index=False)
            _log(f"  {system:12s} {label:13s} steps={len(cl.pred)} "
                 f"VPT={cl.vpt_lyap:.2f} diverged={cl.diverged}")
    _log("done — trajectories in results/trajectories/")


if __name__ == "__main__":
    main()
