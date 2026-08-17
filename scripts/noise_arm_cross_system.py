#!/usr/bin/env python3
"""Finite-shot (noise) arm across all three systems, scored against persistence.

Why this exists. The committed finite-shot evidence (results/concentration/
finite_shots.csv, finite_shot_budget.csv) is Hénon-only, so the obvious question
is whether "shot noise destroys the exact-expectation advantage" is a property of
the method or a property of Hénon. This runs the identical sweep -- same seeds,
same n, same readout sets, same shot budgets -- on Hénon, Lorenz-63 and
Mackey-Glass so the comparison is like-for-like.

Why persistence is in the table. Raw NRMSE is NOT comparable across these three
systems, and reading it as if it were inverts the ranking. One-step prediction on
the two flows is nearly trivial because they are smooth and oversampled: plain
persistence (u(t+1) = u(t)) already scores 0.151 on Mackey-Glass and 0.287 on
Lorenz, while on Hénon -- a broadband chaotic map -- persistence scores 1.600,
i.e. WORSE than predicting the mean. So a 1024-shot Hénon NRMSE of 0.43 looks
much worse than Mackey-Glass's 0.13 while actually representing far more skill.

The comparable quantity is skill over persistence (NRMSE_persistence / NRMSE):
how many times better than the trivial predictor the reservoir is. Reported
alongside the raw NRMSE, never instead of it.

NOTE ON A RETRACTED MOTIVATION. An earlier version of this script existed to
disambiguate a cross-environment "collapse" of the Hénon noise arm (NRMSE 0.0011
-> ~1.00). That collapse was not real: torch's complex 1-D dot path silently
returns 0 in the aarch64 container wheel, so every quantum expectation there was
0 and the model degenerated to the mean predictor. Fixed in qdepipe/models/
_qops.py, which now refuses to import in such an environment. The reproducibility
question was then answered directly on Hénon: with the data held bit-identical
Mackey-Glass reproduces to every printed digit, and Hénon's residual few-percent
drift is trajectory divergence -- its purely CLASSICAL ESN(F) reference moves 20%
too. None of that motivation survives; the cross-system finite-shot measurement
below is worth having on its own terms.

`matched_budget_shots()` is Hénon-hardcoded, so this uses `finite_shot_sweep`,
which takes a system.

Run:  python scripts/noise_arm_cross_system.py [--quick]
Out:  results/concentration/noise_arm_cross_system.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdepipe import concentration as C  # noqa: E402
from qdepipe import metrics  # noqa: E402
from qdepipe.pipeline.embedding import supervised_pairs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "concentration")
SYSTEMS = ("henon", "lorenz", "mackeyglass")


def persistence_nrmse(system, n_points, horizon=1):
    """NRMSE of u(t+1) = u(t) on the SAME series, split and metric the sweep uses.

    Alignment mirrors C._nrmse_from_feats exactly: supervised_pairs gives
    y = u[horizon:], the state row at t predicts t+horizon, so the persistence
    prediction for y is u[:-horizon]; the test slice starts at split.test.start.
    """
    u, split = C._scaled_series(system, n_points)
    _, y = supervised_pairs(u[:, None], u, horizon)
    yhat = u[:-horizon]
    te = slice(split.test.start, len(y))
    return float(metrics.nrmse(y[te], yhat[te]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = (0, 1) if args.quick else (0, 1, 2, 3, 4)
    n_list = [4, 6] if args.quick else [4, 6, 8]

    frames = []
    for system in SYSTEMS:
        print(f"[noise-arm] {system}: seeds={seeds} n={n_list}", flush=True)
        rows = C.finite_shot_sweep(system, seeds=seeds, n_list=n_list,
                                   log=lambda m: print("   ", m, flush=True))
        df = pd.DataFrame(rows)
        if "system" not in df.columns:
            df.insert(0, "system", system)

        # One persistence value per (system, n_points); n_points varies with n.
        pers = {n: persistence_nrmse(system, C.NPOINTS[n]) for n in n_list}
        df["NRMSE_persistence"] = df["n_qubits"].map(pers)
        df["skill_vs_persistence"] = df["NRMSE_persistence"] / df["NRMSE"]
        for n in n_list:
            print(f"     persistence n={n} (npts={C.NPOINTS[n]}): {pers[n]:.4f}",
                  flush=True)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "noise_arm_cross_system.csv")
    out.to_csv(dest, index=False)
    print(f"\nwrote {os.path.relpath(dest, ROOT)} ({len(out)} rows)")

    # Headline: how much of the exact-expectation skill survives sampling.
    print("\nskill over persistence (higher is better):")
    piv = out[out.readout_set == "Z+X+Y+ZZ"].pivot_table(
        index="system", columns="shots", values="skill_vs_persistence",
        aggfunc=np.mean)
    cols = [c for c in ("exact", 8192, 1024, "1024") if c in piv.columns]
    print(piv[cols].to_string(float_format=lambda v: f"{v:9.2f}"))


if __name__ == "__main__":
    main()
