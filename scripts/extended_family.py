#!/usr/bin/env python3
"""Pool every quantum-vs-classical per-seed DM test into one extended family.

The thesis's confirmatory family is 214 tests from the qubit-scaling study
(quantum vs ELM(F), quantum vs NG-RC). This asks what happens if every other
quantum-vs-classical per-seed test in the project joins it: the size-matched ESN
arm, the read-out and virtual-node sweeps, the polynomial-encoding study, the
leaky-filter arm, and Lorenz-96.

Sign convention throughout: POSITIVE dm_stat means the QUANTUM model carried the
larger loss, so the classical model won that test.

Quantum-vs-quantum rows are excluded -- ablations (J=1 vs J=0, leaky vs base,
poly2 vs qrc_base) are not comparisons against a classical comparator and do not
belong in a family about whether quantum beats classical.

Run: .venv/bin/python scripts/extended_family.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSICAL = ("elmf", "ngrc", "esnf", "elm300", "esn300", "esnf96", "esnf128",
             "elmf_same", "elmf_samef", "chebpoly", "esnbest", "ngrcbest")


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


KEEP = ["arm", "opponent", "cell", "seed", "dm_stat", "p", "core"]


def load():
    """Each row is one per-seed test, tagged with the CONDITION it belongs to.

    `cell` is everything that defines the experimental condition except the seed
    -- system, size, read-out, opponent. Seeds are repeats inside a cell, not
    separate experiments, so the condition is the unit that should be counted.
    """
    R = lambda f: pd.read_csv(os.path.join(ROOT, f), float_precision="round_trip")
    out = []

    d = R("results/scaling_proof/dm.csv")
    d["arm"] = "qubit scaling"
    d["opponent"] = d.comparison.str.replace("qrc_vs_", "", regex=False)
    d["cell"] = d.system + "|n=" + d.n.astype(str) + "|" + d.opponent
    d["core"] = d.comparison.isin(["qrc_vs_elmF", "qrc_vs_ngrc"])
    out.append(d[KEEP])

    d = R("results/cheb/dm.csv")
    d = d[d["vs"].str.lower().isin(CLASSICAL)]          # drop cheb-vs-depth (both quantum)
    d = d.assign(arm="polynomial encodings", opponent=d["vs"], core=False,
                 cell=d.system + "|" + d.model + "|" + d["vs"])
    out.append(d[KEEP])

    d = R("results/followup/dm.csv")
    d = d[d["vs"].str.lower().isin(CLASSICAL)]          # drop vs qrc_base / qrc_V4
    d = d.assign(arm="read-out & virtual nodes", opponent=d["vs"], core=False,
                 cell=(d.experiment + "|" + d.system + "|n=" + d.n.astype(str)
                       + "|V=" + d.V.astype(str) + "|" + d.readout + "|" + d["vs"]))
    out.append(d[KEEP])

    d = R("results/leaky/dm.csv")
    d = d[d.comparison.isin(["qrcBest_vs_esnBest", "qrcBest_vs_ngrcBest"])]
    d = d.assign(arm="leaky filter", core=False,
                 opponent=d.comparison.str.replace("qrcBest_vs_", "", regex=False))
    d = d.assign(cell=d.system + "|" + d.opponent)
    out.append(d[KEEP])

    d = R("results/lorenz96/dm.csv")
    d = d.assign(arm="Lorenz-96", core=False,
                 opponent=d.comparison.str.split("_vs_").str[1])
    d = d.assign(cell="lorenz96|" + d.comparison)
    out.append(d[KEEP])

    return pd.concat(out, ignore_index=True)


def by_condition(fam):
    """Collapse seeds to one verdict per condition (majority of seeds)."""
    g = fam.groupby(["arm", "opponent", "cell"]).agg(
        seeds=("dm_stat", "size"),
        classical=("dm_stat", lambda s: int((s > 0).sum())),
        quantum=("dm_stat", lambda s: int((s < 0).sum())))
    g["winner"] = np.where(g.classical > g.quantum, "classical",
                  np.where(g.quantum > g.classical, "quantum", "tied"))
    g["unanimous"] = (g.classical == g.seeds) | (g.quantum == g.seeds)
    return g.reset_index()


def report(fam, label):
    fam = fam.copy()
    fam["holm"] = holm(fam.p.values)
    n = len(fam)
    cls = int((fam.dm_stat > 0).sum())
    qnt = int((fam.dm_stat < 0).sum())
    sig = int((fam.holm < 0.05).sum())
    qsig = int(((fam.dm_stat < 0) & (fam.holm < 0.05)).sum())
    print(f"\n{label}")
    print(f"  tests                          {n}")
    print(f"  classical won                  {cls}  ({cls/n:.1%})")
    print(f"  quantum won                    {qnt}  ({qnt/n:.1%})")
    print(f"  of which significant (Holm)    {qsig}")
    print(f"  significant after Holm         {sig} / {n}")
    print(f"  largest Holm-adjusted p        {fam.holm.max():.3g}")
    return fam


def main():
    fam = load()
    print("=" * 74)
    print("  EXTENDED FAMILY — every quantum-vs-classical per-seed DM test")
    print("=" * 74)
    print("\nby arm:")
    g = fam.groupby("arm").agg(tests=("p", "size"),
                               classical=("dm_stat", lambda s: int((s > 0).sum())),
                               quantum=("dm_stat", lambda s: int((s < 0).sum())))
    g["classical %"] = (100 * g.classical / g.tests).round(1)
    print(g.to_string())

    report(fam[fam.core], "CURRENT confirmatory family (as published), per seed")
    report(fam, "EXTENDED family (everything pooled), per seed")

    # ---------------- the same thing counted by condition ----------------
    for label, sub in (("CURRENT", fam[fam.core]), ("EXTENDED", fam)):
        c = by_condition(sub)
        n = len(c)
        cw = int((c.winner == "classical").sum())
        qw = int((c.winner == "quantum").sum())
        print(f"\n{label} family, counted by CONDITION (seeds collapsed)")
        print(f"  conditions                     {n}")
        print(f"  classical won                  {cw}  ({cw/n:.1%})")
        print(f"  quantum won                    {qw}")
        print(f"  unanimous across their seeds   {int(c.unanimous.sum())} / {n}")
        if qw:
            print("  conditions quantum won:")
            for _, r in c[c.winner == "quantum"].iterrows():
                print(f"      {r.arm:26s} {r.cell:44s} {r.quantum}/{r.seeds} seeds")

    print("\nby opponent, EXTENDED family, counted by condition:")
    c = by_condition(fam)
    o = c.groupby("opponent").agg(conditions=("cell", "size"),
                                  classical=("winner", lambda s: int((s == "classical").sum())))
    o["classical %"] = (100 * o.classical / o.conditions).round(1)
    print(o.sort_values("conditions", ascending=False).to_string())


if __name__ == "__main__":
    main()
