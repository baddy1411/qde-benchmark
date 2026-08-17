#!/usr/bin/env python3
"""Derive the initial-condition-study claims: the 0/60 tally and the
multi-metric IC table of the appendix.

Reads  results/ic_robustness/{onestep,closedloop}.csv
Writes derived/ic_table.csv and prints CLAIM lines.
"""
import os
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "derived")
os.makedirs(OUT, exist_ok=True)

one = pd.read_csv(os.path.join(ROOT, "results/ic_robustness/onestep.csv"))
cl = pd.read_csv(os.path.join(ROOT, "results/ic_robustness/closedloop.csv"))

# Committed convention (summary.csv): one winner per (system, IC), decided on
# the median across seeds -> 20 ICs x 3 systems = 60 cells. Also report the
# finer per-(IC, seed) tally as a robustness check.
med = one.groupby(["system", "ic_index", "model"]).nrmse.median().reset_index()
wins = cells = 0
for (system, ic), g in med.groupby(["system", "ic_index"]):
    g = g.set_index("model").nrmse
    if "qrc_rich" not in g.index:
        continue
    cells += 1
    if g["qrc_rich"] < g.drop("qrc_rich").min():
        wins += 1
fine_w = fine_c = 0
for (system, ic, seed), g in one.groupby(["system", "ic_index", "seed"]):
    g = g.set_index("model").nrmse
    if "qrc_rich" not in g.index:
        continue
    fine_c += 1
    if g["qrc_rich"] < g.drop("qrc_rich").min():
        fine_w += 1

a = one.groupby(["system", "model"]).nrmse.median().rename("nrmse_med")
b = cl.groupby(["system", "model"]).agg(vpt_med=("vpt_lyap", "median"),
                                        div_frac=("diverged", "mean"),
                                        w1_med=("wasserstein", "median"))
tab = a.to_frame().join(b).reset_index()
tab.to_csv(os.path.join(OUT, "ic_table.csv"), index=False)

print(f"CLAIM quantum wins {wins} of {cells} IC cells (thesis: 0 of 60; committed convention)")
print(f"CLAIM finer per-seed tally: {fine_w} of {fine_c} (robustness check)")
print(tab.round(4).to_string(index=False))
h = tab[tab.system == "henon"].set_index("model")
print(f"CLAIM henon W1 medians: qrc_rich {h.loc['qrc_rich'].w1_med:.4f} vs "
      f"ngrc {h.loc['ngrc'].w1_med:.4f} (thesis: 0.0132 vs 0.0136)")
