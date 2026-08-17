#!/usr/bin/env python3
"""Derive every qubit-scaling claim in the thesis from the raw artifacts.

Reads  results/scaling_proof/{dm,scores}.csv
Writes derived/scaling_claims.csv and prints each claim as CLAIM lines that
must match the thesis prose (abstract, sec:results:scaling, Appendix C).
Deterministic: bootstrap uses a fixed seed.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "derived")
os.makedirs(OUT, exist_ok=True)

dm = pd.read_csv(os.path.join(ROOT, "results/scaling_proof/dm.csv"))
sc = pd.read_csv(os.path.join(ROOT, "results/scaling_proof/scores.csv"),
                 keep_default_na=False)

# ---- the pre-specified family: QRC vs the two matched controls -------------
fam = dm[dm.comparison.isin(["qrc_vs_elmF", "qrc_vs_ngrc"])].copy()
n_total = len(fam)
n_qrc_worse = int((fam.dm_stat > 0).sum())
n_qrc_better = int((fam.dm_stat < 0).sum())
rev = fam[fam.dm_stat < 0]

# ---- Holm–Bonferroni over the whole family ---------------------------------
p = np.sort(fam.p.values)
m = len(p)
holm = np.maximum.accumulate(p * (m - np.arange(m)))
all_sig = bool((holm < 0.05).all())

core = fam[fam.n < 12]
p2 = np.sort(core.p.values)
holm2 = np.maximum.accumulate(p2 * (len(p2) - np.arange(len(p2))))

# ---- effect sizes over the paired seed-cells --------------------------------
piv = sc.pivot_table(index=["system", "n_qubits_eq", "seed"],
                     columns="model", values="nrmse").dropna(subset=["qrc"])
rng = np.random.default_rng(0)
rows = []
for rival in ["elmF", "ngrc"]:
    d = (piv.qrc - piv[rival]).values
    r = (piv.qrc / piv[rival]).values
    n = len(r)
    boot = [np.median(rng.choice(r, n, replace=True)) for _ in range(10_000)]
    rows.append({"rival": rival, "n_pairs": n,
                 "median_ratio": float(np.median(r)),
                 "ratio_ci_lo": float(np.percentile(boot, 2.5)),
                 "ratio_ci_hi": float(np.percentile(boot, 97.5)),
                 "median_paired_diff": float(np.median(d))})
eff = pd.DataFrame(rows)

summary = pd.DataFrame([{
    "family_size": n_total, "qrc_worse": n_qrc_worse, "qrc_better": n_qrc_better,
    "max_raw_p": float(fam.p.max()), "max_holm_p": float(holm.max()),
    "all_holm_significant": all_sig,
    "core_family_size": len(core), "core_max_holm_p": float(holm2.max()),
    "reversal_system": rev.system.iloc[0] if len(rev) else "",
    "reversal_n": int(rev.n.iloc[0]) if len(rev) else -1,
    "reversal_comparison": rev.comparison.iloc[0] if len(rev) else "",
}])
summary.to_csv(os.path.join(OUT, "scaling_claims.csv"), index=False)
eff.to_csv(os.path.join(OUT, "scaling_effect_sizes.csv"), index=False)

print(f"CLAIM family = {n_total} tests (thesis: 214)")
print(f"CLAIM qrc significantly worse in {n_qrc_worse} (thesis: 213), "
      f"better in {n_qrc_better} (thesis: 1)")
print(f"CLAIM reversal = {rev.system.iloc[0]} n={int(rev.n.iloc[0])} "
      f"{rev.comparison.iloc[0]} (thesis: henon n=4 vs elmF)")
print(f"CLAIM max raw p = {fam.p.max():.3g} (thesis: 5.6e-5)")
print(f"CLAIM max Holm-adjusted p = {holm.max():.3g} (thesis: 9.7e-5); "
      f"all significant: {all_sig}")
print(f"CLAIM excluding n=12 arm: {len(core)} tests (thesis: 210), "
      f"max Holm p {holm2.max():.3g}")
for _, e in eff.iterrows():
    print(f"CLAIM ratio vs {e.rival}: {e.median_ratio:.2g} "
          f"[{e.ratio_ci_lo:.2g}, {e.ratio_ci_hi:.2g}], "
          f"paired diff {e.median_paired_diff:.2g}, n={int(e.n_pairs)} "
          f"(thesis: elmF 4.6 [3.0,5.2] 3.5e-3; ngrc 7.3 [6.2,9.4] 4.4e-3)")
