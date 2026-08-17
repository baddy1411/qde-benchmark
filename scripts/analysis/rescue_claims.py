#!/usr/bin/env python3
"""Derive every rescue-experiment claim: augmentation parity, V-sweep gap,
polynomial-encoding losses, ChebPoly champion, RF-QRC twin match and shot
robustness, advocate summary.

Reads  results/{followup,cheb,rfqrc,rfqrc_cheb,qadv}/...
Writes derived/rescue_claims.csv and prints CLAIM lines.
"""
import os
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "derived")
os.makedirs(OUT, exist_ok=True)
claims = []


def claim(key, value, thesis):
    claims.append({"key": key, "value": value, "thesis": thesis})
    print(f"CLAIM {key} = {value}   (thesis: {thesis})")


# ---- augmentation + V sweep -------------------------------------------------
f = pd.read_csv(os.path.join(ROOT, "results/followup/scores.csv"),
                keep_default_na=False)
f["nrmse"] = f.nrmse.astype(float)
f = f.drop_duplicates(subset=["experiment", "system", "n", "V", "model",
                              "readout", "seed"], keep="last")
r8 = f[(f.experiment == "readout") & (f.n == 8) & (f.system == "henon")]
med = r8.groupby(["model", "readout"]).nrmse.median()
claim("henon n8 qrc poly2int", f"{med['qrc']['poly2int']:.2e}", "4.4e-5")
claim("henon n8 parity ratio",
      f"{med['qrc']['poly2int']/med['elmF']['poly2int']:.3f}", "1.04")
v = f[f.experiment == "vsweep"]
g = v.groupby(["system", "V", "model"]).nrmse.median()
claim("henon V-sweep gap V4 -> V16",
      f"{g['henon'][4]['qrc']/g['henon'][4]['elmF']:.1f}x -> "
      f"{g['henon'][16]['qrc']/g['henon'][16]['elmF']:.1f}x", "4.4x -> 23x")

# ---- polynomial encodings ----------------------------------------------------
c = pd.read_csv(os.path.join(ROOT, "results/cheb/scores.csv"),
                keep_default_na=False)
c["nrmse"] = c.nrmse.astype(float)
c = c.drop_duplicates(subset=["system", "model", "seed"], keep="last")
cm = c.groupby(["system", "model"]).nrmse.median()
claim("henon chebpoly (champion)", f"{cm['henon']['chebpoly']:.2e}", "9.6e-8")
claim("henon cheb-encoding vs depth",
      f"{cm['henon']['cheb']/cm['henon']['depth']:.1f}x worse", "~5x worse")
dmc = pd.read_csv(os.path.join(ROOT, "results/cheb/dm.csv"))
losses = int(((dmc.p < 0.05) & (dmc.dm_stat > 0)).sum())
claim("cheb DM losing significant", f"{losses}/{len(dmc)}", "60/60")

# ---- RF-QRC -------------------------------------------------------------------
mv = pd.read_csv(os.path.join(ROOT, "results/rfqrc/mv_pilot_convention.csv"))
mvm = mv.groupby("model").vpt_lyap.median()
claim("rfqrc MV VPT", f"{mvm['rfqrc']:.2f}", "8.02")
claim("classical twin MV VPT", f"{mvm['classical_twin']:.2f}", "8.06")
sh = pd.read_csv(os.path.join(ROOT, "results/rfqrc/shots.csv"))
h = sh[sh.system == "henon"]
best8192 = h[h.shots == 8192].test_nrmse.min()
one = pd.read_csv(os.path.join(ROOT, "results/rfqrc/onestep.csv"),
                  keep_default_na=False)
exact = one[(one.system == "henon") & (one.model == "rfqrc")]
exact_best = exact.loc[exact.val_nrmse.astype(float).idxmin()].test_nrmse
claim("rfqrc shot degradation at 8192",
      f"{(float(best8192)/float(exact_best)-1)*100:.0f}%", "within ~8%")

rc = pd.read_csv(os.path.join(ROOT, "results/rfqrc_cheb/mv_closedloop.csv"))
rcm = rc.groupby("model").vpt_lyap.median()
claim("rfqrc-cheb MV collapse",
      f"{rcm['rfqrc_linear']:.2f} -> {rcm['rfqrc_cheb']:.2f}", "8.02 -> 0.09")

# ---- advocate ------------------------------------------------------------------
q = pd.read_csv(os.path.join(ROOT, "results/qadv/scores.csv"),
                keep_default_na=False)
q["test_nrmse"] = q.test_nrmse.astype(float)
q = q.drop_duplicates(subset=["system", "model", "seed"], keep="last")
qm = q.groupby(["system", "model"]).test_nrmse.median()
claim("advocate henon J0", f"{qm['henon']['qadv_J0']:.2e}", "1.43e-8")
claim("advocate vs chebpoly ratio",
      f"{qm['henon']['qadv_J0']/qm['henon']['chebpoly']:.2f}", "~0.66 (tie)")
qdm = pd.read_csv(os.path.join(ROOT, "results/qadv/dm.csv"))
tie = qdm[qdm.vs == "chebpoly"]
claim("advocate vs chebpoly DM significant",
      f"{int((tie.p < 0.05).sum())}/{len(tie)}", "0/15")

pd.DataFrame(claims).to_csv(os.path.join(OUT, "rescue_claims.csv"), index=False)
