#!/usr/bin/env python3
"""Derive the pretrained-baseline claims from results/pretrained/*.csv."""
import os
import sys

import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "derived")
os.makedirs(OUT, exist_ok=True)

z5 = pd.read_csv(os.path.join(ROOT, "results/pretrained/zeroshot_ctx512.csv"))
z20 = pd.read_csv(os.path.join(ROOT, "results/pretrained/zeroshot_ctx2048.csv"))
lead = pd.read_csv(os.path.join(ROOT, "results/cross/mackeyglass_leaderboard.csv"))

h5 = z5.set_index("system")
h20 = z20.set_index("system")
lin = lead.set_index("model").loc["Linear-Ridge", "nrmse_mean"]

rows = []
def claim(k, v, t):
    rows.append({"key": k, "value": v, "thesis": t})
    print(f"CLAIM {k} = {v}   (thesis: {t})")

claim("henon zero-shot NRMSE", f"{h5.loc['henon'].onestep_nrmse:.3f}", "~1.0 (mean level)")
claim("lorenz zero-shot NRMSE", f"{h5.loc['lorenz'].onestep_nrmse:.2f}", "0.22")
claim("mg zero-shot NRMSE", f"{h5.loc['mackeyglass'].onestep_nrmse:.2f}", "0.12")
claim("mg vs untuned linear ridge",
      f"{h5.loc['mackeyglass'].onestep_nrmse/lin:.0f}x", "~40x")
claim("lorenz spectral MSE ctx512 -> ctx2048",
      f"{h5.loc['lorenz'].spectral_mse:.0f} -> {h20.loc['lorenz'].spectral_mse:.1f}",
      "989 -> 5.6")
claim("point accuracy vs context (henon)",
      f"{h5.loc['henon'].onestep_nrmse:.2f} vs {h20.loc['henon'].onestep_nrmse:.2f}",
      "unchanged")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "chronos_claims.csv"), index=False)
