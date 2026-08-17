#!/usr/bin/env python3
"""Run the Part D frequency-domain redundancy test -> results/concentration/frequency_redundancy.csv.

Reports, per system (Hénon, Mackey-Glass): the ZZ residual-power fraction (linear
reachability; threshold-free) and the ZZ new-frequency fraction (at several support
thresholds for robustness). STOPS at the checkpoint; no prose.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from qdepipe import frequency as F

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# QRC-rich per-system best encode_scale (Hénon from adv_A2; MG from matched_budget)
BEST_ENC = {"henon": 0.5, "mackeyglass": 1.0}
THRESHOLDS = [0.001, 0.01, 0.05]


def main():
    os.makedirs(os.path.join(RESULTS, "concentration"), exist_ok=True)
    from qdepipe._provenance import write_versions
    write_versions(os.path.join(os.path.dirname(RESULTS), "versions.txt"))
    rows = []
    for system, enc in BEST_ENC.items():
        S, ZZ = F.observable_timeseries(system, enc)
        rpf = F.residual_power_fraction(S, ZZ)
        nffs = {t: F.new_frequency_fraction(S, ZZ, t) for t in THRESHOLDS}
        rows.append({"system": system, "encode_scale": enc,
                     "zz_residual_power_fraction": rpf,
                     "zz_new_frequency_fraction_thr0.001": nffs[0.001],
                     "zz_new_frequency_fraction_thr0.01": nffs[0.01],
                     "zz_new_frequency_fraction_thr0.05": nffs[0.05]})
        print(f"  {system:12s} (enc={enc}): residual-power={rpf:.4g}  "
              f"new-freq[thr .001/.01/.05]={nffs[0.001]:.4g}/{nffs[0.01]:.4g}/{nffs[0.05]:.4g}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "concentration", "frequency_redundancy.csv"), index=False)

    print("\n================ CHECKPOINT (Part D) ================")
    print(df.to_string(index=False))
    h = df[df.system == "henon"].iloc[0]
    m = df[df.system == "mackeyglass"].iloc[0]
    print(f"\nfour headline fractions (residual-power, new-freq@0.01):")
    print(f"  Hénon: residual-power={h['zz_residual_power_fraction']:.4g}, "
          f"new-freq={h['zz_new_frequency_fraction_thr0.01']:.4g}")
    print(f"  MG:    residual-power={m['zz_residual_power_fraction']:.4g}, "
          f"new-freq={m['zz_new_frequency_fraction_thr0.01']:.4g}")
    print("wrote results/concentration/frequency_redundancy.csv")


if __name__ == "__main__":
    main()
