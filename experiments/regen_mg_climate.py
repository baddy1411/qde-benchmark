#!/usr/bin/env python3
"""Regenerate the Mackey-Glass climate CSVs through the actual code paths, after the
LAMBDA_CONT update (0.0086 -> 0.00838). The closed-loop rollout is deterministic, so
the valid-prediction-time STEP counts are unchanged; only their conversion to Lyapunov
times (VPT = steps x lambda) is rescaled. Re-running through the real pipeline (rather
than hand-rescaling) keeps the CSVs provably faithful.

Reproduces:
  results/cross/mackeyglass_climate.csv          (classical @3 seeds + quantum @5 seeds)
  results/cross/mackeyglass_quantum_climate.csv  (quantum, full climate battery)
"""
from __future__ import annotations

import os
import pandas as pd

import cross_system as CS
import quantum_cross as QC

RESULTS = CS.RESULTS
SYS = "mackeyglass"


def main():
    # classical climate (NG-RC tuned k8d3, ESN, ELM, RandomForest), 3 seeds, n=3000
    cls = CS.climate(SYS, (0, 1, 2), {"k": 8, "degree": 3})
    cls.to_csv(os.path.join(RESULTS, "cross", f"{SYS}_climate.csv"), index=False)
    print("classical climate regenerated:")
    print(cls.to_string(index=False))

    # quantum climate (qrc at MG best encodings from mackeyglass_matched_budget.csv),
    # 5 seeds, n=1500; quantum_climate() writes the quantum CSV and appends to the
    # main climate CSV (drops prior quantum rows first).
    best = {"qrc_v4": {"scaler": "minmax", "encode_scale": 0.5},
            "qrc_v6": {"scaler": "minmax", "encode_scale": 3.0},
            "qrc_rich": {"scaler": "minmax", "encode_scale": 1.0}}
    qdf = QC.quantum_climate(SYS, (0, 1, 2, 3, 4), best)
    print("\nquantum climate regenerated:")
    print(qdf.to_string(index=False))

    print("\nfinal merged cross/mackeyglass_climate.csv:")
    print(pd.read_csv(os.path.join(RESULTS, "cross", f"{SYS}_climate.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
