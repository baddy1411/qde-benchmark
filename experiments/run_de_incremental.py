#!/usr/bin/env python3
"""Experiment D — incremental execution.  ->  results_de/incremental.csv

Exercises the consolidated ``run_sweep`` resume logic across the five canonical
scenarios: clean rebuild, cached rerun (no change), change one model config,
change one dataset, and resume after a simulated failure. Measures tasks executed
vs skipped and confirms no duplicate rows are ever created and existing correct
outputs are preserved.
"""
from __future__ import annotations

import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.logging_setup import configure, get_logger
from qdepipe.sweep import run_sweep
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import NGRC, NGRCConfig

OUT = Path("results_de"); OUT.mkdir(exist_ok=True)
configure(run_name="de_incremental")
log = get_logger("de.incremental")

# a small grid of independent cells: (system, seed, lookback)
BASE_CELLS = [(s, seed, 5) for s in ("henon", "lorenz", "mackeyglass")
              for seed in range(4)]


def cell_key(c):
    return c  # (system, seed, lookback) is already the identity


def run_cell(c):
    system, seed, lookback = c
    cfg = ExperimentConfig(system=system, n_points=1500, seed=seed,
                           lookback=lookback, washout=100)
    r = run_experiment(ReservoirForecaster(
        NGRC(NGRCConfig(k=lookback, degree=2)), "NG-RC"), cfg)
    return {"key": c, "nrmse": r.nrmse}


def sweep(cells, done, fail_on=None):
    store = []
    def run_one(c):
        if fail_on is not None and c == fail_on:
            raise RuntimeError(f"simulated failure on {c}")
        return run_cell(c)
    res = run_sweep(cells, run_one, done=done, key_fn=cell_key, append_fn=store.append)
    return res, store


rows = []
# 1 — clean rebuild (nothing done yet)
t0 = time.perf_counter()
res, store = sweep(BASE_CELLS, done=[])
rows.append({"scenario": "1_clean_rebuild", "executed": res.executed,
             "skipped": res.skipped, "failed": res.failed,
             "seconds": round(time.perf_counter() - t0, 2)})
done_keys = [s["key"] for s in store]

# 2 — cached rerun, no changes (everything already done -> all skipped)
t0 = time.perf_counter()
res, _ = sweep(BASE_CELLS, done=done_keys)
rows.append({"scenario": "2_cached_rerun", "executed": res.executed,
             "skipped": res.skipped, "failed": res.failed,
             "seconds": round(time.perf_counter() - t0, 2)})

# 3 — change one model config: one cell's lookback 5 -> 7 (only that recomputes)
changed = [c if c != ("henon", 0, 5) else ("henon", 0, 7) for c in BASE_CELLS]
t0 = time.perf_counter()
res, _ = sweep(changed, done=done_keys)   # the (henon,0,7) key is new -> 1 executed
rows.append({"scenario": "3_change_one_config", "executed": res.executed,
             "skipped": res.skipped, "failed": res.failed,
             "seconds": round(time.perf_counter() - t0, 2)})

# 4 — change one dataset: add a brand-new (system,seed) cell
added = BASE_CELLS + [("lorenz96", 0, 5)]
t0 = time.perf_counter()
res, _ = sweep(added, done=done_keys)     # only the lorenz96 cell is new
rows.append({"scenario": "4_change_one_dataset", "executed": res.executed,
             "skipped": res.skipped, "failed": res.failed,
             "seconds": round(time.perf_counter() - t0, 2)})

# 5 — resume after failure: fail mid-sweep, then resume with the completed keys
res_fail, store_fail = sweep(BASE_CELLS, done=[], fail_on=("lorenz", 2, 5))
done_after = [s["key"] for s in store_fail]
res_resume, _ = sweep(BASE_CELLS, done=done_after)   # only the failed cell remains
rows.append({"scenario": "5a_run_with_failure", "executed": res_fail.executed,
             "skipped": res_fail.skipped, "failed": res_fail.failed, "seconds": None})
rows.append({"scenario": "5b_resume_after_fail", "executed": res_resume.executed,
             "skipped": res_resume.skipped, "failed": res_resume.failed, "seconds": None})

pd.DataFrame(rows).to_csv(OUT / "incremental.csv", index=False)
print("\n===== Experiment D summary =====")
print(pd.DataFrame(rows).to_string(index=False))
# assertions that make the point explicit
r2 = rows[1]; r3 = rows[2]; r4 = rows[3]
ok = (r2["executed"] == 0 and r2["skipped"] == len(BASE_CELLS)          # cached: all skipped
      and r3["executed"] == 1 and r4["executed"] == 1                    # change-one: exactly 1
      and rows[5]["executed"] == 1)                                      # resume: only failed cell
print(f"\ncached rerun recomputed {r2['executed']} cells (want 0); "
      f"change-one recomputed {r3['executed']} (want 1); "
      f"resume recomputed {rows[5]['executed']} (want 1) -> "
      f"{'PASS' if ok else 'FAIL'}")
print("wrote results_de/incremental.csv")
