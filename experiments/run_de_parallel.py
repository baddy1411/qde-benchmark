#!/usr/bin/env python3
"""Experiment C — parallel execution.  ->  results_de/parallel.csv

The sweep loop over independent (system, seed) cells is embarrassingly parallel:
no shared state, no cross-cell dependency. This measures the speed-up from running
those cells across worker processes with the STDLIB ``concurrent.futures``
ProcessPoolExecutor — deliberately not dask, whose distributed scheduler adds
nothing for a single-machine, independent-task workload (the "no technology for
appearance" principle).

Measured: wall-clock, speed-up (T1/Tn), parallel efficiency (speedup/n),
peak memory, failed tasks.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.logging_setup import configure, get_logger
from qdepipe.instrument import rss_mb

OUT = Path("results_de"); OUT.mkdir(exist_ok=True)


# top-level worker (must be picklable for ProcessPoolExecutor)
def _cell(args) -> dict:
    import warnings as _w; _w.filterwarnings("ignore")
    from qdepipe.experiment import run_experiment, ExperimentConfig
    from qdepipe.forecasters import ReservoirForecaster
    from qdepipe.models import ESN, ESNConfig
    system, seed = args
    cfg = ExperimentConfig(system=system, n_points=20_000, seed=seed,
                           lookback=5, washout=100)
    fc = ReservoirForecaster(ESN(ESNConfig(units=300, seed=seed)), "ESN",
                             external_window=True)
    r = run_experiment(fc, cfg)
    return {"system": system, "seed": seed, "nrmse": r.nrmse}


def run_pool(cells, workers):
    t0 = time.perf_counter()
    failed = 0
    if workers == 1:
        for c in cells:
            try:
                _cell(c)
            except Exception:
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f in [ex.submit(_cell, c) for c in cells]:
                try:
                    f.result()
                except Exception:
                    failed += 1
    return time.perf_counter() - t0, failed


if __name__ == "__main__":     # ProcessPoolExecutor requires the guard
    configure(run_name="de_parallel")
    log = get_logger("de.parallel")

    systems = ["henon", "lorenz", "mackeyglass"]
    cells = [(s, seed) for s in systems for seed in range(8)]   # 24 independent cells
    log.info("workload: %d independent cells (ESN, 20k points each)", len(cells))

    rows = []
    t1 = None
    for w in (1, 2, 4, 8):
        wall, failed = run_pool(cells, w)
        if w == 1:
            t1 = wall
        speedup = t1 / wall
        rows.append({"workers": w, "wall_seconds": round(wall, 2),
                     "speedup": round(speedup, 2),
                     "efficiency": round(speedup / w, 2),
                     "failed_tasks": failed,
                     "peak_memory_mb": round(rss_mb() or 0, 1)})
        log.info("workers=%d wall=%.2fs speedup=%.2fx efficiency=%.2f failed=%d",
                 w, wall, speedup, speedup / w, failed)

    pd.DataFrame(rows).to_csv(OUT / "parallel.csv", index=False)
    print("\n===== Experiment C summary =====")
    print(pd.DataFrame(rows).to_string(index=False))
    best = max(rows, key=lambda r: r["speedup"])
    print(f"\nbest speed-up: {best['speedup']}x at {best['workers']} workers "
          f"(efficiency {best['efficiency']}); {sum(r['failed_tasks'] for r in rows)} total failures")
    print("wrote results_de/parallel.csv")
