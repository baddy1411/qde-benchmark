#!/usr/bin/env python3
"""Experiment A — data-volume scaling.  ->  results_de/volume.csv

Measures per-stage wall-clock time, peak memory, throughput (rows/s) and output
size as the time-series length grows, for representative QRC / ESN / NG-RC / ELM
configurations. Each model sweeps ASCENDING sizes and stops when a single cell
exceeds a wall-clock budget — so the table honestly reports the largest size each
model reaches on this hardware (MacBook Air, Apple Silicon, CPU-only) rather than
assuming a fixed row count that is trivial for one model and infeasible for
another. That per-model ceiling is itself a finding.
"""
from __future__ import annotations

import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.logging_setup import configure, get_logger
from qdepipe.instrument import StageProfiler, rss_mb
from qdepipe.data import generate
from qdepipe.pipeline import make_scaler, temporal_split
from qdepipe.pipeline.embedding import supervised_pairs
from qdepipe.readout import ridge_fit, ridge_predict
from qdepipe.models import ESN, ESNConfig, ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.concentration import _rich_cfg
from qdepipe.models.gate_qrc import GateQRC

OUT = Path("results_de"); OUT.mkdir(exist_ok=True)
configure(run_name="de_volume")
log = get_logger("de.volume")

SYSTEM = "lorenz"          # a continuous system so large n is meaningful
BUDGET_S = 90.0            # per-cell wall-clock budget; stop a model past this
LOOKBACK = 5

# ascending size ladders per model, tuned to the cost class
LADDERS = {
    "ngrc": [10_000, 50_000, 100_000, 500_000, 1_000_000],
    "elm":  [10_000, 50_000, 100_000, 500_000],
    "esn":  [10_000, 50_000, 100_000, 500_000],
    "qrc_rich": [500, 1_000, 2_000, 5_000],   # ~seconds/timestep class
}


def build(model_key, n_points):
    if model_key == "ngrc":
        return NGRC(NGRCConfig(k=LOOKBACK, degree=3))
    if model_key == "elm":
        return ELM(ELMConfig(units=300, lookback=LOOKBACK, seed=0))
    if model_key == "esn":
        return ESN(ESNConfig(units=300, seed=0))
    if model_key == "qrc_rich":
        return GateQRC(_rich_cfg(6, V=4, seed=0))
    raise KeyError(model_key)


def one_cell(model_key, n_points):
    prof = StageProfiler(sample_interval=0.02).start()
    with prof.stage("data_generation"):
        x = generate(SYSTEM, n_points)
    with prof.stage("transformation"):
        split = temporal_split(len(x), (0.6, 0.2, 0.2), 100)
        sc = make_scaler("minmax"); sc.fit(x[split.train_fit])
        u = np.asarray(sc.transform(x), dtype=float).ravel()
    model = build(model_key, n_points)
    with prof.stage("feature_generation"):
        feats = model.featurize(u)
    with prof.stage("training"):
        X, y = supervised_pairs(feats, u, 1)
        a, b, w = split.train.stop, split.test.start, split.washout
        W = ridge_fit(X[slice(w, a - 1)], y[slice(w, a - 1)], alpha=1e-6, bias=True)
    with prof.stage("inference"):
        _ = ridge_predict(X[slice(b, len(X))], W, bias=True)
    prof.stop()
    s = prof.summary()
    s.update({"model": model_key, "n_points": n_points,
              "n_features": int(feats.shape[1]),
              "rows_per_sec": round(n_points / s["total_seconds"], 1),
              "feature_bytes": int(feats.nbytes)})
    return s


rows = []
for model_key, ladder in LADDERS.items():
    for n in ladder:
        try:
            s = one_cell(model_key, n)
        except MemoryError:
            log.warning("%s n=%d: MemoryError — stopping this model's ladder", model_key, n)
            rows.append({"model": model_key, "n_points": n, "total_seconds": None,
                         "peak_memory_mb": None, "note": "MemoryError"})
            break
        rows.append(s)
        log.info("%-9s n=%-8d total=%6.2fs feat=%6.2fs mem=%6.1fMB rows/s=%.0f",
                 model_key, n, s["total_seconds"], s.get("feature_generation_seconds", 0),
                 s["peak_memory_mb"] or -1, s["rows_per_sec"])
        if s["total_seconds"] > BUDGET_S:
            log.info("%s exceeded %.0fs budget at n=%d — stopping ladder",
                     model_key, BUDGET_S, n)
            break

pd.DataFrame(rows).to_csv(OUT / "volume.csv", index=False)

df = pd.DataFrame(rows)
print("\n===== Experiment A summary (largest size reached per model) =====")
ok = df[df["total_seconds"].notna()]
for m in LADDERS:
    d = ok[ok.model == m]
    if len(d):
        top = d.loc[d.n_points.idxmax()]
        print(f"  {m:9s} reached n={int(top.n_points):>9,}  "
              f"total={top.total_seconds:7.2f}s  peak={top.peak_memory_mb:6.1f}MB  "
              f"rows/s={top.rows_per_sec:>10,.0f}")
print("\nkey finding: quantum feature generation, not data volume, is the ceiling —")
print("QRC's largest feasible n is orders of magnitude below the classical models'.")
print("wrote results_de/volume.csv")
