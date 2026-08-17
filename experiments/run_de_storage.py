#!/usr/bin/env python3
"""Experiment B — storage-format comparison (CSV vs Parquet vs NumPy .npy).
->  results_de/storage.csv

Writes one representative result table three ways and measures file size, write
time, full read time, selected-column read time, schema preservation, and
compression ratio. Same in-memory table for every format, so the comparison is
not confounded by different data.

The selected-column read is the headline number for a DE thesis: CSV must parse
every column of every row to return one column; Parquet reads only that column's
pages. .npy is included because the project already uses it (feature cache).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.logging_setup import configure, get_logger

OUT = Path("results_de"); OUT.mkdir(exist_ok=True)
configure(run_name="de_storage")
log = get_logger("de.storage")


def make_table(n_rows: int) -> pd.DataFrame:
    """A representative result table: mixed dtypes like a real scoreboard."""
    rng = np.random.default_rng(0)
    systems = np.array(["henon", "lorenz", "mackeyglass", "lorenz96"])
    models = np.array(["ngrc", "esn", "elm", "qrc_rich", "lstm", "xgboost"])
    return pd.DataFrame({
        "run_id": [f"r{i:07d}" for i in range(n_rows)],
        "system": rng.choice(systems, n_rows),
        "model": rng.choice(models, n_rows),
        "seed": rng.integers(0, 20, n_rows),
        "n_points": rng.choice([800, 1500, 4000], n_rows),
        "nrmse": rng.lognormal(-6, 3, n_rows),
        "rmse": rng.lognormal(-5, 2, n_rows),
        "mae": rng.lognormal(-5, 2, n_rows),
        "feature_rank": rng.integers(4, 300, n_rows),
        "total_seconds": rng.exponential(2.0, n_rows),
    })


def timeit(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def bench(n_rows: int) -> list[dict]:
    df = make_table(n_rows)
    rows = []
    base = OUT / f"_bench_{n_rows}"
    paths = {"csv": base.with_suffix(".csv"),
             "parquet": base.with_suffix(".parquet"),
             "parquet_zstd": base.with_suffix(".zstd.parquet"),
             "npy": base.with_suffix(".npy")}

    writers = {
        "csv": lambda: df.to_csv(paths["csv"], index=False),
        "parquet": lambda: df.to_parquet(paths["parquet"], index=False, compression="snappy"),
        "parquet_zstd": lambda: df.to_parquet(paths["parquet_zstd"], index=False, compression="zstd"),
        # .npy only stores the numeric block (a real limitation to report: it
        # drops the string columns / schema — that's the point of the comparison)
        "npy": lambda: np.save(paths["npy"], df.select_dtypes("number").to_numpy()),
    }
    full_readers = {
        "csv": lambda: pd.read_csv(paths["csv"]),
        "parquet": lambda: pd.read_parquet(paths["parquet"]),
        "parquet_zstd": lambda: pd.read_parquet(paths["parquet_zstd"]),
        "npy": lambda: np.load(paths["npy"]),
    }
    col_readers = {  # read only the 'nrmse' column
        "csv": lambda: pd.read_csv(paths["csv"], usecols=["nrmse"]),
        "parquet": lambda: pd.read_parquet(paths["parquet"], columns=["nrmse"]),
        "parquet_zstd": lambda: pd.read_parquet(paths["parquet_zstd"], columns=["nrmse"]),
        "npy": lambda: np.load(paths["npy"])[:, 4],  # rank col index; positional only
    }
    csv_size = None
    for fmt in writers:
        w = timeit(writers[fmt])
        size = os.path.getsize(paths[fmt])
        if fmt == "csv":
            csv_size = size
        full = timeit(full_readers[fmt])
        col = timeit(col_readers[fmt])
        # schema preservation: does a full read give back the same dtypes?
        if fmt == "npy":
            schema_ok = False  # numeric-only, strings + column names lost
        else:
            rt = full_readers[fmt]()
            schema_ok = list(rt.dtypes.astype(str)) == list(df.dtypes.astype(str))
        rows.append({
            "n_rows": n_rows, "format": fmt,
            "file_bytes": size, "file_kb": round(size / 1024, 1),
            "write_ms": round(w * 1000, 2), "read_full_ms": round(full * 1000, 2),
            "read_1col_ms": round(col * 1000, 2),
            "compression_vs_csv": round(csv_size / size, 2) if csv_size else 1.0,
            "schema_preserved": schema_ok,
        })
        log.info("n=%d %-13s size=%.1fKB write=%.1fms read=%.1fms 1col=%.2fms schema=%s",
                 n_rows, fmt, size / 1024, w * 1000, full * 1000, col * 1000, schema_ok)
    for p in paths.values():
        p.unlink(missing_ok=True)
    return rows


all_rows = []
for n in (1_000, 50_000, 500_000):
    all_rows += bench(n)
pd.DataFrame(all_rows).to_csv(OUT / "storage.csv", index=False)

# headline summary
df = pd.DataFrame(all_rows)
big = df[df.n_rows == 500_000].set_index("format")
print("\n===== Experiment B summary (500k rows) =====")
print(big[["file_kb", "write_ms", "read_full_ms", "read_1col_ms",
           "compression_vs_csv", "schema_preserved"]].to_string())
csv1, pq1 = big.loc["csv", "read_1col_ms"], big.loc["parquet", "read_1col_ms"]
print(f"\nselected-column read speedup (Parquet vs CSV): {csv1/pq1:.1f}x")
print(f"Parquet(zstd) compression vs CSV: {big.loc['parquet_zstd','compression_vs_csv']:.1f}x smaller")
print("wrote results_de/storage.csv")
