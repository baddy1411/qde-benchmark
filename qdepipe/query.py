"""DuckDB query layer over the Parquet registry (and, directly, over legacy CSVs).

DuckDB reads Parquet (and CSV) files with no server, no ingestion step, no
persistent database file required — so this module is thin: it points DuckDB at
``results_registry/*.parquet`` and exposes ``sql()`` plus a few saved queries the
thesis actually asks repeatedly.

Because DuckDB can also query CSV directly, ``sql_over_csv()`` gives an instant
SQL layer over the existing ``results/*.csv`` even before any Parquet backfill —
the cheapest possible query win.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .registry_io import ROOT as REGISTRY_ROOT


def _con(root: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """A connection with the four registry tables registered as views."""
    root = Path(root) if root is not None else REGISTRY_ROOT
    con = duckdb.connect(database=":memory:")
    for tbl in ("datasets", "splits", "runs", "metrics"):
        p = root / f"{tbl}.parquet"
        if p.exists():
            con.execute(f"CREATE VIEW {tbl} AS SELECT * FROM read_parquet('{p}')")
    return con


def sql(query: str, root: Optional[Path] = None) -> pd.DataFrame:
    """Run SQL against the registry views (datasets/splits/runs/metrics)."""
    return _con(root).execute(query).df()


def sql_over_csv(query: str, csv_path: str) -> pd.DataFrame:
    """Run SQL directly over a single CSV (no registry needed). ``{csv}`` in the
    query is substituted with a ``read_csv_auto`` of the path."""
    con = duckdb.connect(database=":memory:")
    q = query.replace("{csv}", f"read_csv_auto('{csv_path}')")
    return con.execute(q).df()


# ---------------------------------------------------------------------------
# saved queries the thesis asks repeatedly
# ---------------------------------------------------------------------------
def best_model_per_system(root: Optional[Path] = None) -> pd.DataFrame:
    """Lowest median NRMSE per (system, model), ranked — the leaderboard, from
    the registry rather than by hand-opening CSVs."""
    return sql("""
        SELECT d.system, r.model, r.model_family,
               median(m.nrmse) AS median_nrmse,
               count(*) AS n_runs
        FROM runs r
        JOIN datasets d ON r.dataset_id = d.dataset_id
        JOIN metrics  m ON r.run_id     = m.run_id
        WHERE r.status = 'success' AND m.nrmse IS NOT NULL
        GROUP BY d.system, r.model, r.model_family
        ORDER BY d.system, median_nrmse
    """, root)


def runs_by_family(root: Optional[Path] = None) -> pd.DataFrame:
    """How many runs, and their NRMSE spread, per model family."""
    return sql("""
        SELECT r.model_family,
               count(*) AS n_runs,
               count(DISTINCT r.dataset_id) AS n_datasets,
               median(m.nrmse) AS median_nrmse,
               min(m.nrmse) AS best_nrmse
        FROM runs r JOIN metrics m ON r.run_id = m.run_id
        WHERE m.nrmse IS NOT NULL
        GROUP BY r.model_family ORDER BY median_nrmse
    """, root)


def lineage_of(run_id: str, root: Optional[Path] = None) -> pd.DataFrame:
    """The full lineage of one run: its dataset, split, config, code commit."""
    return sql(f"""
        SELECT r.run_id, d.system, d.dataset_hash, s.split_id, s.scaler_fit_range,
               r.model, r.configuration_id, r.seed, r.git_commit, m.nrmse
        FROM runs r
        JOIN datasets d ON r.dataset_id = d.dataset_id
        JOIN splits   s ON r.split_id   = s.split_id
        JOIN metrics  m ON r.run_id     = m.run_id
        WHERE r.run_id = '{run_id}'
    """, root)


def leakage_safety_check(root: Optional[Path] = None) -> pd.DataFrame:
    """Every distinct scaler_fit_range in the registry — a one-query audit that
    no headline run used the leakage-unsafe 'global' scope by accident."""
    return sql("""
        SELECT s.scaler_fit_range, count(*) AS n_runs
        FROM runs r JOIN splits s ON r.split_id = s.split_id
        GROUP BY s.scaler_fit_range
    """, root)
