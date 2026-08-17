"""Parquet result registry — the queryable, schema-enforced lineage layer.

Implements ``RESULT_SCHEMA.md`` Part B items 3, 4, 11, 12. ADDITIVE: writes only
to a NEW top-level ``results_registry/`` directory (sibling of ``results/``, never
inside it), so no existing verified CSV is ever touched. The existing CSVs remain
the human-readable source of truth; this registry is the fast, typed, joinable
query layer over the four contracts.

Four tables (one Parquet file each): datasets, splits, runs, metrics — joinable on
``dataset_id`` / ``split_id`` / ``run_id``.

Engineering guarantees:
* **Schema-enforced writes.** Every appended row is validated against its table's
  pandera schema (``contracts.REGISTRY_SCHEMAS``) before it can land — closing
  the gap where the pipeline *has* schemas but ad-hoc scripts bypass them.
* **Atomic writes.** Parquet has no true row-append, so ``append_rows`` does
  read -> concat -> write-temp -> ``os.replace`` (the exact atomic pattern the
  feature store uses for ``.npy``). An interrupted write can never leave a
  half-written/visible Parquet file.
* **Duplicate-row protection.** ``append_rows`` refuses to insert a row whose
  logical key already exists (configurable per table), so re-running a backfill
  or a sweep cannot silently double-count.
* **Overwrite protection for legacy CSVs.** ``guard_csv_write`` refuses to
  overwrite an existing CSV whose content hash differs from its recorded
  manifest — a systemic guard against the filename-collision class of bug that
  hit this project once (06_threats.tex), rather than the one-off fix.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from . import contracts, registry_ids

ROOT = Path(__file__).resolve().parent.parent / "results_registry"

TABLES = ("datasets", "splits", "runs", "metrics")

# logical uniqueness key per table (for duplicate-row protection)
KEYS = {
    "datasets": ["dataset_id"],
    "splits": ["split_id"],
    "runs": ["run_id"],
    "metrics": ["run_id"],
}

# column order / expected columns per table (Contracts A-D)
COLUMNS = {
    "datasets": ["dataset_id", "system", "system_parameters", "initial_condition",
                 "random_seed", "integration_method", "time_step", "sample_count",
                 "transient_removed", "created_at", "code_version", "dataset_hash"],
    "splits": ["split_id", "dataset_id", "train_start", "train_end",
               "validation_start", "validation_end", "test_start", "test_end",
               "scaler_fit_range", "window_length", "prediction_horizon",
               "split_hash"],
    "runs": ["run_id", "dataset_id", "split_id", "model", "model_family",
             "configuration_id", "seed", "forecast_mode", "status",
             "provenance_source", "start_time", "end_time", "git_commit",
             "environment_id", "error_message"],
    "metrics": ["run_id", "nrmse", "mae", "valid_prediction_time", "divergence",
                "feature_count", "feature_rank", "data_generation_seconds",
                "validation_seconds", "transformation_seconds",
                "feature_generation_seconds", "training_seconds",
                "inference_seconds", "total_seconds", "peak_memory_mb"],
}


def _path(table: str, root: Optional[os.PathLike] = None) -> Path:
    base = Path(root) if root is not None else ROOT
    return base / f"{table}.parquet"


def read_table(table: str, root: Optional[os.PathLike] = None) -> pd.DataFrame:
    """Load a registry table (empty DataFrame with the right columns if absent)."""
    p = _path(table, root)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=COLUMNS[table])


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)          # atomic on POSIX — no half-written visible file


def append_rows(table: str, rows: Iterable[dict], *,
                root: Optional[os.PathLike] = None,
                on_duplicate: str = "skip") -> int:
    """Append validated rows to a registry table with duplicate protection.

    ``on_duplicate``: "skip" (default — silently drop rows whose key already
    exists, so a re-run is idempotent), "error" (raise on any duplicate),
    "replace" (overwrite the existing row with the same key).
    Returns the number of rows actually written.
    """
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; valid: {TABLES}")
    new = pd.DataFrame(list(rows))
    if new.empty:
        return 0
    # normalise columns to the contract shape (missing -> NA, extra -> error)
    extra = [c for c in new.columns if c not in COLUMNS[table]]
    if extra:
        raise ValueError(f"unknown column(s) for {table}: {extra}")
    new = new.reindex(columns=COLUMNS[table])
    # SCHEMA-ENFORCED WRITE — the guarantee this module's docstring makes.
    # Runs before duplicate filtering so a malformed row is rejected whether or
    # not it happens to collide with an existing key.
    new = contracts.validate_registry_rows(table, new)

    existing = read_table(table, root)
    key = KEYS[table]
    if not existing.empty:
        merged_keys = set(map(tuple, existing[key].astype(str).values.tolist()))
        dup_mask = new[key].astype(str).apply(tuple, axis=1).isin(merged_keys)
        if dup_mask.any():
            if on_duplicate == "error":
                raise ValueError(
                    f"{int(dup_mask.sum())} duplicate {key} in {table}: "
                    f"{new.loc[dup_mask, key].values.tolist()[:3]}...")
            if on_duplicate == "replace":
                existing = existing[~existing[key].astype(str).apply(tuple, axis=1)
                                    .isin(set(map(tuple, new[key].astype(str)
                                                  .values.tolist())))]
            else:  # skip
                new = new[~dup_mask]
    if new.empty:
        return 0
    out = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    _atomic_write_parquet(out, _path(table, root))
    return int(len(new))


# ---------------------------------------------------------------------------
# overwrite protection for legacy CSV results (systemic guard for the
# filename-collision class of bug documented in 06_threats.tex)
# ---------------------------------------------------------------------------
def _content_hash(path: Path) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_csv_write(path: os.PathLike, *, allow_overwrite: bool = False) -> None:
    """Raise if ``path`` already exists and its content differs from the hash
    recorded in its ``<path>.manifest`` sidecar — preventing an accidental
    overwrite of a verified result file by a differently-producing script.

    Call this immediately BEFORE writing a final result CSV. On the first write
    it records the manifest; on later writes it verifies the existing file is the
    one the manifest describes (i.e. this writer is re-producing the same file,
    not clobbering a different one). ``allow_overwrite=True`` bypasses the guard
    for a deliberate, reviewed replacement and refreshes the manifest.
    """
    path = Path(path)
    manifest = path.with_suffix(path.suffix + ".manifest")
    if path.exists() and manifest.exists() and not allow_overwrite:
        recorded = manifest.read_text().strip()
        actual = _content_hash(path)
        if recorded != actual:
            raise FileExistsError(
                f"refusing to overwrite {path}: on-disk content hash {actual} "
                f"does not match recorded manifest {recorded}. This looks like a "
                f"filename collision (a different producer). Pass "
                f"allow_overwrite=True only after confirming this is intended.")


def record_csv_manifest(path: os.PathLike) -> None:
    """Record/refresh the content-hash manifest AFTER a successful CSV write."""
    path = Path(path)
    if path.exists():
        path.with_suffix(path.suffix + ".manifest").write_text(_content_hash(path))


# ---------------------------------------------------------------------------
# convenience: assemble the four rows for one completed run
# ---------------------------------------------------------------------------
def rows_for_run(cfg, *, model_key: str, model_family: str, forecast_mode: str,
                 metrics: dict, series=None, spec=None, split_fracs=(0.6, 0.2, 0.2),
                 status: str = "success", git_commit: Optional[str] = None,
                 environment_id: Optional[str] = None,
                 error_message: Optional[str] = None,
                 start_time=None, end_time=None,
                 provenance_source: str = "captured") -> dict:
    """Build the {datasets,splits,runs,metrics} row dicts for one run from an
    ExperimentConfig + its metrics. ``series``/``spec`` optional (enable the
    content hashes); everything degrades gracefully if omitted."""
    import json
    dmeta = registry_ids.dataset_metadata(cfg.system, cfg.n_points,
                                          seed=getattr(cfg, "seed", None))
    did = registry_ids.dataset_id(**dmeta)
    dhash = registry_ids.dataset_hash(series) if series is not None else None
    sfr = getattr(cfg, "scaler_scope", "train")
    sid = (registry_ids.split_id_from_spec(
                did, spec, lookback=cfg.lookback, horizon=cfg.horizon,
                split_fracs=split_fracs, scaler_fit_range=sfr)
           if spec is not None else
           registry_ids.split_id(did, split_fracs=split_fracs,
                                 washout=cfg.washout, lookback=cfg.lookback,
                                 horizon=cfg.horizon, scaler_fit_range=sfr))
    rid = registry_ids.run_id()
    cid = registry_ids.configuration_id(cfg)
    if git_commit is None:
        from .tracking import git_sha
        git_commit = git_sha()
    if environment_id is None:
        from .envid import environment_id as _environment_id
        environment_id = _environment_id()

    def js(o):
        return json.dumps(o) if o is not None else None

    dataset_row = {**{k: dmeta[k] for k in ("system", "random_seed",
                                            "integration_method", "time_step",
                                            "sample_count", "transient_removed")},
                   "dataset_id": did,
                   "system_parameters": js(dmeta["system_parameters"]),
                   "initial_condition": js(dmeta["initial_condition"]),
                   "created_at": start_time, "code_version": git_commit,
                   "dataset_hash": dhash}
    split_row = {"split_id": sid, "dataset_id": did,
                 "train_start": spec.train.start if spec else None,
                 "train_end": spec.train.stop if spec else None,
                 "validation_start": spec.val.start if spec else None,
                 "validation_end": spec.val.stop if spec else None,
                 "test_start": spec.test.start if spec else None,
                 "test_end": spec.test.stop if spec else None,
                 "scaler_fit_range": sfr, "window_length": cfg.lookback,
                 "prediction_horizon": cfg.horizon,
                 "split_hash": (registry_ids.split_hash(spec.train, spec.val,
                                                        spec.test) if spec else None)}
    run_row = {"run_id": rid, "dataset_id": did, "split_id": sid,
               "model": model_key, "model_family": model_family,
               "configuration_id": cid, "seed": getattr(cfg, "seed", None),
               "forecast_mode": forecast_mode, "status": status,
               "provenance_source": provenance_source,
               "start_time": start_time, "end_time": end_time,
               "git_commit": git_commit, "environment_id": environment_id,
               "error_message": error_message}
    metric_row = {"run_id": rid,
                  **{k: metrics.get(k) for k in COLUMNS["metrics"] if k != "run_id"}}
    return {"datasets": dataset_row, "splits": split_row,
            "runs": run_row, "metrics": metric_row}
