"""Retroactively populate the Parquet registry from existing result CSVs.

READ-ONLY on every existing CSV — it only ever writes to ``results_registry/``.
This is how the scientific results already produced (Phase 7 of the DE plan) gain
dataset/split/run/config lineage without re-running a single experiment: the IDs
are computed from config fields already recorded in the CSVs, and the (cheap,
deterministic) generator is re-run only to compute each unique dataset's content
hash.

Primary source: ``results/model_scoreboard_all_runs.csv`` — 360 runs carrying the
full ``cfg_*`` config, all seven metrics, feature counts and rank, and a status
column. Additional structured CSVs can be registered via ``SOURCES`` as needed.

Every row written here is marked ``provenance_source="reconstructed"``. Fields a
reconstruction genuinely cannot know are left null rather than filled with a
plausible substitute: timing/memory (those runs predate the instrumentation),
``environment_id`` (the environment running the backfill is not the environment
that ran the experiment), and ``git_commit`` / ``code_version`` (the source CSV
predates the repository being version-controlled, so no commit describes the
code that produced it). Only the execution-time writer,
``registry_io.rows_for_run``, may set those fields; its rows are
``provenance_source="captured"``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from . import registry_ids, registry_io
from .experiment import ExperimentConfig
from .pipeline.split import temporal_split
from .data import generate

ROOT = Path(__file__).resolve().parent.parent
SCOREBOARD = ROOT / "results" / "model_scoreboard_all_runs.csv"

# group -> Contract-C model_family (already present as the scoreboard 'group' col)
_FAMILY = {"reservoir": "reservoir", "quantum": "quantum",
           "statistical": "statistical", "ml": "ml", "dl": "dl"}


def _dataset_hash_cache():
    """Regenerate each unique (system, n_points) series once for its content hash."""
    cache: dict = {}

    def get(system: str, n_points: int) -> Optional[str]:
        key = (system, int(n_points))
        if key not in cache:
            try:
                x = generate(system, int(n_points))
                cache[key] = registry_ids.dataset_hash(x)
            except Exception:
                cache[key] = None
        return cache[key]

    return get


def backfill_scoreboard(root: Optional[Path] = None, source: Path = SCOREBOARD) -> dict:
    """Populate datasets/splits/runs/metrics from the model scoreboard.

    Returns a dict of rows-written per table. Idempotent for datasets/splits
    (content-derived IDs); NOT for runs/metrics, whose ``run_id`` is a fresh
    uuid4 per call — delete ``results_registry/`` before re-running.
    """
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    dhash = _dataset_hash_cache()

    ds_rows, sp_rows, run_rows, met_rows = {}, {}, [], []
    for r in df.itertuples():
        # reconstruct the config actually used (cfg_* columns)
        split_fracs = eval(r.cfg_split_fracs) if isinstance(r.cfg_split_fracs, str) \
            else (0.6, 0.2, 0.2)
        cfg = ExperimentConfig(
            system=r.cfg_system, n_points=int(r.cfg_n_points), seed=int(r.cfg_seed),
            scaler=r.cfg_scaler, scaler_scope=r.cfg_scaler_scope,
            split_fracs=tuple(split_fracs), washout=int(r.cfg_washout),
            horizon=int(r.cfg_horizon), lookback=int(r.cfg_lookback),
            stride=int(r.cfg_stride), windowing=r.cfg_windowing,
            alpha=float(r.cfg_alpha), encode_scale=float(r.cfg_encode_scale))
        spec = temporal_split(cfg.n_points, cfg.split_fracs, cfg.washout)
        dmeta = registry_ids.dataset_metadata(cfg.system, cfg.n_points, seed=cfg.seed)
        did = registry_ids.dataset_id(**dmeta)
        sid = registry_ids.split_id_from_spec(
            did, spec, lookback=cfg.lookback, horizon=cfg.horizon,
            split_fracs=cfg.split_fracs, scaler_fit_range=cfg.scaler_scope)
        rid = registry_ids.run_id()
        cid = registry_ids.configuration_id(cfg)

        # dataset row (dedup by did)
        if did not in ds_rows:
            import json
            ds_rows[did] = {
                "dataset_id": did, "system": cfg.system,
                "system_parameters": json.dumps(dmeta["system_parameters"]),
                "initial_condition": None, "random_seed": cfg.seed,
                "integration_method": dmeta["integration_method"],
                "time_step": dmeta["time_step"], "sample_count": cfg.n_points,
                "transient_removed": dmeta["transient_removed"],
                "created_at": None, "code_version": None,
                "dataset_hash": dhash(cfg.system, cfg.n_points)}
        # split row (dedup by sid)
        if sid not in sp_rows:
            sp_rows[sid] = {
                "split_id": sid, "dataset_id": did,
                "train_start": spec.train.start, "train_end": spec.train.stop,
                "validation_start": spec.val.start, "validation_end": spec.val.stop,
                "test_start": spec.test.start, "test_end": spec.test.stop,
                "scaler_fit_range": cfg.scaler_scope, "window_length": cfg.lookback,
                "prediction_horizon": cfg.horizon,
                "split_hash": registry_ids.split_hash(spec.train, spec.val, spec.test)}
        # run row
        status = "success" if str(r.status) == "ok" else "failed"
        run_rows.append({
            "run_id": rid, "dataset_id": did, "split_id": sid,
            "model": r.model_key, "model_family": _FAMILY.get(r.group, r.group),
            "configuration_id": cid, "seed": cfg.seed, "forecast_mode": "one_step",
            "status": status, "provenance_source": "reconstructed",
            "start_time": None, "end_time": None,
            "git_commit": None, "environment_id": None,
            "error_message": None if status == "success" else str(r.status)})
        # metric row (timing/memory null — not measured for these historical runs)
        def num(v):
            try:
                f = float(v);  return f if np.isfinite(f) else None
            except Exception:
                return None
        met_rows.append({
            "run_id": rid, "nrmse": num(r.nrmse), "mae": num(r.mae),
            "valid_prediction_time": None, "divergence": None,
            "feature_count": num(r.n_features), "feature_rank": num(r.feature_rank),
            "data_generation_seconds": None, "validation_seconds": None,
            "transformation_seconds": None, "feature_generation_seconds": None,
            "training_seconds": None, "inference_seconds": None,
            "total_seconds": None, "peak_memory_mb": None})

    written = {
        "datasets": registry_io.append_rows("datasets", ds_rows.values(), root=root),
        "splits": registry_io.append_rows("splits", sp_rows.values(), root=root),
        "runs": registry_io.append_rows("runs", run_rows, root=root),
        "metrics": registry_io.append_rows("metrics", met_rows, root=root),
    }
    return written


if __name__ == "__main__":
    from .logging_setup import configure, get_logger
    configure(run_name="backfill")
    log = get_logger("qdepipe.backfill")
    w = backfill_scoreboard()
    log.info("backfilled registry from scoreboard: %s", w)
    print("backfill complete:", w)
