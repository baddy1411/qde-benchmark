"""Data-quality + data-engineering test suite (DATA_QUALITY_TEST_PLAN.md).

Tier 1 (unit) + Tier 2 (contract) + Tier 3 (integration) tests for the new DE
layer and the existing pipeline invariants it depends on. Runs under the existing
pytest config; standalone-runnable via ``python tests/test_data_quality.py``.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pytest

from qdepipe import registry_ids as RID
from qdepipe import registry_io as RIO
from qdepipe import instrument
from qdepipe.sweep import run_sweep
from qdepipe.experiment import ExperimentConfig, run_experiment
from qdepipe.pipeline.split import temporal_split
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import NGRC, NGRCConfig


# ============================ Tier 1 — unit ================================
def test_dataset_id_deterministic_and_sensitive():
    m = RID.dataset_metadata("henon", 1500, seed=0)
    assert RID.dataset_id(**m) == RID.dataset_id(**m)
    assert RID.dataset_id(**RID.dataset_metadata("lorenz", 1500, 0)) != RID.dataset_id(**m)


def test_dataset_hash_content_sensitive():
    x = np.linspace(0, 1, 200)
    assert RID.dataset_hash(x) == RID.dataset_hash(x.copy())
    assert RID.dataset_hash(x) != RID.dataset_hash(x + 1e-9)


def test_split_id_captures_scaler_scope():
    """LEAKAGE-CRITICAL: train vs global scope must yield different split ids."""
    did = "abc"
    a = RID.split_id(did, split_fracs=(0.6, 0.2, 0.2), washout=100,
                     lookback=5, horizon=1, scaler_fit_range="train")
    b = RID.split_id(did, split_fracs=(0.6, 0.2, 0.2), washout=100,
                     lookback=5, horizon=1, scaler_fit_range="global")
    assert a != b


def test_split_id_sensitive_to_washout():
    did = "abc"
    a = RID.split_id(did, split_fracs=(0.6, 0.2, 0.2), washout=100, lookback=5, horizon=1)
    b = RID.split_id(did, split_fracs=(0.6, 0.2, 0.2), washout=50, lookback=5, horizon=1)
    assert a != b


def test_run_id_unique():
    assert RID.run_id() != RID.run_id()


def test_temporal_split_no_overlap():
    """Splits must be strictly ordered and non-overlapping for a grid of configs."""
    for n in (500, 1500, 4000):
        for fr in [(0.6, 0.2, 0.2), (0.8, 0.1, 0.1), (0.5, 0.25, 0.25)]:
            s = temporal_split(n, fr, washout=50)
            assert s.train.start < s.train.stop <= s.val.start < s.val.stop <= s.test.start < s.test.stop


def test_temporal_split_rejects_oversized_washout():
    with pytest.raises(ValueError):
        temporal_split(100, (0.6, 0.2, 0.2), washout=100)


def test_nrmse_infinite_on_zero_std():
    from qdepipe import metrics
    y = np.ones(50)  # zero std
    assert not np.isfinite(metrics.nrmse(y, y + 0.1))


def _run_row(**over) -> dict:
    """A complete, contract-valid ``runs`` row; override any field via kwargs."""
    row = {"run_id": "a", "dataset_id": "d", "split_id": "s", "model": "m",
           "model_family": "reservoir", "configuration_id": "c", "seed": 0,
           "forecast_mode": "one_step", "status": "success",
           "provenance_source": "reconstructed", "start_time": None,
           "end_time": None, "git_commit": None, "environment_id": None,
           "error_message": None}
    row.update(over)
    return row


def test_registry_rejects_unknown_column():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            RIO.append_rows("runs", [{"run_id": "x", "bogus": 1}], root=Path(d))


def test_registry_duplicate_protection():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert RIO.append_rows("runs", [_run_row()], root=d) == 1
        assert RIO.append_rows("runs", [_run_row()], root=d) == 0  # skip
        with pytest.raises(ValueError):
            RIO.append_rows("runs", [_run_row()], root=d, on_duplicate="error")


def test_registry_atomic_no_temp_leak():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        RIO.append_rows("runs", [_run_row(run_id=f"r{i}") for i in range(5)], root=d)
        assert not list(d.glob("*.tmp*")), "atomic write leaked a temp file"


def test_registry_validates_rows_on_write():
    """The write path really validates (registry_io's docstring promise)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for bad in (_run_row(status="completed"),          # enum violation
                    _run_row(seed=-1),                     # range violation
                    _run_row(dataset_id=None),             # declared non-nullable
                    _run_row(git_commit="2ba310769dc4")):  # backdated provenance
            with pytest.raises(ValueError):
                RIO.append_rows("runs", [bad], root=d)
        assert not (d / "runs.parquet").exists(), "an invalid row reached disk"


def test_registry_schemas_cover_every_contract_column():
    """strict=True means the schemas and COLUMNS must not drift apart."""
    from qdepipe.contracts import REGISTRY_SCHEMAS
    for tbl in RIO.TABLES:
        assert set(REGISTRY_SCHEMAS[tbl].columns) == set(RIO.COLUMNS[tbl]), tbl


def test_csv_overwrite_guard_catches_collision():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.csv"
        p.write_text("a\n1\n")
        RIO.record_csv_manifest(p)
        RIO.guard_csv_write(p)                    # same content: ok
        p.write_text("a\n999\n")                  # a different producer clobbered it
        with pytest.raises(FileExistsError):
            RIO.guard_csv_write(p)
        RIO.guard_csv_write(p, allow_overwrite=True)  # deliberate override: ok


# ============================ Tier 2 — contract ===========================
def test_rows_for_run_all_four_tables():
    from qdepipe.data import generate
    cfg = ExperimentConfig(system="henon", n_points=800, seed=0, lookback=3, washout=100)
    x = generate("henon", 800)
    spec = temporal_split(len(x), (0.6, 0.2, 0.2), 100)
    rows = RIO.rows_for_run(cfg, model_key="ngrc", model_family="reservoir",
                            forecast_mode="one_step",
                            metrics={"nrmse": 1e-6}, series=x, spec=spec)
    assert set(rows) == {"datasets", "splits", "runs", "metrics"}
    assert rows["splits"]["scaler_fit_range"] == "train"
    assert rows["datasets"]["dataset_hash"] is not None
    # every column of every table is a known contract column
    for tbl, row in rows.items():
        assert set(row).issubset(set(RIO.COLUMNS[tbl])), tbl


# ============================ Tier 3 — integration ========================
def test_instrumentation_is_noop_on_result():
    """Wrapping stages in the profiler must not change the computed NRMSE."""
    cfg = ExperimentConfig(system="henon", n_points=800, seed=0, lookback=3, washout=100)
    plain = run_experiment(
        ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC"), cfg).nrmse
    with instrument.StageProfiler() as p:
        with p.stage("feature_generation"):
            profiled = run_experiment(
                ReservoirForecaster(NGRC(NGRCConfig(k=3, degree=2)), "NG-RC"), cfg).nrmse
    assert plain == profiled, "instrumentation changed the result"
    s = p.summary()
    assert s["total_seconds"] > 0


def test_sweep_resume_and_failure_isolation():
    appended = []

    def run_one(c):
        if c == 2:
            raise RuntimeError("boom")
        return {"c": c}

    res = run_sweep([0, 1, 2, 3], run_one, done=[0], append_fn=appended.append)
    assert res.skipped == 1 and res.executed == 2 and res.failed == 1
    assert [a["c"] for a in appended] == [1, 3]  # 2 failed, sweep continued


def test_backfilled_registry_leakage_audit():
    """Every backfilled run used train-only scaling (the headline safety claim)."""
    from qdepipe import backfill, query
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        w = backfill.backfill_scoreboard(root=d)
        assert w["runs"] == 360, w
        audit = query.leakage_safety_check(root=d)
        assert set(audit["scaler_fit_range"]) == {"train"}, "a non-train scope leaked in"


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
