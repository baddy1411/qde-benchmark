"""Tracking tests — provenance primitives + MLflow logging are observation-only:
they record what ran (git SHA, config hash, run id, params, metrics) and never
change the result. Uses a tmp file-store mlruns/ so nothing global is touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from qdepipe import tracking
from qdepipe.experiment import ExperimentConfig


def test_git_sha_and_dirty_graceful_before_first_commit():
    # repo was `git init`-ed with no commit -> uncommitted + dirty, never raises
    assert tracking.git_sha() in ("uncommitted",) or len(tracking.git_sha()) == 12
    assert isinstance(tracking.git_dirty(), bool)


def test_config_hash_stable_and_sensitive():
    a = ExperimentConfig(system="henon", n_points=1500, seed=0)
    b = ExperimentConfig(system="henon", n_points=1500, seed=0)
    c = ExperimentConfig(system="lorenz", n_points=1500, seed=0)
    assert tracking.config_hash(a) == tracking.config_hash(b)   # stable
    assert tracking.config_hash(a) != tracking.config_hash(c)   # sensitive
    assert len(tracking.config_hash(a)) == 16


def test_provenance_stamp():
    p = tracking.provenance(ExperimentConfig(system="henon"))
    assert set(p) == {"git_sha", "git_dirty", "config_hash"}


@pytest.mark.skipif(not tracking.mlflow_available(), reason="mlflow not installed")
def test_log_run_records_params_metrics_and_provenance_tags(tmp_path):
    import mlflow
    uri = f"file://{tmp_path}/mlruns"
    cfg = ExperimentConfig(system="henon", n_points=300, seed=2, alpha=1e-6)
    metrics = {"nrmse": 0.0123, "rmse": 0.01, "diverged": float("inf")}  # inf must be skipped
    rid = tracking.log_run(cfg, metrics, model="ngrc", tracking_uri=uri, experiment="t")
    assert rid is not None

    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(rid)
    assert run.data.tags["config_hash"] == tracking.config_hash(cfg)
    assert run.data.tags["git_sha"] == tracking.git_sha()
    assert run.data.tags["model"] == "ngrc"
    assert run.data.params["system"] == "henon"
    assert abs(run.data.metrics["nrmse"] - 0.0123) < 1e-9
    assert "diverged" not in run.data.metrics            # non-finite skipped


@pytest.mark.skipif(not tracking.mlflow_available(), reason="mlflow not installed")
def test_cli_track_logs_and_does_not_change_results(tmp_path, monkeypatch):
    """track:true logs one MLflow run per cell AND leaves results byte-identical."""
    import mlflow
    from qdepipe import cli
    base = {"scaler": "minmax", "split_fracs": [0.6, 0.2, 0.2], "washout": 50,
            "lookback": 5, "n_points": 300}
    sweep = {"systems": ["henon"], "seeds": [0, 1], "models": ["ngrc"]}
    uri = f"file://{tmp_path}/mlruns"

    off = cli.task_run({"task": "run", "sweep": sweep, "experiment": base})
    on = cli.task_run({"task": "run", "track": True, "mlflow_uri": uri,
                       "mlflow_experiment": "cli_t", "sweep": sweep, "experiment": base})

    # results identical with tracking on vs off (observation doesn't perturb compute)
    assert np.array_equal(off["nrmse"].to_numpy(), on["nrmse"].to_numpy())

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("cli_t")
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) == 2                                 # one run per matrix cell
