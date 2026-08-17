"""CLI tests. The headline: `task: run` produces numbers byte-identical to calling
run_experiment directly (proving the CLI calls the proven function, reimplements
nothing), and the cache flag is transparent (CLI inherits gate (b))."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from qdepipe import cli
from qdepipe.experiment import run_experiment
from qdepipe.registry import build_forecaster


def _write(tmp_path, spec):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


BASE_EXP = {"scaler": "minmax", "scaler_scope": "train", "split_fracs": [0.6, 0.2, 0.2],
            "washout": 50, "lookback": 5, "horizon": 1, "alpha": 1e-6, "n_points": 300}


# --------------------------------------------------------------------------- #
# 1. CLI `run` == direct run_experiment (calls the proven function, no reimpl)  #
# --------------------------------------------------------------------------- #
def test_run_matrix_equals_direct_calls():
    spec = {"task": "run",
            "sweep": {"systems": ["henon"], "seeds": [0, 1], "models": ["ngrc"]},
            "experiment": BASE_EXP}
    df = cli.task_run(spec)

    # reproduce each cell by calling run_experiment directly
    from qdepipe.contracts import ExperimentConfigModel
    from qdepipe.experiment import ExperimentConfig
    for seed in (0, 1):
        m = ExperimentConfigModel(**BASE_EXP, system="henon", seed=seed)
        cfg = ExperimentConfig(**m.model_dump())
        direct = run_experiment(build_forecaster("ngrc", seed, cfg), cfg).nrmse
        cli_val = df[(df.seed == seed)]["nrmse"].iloc[0]
        assert cli_val == direct, f"CLI != direct for seed {seed}"


# --------------------------------------------------------------------------- #
# 2. cache flag is transparent (CLI inherits gate (b))                          #
# --------------------------------------------------------------------------- #
def test_cache_flag_is_transparent(tmp_path, monkeypatch):
    # FeatureStore()'s default root is repo_root/feature_store (see
    # feature_store._default_root), NOT cwd/feature_store -- so chdir does not
    # isolate this test, and before this was fixed it read and wrote the real
    # repository cache. That went unnoticed because a warm host cache holds the
    # same bytes the test would have computed; it surfaced only when the suite
    # ran in a container whose BLAS produces different bytes from the host that
    # populated the cache. Redirect the default root itself.
    import qdepipe.feature_store as fs
    monkeypatch.setattr(fs, "_default_root", lambda: tmp_path / "feature_store")
    monkeypatch.chdir(tmp_path)
    spec_off = {"task": "run", "cache": False,
                "sweep": {"systems": ["henon"], "seeds": [0], "models": ["qrc_v4"]},
                "experiment": BASE_EXP}
    spec_on = {**spec_off, "cache": True}
    off = cli.task_run(spec_off)["nrmse"].iloc[0]
    on = cli.task_run(spec_on)["nrmse"].iloc[0]
    on2 = cli.task_run(spec_on)["nrmse"].iloc[0]   # warm cache
    assert off == on == on2


# --------------------------------------------------------------------------- #
# 3. validation uses the single source of truth (ExperimentConfigModel)         #
# --------------------------------------------------------------------------- #
def test_validate_rejects_bad_experiment_field(tmp_path):
    bad = {"task": "run", "sweep": {"systems": ["henon"], "seeds": [0], "models": ["ngrc"]},
           "experiment": {**BASE_EXP, "bogus_axis": 1}}     # extra='forbid'
    with pytest.raises(Exception):
        cli.main(["validate", _write(tmp_path, bad)])


def test_validate_rejects_bad_split(tmp_path):
    bad = {"task": "run", "sweep": {"models": ["ngrc"]},
           "experiment": {**BASE_EXP, "split_fracs": [0.7, 0.2, 0.2]}}   # sums to 1.1
    with pytest.raises(Exception):
        cli.main(["validate", _write(tmp_path, bad)])


def test_validate_rejects_unknown_model(tmp_path):
    bad = {"task": "run", "sweep": {"models": ["not_a_model"]}, "experiment": BASE_EXP}
    with pytest.raises(SystemExit):
        cli.main(["validate", _write(tmp_path, bad)])


def test_validate_accepts_good_config(tmp_path, capsys):
    good = {"task": "run", "sweep": {"systems": ["henon"], "seeds": [0], "models": ["ngrc"]},
            "experiment": BASE_EXP}
    assert cli.main(["validate", _write(tmp_path, good)]) == 0
    assert "OK" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 4. end-to-end run writes a CSV; unknown task errors                           #
# --------------------------------------------------------------------------- #
def test_run_writes_csv(tmp_path):
    out = tmp_path / "r.csv"
    spec = {"task": "run", "output": str(out),
            "sweep": {"systems": ["henon"], "seeds": [0], "models": ["ngrc", "esn"]},
            "experiment": BASE_EXP}
    assert cli.main(["run", _write(tmp_path, spec)]) == 0
    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 2 and set(df["model_key"]) == {"ngrc", "esn"}


def test_unknown_task_errors(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", _write(tmp_path, {"task": "nope", "experiment": BASE_EXP})])


def test_list_tasks(capsys):
    assert cli.main(["list-tasks"]) == 0
    out = capsys.readouterr().out
    assert "run" in out and "matched_budget" in out and "qrc_v4" in out
