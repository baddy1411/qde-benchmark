"""CLI stage test — the sweep/run stage is tested as a function through
qdepipe.cli, asserting forecasts persist with the canonical schema. (The
Snakemake DAG tests that previously lived here were removed with the unused
workflow/ directory.)"""
from __future__ import annotations

import os

import pandas as pd

from qdepipe import cli


def test_matrix_stage_persists_forecasts(tmp_path):
    fdir = tmp_path / "fc"
    spec = {"task": "run", "forecasts_dir": str(fdir),
            "sweep": {"systems": ["henon"], "seeds": [0], "models": ["ngrc", "esn"]},
            "experiment": {"scaler": "minmax", "washout": 50, "lookback": 5, "n_points": 300}}
    cli.task_run(spec)
    files = sorted(os.path.basename(p) for p in fdir.glob("*.csv"))
    assert files == ["henon__esn__seed0.csv", "henon__ngrc__seed0.csv"]
    df = pd.read_csv(fdir / "henon__ngrc__seed0.csv")
    assert list(df.columns) == ["t", "y_true", "y_pred"]
