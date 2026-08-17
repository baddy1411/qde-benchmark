"""Contracts test — pins the config-schema↔dataclass sync and the feature-key
include/exclude contract executably, so neither can drift silently."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from qdepipe.contracts import (
    ExperimentConfigModel,
    EXPERIMENT_ROW_SCHEMA,
    FORECAST_SCHEMA,
    feature_key,
    feature_payload,
    validate_config,
)
from qdepipe.experiment import ExperimentConfig
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig


# --------------------------------------------------------------------------- #
# 1. The pydantic mirror must not drift from the dataclass                    #
# --------------------------------------------------------------------------- #
def test_config_schema_matches_dataclass_fields():
    dc = {f.name for f in dataclasses.fields(ExperimentConfig)}
    sc = set(ExperimentConfigModel.model_fields)
    assert dc == sc, f"schema/dataclass drift: only in dataclass={dc - sc}, only in schema={sc - dc}"


def test_defaults_match_dataclass():
    dc = ExperimentConfig()
    sc = ExperimentConfigModel()
    for f in dataclasses.fields(ExperimentConfig):
        dv, sv = getattr(dc, f.name), getattr(sc, f.name)
        assert tuple(dv) == tuple(sv) if isinstance(dv, (tuple, list)) else dv == sv, f.name


def test_valid_and_invalid_configs():
    validate_config(ExperimentConfig())                       # dataclass ok
    validate_config({"system": "lorenz", "n_points": 1500})   # dict ok
    with pytest.raises(Exception):
        validate_config({"system": "not_a_system"})
    with pytest.raises(Exception):
        validate_config({"n_points": -5})
    with pytest.raises(Exception):
        validate_config({"split_fracs": (0.7, 0.2, 0.2)})     # sums to 1.1
    with pytest.raises(Exception):
        validate_config({"bogus_field": 1})                   # extra=forbid


# --------------------------------------------------------------------------- #
# 2. CSV contracts                                                            #
# --------------------------------------------------------------------------- #
def test_forecast_schema():
    good = pd.DataFrame({"t": [0, 1, 2], "y_true": [0.1, 0.2, 0.3], "y_pred": [0.1, 0.25, 0.28]})
    FORECAST_SCHEMA.validate(good)
    # a divergent (non-finite) y_pred is allowed; a non-finite y_true is not
    div = pd.DataFrame({"t": [0, 1], "y_true": [0.1, 0.2], "y_pred": [0.1, float("inf")]})
    FORECAST_SCHEMA.validate(div)
    with pytest.raises(Exception):
        FORECAST_SCHEMA.validate(pd.DataFrame({"t": [0], "y_true": [float("nan")], "y_pred": [0.1]}))


def test_experiment_row_schema():
    good = pd.DataFrame({"model": ["NG-RC"], "n_features": [12], "seed": [0],
                         "nrmse": [1e-7], "rmse": [1e-7], "mae": [1e-7], "mse": [1e-14],
                         "r2": [1.0], "smape": [0.0], "max_error": [1e-6],
                         "vpt": [3.2]})   # extra column allowed (strict=False)
    EXPERIMENT_ROW_SCHEMA.validate(good)
    with pytest.raises(Exception):
        EXPERIMENT_ROW_SCHEMA.validate(pd.DataFrame({"model": ["x"], "n_features": [0], "seed": [0]}))


# --------------------------------------------------------------------------- #
# 3. Feature-key contract — the load-bearing part                             #
# --------------------------------------------------------------------------- #
def _esn(seed=0):
    return ESN(ESNConfig(seed=seed))


def test_feature_key_is_stable_and_deterministic():
    m, c = _esn(), ExperimentConfig()
    assert feature_key(m, c) == feature_key(m, c)
    assert feature_key(_esn(), ExperimentConfig()) == feature_key(_esn(), ExperimentConfig())
    assert len(feature_key(m, c)) == 16


@pytest.mark.parametrize("field,value", [
    ("system", "lorenz"),
    ("n_points", 1500),
    ("scaler", "standard"),
    ("scaler_scope", "global"),
    ("split_fracs", (0.7, 0.15, 0.15)),
])
def test_feature_key_changes_with_feature_affecting_data_fields(field, value):
    base = ExperimentConfig()
    other = ExperimentConfig(**{**base.__dict__, field: value})
    assert feature_key(_esn(), base) != feature_key(_esn(), other), field


@pytest.mark.parametrize("field,value", [
    ("washout", 250),
    ("horizon", 5),
    ("alpha", 1.0),
    ("postprocess", ["zscore"]),
])
def test_feature_key_ignores_readout_only_fields(field, value):
    base = ExperimentConfig()
    other = ExperimentConfig(**{**base.__dict__, field: value})
    assert feature_key(_esn(), base) == feature_key(_esn(), other), field


def test_feature_key_changes_with_model_identity_and_seed():
    c = ExperimentConfig()
    assert feature_key(_esn(0), c) != feature_key(_esn(1), c)          # reservoir seed
    assert feature_key(_esn(0), c) != feature_key(NGRC(NGRCConfig()), c)  # model class


def test_external_window_enters_key_only_when_flagged():
    base = ExperimentConfig()
    wide = ExperimentConfig(**{**base.__dict__, "lookback": 5})
    # internal-window model: lookback must NOT change the key
    assert feature_key(_esn(), base, external_window=False) == \
           feature_key(_esn(), wide, external_window=False)
    # shared-window model: lookback MUST change the key
    assert feature_key(_esn(), base, external_window=True) != \
           feature_key(_esn(), wide, external_window=True)


def test_payload_is_json_serialisable_and_self_describing():
    import json
    p = feature_payload(_esn(), ExperimentConfig())
    json.dumps(p)  # must not raise
    assert p["contract"] == "fkey-v1"
    assert p["model"]["class"] == "ESN"
    assert set(p["data"]) == {"system", "n_points", "scaler", "scaler_scope", "split_fracs"}
