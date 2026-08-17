"""Data contracts, config validation, and the feature-store cache key.

ADDITIVE & OPT-IN. Importing this module has **no** side effect on the pipeline:
it changes nothing about how existing experiments run or what they write. These
are validators you call explicitly, plus the config-hash the quantum feature
store (step 3) will key on. Four things live here:

1. ``ExperimentConfigModel`` — a pydantic *mirror* of
   :class:`qdepipe.experiment.ExperimentConfig`, used to validate a config dict
   (ranges, enums, the split-fraction invariant) before a run. It does **not**
   replace the dataclass; :func:`tests.test_contracts` asserts the two field sets
   stay in sync so they can't drift.

2. ``EXPERIMENT_ROW_SCHEMA`` / ``FORECAST_SCHEMA`` — pandera schemas for the two
   canonical CSV shapes the framework produces (the per-run result row, and the
   persisted ``t,y_true,y_pred`` forecast). They validate framework outputs; they
   are not asserted over every legacy ad-hoc CSV.

3. ``REGISTRY_SCHEMAS`` / :func:`validate_registry_rows` — one pandera schema per
   registry table (Appendix E). Unlike (2) these are **enforced**, not opt-in:
   ``registry_io.append_rows`` validates every batch against them before writing.

4. :func:`feature_key` — the load-bearing cache key. See the long note there for
   exactly which fields are in the key and why.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, List, Literal, Optional, Tuple

import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Valid enum values, mirrored from the code that consumes them (single source of
# truth lives there; these are kept in sync by the drift test).
SCALERS = ("identity", "none", "minmax", "standard", "standardize", "robust")
SYSTEMS = ("henon", "lorenz", "mackeyglass")
SCOPES = ("train", "global")
WINDOWING = ("internal", "shared")
METRIC_COLS = ("nrmse", "rmse", "mae", "mse", "r2", "smape", "max_error")


# ===========================================================================
# 1. Config validation (pydantic mirror of ExperimentConfig) — opt-in
# ===========================================================================
class ExperimentConfigModel(BaseModel):
    """Validation mirror of ``ExperimentConfig``. Field names/defaults must match
    the dataclass exactly (enforced by the drift test)."""

    model_config = ConfigDict(extra="forbid")  # a typo'd field is an error, not silent

    # data
    n_points: int = Field(4000, gt=0)
    seed: int = Field(42, ge=0)
    system: Literal["henon", "lorenz", "mackeyglass", "lorenz96"] = "henon"
    # pre-model data engineering (applies to ALL models)
    scaler: Literal["identity", "none", "minmax", "standard", "standardize", "robust"] = "minmax"
    scaler_scope: Literal["train", "global"] = "train"
    split_fracs: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    washout: int = Field(100, ge=0)
    horizon: int = Field(1, ge=1)
    # windowing
    lookback: int = Field(1, ge=1)
    stride: int = Field(1, ge=1)
    windowing: Literal["internal", "shared"] = "internal"
    # post-model feature engineering
    postprocess: List[Any] = Field(default_factory=list)
    # readout
    alpha: float = Field(1e-6, ge=0)
    # quantum encoding
    encode_scale: float = Field(1.0, gt=0)

    @field_validator("split_fracs")
    @classmethod
    def _split_sums_to_one(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if abs(sum(v) - 1.0) > 1e-9:
            raise ValueError(f"split_fracs must sum to 1.0, got {v} (sum={sum(v)})")
        if any(f <= 0 for f in v):
            raise ValueError(f"split_fracs must be strictly positive, got {v}")
        return v

    # NOTE: scaler_scope="global" is a *valid* (intentionally unsafe) axis the
    # thesis sweeps; leakage-safety is enforced in run_experiment, not here. This
    # schema validates shape, it does not police the leakage invariant.


def validate_config(cfg: Any) -> ExperimentConfigModel:
    """Validate a config (dataclass instance, dict, or kwargs) -> model. Raises
    pydantic.ValidationError on a bad field. Pure check; returns a validated copy."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        cfg = dataclasses.asdict(cfg)
    return ExperimentConfigModel(**cfg)


# ===========================================================================
# 2. CSV contracts (pandera) — opt-in validation of canonical framework outputs
# ===========================================================================
# The per-run result row produced by ExperimentResult.row(). strict=False so the
# climate/extra metric columns some studies add don't fail validation.
EXPERIMENT_ROW_SCHEMA = pa.DataFrameSchema(
    {
        "model": pa.Column(str),
        "n_features": pa.Column(int, pa.Check.gt(0), coerce=True),
        "seed": pa.Column(int, pa.Check.ge(0), coerce=True),
        **{m: pa.Column(float, nullable=True, coerce=True) for m in METRIC_COLS},
    },
    strict=False,
    coerce=True,
    name="experiment_result_row",
)

# The persisted per-seed forecast: t, y_true, y_pred. y_pred deliberately allows
# non-finite values — a blown-up (divergent) forecast is a legitimate persisted
# finding, and the significance/feature machinery handles it downstream.
FORECAST_SCHEMA = pa.DataFrameSchema(
    {
        "t": pa.Column(int, pa.Check.ge(0), coerce=True),
        "y_true": pa.Column(float, pa.Check(lambda s: s.notna().all(), name="y_true_finite"), coerce=True),
        "y_pred": pa.Column(float, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
    name="forecast",
)


def validate_result_rows(df):
    """Validate a result-row DataFrame against EXPERIMENT_ROW_SCHEMA (returns the
    validated frame, raises pandera.errors.SchemaError on violation)."""
    return EXPERIMENT_ROW_SCHEMA.validate(df, lazy=True)


def validate_forecast(df):
    """Validate a t,y_true,y_pred forecast frame against FORECAST_SCHEMA."""
    return FORECAST_SCHEMA.validate(df, lazy=True)


# ===========================================================================
# 3. Registry table contracts (pandera) — ENFORCED in registry_io.append_rows
# ===========================================================================
# One DataFrameSchema per registry table, mirroring Appendix E exactly. These
# are what make the "schema-enforced writes" guarantee in registry_io's
# docstring true: append_rows validates every batch against the schema for its
# table before anything is written.
#
# Nullability follows the honesty rule the registry is built on: a value that is
# genuinely unknowable is NULL, never a plausible substitute. A row therefore
# declares HOW its provenance was obtained in `provenance_source`:
#   "captured"      — written by the run itself, at execution time
#   "reconstructed" — rebuilt from committed artifacts afterwards (backfill)
# and the cross-field checks below make that declaration binding: a captured row
# must carry commit and wall-clock times; a reconstructed row must not claim
# them. (environment_id stays nullable even when captured, because envid returns
# None on an unpinned environment — see qdepipe/envid.py.)

PROVENANCE_SOURCES = ("captured", "reconstructed")
MODEL_FAMILIES = ("statistical", "ml", "dl", "reservoir", "quantum")
FORECAST_MODES = ("one_step", "closed_loop")
RUN_STATUSES = ("success", "failed")

# Fields a reconstruction cannot know and must therefore leave null.
_EXECUTION_ONLY = ("git_commit", "start_time", "end_time", "environment_id")


def _reconstructed_rows_claim_nothing(df):
    """reconstructed => git_commit/start_time/end_time/environment_id are null."""
    recon = df["provenance_source"].eq("reconstructed")
    claimed = df[list(_EXECUTION_ONLY)].notna().any(axis=1)
    return ~(recon & claimed)


def _captured_rows_carry_provenance(df):
    """captured => git_commit and both wall-clock times are present."""
    cap = df["provenance_source"].eq("captured")
    missing = df[["git_commit", "start_time", "end_time"]].isna().any(axis=1)
    return ~(cap & missing)


REGISTRY_SCHEMAS = {
    "datasets": pa.DataFrameSchema(
        {
            "dataset_id":         pa.Column(str, nullable=False, unique=True),
            "system":             pa.Column(str, nullable=False),
            "system_parameters":  pa.Column(str, nullable=False),
            # null when the generator's default initial condition was used
            "initial_condition":  pa.Column(str, nullable=True),
            "random_seed":        pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
            "integration_method": pa.Column(str, nullable=False),
            "time_step":          pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True),
            "sample_count":       pa.Column("int64", pa.Check.gt(0), nullable=False, coerce=True),
            "transient_removed":  pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
            # created_at / code_version: execution-time facts a reconstruction
            # cannot recover (see the run table's provenance_source)
            "created_at":         pa.Column(nullable=True),
            "code_version":       pa.Column(str, nullable=True),
            "dataset_hash":       pa.Column(str, nullable=False),
        },
        strict=True, name="registry_datasets",
    ),
    "splits": pa.DataFrameSchema(
        {
            "split_id":           pa.Column(str, nullable=False, unique=True),
            "dataset_id":         pa.Column(str, nullable=False),
            "train_start":        pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
            "train_end":          pa.Column("int64", pa.Check.gt(0), nullable=False, coerce=True),
            "validation_start":   pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
            "validation_end":     pa.Column("int64", pa.Check.gt(0), nullable=False, coerce=True),
            "test_start":         pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
            "test_end":           pa.Column("int64", pa.Check.gt(0), nullable=False, coerce=True),
            "scaler_fit_range":   pa.Column(str, pa.Check.isin(SCOPES), nullable=False),
            "window_length":      pa.Column("int64", pa.Check.ge(1), nullable=False, coerce=True),
            "prediction_horizon": pa.Column("int64", pa.Check.ge(1), nullable=False, coerce=True),
            "split_hash":         pa.Column(str, nullable=False),
        },
        strict=True, name="registry_splits",
    ),
    "runs": pa.DataFrameSchema(
        {
            "run_id":            pa.Column(str, nullable=False, unique=True),
            "dataset_id":        pa.Column(str, nullable=False),
            "split_id":          pa.Column(str, nullable=False),
            "model":             pa.Column(str, nullable=False),
            "model_family":      pa.Column(str, pa.Check.isin(MODEL_FAMILIES), nullable=False),
            "configuration_id":  pa.Column(str, nullable=False),
            "seed":              pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
            "forecast_mode":     pa.Column(str, pa.Check.isin(FORECAST_MODES), nullable=False),
            "status":            pa.Column(str, pa.Check.isin(RUN_STATUSES), nullable=False),
            "provenance_source": pa.Column(str, pa.Check.isin(PROVENANCE_SOURCES), nullable=False),
            # dtype deliberately unconstrained: object/None while reconstructed,
            # datetime once an instrumented run writes them.
            "start_time":        pa.Column(nullable=True),
            "end_time":          pa.Column(nullable=True),
            "git_commit":        pa.Column(str, nullable=True),
            "environment_id":    pa.Column(str, nullable=True),
            "error_message":     pa.Column(str, nullable=True),
        },
        checks=[
            pa.Check(_reconstructed_rows_claim_nothing,
                     name="reconstructed_rows_claim_no_execution_provenance",
                     error="a reconstructed row must leave git_commit, start_time, "
                           "end_time and environment_id null"),
            pa.Check(_captured_rows_carry_provenance,
                     name="captured_rows_carry_execution_provenance",
                     error="a captured row must record git_commit, start_time and end_time"),
        ],
        strict=True, name="registry_runs",
    ),
    "metrics": pa.DataFrameSchema(
        {
            "run_id":                pa.Column(str, nullable=False, unique=True),
            "nrmse":                 pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
            "mae":                   pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
            "valid_prediction_time": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
            "divergence":            pa.Column(nullable=True),
            "feature_count":         pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True),
            "feature_rank":          pa.Column(float, pa.Check.ge(1), nullable=True, coerce=True),
            **{c: pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True)
               for c in ("data_generation_seconds", "validation_seconds",
                         "transformation_seconds", "feature_generation_seconds",
                         "training_seconds", "inference_seconds", "total_seconds",
                         "peak_memory_mb")},
        },
        strict=True, name="registry_metrics",
    ),
}


def validate_registry_rows(table: str, df):
    """Validate a batch of registry rows for ``table``. Raises ValueError on
    violation (wrapping pandera's SchemaError/SchemaErrors, which are NOT
    ValueErrors) so every registry write failure surfaces as one exception type."""
    import pandera.errors as _pae
    try:
        return REGISTRY_SCHEMAS[table].validate(df, lazy=True)
    except (_pae.SchemaError, _pae.SchemaErrors) as e:
        raise ValueError(f"schema validation failed for registry table "
                         f"{table!r}: {e}") from e


# ===========================================================================
# 4. Feature-store cache key — THE load-bearing contract (wired in step 3)
# ===========================================================================
# Bump when the *meaning* of `reservoir.featurize` changes in a way that
# invalidates cached matrices (e.g. a change to the circuit, encoding, or virtual
# -node layout). Old caches with a different version string are simply never hit.
FEATURE_CONTRACT_VERSION = "fkey-v1"

# The cache boundary is the RAW reservoir feature matrix `reservoir.featurize(inp)`
# — i.e. *before* any post-processing, washout slicing, target shifting, or ridge.
# The key therefore captures exactly what determines that matrix, and nothing else.
#
#   IN THE KEY (changes the feature matrix):
#     data:  system, n_points, scaler, scaler_scope, split_fracs
#            -> the scaled series `u` the reservoir sees depends on ALL of these,
#               because the scaler is fit on the TRAIN slice (so split_fracs and
#               scaler_scope change `u`). This is the subtle, load-bearing part:
#               the plan's short key (system, encoding, qubits, V, seed) is
#               INSUFFICIENT — omitting split_fracs/scaler would alias two runs
#               with different scaled inputs onto the same cache entry.
#     model: class name + the model's full config (asdict) -> captures n_qubits,
#            V, window, readout, coupling, J_strength, channel, gamma, encoding,
#            encode_scale, shots, AND the reservoir seed, for every model family,
#            without enumerating per-model fields.
#     input_window: windowing/lookback/stride, but ONLY when external_window=True
#            (the reservoir consumes shared delay vectors). Internal-window models
#            (QRC's own window, NG-RC/ELM taps) carry that inside the model cfg.
#
#   NOT IN THE KEY (do not change the raw feature matrix; applied after retrieval):
#     washout   -> selects training rows; featurize computes all rows regardless
#     horizon   -> shifts the target, not the features
#     alpha     -> ridge readout only
#     postprocess -> applied on top of the cached matrix (fit on the train slice),
#                    so caching raw features lets one featurize serve many
#                    post-processing variants.
#
# Known minor over-specificity: a deterministic model (NG-RC) still carries its
# unused `seed` in its cfg, so identical features get one cache entry per seed.
# Correct, never wrong; the registry's `deterministic` flag can drop seed in
# step 3 if we want the extra dedup.

FEATURE_KEY_DATA_FIELDS = ("system", "n_points", "scaler", "scaler_scope", "split_fracs")


def _canonical(obj: Any) -> Any:
    """Recursively normalise to a JSON-stable form: sort dict keys, tuples->lists,
    floats->fixed repr (kills 1.0 vs 1.0000000001 jitter), dataclass->dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return format(obj, ".12g")
    return obj


def model_fingerprint(model: Any) -> dict:
    """Stable identity of a reservoir model = class name + canonical config."""
    cfg = getattr(model, "cfg", None)
    return {
        "class": type(model).__name__,
        "name": getattr(model, "name", None),
        "cfg": _canonical(cfg) if cfg is not None else None,
    }


def feature_payload(model: Any, cfg: Any, external_window: bool = False) -> dict:
    """The full, human-readable dict that the cache key hashes (step 3 persists
    this next to the .npy as the manifest, so a cache entry is self-describing)."""
    data = {k: getattr(cfg, k) for k in FEATURE_KEY_DATA_FIELDS}
    win = (
        {"windowing": cfg.windowing, "lookback": cfg.lookback, "stride": cfg.stride}
        if external_window
        else None
    )
    return {
        "contract": FEATURE_CONTRACT_VERSION,
        "model": model_fingerprint(model),
        "data": _canonical(data),
        "input_window": _canonical(win),
    }


def feature_key(model: Any, cfg: Any, external_window: bool = False, length: int = 16) -> str:
    """Stable short hex digest identifying the raw feature matrix for (model, cfg).

    Same (model cfg, data context, windowing) -> same key, across processes and
    runs. `length` is the hex length (default 16 chars = 64-bit; collision-safe
    for any realistic sweep)."""
    blob = json.dumps(feature_payload(model, cfg, external_window), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=max(1, length // 2)).hexdigest()
