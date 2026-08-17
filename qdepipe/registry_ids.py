"""Canonical identifiers and checksums for datasets, splits, and runs.

Implements Contracts A/B/C of ``DATA_CONTRACTS.md`` (the ``*_id`` and ``*_hash``
fields). ADDITIVE: this module is import-only; it changes nothing about how any
existing experiment runs. It reuses the exact same hashing machinery the feature
store already relies on (``contracts._canonical`` + blake2b), so an identity
computed here is computed the same way as a feature-cache key.

Two kinds of identity, deliberately kept distinct (the same distinction the
feature store documents for features):

* ``*_id``   — a CONFIG fingerprint. "Do these two things claim the same recipe?"
               Computed from parameters alone, so it is knowable before any data
               exists and is reproducible across machines.
* ``*_hash`` — a CONTENT checksum of the actual bytes produced. "Did the recipe
               really produce identical data?" Catches a code bug that changes
               the output while leaving the config untouched. Requires the data
               in hand (or a deterministic regeneration), so it is NOT computable
               from a results CSV alone — see ``dataset_hash`` note below.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import contracts


# ---------------------------------------------------------------------------
# low-level: one hashing path for everything (mirrors tracking.config_hash)
# ---------------------------------------------------------------------------
def _hash_obj(obj: Any, length: int = 16) -> str:
    """blake2b over a canonicalised JSON blob. `length` = hex chars (default 16
    = 64-bit). Identical normaliser to the feature-store key, so identities are
    computed the same way everywhere."""
    blob = json.dumps(contracts._canonical(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"),
                           digest_size=max(1, length // 2)).hexdigest()


def _hash_bytes(b: bytes, length: int = 16) -> str:
    return hashlib.blake2b(b, digest_size=max(1, length // 2)).hexdigest()


# ---------------------------------------------------------------------------
# Contract A — dataset identity
# ---------------------------------------------------------------------------
DATASET_ID_FIELDS = ("system", "system_parameters", "initial_condition",
                     "random_seed", "integration_method", "time_step",
                     "sample_count", "transient_removed")


def dataset_id(system: str, *, system_parameters: Optional[Mapping] = None,
               initial_condition: Optional[Mapping] = None,
               random_seed: Optional[int] = None,
               integration_method: Optional[str] = None,
               time_step: Optional[float] = None, sample_count: Optional[int] = None,
               transient_removed: Optional[int] = None, length: int = 16) -> str:
    """Config fingerprint of a raw generated series — independent of any model.

    Only ``system`` is required; the rest default to ``None`` so a caller with
    partial metadata (e.g. reconstructing from an old results CSV that recorded
    only ``system`` + ``n_points``) still gets a stable id over whatever it knows.
    Two datasets with the same recorded fields get the same id; add a field and
    the id changes — deliberately, since a differently-parameterised series is a
    different dataset.
    """
    payload = {
        "contract": "dataset/v1",
        "system": system,
        "system_parameters": dict(system_parameters) if system_parameters else None,
        "initial_condition": dict(initial_condition) if initial_condition else None,
        "random_seed": random_seed,
        "integration_method": integration_method,
        "time_step": time_step,
        "sample_count": sample_count,
        "transient_removed": transient_removed,
    }
    return _hash_obj(payload, length)


def dataset_hash(series: np.ndarray, length: int = 16) -> str:
    """CONTENT checksum of the actual generated array (C-contiguous float64
    bytes). Proves two runs produced bit-identical data, not merely that they
    claimed the same recipe.

    NOTE (honesty, per DATA_CONTRACTS.md): this is NOT computable from a results
    CSV — the raw array is never persisted (it is regenerated in-memory each run).
    To backfill this field for an existing result, the (cheap, deterministic)
    generator must be re-run; the backfill utility does exactly that.
    """
    a = np.ascontiguousarray(np.asarray(series, dtype=np.float64))
    return _hash_bytes(a.tobytes(), length)


# ---------------------------------------------------------------------------
# Contract B — split identity
# ---------------------------------------------------------------------------
def split_id(dataset_id_: str, *, split_fracs: Sequence[float], washout: int,
             lookback: int, horizon: int, scaler_fit_range: str = "train",
             length: int = 16) -> str:
    """Config fingerprint of a temporal split, tied to its dataset.

    ``scaler_fit_range`` is included because it is the single field that
    distinguishes a leakage-safe split ("train") from the deliberately-unsafe
    ablation mode ("global") — two runs identical except for that flag are
    genuinely different splits and must not collide.
    """
    payload = {
        "contract": "split/v1",
        "dataset_id": dataset_id_,
        "split_fracs": list(split_fracs),
        "washout": int(washout),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "scaler_fit_range": scaler_fit_range,
    }
    return _hash_obj(payload, length)


def split_hash(train: slice, val: slice, test: slice, length: int = 16) -> str:
    """CONTENT checksum of the actual index boundaries a split resolved to."""
    payload = [[train.start, train.stop], [val.start, val.stop],
               [test.start, test.stop]]
    return _hash_bytes(json.dumps(payload).encode("utf-8"), length)


def split_id_from_spec(dataset_id_: str, spec, *, lookback: int, horizon: int,
                       split_fracs: Sequence[float], scaler_fit_range: str = "train",
                       length: int = 16) -> str:
    """Convenience: split_id from a ``qdepipe.pipeline.split.SplitSpec``."""
    return split_id(dataset_id_, split_fracs=split_fracs, washout=spec.washout,
                    lookback=lookback, horizon=horizon,
                    scaler_fit_range=scaler_fit_range, length=length)


# ---------------------------------------------------------------------------
# Contract C — run identity
# ---------------------------------------------------------------------------
def run_id() -> str:
    """A unique identifier for one experiment EXECUTION (an event, not a
    fingerprint) — uuid4, so two runs of the identical config are distinct rows
    (they have different start times, environments, and possibly outcomes).
    Uniqueness of the config is captured separately by ``configuration_id``."""
    return uuid.uuid4().hex


def configuration_id(cfg, length: int = 16) -> str:
    """Fingerprint of a full ExperimentConfig. Thin alias over the existing
    ``tracking.config_hash`` so "config identity" has exactly one definition in
    the codebase.

    Caveat (honesty): reconstructing this from an old results CSV is only as
    complete as the columns that CSV recorded — a CSV that omitted, e.g.,
    ``encode_scale`` or ``postprocess`` yields a config_id over a partial config.
    Rows produced going forward (through the instrumented runner) carry the full
    config and are exact.
    """
    from .tracking import config_hash
    return config_hash(cfg, length=length)


# ---------------------------------------------------------------------------
# helper: extract dataset metadata from a system name (for backfill / new runs)
# ---------------------------------------------------------------------------
_SYSTEM_META = {
    # system -> (params, integration_method, time_step, transient_removed)
    "henon":       ({"a": 1.4, "b": 0.3},               "map_iteration", 1.0,   500),
    "lorenz":      ({"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}, "rk4",   0.05,  5000),
    "mackeyglass": ({"beta": 0.2, "gamma": 0.1, "n": 10, "tau": 17}, "euler", 1.0, 1000),
    "lorenz96":    ({"D": 20, "F": 8.0},                "rk4",           0.05,  5000),
}


def dataset_metadata(system: str, n_points: int,
                     seed: Optional[int] = None) -> dict:
    """Assemble the Contract-A field dict for a known system. Values come from
    the generator defaults documented in ``qdepipe/data/*.py``. Effective time
    step for the subsampled continuous systems is ``dt*stride`` (0.01*5 = 0.05)."""
    params, method, dt, transient = _SYSTEM_META.get(
        system, (None, None, None, None))
    return {
        "system": system,
        "system_parameters": params,
        "initial_condition": None,   # default IC; the IC study passes its own
        "random_seed": seed,
        "integration_method": method,
        "time_step": dt,
        "sample_count": int(n_points),
        "transient_removed": transient,
    }
