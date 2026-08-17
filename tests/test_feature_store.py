"""Feature-store tests (standalone — no hot path wired yet).

Gate (a) round-trip exactness is asserted with np.array_equal on float64.
Store-level transparency (miss == direct featurize, hit == miss) is the precursor
to the part-2 gate (b) whole-hot-path test.
"""
from __future__ import annotations

import numpy as np
import pytest

from qdepipe.feature_store import FeatureStore
from qdepipe.experiment import ExperimentConfig
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig


@pytest.fixture
def store(tmp_path):
    return FeatureStore(root=tmp_path, namespace="test")


# --------------------------------------------------------------------------- #
# Gate (a): persist -> load is BIT-IDENTICAL (array_equal, not allclose)        #
# --------------------------------------------------------------------------- #
def test_roundtrip_bit_identical_float64(store):
    adversarial = np.array([5e-324, 1.7976931348623157e308, -0.0, np.pi, 1 / 3,
                            np.nextafter(1.0, 2.0), -2.2250738585072014e-308], dtype=np.float64)
    M = np.outer(adversarial, adversarial).copy()      # 7x7, tricky bits
    store.put("k", M)
    L = store.get("k")
    assert L.dtype == np.float64 == M.dtype
    assert L.shape == M.shape
    assert np.array_equal(L, M)                         # bit-identical, not approximate
    assert L.tobytes() == M.tobytes()                  # byte-identical
    assert np.signbit(L[2, 2]) == np.signbit(M[2, 2])  # -0.0 sign survives


def test_roundtrip_random_matrices(store):
    rng = np.random.default_rng(0)
    for i in range(5):
        M = rng.standard_normal((37, 13)).astype(np.float64)
        store.put(f"k{i}", M)
        assert np.array_equal(store.get(f"k{i}"), M)


# --------------------------------------------------------------------------- #
# Store-level transparency: hit and miss are indistinguishable                  #
# --------------------------------------------------------------------------- #
def test_featurize_miss_equals_direct_and_hit_equals_miss(store):
    u = np.sin(np.linspace(0, 20, 200))
    model = ESN(ESNConfig(seed=0))
    cfg = ExperimentConfig(system="henon", n_points=200)

    direct = np.asarray(model.featurize(u))            # the unwired computation
    miss = store.featurize(model, u, cfg)              # first call -> miss
    hit = store.featurize(model, u, cfg)               # second call -> hit (from disk)

    assert np.array_equal(miss, direct)                # miss == direct featurize
    assert np.array_equal(hit, miss)                   # hit == miss, exactly
    assert store.stats["hits"] == 1 and store.stats["misses"] == 1


def test_compute_runs_exactly_once(store):
    u = np.sin(np.linspace(0, 10, 120))
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.outer(u, u[:5])

    a = store.get_or_compute("once", compute)
    b = store.get_or_compute("once", compute)
    assert calls["n"] == 1                             # computed once, then cached
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Keying: same cfg hits, different cfg never aliases                            #
# --------------------------------------------------------------------------- #
def test_different_config_does_not_alias(store):
    u = np.sin(np.linspace(0, 20, 200))
    model = ESN(ESNConfig(seed=0))
    c_henon = ExperimentConfig(system="henon", n_points=200)
    c_lorenz = ExperimentConfig(system="lorenz", n_points=200)
    store.featurize(model, u, c_henon)
    store.featurize(model, u, c_lorenz)                # different key -> separate entry
    assert store.stats["misses"] == 2 and store.stats["hits"] == 0
    assert store.stats["entries"] == 2


def test_get_missing_returns_none(store):
    assert store.get("nope") is None
    assert store.has("nope") is False


# --------------------------------------------------------------------------- #
# Robustness: atomic writes, corrupt -> miss, manifest, clear                   #
# --------------------------------------------------------------------------- #
def test_no_tmp_left_and_manifest_self_describing(store):
    u = np.sin(np.linspace(0, 5, 80))
    model = NGRC(NGRCConfig())
    cfg = ExperimentConfig(system="henon", n_points=80)
    store.featurize(model, u, cfg)
    key = store.key_for(model, cfg)
    assert list(store.dir.glob("*.tmp")) == []         # atomic write left no temp
    man = store.manifest(key)
    assert man["model"]["class"] == "NGRC"
    assert man["shape"] == list(np.asarray(model.featurize(u)).shape)
    assert man["dtype"] == "float64"
    assert man["key"] == key


def test_corrupt_entry_is_treated_as_miss(store):
    store.put("c", np.ones((3, 3)))
    store._npy("c").write_bytes(b"not a real npy file")  # corrupt it
    assert store.get("c") is None                         # miss, never a wrong hit


def test_clear(store):
    store.put("a", np.ones((2, 2)))
    store.put("b", np.zeros((2, 2)))
    assert store.stats["entries"] == 2
    n = store.clear()
    assert n == 4 and store.stats["entries"] == 0       # 2 npy + 2 json removed
