"""Unit tests for qdepipe.significance (Diebold–Mariano + Model Confidence Set).

Run:  ../venv/bin/python -m pytest tests/test_significance.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from qdepipe.significance import (
    diebold_mariano, dm_matrix, model_confidence_set, errors,
)


# ---------------------------------------------------------------------------
# Diebold–Mariano
# ---------------------------------------------------------------------------
def test_dm_identical_models_is_null():
    """Identical forecasts → loss differential is exactly 0 → DM 0, p 1."""
    rng = np.random.default_rng(0)
    e = rng.normal(size=500)
    dm, p = diebold_mariano(e, e.copy())
    assert dm == 0.0
    assert p == 1.0


def test_dm_doubled_error_is_significant():
    """A model with deliberately doubled errors is significantly worse."""
    rng = np.random.default_rng(1)
    e1 = rng.normal(scale=1.0, size=2000)      # good model
    e2 = 2.0 * rng.normal(scale=1.0, size=2000)  # clearly worse (4x the squared loss)
    dm, p = diebold_mariano(e1, e2)
    # e1 has smaller loss than e2 → d = L(e1)-L(e2) < 0 → negative statistic
    assert dm < 0
    assert p < 0.01


def test_dm_sign_convention():
    """Positive statistic ⇒ first model has the LARGER loss (second is better)."""
    rng = np.random.default_rng(2)
    good = rng.normal(scale=0.5, size=2000)
    bad = rng.normal(scale=2.0, size=2000)
    dm_bad_vs_good, p = diebold_mariano(bad, good)   # arg1 worse → positive
    assert dm_bad_vs_good > 0
    assert p < 0.01


def test_dm_symmetry():
    rng = np.random.default_rng(3)
    a = rng.normal(size=1000)
    b = rng.normal(scale=1.3, size=1000)
    dm_ab, p_ab = diebold_mariano(a, b)
    dm_ba, p_ba = diebold_mariano(b, a)
    assert np.isclose(dm_ab, -dm_ba, atol=1e-9)
    assert np.isclose(p_ab, p_ba, atol=1e-9)


def test_dm_horizon_widens_hac():
    """Larger h uses more HAC lags; the call must run and stay finite."""
    rng = np.random.default_rng(4)
    e1 = rng.normal(size=1500)
    e2 = e1 + rng.normal(scale=0.3, size=1500)
    for h in (1, 3, 5):
        dm, p = diebold_mariano(e1, e2, h=h)
        assert np.isfinite(dm) and 0.0 <= p <= 1.0


def test_dm_matrix_shape_and_diagonal():
    rng = np.random.default_rng(5)
    T = 800
    yt = rng.normal(size=T)
    fc = {
        "good": (yt, yt + rng.normal(scale=0.2, size=T)),
        "mid": (yt, yt + rng.normal(scale=0.6, size=T)),
        "bad": (yt, yt + rng.normal(scale=1.5, size=T)),
    }
    stat, pval, names = dm_matrix(fc)
    assert stat.shape == (3, 3) and pval.shape == (3, 3)
    assert np.allclose(np.diag(stat), 0.0)
    assert np.allclose(np.diag(pval), 1.0)


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------
def _losses(scales, T=1500, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([(rng.normal(scale=s, size=T)) ** 2 for s in scales], axis=0)


def test_mcs_dominant_in_terrible_out():
    """A clearly-dominant model is always in the set; a terrible one is excluded at 0.10."""
    losses = _losses([0.3, 0.9, 4.0], seed=10)     # best, mid, terrible
    res = model_confidence_set(losses, names=["best", "mid", "terrible"],
                               alpha=(0.10, 0.25), n_boot=500, block_length=20, seed=1)
    assert "best" in res["in_set"][0.10]
    assert "terrible" not in res["in_set"][0.10]
    assert res["mcs_pvalue"]["best"] >= res["mcs_pvalue"]["terrible"]


def test_mcs_survivor_pvalue_is_one():
    losses = _losses([0.3, 1.0, 2.0], seed=11)
    res = model_confidence_set(losses, names=["a", "b", "c"], n_boot=400, seed=2)
    assert max(res["mcs_pvalue"].values()) == 1.0


def test_mcs_equal_models_all_retained():
    """If all models share the same loss process, none should be confidently excluded."""
    rng = np.random.default_rng(12)
    base = rng.normal(size=1200) ** 2
    losses = np.stack([base, base, base], axis=0)   # identical
    res = model_confidence_set(losses, names=["x", "y", "z"], n_boot=300, seed=3)
    # identical losses → no model is significantly worse → all in the 0.10 set
    assert set(res["in_set"][0.10]) == {"x", "y", "z"}


def test_errors_helper():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 1.0, 3.0])
    assert np.allclose(errors(yt, yp), [0.5, -1.0, 0.0])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
