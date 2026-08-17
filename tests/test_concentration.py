"""Unit tests for the concentration variance probe.

The probe's core claim is mechanical: an observable whose expectation barely moves
with the input has near-zero across-input variance; one that tracks the input has
clearly non-zero variance. We test that on synthetic feature matrices and on the real
finite-shot path's determinism.

Run:  ../venv/bin/python -m pytest tests/test_concentration.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from qdepipe.concentration import observable_variance, _slice_readout, _rich_cfg
from qdepipe.models.gate_qrc import GateQRC


def test_variance_separates_concentrating_from_tracking():
    """A synthetic dataset: 1-local cols track the input, 2-local cols are ~constant.
    The probe must report near-zero 2-local variance and clearly non-zero 1-local."""
    rng = np.random.default_rng(0)
    T, n = 500, 4
    inp = rng.uniform(-1, 1, size=T)
    tracking = np.stack([inp * (0.5 + 0.1 * i) for i in range(3 * n)], axis=1)   # 1-local
    constant = 0.999 + 1e-4 * rng.standard_normal((T, n))                        # 2-local, concentrated
    feats = np.concatenate([tracking, constant], axis=1)
    v1 = feats[:, :3 * n].var(axis=0)
    v2 = feats[:, 3 * n:].var(axis=0)
    assert v1.mean() > 0.01            # tracking observables carry signal
    assert v2.mean() < 1e-6            # concentrated observables nearly constant
    assert v1.mean() > 1000 * v2.mean()


def test_probe_on_real_reservoir_runs_and_groups():
    """observable_variance on a tiny real reservoir returns the right shapes and
    non-negative variances grouped by weight class."""
    from qdepipe.concentration import _scaled_series
    u, _ = _scaled_series("henon", 300)
    n = 4
    ov = observable_variance(u[:200], n, seed=0)
    assert ov["var_1local"].shape == (3 * n,)
    assert ov["var_2local"].shape == (n,)
    assert ov["mean_var_1local"] >= 0 and ov["mean_var_2local"] >= 0


def test_slice_readout_picks_right_columns():
    """Full (Z,X,Y,ZZ) layout per virtual node is [Z(n) X(n) Y(n) ZZ(n)]; slicing must
    keep the leading observables for each subset."""
    n, V = 3, 2
    block = 4 * n
    feats = np.arange(V * block).reshape(1, V * block).astype(float)
    z = _slice_readout(feats, n, V, "Z")
    zxy = _slice_readout(feats, n, V, "ZXY")
    full = _slice_readout(feats, n, V, "ZXYZZ")
    assert z.shape[1] == V * n
    assert zxy.shape[1] == V * 3 * n
    assert full.shape[1] == V * 4 * n
    # first block's Z columns are 0..n-1, second block's start at `block`
    assert list(z[0]) == [0, 1, 2, block, block + 1, block + 2]


def test_finite_shot_is_noisy_but_reproducible():
    """Finite-shot featurization differs from exact, and is deterministic given seed."""
    from qdepipe.concentration import _scaled_series
    u, _ = _scaled_series("henon", 300)
    exact = GateQRC(_rich_cfg(4, V=4, seed=0)).featurize(u[:60])
    s1 = GateQRC(_rich_cfg(4, V=4, seed=0, shots=1024)).featurize(u[:60])
    s2 = GateQRC(_rich_cfg(4, V=4, seed=0, shots=1024)).featurize(u[:60])
    assert np.allclose(s1, s2)                      # reproducible
    assert not np.allclose(s1, exact, atol=1e-3)    # genuinely noisy
    # shot estimates live in [-1, 1] like the exact Pauli expectations
    assert s1.min() >= -1.0 - 1e-9 and s1.max() <= 1.0 + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
