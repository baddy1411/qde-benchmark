"""The Diebold-Mariano zero-guard must be relative to the loss scale.

Regression cover: the guard was `np.allclose(d, 0.0)`, which against a scalar
reduces to |d| <= atol = 1e-8. With squared-error loss that silently returned
(0.0, 1.0) -- "no significant difference" -- for any pair of accurate models,
without running the test. 36 committed rows were affected.
"""
import numpy as np
import pytest

from qdepipe.significance import diebold_mariano, errors


def _pair(scale, rng_seed=0, offset=0.30):
    """Two error series with a real, consistent difference, at a given scale."""
    rng = np.random.default_rng(rng_seed)
    e1 = rng.normal(0, scale, 400)
    e2 = rng.normal(0, scale * (1 + offset), 400)
    return e1, e2


def test_detects_a_real_difference_at_tiny_scale():
    """The exact case that broke: errors ~1e-5 => squared losses ~1e-10, two
    orders of magnitude under the old 1e-8 floor."""
    e1, e2 = _pair(1e-5)
    stat, p = diebold_mariano(e1, e2)
    assert (stat, p) != (0.0, 1.0), "guard fired instead of running the test"
    assert p < 0.05, f"a 30% loss difference should be detected; got p={p}"


def test_verdict_is_invariant_to_rescaling():
    """Same relative difference at wildly different magnitudes => same verdict.
    A scale-dependent guard breaks exactly this property."""
    out = []
    for scale in (1e-1, 1e-3, 1e-5, 1e-7):
        e1, e2 = _pair(scale)
        stat, p = diebold_mariano(e1, e2)
        out.append((stat, p))
    stats = [s for s, _ in out]
    assert all(p < 0.05 for _, p in out), f"scale-dependent verdicts: {out}"
    # DM is a ratio of a mean to its standard error, so rescaling both series
    # leaves the statistic itself essentially unchanged.
    assert max(stats) - min(stats) < 1e-6, f"statistic drifted with scale: {stats}"


def test_identical_forecasts_still_return_the_degenerate_result():
    """The guard must still fire when there genuinely is nothing to test --
    e.g. leaky eps=1, which is the identity, compared against its own base."""
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1e-4, 200)
    assert diebold_mariano(e, e.copy()) == (0.0, 1.0)


def test_all_zero_errors_return_the_degenerate_result():
    """A perfect forecast on both sides has zero loss scale; no test to run."""
    z = np.zeros(200)
    assert diebold_mariano(z, z) == (0.0, 1.0)


@pytest.mark.parametrize("loss", ["se", "ae"])
def test_holds_for_both_losses(loss):
    e1, e2 = _pair(1e-6)
    stat, p = diebold_mariano(e1, e2, loss=loss)
    assert (stat, p) != (0.0, 1.0)
    assert p < 0.05


def test_errors_helper_roundtrip():
    y = np.linspace(0, 1, 50)
    assert np.allclose(errors(y, y), 0.0)
