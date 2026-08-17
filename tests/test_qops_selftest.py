"""The import-time numerical guard in qdepipe.models._qops.

Regression cover for a real incident: torch's complex 1-D dot path silently
returns 0 in the torch 2.8.0+cpu aarch64-Linux wheel, which turned every quantum
expectation into 0 and made the pipeline report NRMSE ~1.0 (the mean predictor)
for every row -- a broken build that reads like a scientific result.
"""
import math

import pytest
import torch

from qdepipe.models import _qops as Q


def test_selftest_passes_in_this_environment():
    assert Q.selftest_problems() == []


def test_guard_catches_a_stuck_at_zero_expectation(monkeypatch):
    """The exact failure mode that shipped: expectations silently return 0."""
    monkeypatch.setattr(Q, "expectation_psi", lambda psi, op: 0.0)
    problems = Q.selftest_problems()
    assert problems, "a stuck-at-zero expectation must be caught"
    assert any("X_0" in p for p in problems)


def test_guard_catches_a_broken_density_path(monkeypatch):
    monkeypatch.setattr(Q, "expectation_rho", lambda rho, op: 0.0)
    problems = Q.selftest_problems()
    assert any("rho path" in p for p in problems)


def test_guard_catches_lost_normalisation(monkeypatch):
    monkeypatch.setattr(Q, "isingxx_pair",
                        lambda *a, **k: 0.5 * torch.eye(4, dtype=Q.CDTYPE))
    problems = Q.selftest_problems()
    assert any("normalisation" in p for p in problems)


@pytest.mark.parametrize("n", [2, 3])
def test_expectation_paths_agree_on_a_superposition(n):
    """psi and rho paths share no reduction primitive; they must still agree."""
    dev = torch.device("cpu")
    P = Q.paulis(dev)
    H = torch.tensor([[1, 1], [1, -1]], dtype=Q.CDTYPE, device=dev) / math.sqrt(2)
    psi = Q.op_on(H, 0, n, dev) @ Q.zero_state(n, dev)
    rho = psi[:, None] * psi[None, :].conj()
    op = Q.op_on(P["X"], 0, n, dev)
    assert Q.expectation_psi(psi, op) == pytest.approx(1.0, abs=1e-10)
    assert Q.expectation_rho(rho.T, op) == pytest.approx(1.0, abs=1e-10)


def test_fallback_impl_uses_no_dot_call():
    """Guard the fallback itself: 'simplifying' it back to a dot call would
    silently re-break every environment whose complex BLAS dot is wrong."""
    import inspect
    src = inspect.getsource(Q._expectation_psi_safe)
    code = src.split('"""')[2] if src.count('"""') >= 2 else src   # drop docstring
    for banned in ("torch.vdot", "torch.dot", "torch.inner"):
        assert banned not in code, f"{banned} is unsafe on some torch builds"


def test_both_impls_agree_where_native_works():
    """The paths sum in different orders, so they agree to ~1e-9, not exactly.
    Anything larger means one of them is wrong, not merely reordered."""
    if not Q.native_vdot_is_correct():
        pytest.skip("native complex vdot is broken here; only one path is valid")
    dev = torch.device("cpu")
    P = Q.paulis(dev)
    H = torch.tensor([[1, 1], [1, -1]], dtype=Q.CDTYPE, device=dev) / math.sqrt(2)
    psi = Q.op_on(H, 0, 3, dev) @ Q.zero_state(3, dev)
    for op in (Q.op_on(P["X"], 0, 3, dev), Q.zz_pair(0, 1, 3, dev)):
        assert Q._expectation_psi_native(psi, op) == pytest.approx(
            Q._expectation_psi_safe(psi, op), abs=1e-9)


def test_selection_prefers_native_when_it_is_correct():
    """Committed artifacts were produced by the native path; preferring it where
    it is provably right is what keeps regeneration bit-identical."""
    if not Q.native_vdot_is_correct():
        pytest.skip("native complex vdot is broken here")
    assert Q.EXPECTATION_PSI_IMPL == "_expectation_psi_native"


def test_selection_falls_back_when_native_is_broken(monkeypatch):
    """The real failure mode: vdot returns 0 with no error."""
    monkeypatch.setattr(torch, "vdot",
                        lambda a, b: torch.zeros((), dtype=Q.CDTYPE))
    saved = Q.expectation_psi
    try:
        assert not Q.native_vdot_is_correct()
        chosen, problems = Q._select_expectation_psi()
        assert problems == []
        assert chosen == "_expectation_psi_safe"
    finally:
        Q.expectation_psi = saved
