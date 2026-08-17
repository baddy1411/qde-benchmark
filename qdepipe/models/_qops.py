"""Dense quantum operators (PyTorch, complex128) for small-n reservoir circuits.

Statevector and density-matrix building blocks for n ≲ 12 qubits, runnable on
GPU or CPU. Deliberately simple and inspectable — the priority is a faithful,
checkable simulation, not a maximally optimised one. These back the gate-based
QRC reservoir (`gate_qrc.py`).
"""
from __future__ import annotations

import functools
import math
import os

import torch

CDTYPE = torch.complex128


def paulis(device):
    I = torch.tensor([[1, 0], [0, 1]], dtype=CDTYPE, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE, device=device)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def kron_list(mats):
    return functools.reduce(torch.kron, mats)


def op_on(op, q, n, device):
    """Embed a single-qubit operator on qubit q into the n-qubit Hilbert space."""
    P = paulis(device)
    mats = [P["I"]] * n
    mats[q] = op
    return kron_list(mats)


def rx(angle, device):
    return torch.matrix_exp(-0.5j * angle * paulis(device)["X"])


def ry(angle, device):
    return torch.matrix_exp(-0.5j * angle * paulis(device)["Y"])


def rz(angle, device):
    return torch.matrix_exp(-0.5j * angle * paulis(device)["Z"])


def cnot_pair(control, target, n, device):
    """CNOT between arbitrary (control, target) — supports the ring wrap CNOT(n-1,0)."""
    P = paulis(device)
    P0 = torch.tensor([[1, 0], [0, 0]], dtype=CDTYPE, device=device)
    P1 = torch.tensor([[0, 0], [0, 1]], dtype=CDTYPE, device=device)
    return (op_on(P0, control, n, device)
            + op_on(P1, control, n, device) @ op_on(P["X"], target, n, device))


def isingxx_pair(theta, i, j, n, device):
    """exp(-i theta/2 X_i X_j) via (X_iX_j)^2 = I (exact, no matrix_exp)."""
    P = paulis(device)
    XiXj = op_on(P["X"], i, n, device) @ op_on(P["X"], j, n, device)
    eye = torch.eye(2 ** n, dtype=CDTYPE, device=device)
    return math.cos(theta / 2) * eye - 1j * math.sin(theta / 2) * XiXj


def zz_pair(i, j, n, device):
    """Z_i Z_j observable for the ring-ZZ readout."""
    P = paulis(device)
    return op_on(P["Z"], i, n, device) @ op_on(P["Z"], j, n, device)


def zero_state(n, device):
    psi = torch.zeros(2 ** n, dtype=CDTYPE, device=device)
    psi[0] = 1.0
    return psi


def zero_density(n, device):
    rho = torch.zeros((2 ** n, 2 ** n), dtype=CDTYPE, device=device)
    rho[0, 0] = 1.0
    return rho


def _expectation_psi_native(psi, op):
    """BLAS complex-dot path. Correct on most builds, and the one that produced
    every committed artifact -- so it is preferred wherever it is provably right,
    which keeps regeneration bit-identical in the evaluated environment."""
    return torch.real(torch.vdot(psi, op @ psi)).item()


def _expectation_psi_safe(psi, op):
    """Explicit conj-multiply-reduce. Slower to write, correct everywhere.

    Used where the native path is broken: torch's complex 1-D dot (vdot / dot /
    inner / 1-D `@`) silently returns 0 for complex64 and complex128 in the torch
    2.8.0+cpu aarch64-Linux wheel, a complex-BLAS return-value ABI bug. It does
    not raise -- every pure-state expectation just becomes 0, the feature matrix
    goes to all-zeros, and ridge degenerates to the intercept, reporting NRMSE
    ~1.0 (the mean predictor) for every row. Silent, and it looks like a result.

    Do not "simplify" this back to a dot call; that is the whole point of it.
    """
    return torch.real((psi.conj() * (op @ psi)).sum()).item()


# Reference computed in plain Python complex arithmetic, so the check cannot be
# satisfied by the same primitive it is testing.
_PROBE_A = (1 + 2j, 3 - 1j)
_PROBE_B = (0 - 1j, 2 + 0j)
_PROBE_WANT = sum(complex(x).conjugate() * complex(y)
                  for x, y in zip(_PROBE_A, _PROBE_B))


def native_vdot_is_correct():
    """Is torch's complex vdot trustworthy in THIS environment?"""
    try:
        a = torch.tensor(_PROBE_A, dtype=CDTYPE)
        b = torch.tensor(_PROBE_B, dtype=CDTYPE)
        return abs(complex(torch.vdot(a, b).item()) - _PROBE_WANT) < 1e-12
    except Exception:
        return False


# Bound below, after selftest_problems() is defined, by _select_expectation_psi().
expectation_psi = _expectation_psi_native


def expectation_rho(rho, op):
    return torch.real(torch.trace(op @ rho)).item()


def amplitude_damping_kraus(gamma, device):
    g = float(gamma)
    K0 = torch.tensor([[1, 0], [0, (1 - g) ** 0.5]], dtype=CDTYPE, device=device)
    K1 = torch.tensor([[0, g ** 0.5], [0, 0]], dtype=CDTYPE, device=device)
    return [K0, K1]


def dephasing_kraus(gamma, device):
    g = float(gamma)
    K0 = torch.tensor([[1, 0], [0, (1 - g) ** 0.5]], dtype=CDTYPE, device=device)
    K1 = torch.tensor([[0, 0], [0, g ** 0.5]], dtype=CDTYPE, device=device)
    return [K0, K1]


# ---------------------------------------------------------------------------
# import-time known-answer guard
# ---------------------------------------------------------------------------
def selftest_problems():
    """Known-answer checks on the primitives this module is built from.

    Motivated by a real incident: torch's complex 1-D dot path (vdot/dot/inner/
    1-D `@`) silently returns 0 in the torch 2.8.0+cpu aarch64-Linux wheel. It
    raises nothing. Every expectation became 0, the feature matrix went to
    all-zeros, and the pipeline reported NRMSE ~1.0 -- the mean predictor -- for
    every row, which reads like a finding rather than a broken build.

    A wrong numerical primitive must stop the run, not flow into a CSV. These
    checks are 2-qubit and cost microseconds. Returns a list of problem strings.
    """
    dev = torch.device("cpu")
    P = paulis(dev)
    bad = []

    H = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE, device=dev) / math.sqrt(2)
    psi = op_on(H, 0, 2, dev) @ zero_state(2, dev)          # |+0>

    # A superposition, so a stuck-at-zero result cannot pass by coincidence the
    # way <0|X|0> = 0 would.
    for label, op, want in (("<+0|X_0|+0>", op_on(P["X"], 0, 2, dev), 1.0),
                            ("<+0|Z_0|+0>", op_on(P["Z"], 0, 2, dev), 0.0),
                            ("<+0|Z_1|+0>", op_on(P["Z"], 1, 2, dev), 1.0)):
        got = expectation_psi(psi, op)
        if not abs(got - want) < 1e-10:
            bad.append(f"expectation_psi {label} = {got!r}, expected {want}")

    # The statevector and density paths share no reduction primitive
    # (elementwise-sum vs trace). Requiring agreement catches a silent failure in
    # either one without having to predict which primitive breaks next.
    rho = psi[:, None] * psi[None, :].conj()
    for label, op in (("X_0", op_on(P["X"], 0, 2, dev)),
                      ("Z_0Z_1", zz_pair(0, 1, 2, dev))):
        a, b = expectation_psi(psi, op), expectation_rho(rho.T, op)
        if not abs(a - b) < 1e-10:
            bad.append(f"psi path ({a!r}) != rho path ({b!r}) for {label}")

    # Unitary evolution must preserve the norm.
    out = isingxx_pair(0.7, 0, 1, 2, dev) @ psi
    nrm = float((out.conj() * out).sum().real)
    if not abs(nrm - 1.0) < 1e-10:
        bad.append(f"isingxx_pair broke normalisation: |psi|^2 = {nrm!r}")

    return bad


def _select_expectation_psi():
    """Pick the fastest expectation path that is provably correct here.

    Native first, deliberately: it is what produced every committed artifact, so
    preferring it keeps regeneration bit-identical in the evaluated environment
    (the two paths sum in different orders and differ at ~1e-9 relative in the
    exact arm, which finite-shot resampling then amplifies to ~5e-3 -- a discrete
    count moves by a whole shot, it has no small perturbation).

    The choice is made by evidence, not by platform string: a cheap known-answer
    probe selects, then the full selftest VALIDATES the selection. If the native
    path passes the probe but fails the selftest, fall back rather than trust it.
    Returns (chosen_name, problems); non-empty problems means nothing works.
    """
    global expectation_psi
    candidates = ([_expectation_psi_native] if native_vdot_is_correct() else [])
    candidates.append(_expectation_psi_safe)
    if os.environ.get("QDE_SKIP_QOPS_SELFTEST") == "1":
        expectation_psi = candidates[0]
        return expectation_psi.__name__, []
    problems = []
    for impl in candidates:
        expectation_psi = impl
        problems = selftest_problems()
        if not problems:
            return impl.__name__, []
    return None, problems


EXPECTATION_PSI_IMPL, _problems = _select_expectation_psi()
if _problems:
    raise RuntimeError(
        "qdepipe._qops: this environment computes quantum primitives "
        "INCORRECTLY by every available code path, so any downstream number "
        "would be invalid:\n  - " + "\n  - ".join(_problems)
        + f"\n(torch {torch.__version__}). Refusing to import. Set "
          "QDE_SKIP_QOPS_SELFTEST=1 only if you know why this is wrong.")
