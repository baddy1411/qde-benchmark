"""Lorenz-96 — higher-dimensional chaos, the stress test.

    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F        (cyclic, i = 0..D-1)

D=20 sites, forcing F=8 (the standard strongly-chaotic regime). RK4 with
dt=0.01, subsampled by stride=5 (sampling interval 0.05 MTU — the same Δ
convention as our Lorenz-63). Univariate target: x_0, so the entire pipeline
is unchanged.

Why this system: the literature's strongest quantum-reservoir claims (RF-QRC,
correlated-spin experiments) live in HIGHER-dimensional chaos, where classical
reservoirs need many nodes. A negative result here extends the thesis null to
that regime; a gap-closing result locates the regime boundary. Either outcome
is a finding.

LAMBDA_CONT provenance: largest LE per unit time for D=20, F=8 AS INTEGRATED
here, estimated with the Benettin two-trajectory method (experiments/estimate_lyapunov.py
pattern; experiments/run_lorenz96.py --lyapunov reproduces it). Literature for L96 F=8
reports λ₁ ≈ 1.5-1.7 depending on D; the value below is our measured one and
the data-generation gate asserts the estimate stays within ±10%.
"""
from __future__ import annotations

import numpy as np

D_SITES = 20
FORCING = 8.0
# Benettin two-trajectory estimate on THIS integrator (D=20, F=8, RK4 dt=0.01):
# 1.5451 / 1.5444 / 1.5453 across seeds 0/1/2 (n=100k, std 4e-4). Literature for
# L96 F=8 reports lambda_1 ~ 1.5-1.7 depending on D — consistent.
LAMBDA_CONT = 1.5449


def _deriv(x: np.ndarray, F: float) -> np.ndarray:
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def _rk4(x: np.ndarray, dt: float, F: float) -> np.ndarray:
    k1 = _deriv(x, F)
    k2 = _deriv(x + 0.5 * dt * k1, F)
    k3 = _deriv(x + 0.5 * dt * k2, F)
    k4 = _deriv(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_lorenz96(n_points: int, D: int = D_SITES, F: float = FORCING,
                      dt: float = 0.01, stride: int = 5, transient: int = 5000,
                      s0: np.ndarray | None = None):
    """Integrate L96, drop `transient` steps, subsample by `stride`.
    Returns (x0_series, full_state (n_points, D)). Sampling interval dt*stride."""
    x = (np.full(D, F) if s0 is None else np.asarray(s0, dtype=float)).copy()
    if s0 is None:
        x[0] += 0.01                      # canonical deterministic kick off the fixed point
    total = transient + n_points * stride
    traj = np.empty((n_points, D))
    rec = 0
    for k in range(total):
        x = _rk4(x, dt, F)
        if k >= transient and (k - transient) % stride == 0 and rec < n_points:
            traj[rec] = x
            rec += 1
    return traj[:, 0], traj


def benettin_lyapunov(D: int = D_SITES, F: float = FORCING, dt: float = 0.01,
                      n: int = 200_000, transient: int = 20_000,
                      renorm: int = 10, d0: float = 1e-9, seed: int = 0) -> float:
    """Largest LE per unit time via the Benettin two-trajectory method, on the
    system exactly as integrated above (same RK4, same dt)."""
    rng = np.random.default_rng(seed)
    a = np.full(D, F)
    a[0] += 0.01
    for _ in range(transient):
        a = _rk4(a, dt, F)
    delta = rng.normal(size=D)
    b = a + d0 * delta / np.linalg.norm(delta)
    s = 0.0
    steps = 0
    for k in range(n):
        a = _rk4(a, dt, F)
        b = _rk4(b, dt, F)
        if (k + 1) % renorm == 0:
            d = np.linalg.norm(b - a)
            if d > 0:
                s += np.log(d / d0)
                b = a + d0 * (b - a) / d
                steps += renorm
    return s / (steps * dt)


def lyapunov_step(x=None, dt: float = 0.01, stride: int = 5) -> float:
    """Largest LE per *sample* for VPT in Lyapunov times."""
    return LAMBDA_CONT * dt * stride


def lyapunov_gate(tol: float = 0.10, n: int = 50_000) -> tuple[float, bool]:
    """Data-generation sanity gate: a fresh (cheap) Benettin estimate must sit
    within `tol` (relative) of the calibrated LAMBDA_CONT."""
    le = benettin_lyapunov(n=n)
    return le, bool(abs(le - LAMBDA_CONT) / LAMBDA_CONT < tol)
