"""Lorenz-63 — a continuous chaotic system, the second benchmark.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

Standard chaotic parameters sigma=10, rho=28, beta=8/3. We integrate with fixed-step
RK4 and subsample, then forecast the **x-component** as the 1-D target — the same
univariate setup as Hénon, so the pipeline is unchanged.

Why this system: unlike Hénon, the x-series is **not** a clean low-order polynomial of
a couple of its own lags, so it breaks the "Hénon is exactly quadratic" confound that
makes NG-RC look unbeatable. The largest Lyapunov exponent is ~0.9056 per unit time
(Sprott); the per-sample value used for VPT is that times the sampling interval.
"""
from __future__ import annotations

import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
LAMBDA_CONT = 0.9056          # largest LE per unit time (standard reference)


def _deriv(s, sigma, rho, beta):
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def generate_lorenz(n_points: int, sigma: float = SIGMA, rho: float = RHO,
                    beta: float = BETA, dt: float = 0.01, stride: int = 5,
                    transient: int = 5000, s0=(1.0, 1.0, 1.0)):
    """RK4-integrate Lorenz-63, drop `transient` warm-up samples, subsample by
    `stride`. Returns (x_series, full_state[:, 3]). Sampling interval = dt*stride."""
    s = np.array(s0, dtype=float)
    total = transient + n_points * stride
    traj = np.empty((n_points, 3))
    rec = 0
    for k in range(total):
        k1 = _deriv(s, sigma, rho, beta)
        k2 = _deriv(s + 0.5 * dt * k1, sigma, rho, beta)
        k3 = _deriv(s + 0.5 * dt * k2, sigma, rho, beta)
        k4 = _deriv(s + dt * k3, sigma, rho, beta)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if k >= transient and (k - transient) % stride == 0 and rec < n_points:
            traj[rec] = s
            rec += 1
    return traj[:, 0], traj


def lyapunov_step(x=None, dt: float = 0.01, stride: int = 5) -> float:
    """Largest LE per *sample* (= per forecasting step), for VPT in Lyapunov times."""
    return LAMBDA_CONT * dt * stride
