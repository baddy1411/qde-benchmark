"""Mackey–Glass — a delay-differential chaotic system, the third benchmark.

    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

Field-standard chaotic parameters: beta=0.2, gamma=0.1, n=10, tau=17 (the canonical
MG-17 benchmark). Integrated with fixed step dt=1.0 (the conventional discretization).

Why this system: its dynamics live on a high-dimensional delay manifold, so there is
**no obvious low-degree polynomial of a few lags** that reconstructs it. This is the key
stress test for the thesis's "it's just quadratic features" story — if a hand-designed
polynomial (NG-RC) no longer dominates here while a high-dimensional feature map does,
that is a finding, not a bug. Largest LE ~0.00838 per time unit (numerically
estimated for this integration; see LAMBDA_CONT below).
"""
from __future__ import annotations

import numpy as np

BETA, GAMMA, N, TAU = 0.2, 0.1, 10, 17
# Largest LE per unit time, estimated numerically by the Benettin two-trajectory
# method on the system AS INTEGRATED here (Euler, dt=1); stable to 3 s.f. across
# seeds (experiments/estimate_lyapunov.py). Specific to this integration — slightly below
# coarser literature values for abstract MG-17. (Was a sourceless 0.0086.)
LAMBDA_CONT = 0.00838


def generate_mackeyglass(n_points: int, tau: int = TAU, beta: float = BETA,
                         gamma: float = GAMMA, n: int = N, dt: float = 1.0,
                         transient: int = 1000, x0: float = 1.2):
    """Integrate Mackey–Glass with an explicit delay buffer, drop `transient` warm-up
    samples. Returns (x_series, None). Sampling interval = dt."""
    delay = int(round(tau / dt))
    total = transient + n_points
    buf = np.empty(total + delay)
    buf[: delay + 1] = x0
    for t in range(delay, total + delay - 1):
        xtau = buf[t - delay]
        buf[t + 1] = buf[t] + dt * (beta * xtau / (1.0 + xtau ** n) - gamma * buf[t])
    series = buf[delay + transient: delay + transient + n_points]
    return series, None


def lyapunov_step(x=None, dt: float = 1.0) -> float:
    """Largest LE per *sample* (= per forecasting step), for VPT in Lyapunov times."""
    return LAMBDA_CONT * dt
