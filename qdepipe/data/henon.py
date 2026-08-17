"""Hénon map — the single benchmark signal.

    x_{n+1} = 1 - a x_n^2 + y_n
    y_{n+1} = b x_n

Standard chaotic parameters a=1.4, b=0.3 give the canonical attractor with a
largest Lyapunov exponent ≈ 0.4211 (so one Lyapunov time ≈ 2.37 steps). The
`x` series is the 1-D forecasting target throughout the thesis.

Provenance carried from the prior validated project (do not re-derive):
  a=1.4, b=0.3, x0=y0=0, washout 500; LE_true = 0.4211 (QR-Jacobian).
"""
from __future__ import annotations

import numpy as np

HENON_A = 1.4
HENON_B = 0.3
WASHOUT = 500


def generate_henon(n_points: int, a: float = HENON_A, b: float = HENON_B,
                   x0: float = 0.0, y0: float = 0.0, washout: int = WASHOUT):
    """Iterate the Hénon map, drop `washout` transient steps. Returns (x, y)."""
    total = n_points + washout
    x = np.empty(total)
    y = np.empty(total)
    x[0], y[0] = x0, y0
    for n in range(total - 1):
        x[n + 1] = 1.0 - a * x[n] ** 2 + y[n]
        y[n + 1] = b * x[n]
    return x[washout:], y[washout:]


def lyapunov_qr(x, a: float = HENON_A, b: float = HENON_B) -> float:
    """Largest Lyapunov exponent via QR decomposition of the Jacobian product
    along the trajectory (≈0.4211 for the standard map). Used as a data sanity
    gate and to express VPT in Lyapunov times."""
    Q = np.eye(2)
    s = 0.0
    for k in range(len(x) - 1):
        J = np.array([[-2 * a * x[k], 1.0], [b, 0.0]])
        Q, R = np.linalg.qr(J @ Q)
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q, R = Q * d, R * d[:, None]
        s += np.log(abs(R[0, 0]))
    return s / (len(x) - 1)


def lyapunov_gate(x, ref: float = 0.4211, tol: float = 0.03):
    """Return (LE, lyapunov_time, passed) — passed if the estimate is within
    `tol` of the reference. A failing gate means the data generation is wrong."""
    le = lyapunov_qr(x)
    return le, 1.0 / le, bool(abs(le - ref) < tol)
