#!/usr/bin/env python3
"""Numerically estimate the largest Lyapunov exponent of the Lorenz-63 and
Mackey-Glass systems AS IMPLEMENTED in qdepipe (same parameters, integrator, dt),
via the Benettin two-trajectory algorithm (perturb, evolve, measure separation
growth, renormalise). Reports λ per unit time, to verify/replace the hardcoded
constants that the VPT climate metric depends on.

Hénon is not included: its λ is already computed analytically from the Jacobian
(qdepipe/data/henon.py:lyapunov_qr) along each trajectory, not hardcoded.
"""
from __future__ import annotations

import numpy as np


# ---- Lorenz-63 (RK4, matching qdepipe/data/lorenz.py) ----------------------
def _lorenz_deriv(s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def _lorenz_rk4(s, dt):
    k1 = _lorenz_deriv(s)
    k2 = _lorenz_deriv(s + 0.5 * dt * k1)
    k3 = _lorenz_deriv(s + 0.5 * dt * k2)
    k4 = _lorenz_deriv(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lyap_lorenz(dt=0.01, n=400000, transient=50000, renorm=10, d0=1e-9, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 1.0, 1.0])
    for _ in range(transient):
        s = _lorenz_rk4(s, dt)
    sp = s + d0 * rng.standard_normal(3)
    acc, count = 0.0, 0
    for k in range(n):
        s = _lorenz_rk4(s, dt)
        sp = _lorenz_rk4(sp, dt)
        if (k + 1) % renorm == 0:
            d = np.linalg.norm(sp - s)
            acc += np.log(d / d0)
            count += 1
            sp = s + (d0 / d) * (sp - s)
    return acc / (count * renorm * dt)          # λ per unit time


# ---- Mackey-Glass (Euler, matching qdepipe/data/mackeyglass.py) -------------
def _mg_step(buf, t, delay, dt=1.0, beta=0.2, gamma=0.1, n=10):
    xtau = buf[t - delay]
    return buf[t] + dt * (beta * xtau / (1.0 + xtau ** n) - gamma * buf[t])


def lyap_mackeyglass(tau=17, dt=1.0, total=600000, transient=20000,
                     renorm=10, d0=1e-9, seed=0):
    rng = np.random.default_rng(seed)
    delay = int(round(tau / dt))
    # the dynamical state is the length-(delay+1) history window
    win = delay + 1
    a = np.empty(total + delay + 1)
    a[:win] = 1.2
    # warm up the reference on a single buffer
    for t in range(delay, transient + delay):
        a[t + 1] = _mg_step(a, t, delay, dt)
    # initialise two co-evolving windows from the warmed state
    base = a[transient: transient + win].copy()
    pert = base + d0 * rng.standard_normal(win)
    acc, count = 0.0, 0
    for k in range(total):
        # step both windows forward one MG step (index `delay` = current end)
        nb = _mg_step(base, delay, delay, dt)
        npv = _mg_step(pert, delay, delay, dt)
        base = np.concatenate([base[1:], [nb]])
        pert = np.concatenate([pert[1:], [npv]])
        if (k + 1) % renorm == 0:
            d = np.linalg.norm(pert - base)
            acc += np.log(d / d0)
            count += 1
            pert = base + (d0 / d) * (pert - base)
    return acc / (count * renorm * dt)          # λ per unit time


if __name__ == "__main__":
    print("Benettin two-trajectory largest-Lyapunov estimates (as implemented):\n")
    ll = lyap_lorenz()
    print(f"  Lorenz-63 (σ=10,ρ=28,β=8/3, RK4 dt=0.01): λ ≈ {ll:.4f} /unit time")
    print(f"      hardcoded LAMBDA_CONT = 0.9056   (Sprott reference)")
    for s in (0, 1, 2):
        lm = lyap_mackeyglass(seed=s)
        print(f"  Mackey-Glass MG-17 (Euler dt=1) seed{s}: λ ≈ {lm:.5f} /unit time")
    print(f"      hardcoded LAMBDA_CONT = 0.0086   (no specific source)")
