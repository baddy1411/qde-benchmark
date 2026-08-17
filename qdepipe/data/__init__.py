"""Signal sources for the benchmark — a small registry of chaotic systems behind a
uniform interface so the shared pipeline stays system-agnostic.

    generate(system, n_points) -> x         the 1-D forecasting target
    lyapunov_step(system, x)   -> float      largest LE per forecasting step (for VPT)

Hénon's LE is computed from the trajectory (QR-Jacobian); the continuous systems
(Lorenz, Mackey–Glass) carry a reference per-sample constant. Add a system by
registering its generator + LE here — nothing in experiment.py/closedloop.py changes.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .henon import generate_henon, lyapunov_gate, lyapunov_qr  # noqa: F401
from .lorenz import generate_lorenz
from .mackeyglass import generate_mackeyglass
from .lorenz96 import generate_lorenz96
from . import lorenz as _lorenz
from . import mackeyglass as _mg
from . import lorenz96 as _l96

# name -> (generator returning (series, aux), per-step Lyapunov function).
# The lorenz_d* entries are the SAME system at different sampling intervals
# (stride; Δ = dt·stride) — used only for the Lorenz Δ-sensitivity check, i.e. to
# test whether the NG-RC vs ESN ordering depends on the sampling rate. (Outcome:
# it does not — tuned NG-RC wins at every Δ; no reversal.)
SYSTEMS = {
    "henon":       (generate_henon,       lambda x: lyapunov_qr(x)),
    "lorenz":      (generate_lorenz,      lambda x: _lorenz.lyapunov_step(stride=5)),
    "lorenz_d02":  (partial(generate_lorenz, stride=2),  lambda x: _lorenz.lyapunov_step(stride=2)),
    "lorenz_d10":  (partial(generate_lorenz, stride=10), lambda x: _lorenz.lyapunov_step(stride=10)),
    "mackeyglass": (generate_mackeyglass, lambda x: _mg.lyapunov_step()),
    # higher-dimensional stress test: D=20 sites, F=8, x_0 target
    "lorenz96":    (generate_lorenz96,    lambda x: _l96.lyapunov_step()),
}

__all__ = ["generate_henon", "lyapunov_qr", "lyapunov_gate", "generate_lorenz",
           "generate_mackeyglass", "generate_lorenz96", "SYSTEMS", "generate",
           "lyapunov_step", "system_names"]


def system_names():
    return list(SYSTEMS)


def generate(system: str, n_points: int) -> np.ndarray:
    """The 1-D forecasting target series for a named system."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; known: {system_names()}")
    gen, _ = SYSTEMS[system]
    x = gen(n_points)[0]                    # every generator returns (series, aux)
    return np.asarray(x, dtype=float).ravel()


def lyapunov_step(system: str, x: np.ndarray) -> float:
    """Largest Lyapunov exponent per forecasting step for the named system."""
    _, le_fn = SYSTEMS[system]
    return float(le_fn(x))
