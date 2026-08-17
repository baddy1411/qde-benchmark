"""Classical leaky Echo State Network — the baseline reservoir.

Self-contained NumPy implementation (no reservoirpy dependency) so every knob is
under our control and reproducible from a seed — important when the *pipeline*,
not the model, is the variable under study. Standard leaky-integrator update:

    x_t = (1-α)·x_{t-1} + α·tanh(W_in·u_t + W·x_{t-1} + b)

with W rescaled to a target spectral radius (the echo-state-property knob).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ReservoirModel


@dataclass
class ESNConfig:
    units: int = 300
    spectral_radius: float = 0.9    # echo-state property: keep < ~1.0
    leak_rate: float = 0.3          # α — temporal smoothing / memory
    input_scaling: float = 0.5      # how hard the input drives the reservoir
    density: float = 0.1            # fraction of non-zero recurrent weights
    bias_scaling: float = 0.0       # constant drive
    seed: int = 42


class ESN(ReservoirModel):
    name = "ESN"

    def __init__(self, cfg: ESNConfig | None = None):
        self.cfg = cfg or ESNConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.units

        self.bias = self._rng.uniform(-1, 1, size=n) * self.cfg.bias_scaling

        # sparse recurrent matrix, rescaled to the target spectral radius
        W = self._rng.uniform(-1, 1, size=(n, n))
        mask = self._rng.random((n, n)) < self.cfg.density
        W *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W *= self.cfg.spectral_radius / radius
        self.W = W

        # W_in is built lazily per observed input dimension (so the same ESN can
        # run on a scalar series — internal windowing — or an L-channel delay
        # vector — shared windowing, axis 5b). Cached per input_dim.
        self._W_in: dict[int, np.ndarray] = {}

    def _w_in(self, input_dim: int) -> np.ndarray:
        if input_dim not in self._W_in:
            self._W_in[input_dim] = (
                self._rng.uniform(-1, 1, size=(self.cfg.units, input_dim))
                * self.cfg.input_scaling
            )
        return self._W_in[input_dim]

    @property
    def n_features(self) -> int:
        return self.cfg.units

    def featurize(self, u: np.ndarray) -> np.ndarray:
        """u may be (T,) scalar (internal windowing) or (T, L) delay vectors
        (shared windowing). Returns reservoir states (T, units)."""
        U = np.asarray(u, dtype=float)
        if U.ndim == 1:
            U = U[:, None]
        W_in = self._w_in(U.shape[1])
        n, a = self.cfg.units, self.cfg.leak_rate
        states = np.empty((U.shape[0], n))
        x = np.zeros(n)
        for t in range(U.shape[0]):
            pre = W_in @ U[t] + self.W @ x + self.bias
            x = (1 - a) * x + a * np.tanh(pre)
            states[t] = x
        return states
