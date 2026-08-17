"""Extreme Learning Machine reservoir (random nonlinear projection + ridge).

An ELM maps the (windowed) input through a single random, *untrained* nonlinear
layer and trains only a linear readout:

    h_t = tanh(W_in · x_t + b),   W_in, b random and fixed.

It is the canonical "random features" control: same random-projection + linear
readout recipe as a QRC (whose random quantum dynamics replace W_in·x), but with
no recurrence and no quantum structure. If a QRC cannot beat a matched-width ELM,
the quantum dynamics are adding nothing — making ELM an essential benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ReservoirModel
from ..pipeline.embedding import delay_embedding


@dataclass
class ELMConfig:
    units: int = 300
    lookback: int = 2          # delay-window length fed to the random layer
    stride: int = 1
    input_scaling: float = 1.0
    bias_scaling: float = 1.0
    seed: int = 42


class ELM(ReservoirModel):
    name = "ELM"

    def __init__(self, cfg: ELMConfig | None = None):
        self.cfg = cfg or ELMConfig()
        rng = np.random.default_rng(self.cfg.seed)
        self.W_in = rng.uniform(-1, 1, size=(self.cfg.units, self.cfg.lookback)) * self.cfg.input_scaling
        self.bias = rng.uniform(-1, 1, size=self.cfg.units) * self.cfg.bias_scaling

    @property
    def n_features(self) -> int:
        return self.cfg.units

    @property
    def memory_window(self) -> int:
        return self.cfg.lookback * self.cfg.stride + 1

    def featurize(self, u: np.ndarray) -> np.ndarray:
        U = np.asarray(u, dtype=float)
        if U.ndim == 1:
            U = delay_embedding(U, self.cfg.lookback, self.cfg.stride)
        # random nonlinear projection, no recurrence
        return np.tanh(U @ self.W_in.T + self.bias)
