"""Next-Generation Reservoir Computing (Gauthier et al., Nat. Commun. 2021).

NG-RC replaces the random recurrent reservoir with an *explicit, deterministic*
feature map: a linear part (a length-`k` delay vector of the input) plus a
nonlinear part (unique monomials of those delays up to a chosen degree). With a
ridge readout it matches or beats a trained ESN at a fraction of the parameters
— so it is the strongest "feature-engineering instead of a reservoir" classical
benchmark, and a natural bridge to the QRC readout (whose ZZ correlators play the
role of NG-RC's quadratic terms).

This is a *shared-windowing* reservoir by construction (its memory is the delay
vector), the counterpart to the ESN's internal recurrence (axis 5b).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .base import ReservoirModel


@dataclass
class NGRCConfig:
    k: int = 2          # number of delay taps (lookback)
    stride: int = 1     # spacing between taps
    degree: int = 2     # max monomial degree of the nonlinear part
    include_linear: bool = True
    include_bias: bool = True
    seed: int = 0       # unused (deterministic) — kept for interface symmetry


class NGRC(ReservoirModel):
    name = "NG-RC"

    def __init__(self, cfg: NGRCConfig | None = None):
        self.cfg = cfg or NGRCConfig()
        self._F: int | None = None

    def _delays(self, u: np.ndarray) -> np.ndarray:
        """Causal delay matrix (T, k): column j = u shifted by j*stride, padded."""
        s = np.asarray(u, dtype=float).ravel()
        T = len(s)
        D = np.empty((T, self.cfg.k))
        for j in range(self.cfg.k):
            sh = j * self.cfg.stride
            col = np.empty(T)
            col[:sh] = s[0]
            col[sh:] = s[:T - sh] if sh else s
            D[:, j] = col
        return D

    @property
    def n_features(self) -> int:
        if self._F is None:
            raise RuntimeError("call featurize once to fix the feature dimension")
        return self._F

    @property
    def memory_window(self) -> int:
        return self.cfg.k * self.cfg.stride + 1

    def featurize(self, u: np.ndarray) -> np.ndarray:
        D = self._delays(u)                  # (T, k) linear features
        T, k = D.shape
        cols = []
        if self.cfg.include_bias:
            cols.append(np.ones((T, 1)))
        if self.cfg.include_linear:
            cols.append(D)
        # unique monomials of degree 2..degree over the k delay taps
        for deg in range(2, self.cfg.degree + 1):
            for combo in combinations_with_replacement(range(k), deg):
                term = np.prod(D[:, combo], axis=1, keepdims=True)
                cols.append(term)
        feats = np.concatenate(cols, axis=1)
        self._F = feats.shape[1]
        return feats
