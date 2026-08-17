"""Feature post-processing (axes 10-12) — model-agnostic feature engineering.

These act on the reservoir feature matrix *before* the readout, so they apply
identically to ESN and QRC features. This is the highest-leverage, least
standardised part of the QRC pipeline (RF-QRC / memory-augmented QRC; NG-RC).
"""
from __future__ import annotations

import numpy as np

from .base import Transformer


class LeakyMemory(Transformer):
    """Classical leaky integrator on features:  r_t = (1-ε)·r_{t-1} + ε·m_t.

    Restores fading memory on top of a (possibly memoryless) quantum reservoir
    without any extra circuit — the cheap, model-agnostic memory knob (axis 10).
    Stateless to fit; carries no train statistics, but implemented as a
    Transformer so it slots into the pipeline uniformly.
    """

    def __init__(self, eps: float = 0.1):
        if not 0 < eps <= 1:
            raise ValueError("eps must be in (0, 1]")
        self.eps = eps

    def fit(self, x):
        self.fitted = True
        return self

    def transform(self, x):
        m = np.asarray(x, dtype=float)
        out = np.empty_like(m)
        r = np.zeros(m.shape[1])
        for t in range(m.shape[0]):
            r = (1 - self.eps) * r + self.eps * m[t]
            out[t] = r
        return out


class Polynomial(Transformer):
    """Degree-2 (Volterra-style) feature expansion (axis 11).

    Appends squares and, optionally, pairwise products — the principled version
    of the NG-RC nonlinear feature set. Column order is fixed at fit time so
    train/test feature spaces match exactly.
    """

    def __init__(self, interactions: bool = False, include_linear: bool = True):
        self.interactions = interactions
        self.include_linear = include_linear

    def fit(self, x):
        self.n = np.asarray(x).shape[1]
        self.fitted = True
        return self

    def transform(self, x):
        self._check_fitted()
        m = np.asarray(x, dtype=float)
        cols = [m] if self.include_linear else []
        cols.append(m ** 2)
        if self.interactions:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    cols.append((m[:, i] * m[:, j])[:, None])
        return np.concatenate(cols, axis=1)


class FeatureStandardize(Transformer):
    """Per-feature z-score (axis 12), stats from train only.

    Conditions the readout's ridge problem; relevant when QRC observables live on
    very different scales (e.g. Z near ±1 vs ZZ correlators near 0).
    """

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(axis=0)
        self.std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        self.fitted = True
        return self

    def transform(self, x):
        self._check_fitted()
        return (np.asarray(x, dtype=float) - self.mean) / self.std
