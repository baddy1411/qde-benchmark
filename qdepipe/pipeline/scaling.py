"""Scaling stages (axis 1) — all leakage-safe (fit stats from train only).

The *scope* of the fit (train-only vs global) is axis 2: it is enforced by
*what you pass to* `fit`, not by the scaler itself. The experiment runner always
fits on the training slice; a 'global' ablation simply fits on the full series.

Why scaling is not neutral for QRC: the scaled value becomes a rotation angle,
so the scaler chooses *which part of the encoding's sinusoidal feature map* the
data occupies (Schuld–Sweke–Meyer 2021). A min-max to [0,1] and a standardize to
mean-0 land on completely different arcs of that map.
"""
from __future__ import annotations

import numpy as np

from .base import Transformer, Identity


class MinMax(Transformer):
    """Affine map of [data_min, data_max] -> [lo, hi]. Default target [0, 1]."""

    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        self.lo, self.hi = lo, hi

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.dmin, self.dmax = x.min(axis=0), x.max(axis=0)
        self.fitted = True
        return self

    def _denom(self):
        d = self.dmax - self.dmin
        return np.where(d == 0, 1.0, d)

    def transform(self, x):
        self._check_fitted()
        z = (np.asarray(x, dtype=float) - self.dmin) / self._denom()
        return z * (self.hi - self.lo) + self.lo

    def inverse(self, z):
        u = (np.asarray(z, dtype=float) - self.lo) / (self.hi - self.lo)
        return u * self._denom() + self.dmin


class Standardize(Transformer):
    """Z-score: (x - mean) / std, stats from train."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(axis=0)
        self.std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        self.fitted = True
        return self

    def transform(self, x):
        self._check_fitted()
        return (np.asarray(x, dtype=float) - self.mean) / self.std

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.std + self.mean


class Robust(Transformer):
    """(x - median) / IQR — heavy-tail-robust, relevant for chaotic spikes."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.median = np.median(x, axis=0)
        q75, q25 = np.percentile(x, 75, axis=0), np.percentile(x, 25, axis=0)
        iqr = q75 - q25
        self.iqr = np.where(iqr == 0, 1.0, iqr)
        self.fitted = True
        return self

    def transform(self, x):
        self._check_fitted()
        return (np.asarray(x, dtype=float) - self.median) / self.iqr

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.iqr + self.median


def make_scaler(kind: str, **kw) -> Transformer:
    """Factory for axis-1 sweeps: 'identity' | 'minmax' | 'standard' | 'robust'."""
    table = {
        "identity": Identity,
        "none": Identity,
        "minmax": MinMax,
        "standard": Standardize,
        "standardize": Standardize,
        "robust": Robust,
    }
    key = kind.lower()
    if key not in table:
        raise ValueError(f"unknown scaler {kind!r}; choose from {sorted(table)}")
    return table[key](**kw) if key in ("minmax",) else table[key]()
