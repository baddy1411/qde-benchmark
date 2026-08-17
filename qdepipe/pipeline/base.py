"""The leakage-safe transformer contract.

A `Transformer` separates *learning parameters from data* (`fit`, train split
only) from *applying* them (`transform`, any split). Composing transformers
yields a pipeline that is leakage-safe by construction: as long as `fit` only
ever sees training data, no test statistic can leak backwards.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Transformer(ABC):
    """Base class: fit on TRAIN only, then transform any split."""

    fitted: bool = False

    @abstractmethod
    def fit(self, x: np.ndarray) -> "Transformer":
        ...

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse(self, z: np.ndarray) -> np.ndarray:  # optional
        raise NotImplementedError(f"{type(self).__name__} has no inverse")

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(f"{type(self).__name__}.transform called before fit")


class Identity(Transformer):
    """No-op stage — the control level for any axis ('off')."""

    def fit(self, x):
        self.fitted = True
        return self

    def transform(self, x):
        return np.asarray(x)

    def inverse(self, z):
        return np.asarray(z)


class _Composed(Transformer):
    """Sequential composition; `fit` threads transformed train data through."""

    def __init__(self, stages):
        self.stages = list(stages)

    def fit(self, x):
        cur = np.asarray(x)
        for s in self.stages:
            cur = s.fit_transform(cur)
        self.fitted = True
        return self

    def transform(self, x):
        self._check_fitted()
        cur = np.asarray(x)
        for s in self.stages:
            cur = s.transform(cur)
        return cur

    def inverse(self, z):
        cur = np.asarray(z)
        for s in reversed(self.stages):
            cur = s.inverse(cur)
        return cur


def compose(*stages: Transformer) -> Transformer:
    """Compose transformers left-to-right into one leakage-safe stage."""
    real = [s for s in stages if not isinstance(s, Identity)]
    return real[0] if len(real) == 1 else _Composed(real or [Identity()])
