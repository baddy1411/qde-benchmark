"""The model interface that makes classical and quantum reservoirs comparable.

Every reservoir — ESN, gate-QRC, TFIM — exposes exactly one mapping:

    featurize(u) -> states of shape (T, F)

where `u` is the (already scaled) 1-D input series and row t holds the reservoir
features available after seeing inputs u[0..t]. The readout, metrics, pipeline,
and experiment runner all depend only on this contract, so swapping a classical
reservoir for a quantum one changes nothing downstream — which is the whole point
of a *fair* data-engineering comparison.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ReservoirModel(ABC):
    name: str = "reservoir"

    # Finite fading-memory length: the last feature row depends only on the last
    # `memory_window` inputs (None = unbounded / recurrent, e.g. ESN). Used to
    # make closed-loop rollout cheap for finite-memory reservoirs (QRC, NG-RC,
    # ELM) by re-featurizing only a short suffix instead of the whole history.
    memory_window: int | None = None

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of feature columns F that `featurize` returns."""

    @abstractmethod
    def featurize(self, u: np.ndarray) -> np.ndarray:
        """Map a scaled 1-D series u (T,) to a feature matrix (T, F),
        causally aligned so row t depends only on u[0..t]."""
