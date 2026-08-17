"""reservoirpy-backed ESN — an independent implementation cross-check.

Our `ESN` is a from-scratch NumPy reservoir; this wraps the community-standard
`reservoirpy` library behind the same `ReservoirModel.featurize` interface. Having
both lets a notebook show that a data-engineering finding is a property of the
*method*, not of our particular ESN code (a reproducibility safeguard).
"""
from __future__ import annotations

import numpy as np

from .base import ReservoirModel

try:
    import reservoirpy as _rpy
except Exception:  # pragma: no cover
    _rpy = None
else:
    # silence progress output where the API exists; it was removed in
    # ReservoirPy 0.4.0, so a failure here must NOT disable the import.
    try:
        _rpy.verbosity(0)
    except Exception:
        pass


class RPyESN(ReservoirModel):
    name = "ESN-rpy"

    def __init__(self, units: int = 300, sr: float = 0.9, lr: float = 0.3,
                 input_scaling: float = 0.5, seed: int = 42):
        if _rpy is None:
            raise RuntimeError("reservoirpy is not installed")
        from reservoirpy.nodes import Reservoir
        self.units = units
        self._res = Reservoir(units, sr=sr, lr=lr, input_scaling=input_scaling, seed=seed)

    @property
    def n_features(self) -> int:
        return self.units

    def featurize(self, u: np.ndarray) -> np.ndarray:
        U = np.asarray(u, dtype=float)
        if U.ndim == 1:
            U = U[:, None]
        # ReservoirPy 0.4.x: reset() only works once the node has been run; a
        # fresh node already starts from the zero state, so guard the call.
        try:
            self._res.reset()
        except Exception:
            pass
        return np.asarray(self._res.run(U), dtype=float)
