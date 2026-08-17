"""Temporal splitting + washout (axes 3 & 4) — strictly causal, never shuffled.

For time series the split MUST be contiguous in time: shuffling or K-fold leaks
the future into the past. The washout (warm-up) discards the first `washout`
reservoir states from *training* so the readout never sees transient,
initial-state-dependent features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitSpec:
    """Index ranges for the three temporal splits, plus the train washout.

    Convention: half-open ranges [start, stop). `train_fit` is the slice used to
    FIT data-engineering stages (washout excluded) — the leakage-safe scope.
    """
    train: slice
    val: slice
    test: slice
    washout: int

    @property
    def train_fit(self) -> slice:
        """Training slice with the washout removed (used for scaler/readout fit)."""
        return slice(self.train.start + self.washout, self.train.stop)


def temporal_split(n: int, fracs=(0.6, 0.2, 0.2), washout: int = 100) -> SplitSpec:
    """Split `n` timesteps into contiguous train/val/test by `fracs`.

    fracs=(0.6,0.2,0.2) is the default; pass (0.8,0.0,0.2) for the legacy 80/20.
    """
    f_tr, f_va, _ = fracs
    a = int(f_tr * n)
    b = int((f_tr + f_va) * n)
    if washout >= a:
        raise ValueError(f"washout {washout} >= train length {a}")
    return SplitSpec(slice(0, a), slice(a, b), slice(b, n), washout)
