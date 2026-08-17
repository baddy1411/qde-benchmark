"""Composable, leakage-safe data-engineering stages — the object of study.

Every stateful stage is a `Transformer`: it is **fit on the training split only**
and then applied to all splits. That single discipline is what prevents the
preprocessing-before-split leakage that the methodology literature (Kapoor &
Narayanan 2023; Bowles et al. 2024) identifies as the dominant pitfall — and
that produced the spurious "advantage" in the prior version of this thesis.
"""
from __future__ import annotations

from .base import Transformer, Identity, compose  # noqa: F401
from .scaling import MinMax, Standardize, Robust, make_scaler  # noqa: F401
from .split import temporal_split, SplitSpec  # noqa: F401
from .embedding import delay_embedding, supervised_pairs  # noqa: F401
from .postprocess import LeakyMemory, Polynomial, FeatureStandardize  # noqa: F401

__all__ = [
    "Transformer", "Identity", "compose",
    "MinMax", "Standardize", "Robust", "make_scaler",
    "temporal_split", "SplitSpec",
    "delay_embedding", "supervised_pairs",
    "LeakyMemory", "Polynomial", "FeatureStandardize",
]
