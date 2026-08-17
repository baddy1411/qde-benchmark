"""Reservoir models — one `featurize(u) -> (T, F)` interface, many physics."""
from __future__ import annotations

from .base import ReservoirModel  # noqa: F401
from .esn import ESN, ESNConfig  # noqa: F401
from .ngrc import NGRC, NGRCConfig  # noqa: F401
from .qngrc import QNGRC, QNGRCConfig  # noqa: F401
from .elm import ELM, ELMConfig  # noqa: F401

__all__ = ["ReservoirModel", "ESN", "ESNConfig", "NGRC", "NGRCConfig",
           "QNGRC", "QNGRCConfig", "ELM", "ELMConfig"]
