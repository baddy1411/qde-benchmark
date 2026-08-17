"""qdepipe — data-engineering pipeline study for reservoir forecasting.

The package is organised around the idea that **data engineering is the object
of study**, not a preamble to it:

    data.henon        -> raw signal
    pipeline.*        -> leakage-safe, composable data-engineering stages
    encoding          -> how a scalar becomes a quantum rotation angle
    models.*          -> reservoirs with one featurize(u) -> (T, F) interface
    readout           -> the only trained part (ridge)
    experiment        -> config-driven runner that wires it all together

Nothing here imports the legacy `unified/qrclab` code; the quantum engines are
re-derived fresh and validated against the prior project's verified numbers.
"""
from __future__ import annotations

from . import device, metrics  # noqa: F401

__all__ = ["device", "metrics"]
__version__ = "0.1.0"
