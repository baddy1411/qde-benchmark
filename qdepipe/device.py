"""Compute environment + reproducible seeding.

GPU-first (the prior project validated on an RTX 3060 Ti, torch cu128). We never
silently fall back to CPU for the quantum models — a CPU run must be explicit so
it can't masquerade as a GPU result in the thesis tables.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

DEFAULT_SEED = 42


def get_device(require_cuda: bool = False):
    """Return the torch device. If `require_cuda`, raise when no GPU is present."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed in this environment.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("GPU required but CUDA is unavailable — check WSL/driver.")
    return torch.device("cpu")


def seed_everything(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Seed numpy + torch (CPU & CUDA) and return a numpy Generator.

    Every experiment threads this Generator so a run is reproducible from its
    seed alone — a requirement for the multi-seed mean±std tables.
    """
    rng = np.random.default_rng(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return rng


def info() -> dict:
    """Print and return a summary of the compute environment."""
    import sys

    d = {"python": sys.version.split()[0], "numpy": np.__version__}
    if torch is not None:
        d["torch"] = torch.__version__
        d["cuda_available"] = bool(torch.cuda.is_available())
        d["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    for k, v in d.items():
        print(f"{k:16s}: {v}")
    return d
