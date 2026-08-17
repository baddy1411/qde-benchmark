"""Reproducibility provenance — write a versions.txt so any committed result can be
traced to the exact toolchain that produced it."""
from __future__ import annotations

import os
import platform
import sys


def write_versions(path: str = "versions.txt") -> str:
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch",
            "statsmodels", "xgboost", "lightgbm", "reservoirpy"]
    lines = [f"python: {sys.version.split()[0]}",
             f"platform: {platform.platform()}", ""]
    try:
        from importlib.metadata import version, PackageNotFoundError
        for p in pkgs:
            try:
                lines.append(f"{p}: {version(p)}")
            except PackageNotFoundError:
                lines.append(f"{p}: (not installed)")
    except Exception as e:  # pragma: no cover
        lines.append(f"(version lookup failed: {e})")
    out = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(out)
    return os.path.abspath(path)
