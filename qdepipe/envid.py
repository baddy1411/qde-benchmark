"""Environment identity: the value the run contract declares and never had.

Appendix E defines ``environment_id`` as a hash of the pinned dependency
lockfile, and declares it non-nullable. Until this module existed nothing in the
codebase computed it -- it was passed as ``None`` everywhere, so every registry
row violated that part of the contract. See §10.5.

The lockfile is the full resolved transitive closure, not the direct-dependency
list, because that is what actually determines numerics. Inside the reproduction
image it is written to ``/opt/requirements.lock.txt`` at build time; on a host it
is ``docker/requirements.lock.txt`` if one has been exported.

``environment_id`` identifies the *package closure* only. It deliberately does
not cover the BLAS implementation or the CPU architecture, which are recorded
separately by :func:`environment_fingerprint`, because two environments can share
a lockfile and still differ in the last bits of a floating-point reduction.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# In-image location first, then an exported host lockfile.
_LOCK_CANDIDATES = (
    Path("/opt/requirements.lock.txt"),
    ROOT / "docker" / "requirements.lock.txt",
)


def lockfile_path() -> Optional[Path]:
    """The lockfile this environment is identified by, or None if unpinned."""
    override = os.environ.get("QDE_LOCKFILE")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    for p in _LOCK_CANDIDATES:
        if p.is_file():
            return p
    return None


def environment_id(short: int = 12) -> Optional[str]:
    """SHA-256 of the pinned lockfile, truncated. None when nothing is pinned.

    Returning None on an unpinned environment is deliberate: a run that cannot
    name its dependency closure should record that it cannot, rather than
    record a hash of whatever happened to be installed.
    """
    p = lockfile_path()
    if p is None:
        return None
    # normalise line endings so the same lock hashes identically across hosts
    data = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:short]


def blas_name() -> str:
    """Best-effort name of the BLAS NumPy is linked against."""
    try:
        import numpy as np

        cfg = np.__config__.show(mode="dicts")  # numpy >= 2
        blas = (cfg.get("Build Dependencies") or {}).get("blas") or {}
        name = blas.get("name")
        if name:
            version = blas.get("version") or "unknown"
            return f"{name}-{version}"
    except Exception:
        pass
    return "unknown"


def environment_fingerprint() -> dict:
    """Everything needed to say whether two runs should agree bit for bit.

    The lockfile hash pins the packages; the remaining fields pin the parts of
    the environment that a lockfile cannot, and that are the usual reason two
    correctly-pinned runs still disagree in the final digits.
    """
    return {
        "environment_id": environment_id(),
        "lockfile": str(lockfile_path()) if lockfile_path() else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas": blas_name(),
        "containerised": Path("/.dockerenv").exists(),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")
        or os.environ.get("OMP_NUM_THREADS"),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(environment_fingerprint(), indent=2))
