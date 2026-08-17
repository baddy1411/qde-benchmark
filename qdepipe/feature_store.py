"""Quantum feature store — persist, version, and reuse reservoir feature matrices.

STANDALONE & ADDITIVE. This module touches no existing call site; it is a cache
layer you opt into. The expensive thing in this project is `reservoir.featurize`
(exact statevector at 10 qubits is ~30x an 8-qubit run), and the same feature
matrix is recomputed across the DM, MCS, ablation and finite-shot passes. The
store computes each matrix once, keyed by the step-1 :func:`contracts.feature_key`,
and reuses it thereafter.

Design invariants (both gate the wiring in part 2):

* **Round-trip exactness.** Persist -> load is bit-identical. Matrices are saved
  with ``numpy.save`` (the binary ``.npy`` format), which preserves dtype and
  every bit of a float64 array (verified in tests with ``np.array_equal``, not
  ``allclose``). The store never coerces dtype or shape.
* **Cache transparency.** On a miss the store returns *exactly* what
  ``model.featurize(inp)`` returns; on a hit it returns ``np.load`` of the bytes
  it wrote on that miss. Since the bytes are identical and featurize output is
  C-contiguous float64, a hit and a miss are indistinguishable downstream.

A cache hit is therefore *provably the same computation* as a miss — never a
metadata-trusted guess. The key (from :mod:`qdepipe.contracts`) already captures
everything that changes the matrix, including the split-dependent scaling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import contracts


def _default_root() -> Path:
    # repo_root/feature_store  (sibling of results/; visible so step-5 DVC can track it)
    return Path(__file__).resolve().parent.parent / "feature_store"


class FeatureStore:
    """Content-addressed cache of reservoir feature matrices.

    Layout under ``root/namespace/``:  ``<key>.npy`` (the matrix) and
    ``<key>.json`` (the self-describing manifest = the feature payload + shape +
    dtype). Writes are atomic (temp file + ``os.replace``) so an interrupted run
    can never leave a half-written matrix that a later run would treat as a hit.
    """

    def __init__(self, root: Optional[os.PathLike | str] = None, namespace: str = "reservoir"):
        self.root = Path(root) if root is not None else _default_root()
        self.namespace = namespace
        self.dir = self.root / namespace
        self._hits = 0
        self._misses = 0

    # -- paths ---------------------------------------------------------------
    def _npy(self, key: str) -> Path:
        return self.dir / f"{key}.npy"

    def _manifest(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    # -- key -----------------------------------------------------------------
    def key_for(self, model, cfg, external_window: bool = False) -> str:
        return contracts.feature_key(model, cfg, external_window=external_window)

    # -- low-level get/put ---------------------------------------------------
    def has(self, key: str) -> bool:
        return self._npy(key).exists()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Load the matrix for `key`, or None on a miss (also None if the file is
        unreadable — a corrupt entry is treated as a miss, never as a wrong hit)."""
        p = self._npy(key)
        if not p.exists():
            return None
        try:
            return np.load(p, allow_pickle=False)
        except Exception:
            return None

    def put(self, key: str, matrix: np.ndarray, manifest: Optional[dict] = None) -> None:
        """Persist `matrix` under `key` atomically, with an optional manifest.
        The array is written exactly as given — no dtype/shape coercion."""
        self.dir.mkdir(parents=True, exist_ok=True)
        npy = self._npy(key)
        tmp = npy.with_suffix(".npy.tmp")
        # numpy appends .npy to a str path without suffix; use a file handle to
        # keep the exact temp name, then atomically replace.
        with open(tmp, "wb") as fh:
            np.save(fh, matrix, allow_pickle=False)
        os.replace(tmp, npy)
        meta = dict(manifest or {})
        meta.update({"key": key, "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                     "nbytes": int(matrix.nbytes)})
        mtmp = self._manifest(key).with_suffix(".json.tmp")
        with open(mtmp, "w") as fh:
            json.dump(meta, fh, sort_keys=True, indent=2)
        os.replace(mtmp, self._manifest(key))

    # -- high-level read-through --------------------------------------------
    def get_or_compute(self, key: str, compute: Callable[[], np.ndarray],
                       manifest: Optional[dict] = None) -> np.ndarray:
        """Return the cached matrix for `key`, else `compute()` it, persist, and
        return it. On a miss the returned object is exactly `compute()`'s output."""
        cached = self.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        matrix = compute()
        self.put(key, matrix, manifest)
        return matrix

    def featurize(self, model, inp: np.ndarray, cfg, external_window: bool = False) -> np.ndarray:
        """Cached ``model.featurize(inp)``.

        Precondition (held by the pipeline): `inp` is the deterministic function of
        `cfg` (the scaled series, or its delay embedding when `external_window`),
        so the key — derived from (model, cfg) — fully determines `inp`. The store
        does not re-derive `inp`; it caches the result of featurizing the `inp`
        you pass. On a miss it returns exactly ``model.featurize(inp)``.
        """
        key = self.key_for(model, cfg, external_window)
        payload = contracts.feature_payload(model, cfg, external_window)
        return self.get_or_compute(key, lambda: np.asarray(model.featurize(inp)), payload)

    # -- introspection -------------------------------------------------------
    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "entries": sum(1 for _ in self.dir.glob("*.npy")) if self.dir.exists() else 0}

    def reset_stats(self) -> None:
        self._hits = self._misses = 0

    def manifest(self, key: str) -> Optional[dict]:
        p = self._manifest(key)
        return json.loads(p.read_text()) if p.exists() else None

    def clear(self) -> int:
        """Delete all entries in this namespace. Returns the number removed."""
        n = 0
        if self.dir.exists():
            for f in list(self.dir.glob("*.npy")) + list(self.dir.glob("*.json")):
                f.unlink(); n += 1
        return n


def maybe_cached_featurize(store: Optional[FeatureStore], model, inp: np.ndarray,
                           cfg, external_window: bool = False) -> np.ndarray:
    """The one-line, default-off wiring hook for a hot path.

    With ``store=None`` this returns *exactly* ``model.featurize(inp)`` — the
    unwired computation, byte-for-byte — so a path stays provably unchanged with
    the cache off. With a store it routes through the (transparent) cache. Keeping
    the default off permanently means the cache's transparency is re-verifiable on
    demand by flipping this one argument.
    """
    if store is None:
        return model.featurize(inp)
    return store.featurize(model, inp, cfg, external_window=external_window)
