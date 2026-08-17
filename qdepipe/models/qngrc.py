"""Quantum Next-Generation Reservoir Computing (qNG-RC).

A faithful, exact-expectation implementation of the classical-input quantum NG-RC
of Wang, Sun, Kong, Sun and Zhang, "Quantum Next-Generation Reservoir Computing
and Its Quantum Optical Implementation", Phys. Rev. A 111, 022609 (2025),
arXiv:2502.16938, which ports Gauthier's NVAR construction to
a quantum feature map. Distinct from the gate-based QRC in this repo: there is NO
reservoir dynamics, NO time-evolution and NO Pauli-observable readout. Instead:

  1. amplitude-encode the length-k delay window d_t into a state |psi> = d_t/||d_t||
     (its amplitudes ARE the normalised delayed inputs);
  2. build the nonlinear NVAR features as *tensor products* of that state,
     |psi>^{\\otimes s}, whose amplitudes are the degree-s monomials of the
     normalised inputs;
  3. read those amplitudes out and hand them to a ridge readout.

The one thing that makes this genuinely *quantum-NG-RC* and not classical NG-RC is
the **amplitude normalisation** by ||d_t|| that encoding-then-measuring any state
forces: the feature basis is normalised monomials {d_i/||d||, d_i d_j/||d||^2, ...},
not the raw monomials {d_i, d_i d_j, ...} NG-RC uses. The norm ||d_t|| is exposed
as a side feature so the readout has access to the discarded scale.

Idealisation (stated honestly, matching this thesis's treatment of gate-QRC): this
reads the *signed* amplitudes of the tensor-encoded state, i.e. it assumes the
measurement protocol (interference / ancilla) recovers signs exactly and in the
infinite-shot limit. A basis-projection measurement on hardware returns unsigned
probabilities under shot noise; that is exactly the degradation the finite-shot
chapter models for the gate-QRC and is left to a follow-up here. So, as with the
gate-QRC, this exact variant can only *overstate* the quantum model's accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .base import ReservoirModel


@dataclass
class QNGRCConfig:
    k: int = 2            # number of delay taps (lookback) — the amplitude register
    stride: int = 1       # spacing between taps
    degree: int = 2       # max tensor order s = max monomial degree
    include_linear: bool = True
    include_bias: bool = True
    include_norm: bool = True    # expose ||d_t|| (the discarded encoding scale)
    eps: float = 1e-12           # guards divide-by-zero when a window is all-zero
    seed: int = 0                # unused (deterministic) — interface symmetry


class QNGRC(ReservoirModel):
    name = "qNG-RC"

    def __init__(self, cfg: QNGRCConfig | None = None):
        self.cfg = cfg or QNGRCConfig()
        self._F: int | None = None

    def _delays(self, u: np.ndarray) -> np.ndarray:
        """Causal delay matrix (T, k): column j = u shifted by j*stride, padded."""
        s = np.asarray(u, dtype=float).ravel()
        T = len(s)
        D = np.empty((T, self.cfg.k))
        for j in range(self.cfg.k):
            sh = j * self.cfg.stride
            col = np.empty(T)
            col[:sh] = s[0]
            col[sh:] = s[:T - sh] if sh else s
            D[:, j] = col
        return D

    @property
    def n_features(self) -> int:
        if self._F is None:
            raise RuntimeError("call featurize once to fix the feature dimension")
        return self._F

    @property
    def memory_window(self) -> int:
        return self.cfg.k * self.cfg.stride + 1

    def featurize(self, u: np.ndarray) -> np.ndarray:
        D = self._delays(u)                       # (T, k) raw delay window
        T, k = D.shape
        norm = np.sqrt(np.sum(D * D, axis=1, keepdims=True)) + self.cfg.eps
        A = D / norm                              # amplitude-encoded state |psi> (unit norm)

        cols = []
        if self.cfg.include_bias:
            cols.append(np.ones((T, 1)))          # the |0..0> reference amplitude
        if self.cfg.include_norm:
            cols.append(norm)                     # the discarded encoding scale ||d||
        if self.cfg.include_linear:
            cols.append(A)                        # amplitudes of |psi>  (degree-1 monomials)
        # tensor orders s = 2..degree: amplitudes of |psi>^{⊗ s} are the unique
        # degree-s monomials of the normalised inputs (products of amplitudes).
        for s in range(2, self.cfg.degree + 1):
            for combo in combinations_with_replacement(range(k), s):
                cols.append(np.prod(A[:, combo], axis=1, keepdims=True))
        feats = np.concatenate(cols, axis=1)
        self._F = feats.shape[1]
        return feats
