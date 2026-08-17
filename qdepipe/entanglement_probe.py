"""Entanglement probe — half-chain von Neumann entropy of the reservoir state.

Measures the entanglement the circuit actually generates, so the entanglement
ablation can state not just "removing coupling changes nothing" but
"entanglement of S ≈ x ebits was present and contributed nothing" — the
Palacios-style resource-vs-performance link, applied to our reservoir.

Read-only companion to `models/gate_qrc.py`: it reuses the model's own fixed
layer and encoding unitaries (private-method reuse, no core edits) to evolve the
pure state and compute S(rho_A) for the A = first-half-of-qubits bipartition.
Pure-state (channel='none') configs only — the QRC-v6/rich family.

Kron-ordering note: `_qops.op_on` places qubit q at position q of the Kronecker
product, so qubit 0 is the most significant factor. Reshaping psi to
(2^k, 2^(n-k)) therefore puts qubits 0..k-1 on the rows — the A subsystem.
"""
from __future__ import annotations

import numpy as np
import torch

from .models.gate_qrc import GateQRC
from .models import _qops as Q


def half_chain_entropy(psi: torch.Tensor, n: int, k: int | None = None) -> float:
    """S(rho_A) in bits for the bipartition A = qubits 0..k-1 (default n//2).

    Pure state: S = -sum p_i log2 p_i with p_i the squared Schmidt values of
    psi reshaped to (2^k, 2^(n-k)). Zero iff the state is a product across the cut.
    """
    k = n // 2 if k is None else k
    M = psi.reshape(2 ** k, 2 ** (n - k))
    s = torch.linalg.svdvals(M)
    p = (s ** 2).real
    p = p[p > 1e-15]
    return float(-(p * torch.log2(p)).sum())


def reservoir_entropy_series(model: GateQRC, series, max_t: int | None = None) -> np.ndarray:
    """Half-chain entropy of the full-window circuit state at each timestep.

    Evolves the same pure-state circuit `featurize` runs for the longest
    (v = V) sub-window — the state whose measurement produces the features —
    and records S(rho_A) instead of expectation values.
    """
    if model.cfg.channel != "none":
        raise ValueError("entropy probe requires a pure-state (channel='none') config")
    s = np.asarray(series, dtype=float).ravel()
    T = len(s) if max_t is None else min(len(s), max_t)
    n, dev = model.cfg.n_qubits, model.device
    out = np.empty(T)
    for t in range(T):
        lo = max(0, t - model.cfg.window + 1)
        window = s[lo:t + 1]
        psi = Q.zero_state(n, dev)
        for x in window:
            U = model.U_fixed @ model._encode_unitary(x)
            psi = U @ psi
        out[t] = half_chain_entropy(psi, n)
    return out


def mean_reservoir_entropy(model: GateQRC, series, sample: int = 200,
                           rng_seed: int = 0) -> dict:
    """Mean/std/max of S(rho_A) over `sample` timesteps drawn evenly from the
    series (evenly + deterministically, so runs are reproducible)."""
    s = np.asarray(series, dtype=float).ravel()
    idx = np.linspace(model.cfg.window, len(s) - 1, num=min(sample, len(s)),
                      dtype=int)
    vals = np.empty(len(idx))
    n, dev = model.cfg.n_qubits, model.device
    for j, t in enumerate(idx):
        lo = max(0, t - model.cfg.window + 1)
        window = s[lo:t + 1]
        psi = Q.zero_state(n, dev)
        for x in window:
            U = model.U_fixed @ model._encode_unitary(x)
            psi = U @ psi
        vals[j] = half_chain_entropy(psi, n)
    max_bits = min(model.cfg.n_qubits // 2, model.cfg.n_qubits - model.cfg.n_qubits // 2)
    return {"S_mean": float(vals.mean()), "S_std": float(vals.std()),
            "S_max_observed": float(vals.max()), "S_max_possible": float(max_bits),
            "n_sampled": int(len(idx))}
