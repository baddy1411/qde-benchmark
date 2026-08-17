"""Delay embedding & supervised-pair construction (axes 5 & 6).

Takens (1981): a length-`L` delay vector reconstructs the attractor, so the
lookback window is a genuine data-engineering lever. `delay_embedding` is the
optional *shared* embedding that lets every model — ESN and QRC — consume the
same vector input (the fairness choice discussed in THESIS_PLAN §7).

`supervised_pairs` turns an aligned (state, input) stream into (X, y) for a
horizon-`k` forecast, handling the causal off-by-one so no target leaks into its
own features.
"""
from __future__ import annotations

import numpy as np


def delay_embedding(series, L: int = 1, stride: int = 1) -> np.ndarray:
    """Causal delay vectors: row t = [s_{t-(L-1)·stride}, …, s_t].

    Returns shape (T, L); the first (L-1)·stride rows are left-padded with the
    earliest value so the output stays aligned to the input time axis (time t of
    the output corresponds to time t of the input).
    """
    s = np.asarray(series, dtype=float).ravel()
    T = len(s)
    out = np.empty((T, L))
    for j in range(L):
        shift = (L - 1 - j) * stride
        col = np.empty(T)
        col[:shift] = s[0]
        col[shift:] = s[: T - shift] if shift else s
        out[:, j] = col
    return out


def supervised_pairs(states: np.ndarray, target: np.ndarray, horizon: int = 1):
    """Align reservoir states to a horizon-`k` target.

    state row t (features after seeing input up to time t) predicts target[t+k].
    Returns (X, y) with X = states[:-k], y = target[k:].
    """
    states = np.asarray(states, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return states[:-horizon], target[horizon:]
