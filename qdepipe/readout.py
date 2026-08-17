"""Ridge readout — the only trained part of any reservoir model.

Keeping training to a single linear layer is what makes the comparison about the
*reservoir + pipeline*, not about a deep model. `alpha` is swept, never fixed:
the prior project showed the apparent quantum advantage flips with `alpha`, so it
is treated as a first-class data-engineering / model-selection variable (axis 13).
"""
from __future__ import annotations

import numpy as np


def ridge_fit(X, y, alpha: float = 1e-6, bias: bool = True) -> np.ndarray:
    """Closed-form ridge weights for ||XW - y||^2 + alpha||W||^2.

    If `bias`, a column of ones is appended to X (the intercept is not penalised
    in the usual sense here but the extra column is cheap and improves fits). The
    returned W has shape (F[+1], targets).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if bias:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def ridge_predict(X, W, bias: bool = True) -> np.ndarray:
    """Apply ridge weights from `ridge_fit` (must match the `bias` used to fit)."""
    X = np.asarray(X, dtype=float)
    if bias:
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X @ W
