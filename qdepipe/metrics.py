"""Evaluation metrics — identical scoring for every model.

1-step accuracy (NRMSE), linear memory capacity (Jaeger 2002), feature rank, and
the closed-loop climate battery (valid prediction time, log-spectral MSE,
Wasserstein-1). Centralising these is what makes the classical-vs-quantum
comparison fair.
"""
from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def max_error(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.max(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination. 1=perfect, 0=predicts the mean, <0=worse."""
    yt, yp = _pair(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else -np.inf


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0,200]%. Safe near zero (Hénon x crosses 0)."""
    yt, yp = _pair(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(2 * np.abs(yt - yp) / denom))


def mape(y_true, y_pred, eps: float = 1e-3) -> float:
    """Mean absolute percentage error (%), guarded against |y|<eps (reported but
    unreliable on signals that cross zero — prefer sMAPE / NRMSE here)."""
    yt, yp = _pair(y_true, y_pred)
    mask = np.abs(yt) >= eps
    if not np.any(mask):
        return np.inf
    return float(100.0 * np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def nrmse(y_true, y_pred) -> float:
    """Normalised RMSE = RMSE / std(y_true). Lower is better; scale-robust.
    The primary 1-step metric throughout the thesis."""
    yt, yp = _pair(y_true, y_pred)
    sd = np.std(yt)
    return float(rmse(yt, yp) / sd) if sd > 0 else np.inf


def nrmse_vector(Y_true, Y_pred):
    """Multivariate NRMSE for vector targets Y (n, d). Returns (aggregate,
    per_component). The aggregate normalises the total squared error by the total
    variance about the per-component mean (so all components contribute on a common
    scale); per_component is the scalar NRMSE of each column. Used by the
    multivariate Lorenz pilot only; the scalar `nrmse` above is unchanged."""
    Yt = np.asarray(Y_true, dtype=float)
    Yp = np.asarray(Y_pred, dtype=float)
    if Yt.ndim == 1:
        Yt, Yp = Yt[:, None], Yp[:, None]
    num = np.sum((Yp - Yt) ** 2)
    den = np.sum((Yt - Yt.mean(axis=0, keepdims=True)) ** 2)
    agg = float(np.sqrt(num / den)) if den > 0 else np.inf
    per = [nrmse(Yt[:, c], Yp[:, c]) for c in range(Yt.shape[1])]
    return agg, per


def regression_metrics(y_true, y_pred) -> dict:
    """All point-forecast metrics in one dict — used by every experiment table so
    findings can be read on whichever metric the reader trusts."""
    return {
        "nrmse": nrmse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }


def _pair(y_true, y_pred):
    return (np.asarray(y_true, dtype=float).ravel(),
            np.asarray(y_pred, dtype=float).ravel())


def valid_prediction_time(y_true, y_pred, threshold: float = 0.4) -> int:
    """First step where normalised error exceeds `threshold` (Köster VPT)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sd = np.std(y_true) or 1.0
    err = np.abs(y_true - y_pred) / sd
    over = np.where(err > threshold)[0]
    return int(over[0]) if len(over) else len(y_true)


def valid_prediction_time_mv(Y_true, Y_pred, threshold: float = 0.4) -> int:
    """Multivariate VPT: first step where the normalised state error exceeds
    `threshold`. The per-step error ||y_true - y_pred|| is normalised by the RMS
    norm of the centred true trajectory --- the vector generalisation of the
    univariate `valid_prediction_time` (which divides by std). Same threshold
    convention. Used by the closed-loop multivariate pilot only."""
    Yt = np.asarray(Y_true, dtype=float)
    Yp = np.asarray(Y_pred, dtype=float)
    if Yt.ndim == 1:
        Yt, Yp = Yt[:, None], Yp[:, None]
    center = Yt.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum((Yt - center) ** 2, axis=1)))) or 1.0
    err = np.sqrt(np.sum((Yt - Yp) ** 2, axis=1)) / rms
    over = np.where(err > threshold)[0]
    return int(over[0]) if len(over) else len(Yt)


def spectral_mse(y_true, y_pred) -> float:
    """MSE between log power spectra (frequency-domain climate match)."""

    def logpsd(z):
        z = np.asarray(z, dtype=float)
        ps = np.abs(np.fft.rfft(z - np.mean(z))) ** 2
        return np.log10(ps + 1e-12)

    a, b = logpsd(y_true), logpsd(y_pred)
    m = min(len(a), len(b))
    return float(np.mean((a[:m] - b[:m]) ** 2))


def wasserstein1(y_true, y_pred) -> float:
    """Wasserstein-1 between invariant-density histograms. Large (>~10) usually
    flags a numerically exploded closed-loop trajectory."""
    from scipy.stats import wasserstein_distance

    yp = np.asarray(y_pred, dtype=float)
    yp = yp[np.isfinite(yp)]
    if len(yp) == 0:
        return np.inf
    return float(wasserstein_distance(np.asarray(y_true, dtype=float).ravel(), yp.ravel()))


def feature_rank(states, tol=None) -> int:
    """Numerical rank of the feature matrix — the readout's usable dimension."""
    return int(np.linalg.matrix_rank(np.asarray(states, dtype=float), tol=tol))


def memory_capacity(states, inputs, k_max: int = 20, alpha: float = 1e-6):
    """Linear memory capacity (Jaeger): sum_k r^2 between target u(n-k) and the
    ridge reconstruction from reservoir states. Returns (total, per_k)."""
    from .readout import ridge_fit

    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float).ravel()
    per_k = []
    for k in range(1, k_max + 1):
        Xk, yk = states[k:], inputs[:-k]
        m = min(len(Xk), len(yk))
        Xk, yk = Xk[:m], yk[:m]
        W = ridge_fit(Xk, yk, alpha=alpha, bias=False)
        pred = Xk @ W
        c = np.corrcoef(pred.ravel(), yk)[0, 1]
        per_k.append(max(0.0, c ** 2) if np.isfinite(c) else 0.0)
    return float(np.sum(per_k)), np.array(per_k)
