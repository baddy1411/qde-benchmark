"""Statistical-significance tests for forecast comparisons.

The benchmark reports mean NRMSE ± std over seeds, which says nothing about
whether a *difference* between two models exceeds seed/sampling noise. This module
adds the two standard tools for that:

  * Diebold–Mariano (DM)  — pairwise test of equal predictive accuracy, operating
    on the per-step loss differential with a HAC (Newey–West) variance and the
    Harvey–Leybourne–Newbold small-sample correction.
  * Model Confidence Set (MCS, Hansen–Lunde–Nason 2011) — the set of models that
    are statistically indistinguishable from the best, via a block-bootstrap of
    the range statistic.

Both operate on **forecast errors / losses**, not on the raw target — and the
tests make no normality assumption about the errors themselves (DM tests the mean
of the loss *differential*, whose sampling distribution is asymptotically normal
by a CLT for dependent series; this is documented per function).

Everything here is pure numpy + a provided RNG seed, so results are deterministic.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

LossName = str  # "se" (squared error) | "ae" (absolute error)


# ---------------------------------------------------------------------------
# loss functions on forecast errors
# ---------------------------------------------------------------------------
def _loss(err: np.ndarray, loss: LossName) -> np.ndarray:
    """Map a per-step forecast-error series to a per-step loss series."""
    err = np.asarray(err, dtype=float)
    if loss == "se":
        return err ** 2
    if loss == "ae":
        return np.abs(err)
    raise ValueError(f"unknown loss {loss!r}; use 'se' or 'ae'")


def errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Forecast errors e_t = y_pred - y_true (sign irrelevant for se/ae losses)."""
    return np.asarray(y_pred, dtype=float).ravel() - np.asarray(y_true, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Diebold–Mariano
# ---------------------------------------------------------------------------
def _newey_west_var(d: np.ndarray, lag: int) -> float:
    """HAC (Bartlett-kernel / Newey–West) estimate of Var(mean(d)).

    Returns the variance OF THE MEAN of d (i.e. long-run variance / T), so the DM
    statistic is mean(d) / sqrt(that).
    """
    d = np.asarray(d, dtype=float)
    T = len(d)
    dm = d - d.mean()
    gamma0 = np.dot(dm, dm) / T
    s = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)             # Bartlett weight
        gamma_k = np.dot(dm[k:], dm[:-k]) / T
        s += 2.0 * w * gamma_k
    return s / T                            # variance of the mean


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1,
                    loss: LossName = "se") -> tuple[float, float]:
    """Diebold–Mariano test of equal predictive accuracy of two forecasts.

    Parameters
    ----------
    e1, e2 : aligned per-step forecast-error series of the two models on the SAME
        test set. (Pass `errors(y_true, y_pred)` for each.)
    h : forecast horizon. The loss differential of an optimal h-step forecast is
        MA(h-1)-correlated, so the HAC variance uses lag = h-1.
    loss : "se" (squared error, default) or "ae".

    Returns
    -------
    (dm_stat, p_value) : the HLN small-sample-corrected DM statistic and its
        two-sided p-value under a Student-t(T-1) reference (HLN recommendation).

    Notes
    -----
    Null hypothesis: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t). A positive statistic
    means model 1 has the LARGER loss (i.e. model 2 is better). No normality of the
    errors is assumed — the test is on the mean of the loss differential, which is
    asymptotically normal by a CLT for dependent stationary series; the HLN
    correction + t-reference handle finite samples.
    """
    from scipy import stats

    l1, l2 = _loss(e1, loss), _loss(e2, loss)
    d = np.asarray(l1 - l2, dtype=float)
    T = len(d)
    if T < 2:
        return float("nan"), float("nan")
    # Identical forecasts leave no loss differential and no test to run. This
    # check MUST be relative to the loss scale. It used to be `np.allclose(d, 0)`,
    # which against a scalar reduces to |d| <= atol = 1e-8 -- an absolute floor on
    # a quantity whose magnitude it never looks at. With squared-error loss, two
    # models at NRMSE 4e-5 on a [0,1]-scaled series have losses ~1e-10, so every
    # differential sat two orders of magnitude below the floor and the function
    # returned "no significant difference" WITHOUT EVER RUNNING THE TEST. It
    # failed hardest exactly where the models were most accurate.
    scale = max(float(np.mean(np.abs(l1))), float(np.mean(np.abs(l2))))
    if scale == 0.0 or float(np.max(np.abs(d))) <= 1e-12 * scale:
        return 0.0, 1.0

    lag = max(0, int(h) - 1)
    var_mean = _newey_west_var(d, lag)
    if var_mean <= 0:
        return float("nan"), float("nan")

    dm = d.mean() / np.sqrt(var_mean)
    # Harvey–Leybourne–Newbold (1997) small-sample correction
    corr = np.sqrt((T + 1 - 2 * lag + lag * (lag - 1) / T) / T)
    dm_hln = dm * corr
    p = 2.0 * stats.t.cdf(-abs(dm_hln), df=T - 1)
    return float(dm_hln), float(p)


def dm_matrix(forecasts: dict[str, tuple[np.ndarray, np.ndarray]], h: int = 1,
              loss: LossName = "se") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pairwise DM over a set of models.

    forecasts : {model_name: (y_true, y_pred)} all on the same test set.
    Returns (stat, pval, names): square matrices with stat[i,j] = DM(model_i,
    model_j) (positive ⇒ j better than i) and pval[i,j] the two-sided p-value;
    diagonal is (0, 1).
    """
    names = list(forecasts)
    errs = {m: errors(*forecasts[m]) for m in names}
    n = len(names)
    stat = np.zeros((n, n))
    pval = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s, p = diebold_mariano(errs[names[i]], errs[names[j]], h=h, loss=loss)
            stat[i, j], pval[i, j] = s, p
    return stat, pval, names


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen–Lunde–Nason 2011), range statistic + block bootstrap
# ---------------------------------------------------------------------------
def _block_bootstrap_indices(T: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block-bootstrap index vector of length T."""
    n_blocks = int(np.ceil(T / block_length))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block_length) % T) for s in starts])
    return idx[:T]


def model_confidence_set(losses: np.ndarray, names: Sequence[str] | None = None,
                         alpha: Iterable[float] = (0.10, 0.25),
                         n_boot: int = 1000, block_length: int = 10,
                         seed: int = 0) -> dict:
    """Model Confidence Set via the range statistic T_R and a block bootstrap.

    Parameters
    ----------
    losses : (n_models, T) per-step loss matrix (e.g. squared errors) on a common
        test set. Lower loss = better.
    names : model labels (default "m0".. ).
    alpha : confidence levels; the MCS at level a is {models with mcs_pvalue >= a}.
    n_boot, block_length : bootstrap controls (stationary/circular block bootstrap).
    seed : RNG seed for reproducibility.

    Returns
    -------
    dict with:
        'names'            : ordered list (elimination order, worst eliminated first)
        'mcs_pvalue'       : {name: MCS p-value}
        'in_set'           : {alpha: [names in the set at that level]}

    Method (range statistic): iteratively, for the surviving set M, form the matrix
    of mean loss differences d_ij = mean_t(L_i - L_j); the test statistic is
    T_R = max_{i,j in M} |d_ij| / sqrt(Var*(d_ij)), with Var* from the bootstrap.
    Eliminate the worst model (max average standardized excess loss) until the null
    of equal predictive ability is not rejected. The MCS p-value of an eliminated
    model is the running max of the rejection p-values up to its elimination.
    """
    losses = np.asarray(losses, dtype=float)
    M0, T = losses.shape
    names = list(names) if names is not None else [f"m{i}" for i in range(M0)]
    rng = np.random.default_rng(seed)

    # Precompute B bootstrap resamples of the per-step losses (shared across rounds).
    boot_idx = [_block_bootstrap_indices(T, block_length, rng) for _ in range(n_boot)]
    boot_means = np.stack([losses[:, idx].mean(axis=1) for idx in boot_idx], axis=0)  # (B, M0)
    full_means = losses.mean(axis=1)                                                   # (M0,)

    alive = list(range(M0))
    elim_order: list[int] = []
    mcs_p: dict[int, float] = {}
    running = 0.0

    while len(alive) > 1:
        a = np.array(alive)
        means = full_means[a]
        bmeans = boot_means[:, a]                       # (B, k)
        # bootstrap variance of pairwise mean-difference d_ij = mean_i - mean_j
        # var*(d_ij) = Var_b[(bmean_i - full_i) - (bmean_j - full_j)]
        centered = bmeans - means[None, :]              # (B, k)
        # standardized range statistic over all pairs
        k = len(a)
        t_obs = 0.0
        # per-model statistic: standardized excess over the set average (for elimination)
        # t_i = (mean_i - mean_bar) / sd*(mean_i - mean_bar)
        mean_bar = means.mean()
        dev = means - mean_bar
        boot_dev = centered - centered.mean(axis=1, keepdims=True)   # (B, k)
        sd = boot_dev.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.inf)
        t_i = dev / sd
        # range statistic T_R and its bootstrap distribution
        diff = means[:, None] - means[None, :]                       # (k,k) observed d_ij
        # bdiff is ALREADY the bootstrap deviation of d_ij from its observed value
        # (centered = bmean - full_mean), so it is the null-centered statistic.
        bdiff = centered[:, :, None] - centered[:, None, :]          # (B,k,k)
        sd_ij = bdiff.std(axis=0, ddof=1)                            # bootstrap SE of d_ij
        with np.errstate(divide="ignore", invalid="ignore"):
            T_R = np.nanmax(np.abs(diff) / np.where(sd_ij > 0, sd_ij, np.inf))
            boot_TR = np.nanmax(np.abs(bdiff) /
                                np.where(sd_ij[None] > 0, sd_ij[None], np.inf), axis=(1, 2))
        p = float(np.mean(boot_TR >= T_R))
        running = max(running, p)

        worst = a[int(np.argmax(t_i))]      # highest standardized excess loss
        mcs_p[worst] = running
        elim_order.append(worst)
        alive.remove(worst)

    # the survivor is in the set at every level
    last = alive[0]
    mcs_p[last] = 1.0
    elim_order.append(last)

    pval = {names[i]: mcs_p[i] for i in range(M0)}
    in_set = {al: [names[i] for i in range(M0) if mcs_p[i] >= al] for al in alpha}
    order = [names[i] for i in elim_order]
    return {"names": order, "mcs_pvalue": pval, "in_set": in_set}


# ---------------------------------------------------------------------------
# paired bootstrap CI over seeds (used by the scaling proof,
# leaky study and IC-robustness reporting)
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, stat: str = "median",
                        n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> dict:
    """Bootstrap CI on stat(a - b) for PAIRED per-seed scores (same seeds, same
    data). a, b are equal-length arrays (e.g. per-seed NRMSE of two models).

    Resamples seed indices with replacement, so it respects the pairing and
    makes no distributional assumption. Returns the point estimate, the
    (1-alpha) percentile CI, and whether the CI excludes zero — the review's
    "bootstrap confidence intervals on the difference between methods".
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal length")
    fn = {"median": np.median, "mean": np.mean}[stat]
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = fn(d[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"stat": stat, "estimate": float(fn(d)), "ci_lo": float(lo),
            "ci_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_pairs": int(len(d)), "n_boot": int(n_boot)}


def median_iqr(x: np.ndarray) -> dict:
    """Median and interquartile range — the review-preferred summary over seeds."""
    x = np.asarray(x, dtype=float).ravel()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return {"median": float(q2), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "n": int(len(x))}
