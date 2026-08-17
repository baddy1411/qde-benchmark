#!/usr/bin/env python3
"""Full RF-QRC protocol (Ahmed, Tennie & Magri, Phys. Rev. Research 6, 043082):
recurrence-free quantum reservoir computing, implemented faithfully to the
published text and evaluated through this project's leakage-safe conventions.

Construction (per the paper's text):
  |psi(t)> = V(alpha) . Phi(u_t) . Phi(u_t) |0>^n        (NO recurrence)
  Phi(u)   = single-qubit RY rotations from the CURRENT input + CNOT chain
             ("fully entangled" feature map), applied twice
  V(alpha) = random parameterised circuit, n parameters ~ U[0, 4*pi] at a
             fixed seed (RY per qubit + CNOT chain)
  features = computational-basis probabilities |<b|psi>|^2  (2^n values)
  memory   = classical leaky integration r_t = (1-eps) r_{t-1} + eps rhat_t
  read-out = ridge (Tikhonov beta), trained open-loop; prediction closed-loop
  tuned    = eps in {0.05, 0.1, 0.2, 0.3} (their [0.05, 0.3] range),
             beta in {1e-6, 1e-9, 1e-12} (their values), on the VALIDATION set
  exact statevector (their main results are noise-free); they state 1e4-1e5
             shots would be needed physically -> tested in the shots phase.

Disclosed implementation choices where the paper's Appendix A figures are not
in the text: RY as the rotation gate, a linear CNOT chain as the entangler,
one rotation per qubit in V(alpha), encode angle = pi * u (inputs are already
min-max scaled to [0,1] by the pipeline; multivariate dims assigned to qubits
cyclically). n_qubits = 8 (their 8-11), F = 2^8 = 256 features.

Phases:
  1  univariate one-step battery (henon/lorenz/mackeyglass), 5 alpha-seeds,
     (eps, beta) tuned on validation; controls ELM(F=256) and NG-RC under the
     IDENTICAL manual pipeline (train-only scaling, same splits, same ridge).
  2  univariate closed-loop via the committed run_closed_loop harness,
     VPT thresholds 0.4 (project convention) and 0.5 (their convention).
  3  multivariate Lorenz-63 closed-loop -- their headline task -- through the
     MV pilot's symmetric feedback-loop conventions (anchor at test start,
     horizon 400, freeze-on-non-finite, divergence at 10x true range).
  4  finite shots: multinomial sampling of the basis probabilities at
     S in {8192, 1e4, 1e5}, with and without SVD-truncation denoising
     (rank chosen on validation), beta re-tuned under noise.

Everything fresh (no feature cache). Outputs -> results/rfqrc/*.csv.
Committed artifacts are not modified.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe import metrics
from qdepipe.concentration import _scaled_series
from qdepipe.experiment import ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.closedloop import run_closed_loop
from qdepipe.models import ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.base import ReservoirModel
from qdepipe.readout import ridge_fit, ridge_predict

OUT = "results/rfqrc"
os.makedirs(OUT, exist_ok=True)

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
NGRC_DEGREE = {"henon": 2, "lorenz": 3, "mackeyglass": 2}
SEEDS = [0, 1, 2, 3, 4]
NPTS = 3000
WASHOUT = 100
NQ = 8
F = 2 ** NQ
EPS_GRID = [0.05, 0.1, 0.2, 0.3]
BETA_GRID = [1e-6, 1e-9, 1e-12]
ENC = np.pi                       # angle scale on [0,1]-scaled inputs


# ---------------------------------------------------------------- statevector
def _apply_ry(psi, q, theta, n):
    """RY on qubit q of an n-qubit state (2^n complex), via reshape."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    psi = psi.reshape((2 ** q, 2, 2 ** (n - q - 1)))
    a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
    psi[:, 0, :] = c * a - s * b
    psi[:, 1, :] = s * a + c * b
    return psi.reshape(-1)


def _apply_cnot(psi, ctrl, targ, n):
    psi = psi.reshape([2] * n)
    idx_c1 = [slice(None)] * n
    idx_c1[ctrl] = 1
    sub = psi[tuple(idx_c1)]
    psi[tuple(idx_c1)] = np.flip(sub, axis=targ - (1 if targ > ctrl else 0))
    return psi.reshape(-1)


class RFQRC(ReservoirModel):
    """Recurrence-free QRC per Ahmed et al.; leaky memory -> unbounded window."""
    memory_window = None

    def __init__(self, n_qubits=NQ, eps=0.1, enc_scale=ENC, seed=0, input_dim=1):
        self.n = n_qubits
        self.eps = eps
        self.enc = enc_scale
        self.seed = seed
        self.d = input_dim
        rng = np.random.default_rng(seed)
        self.alpha = rng.uniform(0, 4 * np.pi, n_qubits)
        self.name = f"RF-QRC(n={n_qubits},eps={eps})"

    @property
    def n_features(self):
        return 2 ** self.n

    def _probs_row(self, u_t):
        u_t = np.atleast_1d(u_t)
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for _ in range(2):                                   # Phi applied twice
            for q in range(self.n):
                psi = _apply_ry(psi, q, self.enc * u_t[q % len(u_t)], self.n)
            for q in range(self.n - 1):
                psi = _apply_cnot(psi, q, q + 1, self.n)
        for q in range(self.n):                              # V(alpha)
            psi = _apply_ry(psi, q, self.alpha[q], self.n)
        for q in range(self.n - 1):
            psi = _apply_cnot(psi, q, q + 1, self.n)
        return np.abs(psi) ** 2

    def raw_probs(self, u):
        u = np.asarray(u, dtype=float)
        rows = u if u.ndim == 2 else u[:, None]
        return np.stack([self._probs_row(r) for r in rows])

    def featurize(self, u):
        return leaky(self.raw_probs(u), self.eps)


def leaky(P, eps):
    out = np.empty_like(P)
    r = np.zeros(P.shape[1])
    for t in range(P.shape[0]):
        r = (1 - eps) * r + eps * P[t]
        out[t] = r
    return out


# ------------------------------------------------------ manual one-step frame
def onestep_eval(X, u, split, beta):
    """Fit rows [washout, train_end-1) -> target u[t+1]; return val/test NRMSE."""
    a, b, n = split.train.stop, split.val.stop, len(u)
    tr = np.arange(WASHOUT, a - 1)
    va = np.arange(a, b - 1)
    te = np.arange(b, n - 1)
    W = ridge_fit(X[tr], u[tr + 1], alpha=beta)
    val = metrics.nrmse(u[va + 1], ridge_predict(X[va], W))
    test = metrics.nrmse(u[te + 1], ridge_predict(X[te], W))
    return val, test


# ================================================================= Phase 1
def phase1():
    rows, best = [], {}
    for system in SYSTEMS:
        for seed in SEEDS:
            u, split = _scaled_series(system, NPTS)
            model = RFQRC(seed=seed)
            P = model.raw_probs(u)
            cands = []
            for eps in EPS_GRID:
                X = leaky(P, eps)
                for beta in BETA_GRID:
                    val, test = onestep_eval(X, u, split, beta)
                    cands.append((val, test, eps, beta))
                    rows.append({"system": system, "seed": seed, "model": "rfqrc",
                                 "eps": eps, "beta": beta, "val_nrmse": val,
                                 "test_nrmse": test, "n_features": F})
            val, test, eps, beta = min(cands)
            best[(system, seed)] = (eps, beta)
            print(f"[P1] {system:12s} seed{seed} RF-QRC best eps={eps} beta={beta:g} "
                  f"val={val:.4g} test={test:.4g}", flush=True)
            # controls, identical frame, standard beta
            for label, mk in [
                ("elmF", lambda: ELM(ELMConfig(units=F, lookback=5, seed=seed))),
                ("ngrc", lambda: NGRC(NGRCConfig(k=5, degree=NGRC_DEGREE[system]))),
            ]:
                Xc = mk().featurize(u)
                val, test = onestep_eval(Xc, u, split, 1e-6)
                rows.append({"system": system, "seed": seed, "model": label,
                             "eps": "", "beta": 1e-6, "val_nrmse": val,
                             "test_nrmse": test, "n_features": Xc.shape[1]})
                print(f"[P1] {system:12s} seed{seed} {label:5s} test={test:.4g}",
                      flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "onestep.csv"), index=False)
    return best


# ================================================================= Phase 2
def phase2(best):
    rows = []
    for system in SYSTEMS:
        for seed in [0, 1, 2]:
            eps, beta = best[(system, seed)]
            for thr in [0.4, 0.5]:
                for label, fc in [
                    ("rfqrc", ReservoirForecaster(
                        RFQRC(eps=eps, seed=seed), f"RF-QRC", store=None)),
                    ("elmF", ReservoirForecaster(
                        ELM(ELMConfig(units=F, lookback=5, seed=seed)),
                        f"ELM(F={F})", store=None)),
                    ("ngrc", ReservoirForecaster(
                        NGRC(NGRCConfig(k=5, degree=NGRC_DEGREE[system])),
                        "NG-RC", store=None)),
                ]:
                    cfg = ExperimentConfig(system=system, n_points=NPTS, seed=seed,
                                           lookback=5, washout=WASHOUT,
                                           alpha=(beta if label == "rfqrc" else 1e-6))
                    cl = run_closed_loop(fc, cfg, n_steps=300, vpt_threshold=thr)
                    rows.append({"system": system, "seed": seed, "model": label,
                                 "vpt_threshold": thr, "vpt_lyap": cl.vpt_lyap,
                                 "spectral_mse": cl.spectral_mse,
                                 "wasserstein": cl.wasserstein,
                                 "diverged": cl.diverged})
                    print(f"[P2] {system:12s} seed{seed} thr={thr} {label:5s} "
                          f"VPT={cl.vpt_lyap:.3f} div={cl.diverged}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "closedloop.csv"), index=False)


# ================================================================= Phase 3
def phase3():
    """Multivariate Lorenz-63 closed-loop -- Ahmed et al.'s headline task --
    with the MV pilot's symmetric loop conventions."""
    import lorenz_mv_pilot as P

    state = P.lorenz_state()
    U, split = P.scale_split(state)          # U: (T,3) train-only scaled
    a, b = split.train.stop, split.val.stop
    horizon = 400
    rows = []
    for seed in SEEDS:
        model = RFQRC(seed=seed, input_dim=3)
        Praw = model.raw_probs(U)
        # tune (eps, beta) on validation, multivariate one-step
        cands = []
        for eps in EPS_GRID:
            X = leaky(Praw, eps)
            tr = np.arange(WASHOUT, a - 1)
            va = np.arange(a, b - 1)
            for beta in BETA_GRID:
                W = ridge_fit(X[tr], U[tr + 1], alpha=beta)
                v = float(np.mean([metrics.nrmse(U[va + 1, j],
                                                 ridge_predict(X[va], W)[:, j])
                                   for j in range(3)]))
                cands.append((v, eps, beta))
        v, eps, beta = min(cands)
        X = leaky(Praw, eps)
        tr = np.arange(WASHOUT, a - 1)
        W = ridge_fit(X[tr], U[tr + 1], alpha=beta)
        model_cl = RFQRC(seed=seed, eps=eps, input_dim=3)

        # symmetric feedback loop (identical conventions to the MV pilot)
        anchor = split.test.start
        horizon = min(horizon, len(U) - anchor)
        hist = U[:anchor].copy()
        true = U[anchor:anchor + horizon]
        preds = []
        for _ in range(horizon):
            feats = model_cl.featurize(hist)
            nxt = ridge_predict(feats[-1:], W)[0]
            if not np.all(np.isfinite(nxt)):
                nxt = hist[-1]
            preds.append(nxt)
            hist = np.vstack([hist, nxt])
        preds = np.asarray(preds)
        err = np.linalg.norm(preds - true, axis=1) / (
            np.linalg.norm(true, axis=1).mean() + 1e-12)
        le = P.LYAP_STEP if hasattr(P, "LYAP_STEP") else None
        # VPT in steps at both thresholds; Lyapunov conversion via pilot value
        from qdepipe.data import lyapunov_step
        lam = lyapunov_step("lorenz", state[:, 0])
        ly_time = 1.0 / lam if lam and lam > 0 else np.nan
        for thr in [0.4, 0.5]:
            below = np.where(err > thr)[0]
            vpt_steps = int(below[0]) if len(below) else horizon
            rows.append({"seed": seed, "eps": eps, "beta": beta,
                         "vpt_threshold": thr, "vpt_steps": vpt_steps,
                         "vpt_lyap": vpt_steps / ly_time if ly_time else np.nan,
                         "diverged": bool(np.max(np.abs(preds)) >
                                          10 * np.max(np.abs(true)))})
        print(f"[P3] MV seed{seed} eps={eps} beta={beta:g} "
              f"VPT(0.4)={rows[-2]['vpt_lyap']:.3f} div={rows[-1]['diverged']}",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mv_closedloop.csv"), index=False)


# ================================================================= Phase 4
def phase4(best):
    rng = np.random.default_rng(123)
    rows = []
    RANKS = [8, 16, 32, 64, None]
    for system in ["henon", "lorenz"]:
        seed = 0
        eps, _ = best[(system, seed)]
        u, split = _scaled_series(system, NPTS)
        P = RFQRC(seed=seed).raw_probs(u)
        for S in [8192, 10_000, 100_000]:
            Phat = np.stack([rng.multinomial(S, p / p.sum()) / S for p in P])
            X = leaky(Phat, eps)
            for rank in RANKS:
                if rank is None:
                    Xr = X
                else:                                   # SVD-truncation denoise
                    a = split.train.stop
                    U_, s_, Vt = np.linalg.svd(X[WASHOUT:a - 1], full_matrices=False)
                    proj = Vt[:rank].T @ Vt[:rank]
                    Xr = X @ proj
                cands = []
                for beta in [1e-6, 1e-4, 1e-2]:
                    val, test = onestep_eval(Xr, u, split, beta)
                    cands.append((val, test, beta))
                val, test, beta = min(cands)
                rows.append({"system": system, "shots": S,
                             "rank": rank if rank else "full", "eps": eps,
                             "beta": beta, "val_nrmse": val, "test_nrmse": test})
                print(f"[P4] {system:8s} S={S:6d} rank={str(rank):4s} "
                      f"test={test:.4g}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "shots.csv"), index=False)


if __name__ == "__main__":
    one = os.path.join(OUT, "onestep.csv")
    if os.path.exists(one):                       # resume: rebuild tuning choices
        d = pd.read_csv(one)
        r = d[d.model == "rfqrc"]
        best = {}
        for (system, seed), g in r.groupby(["system", "seed"]):
            row = g.loc[g.val_nrmse.idxmin()]
            best[(system, int(seed))] = (float(row.eps), float(row.beta))
        print(f"resume: tuning restored for {len(best)} cells", flush=True)
    else:
        best = phase1()
    if not os.path.exists(os.path.join(OUT, "closedloop.csv")):
        phase2(best)
    phase3()
    phase4(best)
    print("\nRF-QRC protocol complete -> results/rfqrc/")
