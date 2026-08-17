#!/usr/bin/env python3
"""RF-QRC + Chebyshev encoding: the combination the two prior diagnoses predict.

Why this pairing. The Chebyshev-encoding experiment showed the encoding does
deliver exact polynomial features (<Z_q> = T_{2(q+1)}(z), machine precision)
but that QRC's SEQUENTIAL RE-UPLOAD of a window destroys the identity, because
cos(a+b) mixes polynomials with sqrt(1-z^2) terms. RF-QRC has no window
re-upload: Phi is applied twice to the SAME current input, so composition only
doubles the angle and cos(2a) = T_{4m}(z) is still exactly Chebyshev. Memory
comes from the classical leaky filter AFTER measurement, which cannot scramble
the basis.

Second reason to expect a fit: RF-QRC's features are computational-basis
probabilities. For a product state |<b|psi>|^2 = prod_q p_q(b_q) with
p_q(0) = (1 + T_{2m_q}(z_q))/2, so the features are polynomial PRODUCTS across
the encoded variables -- exactly the cross-term structure Lorenz-63 needs
(its rule contains xz and xy).

Arms (all fresh, no cache):
  rfqrc_linear   RF-QRC as published (theta = pi*u)                  [reference]
  rfqrc_cheb     RF-QRC with per-qubit Chebyshev tower encoding      [the test]
  cheb_twin      CLASSICAL: explicit Chebyshev polys of the same
                 current state + the SAME leaky memory + ridge       [the control]
  elmF / ngrc    standing controls (univariate phase only)

Phases: (1) univariate one-step, 3 systems x 5 seeds, (eps,beta) tuned on
validation with the paper's grids; (2) multivariate Lorenz-63 closed-loop --
RF-QRC's own headline regime and the place this encoding should be strongest --
through the committed MV pilot's exact loop.

Outputs -> results/rfqrc_cheb/{onestep,mv_closedloop}.csv
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import lorenz_mv_pilot as P
import lorenz_mv_closedloop as CL
from qdepipe import metrics
from qdepipe.concentration import _scaled_series
from qdepipe.models import ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.readout import ridge_fit, ridge_predict

# reuse the faithful RF-QRC implementation + helpers (no phases executed)
exec(open("run_rfqrc.py").read()
     .split("# ================================================================= Phase 1")[0]
     .split("if __name__")[0])

OUT = "results/rfqrc_cheb"
os.makedirs(OUT, exist_ok=True)
EPS_CLIP = 1e-6


def to_pm1(x):
    return np.clip(2.0 * np.asarray(x, dtype=float) - 1.0,
                   -1.0 + EPS_CLIP, 1.0 - EPS_CLIP)


class RFQRCCheb(RFQRC):
    """RF-QRC whose feature map Phi is a per-qubit Chebyshev tower.

    Qubit q encodes input dimension (q mod d) at Chebyshev degree
    m_q = 1 + q//d, i.e. angle 2*m_q*arccos(z). Phi is applied twice exactly as
    in the paper; for the same input that doubles the angle, so the measured
    single-qubit terms remain exact Chebyshev polynomials T_{4 m_q}(z).
    """

    def __init__(self, n_qubits=NQ, eps=0.1, seed=0, input_dim=1):
        super().__init__(n_qubits=n_qubits, eps=eps, seed=seed, input_dim=input_dim)
        self.name = f"RF-QRC-cheb(n={n_qubits},eps={eps})"

    def _probs_row(self, u_t):
        u_t = np.atleast_1d(u_t)
        d = len(u_t)
        z = to_pm1(u_t)
        ac = np.arccos(z)
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for _ in range(2):                                   # Phi applied twice
            for q in range(self.n):
                m = 1 + q // d
                psi = _apply_ry(psi, q, 2.0 * m * ac[q % d], self.n)
            for q in range(self.n - 1):
                psi = _apply_cnot(psi, q, q + 1, self.n)
        for q in range(self.n):                              # V(alpha)
            psi = _apply_ry(psi, q, self.alpha[q], self.n)
        for q in range(self.n - 1):
            psi = _apply_cnot(psi, q, q + 1, self.n)
        return np.abs(psi) ** 2


class ChebTwin:
    """Classical control: Chebyshev polynomials of the CURRENT state (same
    degrees the quantum tower uses) plus all pairwise products -- i.e. the same
    function class, computed directly -- then the same leaky memory."""

    def __init__(self, input_dim=1, n_qubits=NQ, **_):
        self.d = input_dim
        self.degs = [1 + q // input_dim for q in range(n_qubits)]
        self.dims = [q % input_dim for q in range(n_qubits)]

    def raw(self, U):
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[:, None]
        Z = to_pm1(U)
        base = np.stack([np.cos(4.0 * m * np.arccos(Z[:, dim]))
                         for m, dim in zip(self.degs, self.dims)], axis=1)
        b = base.shape[1]
        prods = np.stack([base[:, i] * base[:, j]
                          for i in range(b) for j in range(i, b)], axis=1)
        return np.concatenate([np.ones((len(base), 1)), base, prods], axis=1)


# ============================================================ Phase 1
def phase1():
    rows = []
    for system in SYSTEMS:
        for seed in SEEDS:
            u, split = _scaled_series(system, NPTS)
            arms = {
                "rfqrc_linear": RFQRC(seed=seed).raw_probs(u),
                "rfqrc_cheb": RFQRCCheb(seed=seed).raw_probs(u),
                "cheb_twin": ChebTwin(input_dim=1).raw(u),
            }
            for label, Praw in arms.items():
                cands = []
                for eps in EPS_GRID:
                    X = leaky(Praw, eps)
                    for beta in BETA_GRID:
                        val, test = onestep_eval(X, u, split, beta)
                        cands.append((val, test, eps, beta))
                val, test, eps, beta = min(cands)
                rows.append({"system": system, "seed": seed, "model": label,
                             "eps": eps, "beta": beta, "val_nrmse": val,
                             "test_nrmse": test, "n_features": Praw.shape[1]})
                print(f"[P1] {system:12s} seed{seed} {label:13s} eps={eps} "
                      f"beta={beta:g} test={test:.4e}", flush=True)
            for label, mk in [
                ("elmF", lambda: ELM(ELMConfig(units=F, lookback=5, seed=seed))),
                ("ngrc", lambda: NGRC(NGRCConfig(k=5, degree=NGRC_DEGREE[system]))),
            ]:
                Xc = mk().featurize(u)
                val, test = onestep_eval(Xc, u, split, 1e-6)
                rows.append({"system": system, "seed": seed, "model": label,
                             "eps": "", "beta": 1e-6, "val_nrmse": val,
                             "test_nrmse": test, "n_features": Xc.shape[1]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "onestep.csv"), index=False)


# ============================================================ Phase 2
def phase2():
    state = P.lorenz_state()
    U, split = P.scale_split(state)
    a, b = split.train.stop, split.val.stop
    rows = []
    for seed in SEEDS:
        makers = {
            "rfqrc_linear": lambda s=seed: RFQRC(seed=s, input_dim=3),
            "rfqrc_cheb": lambda s=seed: RFQRCCheb(seed=s, input_dim=3),
            "cheb_twin": lambda s=seed: ChebTwin(input_dim=3),
        }
        for label, mk in makers.items():
            model = mk()
            raw = model.raw_probs(U) if hasattr(model, "raw_probs") else model.raw(U)
            tr, va = np.arange(WASHOUT, a - 1), np.arange(a, b - 1)
            cands = []
            for eps in EPS_GRID:
                X = leaky(raw, eps)
                for beta in BETA_GRID:
                    W = ridge_fit(X[tr], U[tr + 1], alpha=beta)
                    v = float(np.mean([metrics.nrmse(U[va + 1, j],
                                                     ridge_predict(X[va], W)[:, j])
                                       for j in range(3)]))
                    cands.append((v, eps, beta))
            v, eps, beta = min(cands)
            X = leaky(raw, eps)
            W = ridge_fit(X[tr], U[tr + 1], alpha=beta)
            m2 = mk()
            rawf = (lambda H: m2.raw_probs(H)) if hasattr(m2, "raw_probs") else (lambda H: m2.raw(H))

            def step(h, W=W, eps=eps, rawf=rawf):
                return ridge_predict(leaky(rawf(np.asarray(h)), eps)[-1:], W)[0]

            res, _, _ = CL.closed_loop_mv(step, U, split, n_steps=400)
            res.update({"model": label, "seed": seed, "eps": eps, "beta": beta,
                        "val_nrmse": v})
            rows.append(res)
            print(f"[P2] MV seed{seed} {label:13s} eps={eps} val={v:.3e} "
                  f"VPT={res['vpt_lyap']:.3f} div={res['diverged']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "mv_closedloop.csv"), index=False)
    print("\nMV medians:")
    print(df.groupby("model").agg(vpt=("vpt_lyap", "median"), div=("diverged", "mean"),
                                  val=("val_nrmse", "median")).to_string())


if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUT, "onestep.csv")):
        phase1()
    phase2()
    print("\ndone -> results/rfqrc_cheb/")
