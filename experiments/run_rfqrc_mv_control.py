#!/usr/bin/env python3
"""RF-QRC on the multivariate Lorenz headline task, evaluated through the MV
pilot's EXACT loop (closed_loop_mv: same error metric, VPT threshold 0.4,
divergence rule, freeze rule) so the numbers are directly comparable with the
committed lorenz_mv_cl_vpt.csv rows -- plus the decisive classical twin:

  RF-QRC       = random CIRCUIT features of the current state + leaky memory
  classical twin = random PROJECTION features of the current state + leaky
                   memory (tanh(W_r u_t + b), 256 features, no window)

Both tuned identically ((eps, beta) on validation, their grids), 5 seeds.
If the twin matches RF-QRC, the architecture (current-state encoding + leaky
+ closed loop) explains the performance, not quantumness.
Output: results/rfqrc/mv_pilot_convention.csv
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
from qdepipe.readout import ridge_fit, ridge_predict

exec(open("run_rfqrc.py").read().split("# ================================================================= Phase 1")[0]
     .split('if __name__')[0])   # reuse RFQRC, leaky, grids (no phases run)

OUT = "results/rfqrc"
os.makedirs(OUT, exist_ok=True)
WASHOUT = 100


class ClassicalTwin:
    """tanh(W_r u_t + b): random projection of the CURRENT state only."""

    def __init__(self, n_feat=256, seed=0, input_dim=3):
        rng = np.random.default_rng(1000 + seed)
        self.Wr = rng.normal(0, 1.0, (n_feat, input_dim))
        self.b = rng.uniform(-1, 1, n_feat)
        self.eps = 0.1

    def raw(self, U):
        return np.tanh(np.asarray(U) @ self.Wr.T + self.b)

    def featurize(self, U):
        return leaky(self.raw(U), self.eps)


def tune_and_loop(label, raw_of, make_featurizer, U, split, seed):
    a, b = split.train.stop, split.val.stop
    Praw = raw_of(U)
    tr = np.arange(WASHOUT, a - 1)
    va = np.arange(a, b - 1)
    cands = []
    for eps in EPS_GRID:
        X = leaky(Praw, eps)
        for beta in BETA_GRID:
            W = ridge_fit(X[tr], U[tr + 1], alpha=beta)
            v = float(np.mean([metrics.nrmse(U[va + 1, j],
                                             ridge_predict(X[va], W)[:, j])
                               for j in range(3)]))
            cands.append((v, eps, beta))
    v, eps, beta = min(cands)
    X = leaky(Praw, eps)
    W = ridge_fit(X[tr], U[tr + 1], alpha=beta)
    feat = make_featurizer(eps)

    def step(h):
        return ridge_predict(feat(np.asarray(h))[-1:], W)[0]

    res, _, _ = CL.closed_loop_mv(step, U, split, n_steps=400)
    res.update({"model": label, "seed": seed, "eps": eps, "beta": beta,
                "val_nrmse": v})
    print(f"[MV] {label:14s} seed{seed} eps={eps} beta={beta:g} "
          f"VPT={res['vpt_lyap']:.3f} div={res['diverged']}", flush=True)
    return res


def main():
    state = P.lorenz_state()
    U, split = P.scale_split(state)
    rows = []
    for seed in [0, 1, 2, 3, 4]:
        q = RFQRC(seed=seed, input_dim=3)
        rows.append(tune_and_loop(
            "rfqrc", q.raw_probs,
            lambda eps, q=q: (lambda H: leaky(q.raw_probs(H), eps)),
            U, split, seed))
        t = ClassicalTwin(seed=seed)
        rows.append(tune_and_loop(
            "classical_twin", t.raw,
            lambda eps, t=t: (lambda H: leaky(t.raw(H), eps)),
            U, split, seed))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "mv_pilot_convention.csv"), index=False)
    print("\nmedians:")
    print(df.groupby("model").agg(vpt=("vpt_lyap", "median"),
                                  div=("diverged", "mean"),
                                  spec=("spectral_mse", "median"),
                                  w1=("wasserstein", "median")).to_string())


if __name__ == "__main__":
    main()
