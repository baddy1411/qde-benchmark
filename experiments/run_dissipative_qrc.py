#!/usr/bin/env python3
"""Dissipative quantum-memory reservoir (Fujii-Nakajima style): the one family
this project never evaluated, and the literature's strongest remaining
temporal-processing case for QRC.

Architecture (genuinely recurrent -- no window, no classical leaky filter):
  state:    density matrix rho_t carried ACROSS time steps
  inject:   rho <- |psi(u_t)><psi(u_t)| (x) Tr_0[rho]   (erase-and-write on
            qubit 0; the remaining qubits keep entangled memory of the past)
  evolve:   U_sub = exp(-i H tau/V), H = random transverse-field Ising
            (XX couplings ~ N(0,1)/sqrt(n) + Z fields, seeded); V sub-steps
  dissipate: per-qubit amplitude damping, rate gamma per step (the memory/
            echo-state knob; gamma=0 is the pure Fujii-Nakajima limit)
  measure:  <Z_i>, <X_i>, <Y_i> after each sub-step  ->  F = V * 3n features

Grids (validation-tuned, rescue-experiment conventions): tau in {1,4,10},
gamma in {0,0.02,0.05,0.1,0.2}, ridge beta in {1e-6,1e-9,1e-12}. Controls at
matched F: ELM(F), ESN(F units) as the classical-recurrent twin, NG-RC
reference. npts=1500, washout 100, 5 seeds. Exact expectations.

Phases:
  0  sanity gates (printed, must pass before anything runs):
     trace preservation, echo-state convergence from different initial
     states, and MEMORY BEYOND ANY WINDOW: two inputs identical in their
     last 25 steps but different before must yield different features.
  1  n=4 full grid, univariate one-step battery, 3 systems x 5 seeds.
  2  n=5 confirmation arm at each system's best (tau, gamma).
  3  memory-capacity task (iid inputs, reconstruct u_{t-d}, MC = sum r^2,
     d = 1..15): dissipative QRC (best gamma and gamma=0) vs ESN(F) vs
     ELM(F) -- the family's home-ground comparison.
  4  closed-loop on Henon at best config, INCREMENTAL rollout (the recurrent
     state is carried, so autonomous steps cost O(1)) vs ESN(F).

Outputs -> results/dissipative/*.csv. Fresh compute; committed artifacts
untouched.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.linalg import expm

from qdepipe import metrics
from qdepipe.concentration import _scaled_series
from qdepipe.models import ELM, ELMConfig, ESN, ESNConfig, NGRC, NGRCConfig
from qdepipe.readout import ridge_fit, ridge_predict

OUT = "results/dissipative"
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
NGRC_DEGREE = {"henon": 2, "lorenz": 3, "mackeyglass": 2}
SEEDS = [0, 1, 2, 3, 4]
NPTS = 1500
WASHOUT = 100
V = 4
TAUS = [1.0, 4.0, 10.0]
GAMMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
BETAS = [1e-6, 1e-9, 1e-12]


# ------------------------------------------------------------ pauli helpers
def _paulis():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def _op_on(op, q, n):
    I = np.eye(2, dtype=complex)
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, op if i == q else I)
    return out


class DissipativeQRC:
    """Recurrent dissipative quantum reservoir; memory lives in rho."""

    def __init__(self, n=4, V=V, tau=4.0, gamma=0.05, seed=0):
        self.n, self.V, self.tau, self.gamma, self.seed = n, V, tau, gamma, seed
        I, X, Y, Z = _paulis()
        rng = np.random.default_rng(seed)
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n):                                   # transverse fields
            H += rng.normal(0, 1) * _op_on(Z, i, n)
        for i in range(n):                                   # random XX couplings
            for j in range(i + 1, n):
                H += (rng.normal(0, 1) / np.sqrt(n)) * (
                    _op_on(X, i, n) @ _op_on(X, j, n))
        self.U = expm(-1j * H * (tau / V))
        self.Ud = self.U.conj().T
        # observables (dense, precomputed)
        self.obs = ([_op_on(Z, q, n) for q in range(n)]
                    + [_op_on(X, q, n) for q in range(n)]
                    + [_op_on(Y, q, n) for q in range(n)])
        # per-qubit amplitude-damping Kraus, embedded
        g = gamma
        E0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
        self.kraus = [( _op_on(E0, q, n), _op_on(E1, q, n)) for q in range(n)] \
            if g > 0 else []
        self.name = f"dQRC(n={n},tau={tau},g={gamma})"

    @property
    def n_features(self):
        return self.V * 3 * self.n

    def _inject(self, rho, u):
        """rho <- |psi(u)><psi(u)| (x) Tr_qubit0[rho]."""
        n, dim = self.n, 2 ** self.n
        r = rho.reshape(2, dim // 2, 2, dim // 2)
        reduced = r[0, :, 0, :] + r[1, :, 1, :]              # Tr over qubit 0
        a, b = np.sqrt(1 - u), np.sqrt(u)
        psi = np.array([[a * a, a * b], [a * b, b * b]], dtype=complex)
        return np.kron(psi, reduced)

    def _step(self, rho, u):
        rho = self._inject(rho, float(np.clip(u, 0.0, 1.0)))
        feats = []
        for _ in range(self.V):
            rho = self.U @ rho @ self.Ud
            for E0, E1 in self.kraus:
                rho = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
            feats.extend(float(np.trace(rho @ O).real) for O in self.obs)
        return rho, np.asarray(feats)

    def featurize(self, u, rho0=None):
        u = np.asarray(u, dtype=float).ravel()
        dim = 2 ** self.n
        rho = (np.eye(dim, dtype=complex) / dim) if rho0 is None else rho0
        X = np.zeros((len(u), self.n_features))
        for t, x in enumerate(u):
            rho, X[t] = self._step(rho, x)
        self._last_rho = rho
        return X


# ------------------------------------------------------------- Phase 0 gates
def sanity():
    m = DissipativeQRC(n=4, tau=4.0, gamma=0.05, seed=0)
    rng = np.random.default_rng(1)
    u = rng.random(120)
    X1 = m.featurize(u)
    tr = float(np.trace(m._last_rho).real)
    print(f"[gate] trace preserved: {tr:.12f} (want 1)")
    # echo-state: different initial states converge
    dim = 16
    rho_alt = np.zeros((dim, dim), dtype=complex); rho_alt[0, 0] = 1.0
    X2 = m.featurize(u, rho0=rho_alt)
    d_early = np.abs(X1[:5] - X2[:5]).max()
    d_late = np.abs(X1[-5:] - X2[-5:]).max()
    print(f"[gate] echo state: early diff {d_early:.2e} -> late diff {d_late:.2e} (want -> 0)")
    # memory beyond any window: same last 25 inputs, different before
    u2 = u.copy(); u2[:-25] = rng.random(len(u) - 25)
    Y1, Y2 = m.featurize(u), m.featurize(u2)
    dmem = np.abs(Y1[-1] - Y2[-1]).max()
    print(f"[gate] memory beyond window: final-row diff {dmem:.2e} (want > 0; window-based models give 0)")
    ok = abs(tr - 1) < 1e-9 and d_late < 1e-6 and dmem > 1e-8
    print(f"[gate] ALL {'PASS' if ok else 'FAIL'}")
    return ok


def onestep_eval(X, u, split, beta):
    a, b, n = split.train.stop, split.val.stop, len(u)
    tr = np.arange(WASHOUT, a - 1)
    va = np.arange(a, b - 1)
    te = np.arange(b, n - 1)
    W = ridge_fit(X[tr], u[tr + 1], alpha=beta)
    return (metrics.nrmse(u[va + 1], ridge_predict(X[va], W)),
            metrics.nrmse(u[te + 1], ridge_predict(X[te], W)))


def row_done(df, **kv):
    if df is None:
        return False
    m = np.ones(len(df), dtype=bool)
    for k, v in kv.items():
        m &= (df[k].astype(str) == str(v))
    return bool(m.any())


def log_row(row):
    pd.DataFrame([row]).to_csv(SCORES, mode="a", index=False,
                               header=not os.path.exists(SCORES))
    print("  ".join(f"{k}={v}" for k, v in row.items()), flush=True)


# ------------------------------------------------------------------- phases
def main():
    if not sanity():
        raise SystemExit("sanity gates failed")

    done = pd.read_csv(SCORES, keep_default_na=False) if os.path.exists(SCORES) else None

    # Phase 1: n=4 full grid + controls
    best = {}
    for system in SYSTEMS:
        for seed in SEEDS:
            u, split = _scaled_series(system, NPTS)
            cands = []
            for tau in TAUS:
                for g in GAMMAS:
                    if row_done(done, phase="grid4", system=system, seed=seed,
                                tau=tau, gamma=g):
                        dg = done[done.phase == "grid4"]
                        r = dg[(dg.system == system)
                               & (dg.seed.astype(int) == seed)
                               & (dg.tau.astype(float) == tau)
                               & (dg.gamma.astype(float) == g)].iloc[0]
                        cands.append((float(r.val_nrmse), float(r.test_nrmse),
                                      tau, g, float(r.beta)))
                        continue
                    X = DissipativeQRC(n=4, tau=tau, gamma=g, seed=seed).featurize(u)
                    vals = [(onestep_eval(X, u, split, b_) + (b_,)) for b_ in BETAS]
                    v, t_, b_ = min(vals, key=lambda c: c[0])
                    cands.append((v, t_, tau, g, b_))
                    log_row({"phase": "grid4", "system": system, "model": "dqrc4",
                             "seed": seed, "tau": tau, "gamma": g, "beta": b_,
                             "val_nrmse": v, "test_nrmse": t_, "n_features": 48})
            v, t_, tau, g, b_ = min(cands, key=lambda c: c[0])
            best[(system, seed)] = (tau, g, b_)
            # matched controls at F=48 (once per seed)
            for label, mk in [
                ("elmF", lambda: ELM(ELMConfig(units=48, lookback=5, seed=seed))),
                ("esnF", lambda: ESN(ESNConfig(units=48, seed=seed))),
                ("ngrc", lambda: NGRC(NGRCConfig(k=5, degree=NGRC_DEGREE[system]))),
            ]:
                if row_done(done, phase="ctrl4", system=system, seed=seed, model=label):
                    continue
                Xc = mk().featurize(u)
                v2, t2 = onestep_eval(Xc, u, split, 1e-6)
                log_row({"phase": "ctrl4", "system": system, "model": label,
                         "seed": seed, "tau": "", "gamma": "", "beta": 1e-6,
                         "val_nrmse": v2, "test_nrmse": t2,
                         "n_features": Xc.shape[1]})

    # Phase 2: n=5 confirmation at best (tau, gamma) per (system, seed)
    for system in SYSTEMS:
        for seed in SEEDS:
            tau, g, _ = best[(system, seed)]
            if row_done(done, phase="confirm5", system=system, seed=seed):
                continue
            u, split = _scaled_series(system, NPTS)
            X = DissipativeQRC(n=5, tau=tau, gamma=g, seed=seed).featurize(u)
            vals = [(onestep_eval(X, u, split, b_) + (b_,)) for b_ in BETAS]
            v, t_, b_ = min(vals, key=lambda c: c[0])
            log_row({"phase": "confirm5", "system": system, "model": "dqrc5",
                     "seed": seed, "tau": tau, "gamma": g, "beta": b_,
                     "val_nrmse": v, "test_nrmse": t_, "n_features": 60})

    # Phase 3: memory capacity (iid inputs), d = 1..15
    mc_rows = []
    rng = np.random.default_rng(7)
    u = rng.random(NPTS)
    from qdepipe.pipeline import temporal_split
    split = temporal_split(NPTS, (0.6, 0.2, 0.2), WASHOUT)
    a, b, nlen = split.train.stop, split.val.stop, NPTS
    tr = np.arange(WASHOUT, a); te = np.arange(b, nlen)
    def mem_cap(X):
        total = 0.0; per = []
        for d in range(1, 16):
            tr_d, te_d = tr[tr >= WASHOUT + d], te[te >= b + d]
            W = ridge_fit(X[tr_d], u[tr_d - d], alpha=1e-9)
            pred = np.asarray(ridge_predict(X[te_d], W)).ravel()
            targ = u[te_d - d]
            r = np.corrcoef(pred, targ)[0, 1]
            r2 = 0.0 if not np.isfinite(r) else max(r, 0.0) ** 2
            per.append(r2); total += r2
        return total, per
    for label, X in [
        ("dqrc4_g0.05", DissipativeQRC(n=4, tau=4.0, gamma=0.05, seed=0).featurize(u)),
        ("dqrc4_g0", DissipativeQRC(n=4, tau=4.0, gamma=0.0, seed=0).featurize(u)),
        ("esnF", ESN(ESNConfig(units=48, seed=0)).featurize(u)),
        ("elmF", ELM(ELMConfig(units=48, lookback=5, seed=0)).featurize(u)),
    ]:
        total, per = mem_cap(X)
        mc_rows.append({"model": label, "memory_capacity": total,
                        **{f"r2_d{d}": per[d-1] for d in range(1, 16)}})
        print(f"[MC] {label:12s} capacity={total:.2f}", flush=True)
    pd.DataFrame(mc_rows).to_csv(os.path.join(OUT, "memory_capacity.csv"), index=False)

    # Phase 4: incremental closed-loop on Henon, best config vs ESN(F)
    cl_rows = []
    system = "henon"
    u, split = _scaled_series(system, NPTS)
    from qdepipe.data import lyapunov_step
    lam = lyapunov_step(system, u)
    for seed in [0, 1, 2]:
        tau, g, b_ = best[(system, seed)]
        m = DissipativeQRC(n=4, tau=tau, gamma=g, seed=seed)
        X = m.featurize(u[:split.test.start])
        trn = np.arange(WASHOUT, split.train.stop - 1)
        W = ridge_fit(X[trn], u[trn + 1], alpha=b_)
        rho = m._last_rho
        true = u[split.test.start:split.test.start + 300]
        preds = []
        x = u[split.test.start - 1]
        for _ in range(300):
            rho, f = m._step(rho, x)
            x = float(ridge_predict(f[None, :], W)[0])
            if not np.isfinite(x):
                x = preds[-1] if preds else 0.5
            preds.append(x)
        preds = np.asarray(preds)
        vpt = metrics.valid_prediction_time(true, preds, 0.4)
        cl_rows.append({"system": system, "seed": seed, "model": "dqrc4",
                        "vpt_steps": vpt, "vpt_lyap": vpt * lam,
                        "diverged": bool(np.max(np.abs(preds)) >
                                         10 * np.max(np.abs(true)))})
        print(f"[CL] henon seed{seed} dqrc4 VPT={vpt*lam:.3f}L", flush=True)
    pd.DataFrame(cl_rows).to_csv(os.path.join(OUT, "closedloop.csv"), index=False)
    print("\ndone -> results/dissipative/")


if __name__ == "__main__":
    main()
