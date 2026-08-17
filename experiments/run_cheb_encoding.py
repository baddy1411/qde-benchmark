#!/usr/bin/env python3
"""Steelman experiment: polynomial-friendly encodings (arcsin / Chebyshev tower).

Motivation. The thesis shows that (i) the encoding is the biggest quantum-side
lever and (ii) the deficit on Henon is *linear accessibility of second-order
structure* -- a classical quadratic read-out closes the gap (augmentation
experiment). Both point to the same theoretical question: what if the ENCODING
itself is chosen so the measured expectations are genuine polynomials of the
input, rather than sines of a linearly-scaled input?

The algebra (verified numerically before implementing):
  RY(theta)|0>  ->  <Z> = cos(theta),  <X> = sin(theta)
  arcsin map:   theta = 2*arcsin(z)     =>  <Z> = 1 - 2 z^2   (exact quadratic)
  Chebyshev:    theta_q = 2(q+1)*arccos(z) =>  <Z_q> = T_{2(q+1)}(z)  (exact
                Chebyshev polynomial of degree 2(q+1) -- a "Chebyshev tower")
Products of such features under the entangling layer are then genuine
polynomials in the (delayed) inputs, i.e. the accessible function class is
reshaped to contain what NG-RC uses.

Design. Everything except the encoding function is IDENTICAL to the thesis's
QRC-rich construction (n=6, V=4, IsingXX J=1, Z+X+Y+ZZ read-out, window=5,
F=96), so any difference is attributable to the encoding alone. The committed
GateQRC is NOT modified: the new encodings are added by subclassing and
overriding `_encode_unitary`.

Controls, all through the identical pipeline:
  ELM(F=96)   the standing matched random-projection control
  NG-RC       the standing tuned strong reference
  ChebPoly    the decisive twin: explicit CLASSICAL Chebyshev features of the
              same delay window (T_1..T_d of each delay + pairwise products).
              If the quantum Chebyshev encoding works, this hands you the same
              function class classically and for free.

Outputs -> results/cheb/{scores,dm}.csv. Fresh compute (no feature cache).
"""
from __future__ import annotations

import math
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

from qdepipe.concentration import _rich_cfg, NPOINTS
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ELM, ELMConfig, NGRC, NGRCConfig
from qdepipe.models.base import ReservoirModel
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.models import _qops as Q
from qdepipe.significance import diebold_mariano, errors

OUT = "results/cheb"
os.makedirs(OUT, exist_ok=True)
SCORES = os.path.join(OUT, "scores.csv")

SYSTEMS = ["henon", "lorenz", "mackeyglass"]
SEEDS = [0, 1, 2, 3, 4]
NQ = 6                      # F = V(4) * m(4) * n(6) = 96, the matched-budget setting
LOOKBACK = 5
NGRC_DEGREE = {"henon": 2, "lorenz": 3, "mackeyglass": 2}
EPS_CLIP = 1e-6             # keep z strictly inside [-1, 1] (arcsin/arccos edges)


def _to_pm1(x):
    """Pipeline scales inputs to [0,1]; map to [-1,1] and clip off the singular
    endpoints where d(arcsin)/dz diverges."""
    z = 2.0 * float(x) - 1.0
    return float(np.clip(z, -1.0 + EPS_CLIP, 1.0 - EPS_CLIP))


class PolyEncodedQRC(GateQRC):
    """QRC-rich with a polynomial-friendly encoding. Only `_encode_unitary`
    differs from the committed model; circuit, read-out, V and window are
    inherited unchanged."""

    def __init__(self, cfg, poly_mode="arcsin"):
        super().__init__(cfg)
        self.poly_mode = poly_mode
        self.name = f"QRC-rich({poly_mode})"

    def _encode_unitary(self, x):
        n, dev = self.cfg.n_qubits, self.device
        z = _to_pm1(x)
        if self.poly_mode == "arcsin":
            # depth-style: single qubit, <Z> = 1 - 2 z^2 exactly
            theta = 2.0 * math.asin(z)
            u2 = self._rx5 @ Q.ry(theta, dev)
            return Q.op_on(u2, 0, n, dev)
        if self.poly_mode == "cheb":
            # width-style Chebyshev tower: <Z_q> = T_{2(q+1)}(z) exactly
            U = torch.eye(2 ** n, dtype=Q.CDTYPE, device=dev)
            ac = math.acos(z)
            for q in range(n):
                U = Q.op_on(Q.ry(2.0 * (q + 1) * ac, dev), q, n, dev) @ U
            return U
        raise ValueError(self.poly_mode)


class ChebPoly(ReservoirModel):
    """Classical twin: explicit Chebyshev features of the delay window.
    T_1..T_d of each of k delays, plus all pairwise products."""

    def __init__(self, k=LOOKBACK, degree=2):
        self.k, self.degree = k, degree
        self.name = f"ChebPoly(k={k},d={degree})"
        self.memory_window = k
        b = k * degree
        self._F = 1 + b + b * (b + 1) // 2

    @property
    def n_features(self):
        return self._F

    def featurize(self, series):
        s = np.asarray(series, dtype=float).ravel()
        T = len(s)
        z = np.clip(2.0 * s - 1.0, -1.0 + EPS_CLIP, 1.0 - EPS_CLIP)
        base = np.zeros((T, self.k * self.degree))
        for i in range(self.k):                       # delay i
            zi = np.concatenate([np.full(min(i, T), z[0]), z[:max(T - i, 0)]])[:T]
            for j in range(1, self.degree + 1):       # Chebyshev degree j
                base[:, i * self.degree + (j - 1)] = np.cos(j * np.arccos(zi))
        cols = [np.ones((T, 1)), base]
        b = base.shape[1]
        cols.append(np.stack([base[:, a] * base[:, c]
                              for a in range(b) for c in range(a, b)], axis=1))
        return np.concatenate(cols, axis=1)


def build(key, system, seed):
    F = 16 * NQ                                        # 96
    if key in ("depth", "width", "arcsin", "cheb"):
        if key in ("depth", "width"):
            cfg = _rich_cfg(NQ, V=4, seed=seed, encoding=key,
                            r=(NQ if key == "width" else 1))
            return ReservoirForecaster(GateQRC(cfg), f"QRC-rich({key})", store=None)
        cfg = _rich_cfg(NQ, V=4, seed=seed)            # encoding overridden below
        return ReservoirForecaster(PolyEncodedQRC(cfg, key),
                                   f"QRC-rich({key})", store=None)
    if key == "elmF":
        return ReservoirForecaster(ELM(ELMConfig(units=F, lookback=LOOKBACK, seed=seed)),
                                   f"ELM(F={F})", store=None)
    if key == "ngrc":
        return ReservoirForecaster(NGRC(NGRCConfig(k=LOOKBACK, degree=NGRC_DEGREE[system])),
                                   "NG-RC", store=None)
    if key == "chebpoly":
        return ReservoirForecaster(ChebPoly(degree=NGRC_DEGREE[system]),
                                   "ChebPoly", store=None)
    raise KeyError(key)


MODELS = ["depth", "width", "arcsin", "cheb", "elmF", "ngrc", "chebpoly"]

done = set()
if os.path.exists(SCORES):
    d = pd.read_csv(SCORES, keep_default_na=False)
    done = {(r.system, r.model, int(r.seed)) for r in d.itertuples()}
    print(f"resuming: {len(done)} cells done", flush=True)

bank = {}
for system in SYSTEMS:
    npts = NPOINTS.get(NQ, 1500)
    for key in MODELS:
        for seed in SEEDS:
            if (system, key, seed) in done:
                continue
            cfg = ExperimentConfig(system=system, n_points=npts, seed=seed,
                                   lookback=LOOKBACK, washout=100)
            r = run_experiment(build(key, system, seed), cfg, keep_forecasts=True)
            bank[(system, key, seed)] = (r.y_true, r.y_pred)
            pd.DataFrame([{"system": system, "model": key, "seed": seed,
                           "nrmse": r.nrmse, "n_features": r.n_features,
                           "n_points": npts}]).to_csv(
                SCORES, mode="a", index=False, header=not os.path.exists(SCORES))
            print(f"{system:12s} {key:9s} seed{seed} NRMSE={r.nrmse:.4e} "
                  f"(F={r.n_features})", flush=True)


def fc(system, key, seed):
    if (system, key, seed) not in bank:
        cfg = ExperimentConfig(system=system, n_points=NPOINTS.get(NQ, 1500),
                               seed=seed, lookback=LOOKBACK, washout=100)
        r = run_experiment(build(key, system, seed), cfg, keep_forecasts=True)
        bank[(system, key, seed)] = (r.y_true, r.y_pred)
    return bank[(system, key, seed)]


dm_rows = []
for system in SYSTEMS:
    for key in ["arcsin", "cheb"]:
        for rival in ["depth", "elmF", "ngrc", "chebpoly"]:
            for seed in SEEDS:
                t1, p1 = fc(system, key, seed)
                t2, p2 = fc(system, rival, seed)
                stat, p = diebold_mariano(errors(t1, p1), errors(t2, p2))
                dm_rows.append({"system": system, "model": key, "vs": rival,
                                "seed": seed, "dm_stat": stat, "p": p})
pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "dm.csv"), index=False)

sc = pd.read_csv(SCORES, keep_default_na=False)
sc["nrmse"] = sc.nrmse.astype(float)
sc = sc.drop_duplicates(subset=["system", "model", "seed"], keep="last")
print("\n===== median NRMSE =====")
print(sc.groupby(["system", "model"]).nrmse.median().unstack("model").to_string(
    float_format="%.3e"))
print("\nwrote", SCORES, "and dm.csv")
