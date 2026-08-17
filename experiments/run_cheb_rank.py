"""Feature-matrix rank diagnostic for the polynomial-encoding experiment.

The thesis states that the Chebyshev encoding improves the conditioning of the
feature matrix even though it worsens accuracy. That diagnostic was computed
during the original run but never persisted, so the claim had no artifact. This
script regenerates it from the committed model definitions.

Self-contained by design: it re-declares the encoder rather than importing
run_cheb_encoding.py, because that module (like the other runner scripts)
executes its full experiment at import time.

For each encoding it builds the training feature matrix under the same protocol
as run_cheb_encoding.py (n_q=6, V=4, F=96, same scaled series and split) and
reports the effective rank and the condition number.

Output: results/cheb/rank.csv
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch

from qdepipe.models import _qops as Q
from qdepipe.concentration import NPOINTS, _rich_cfg, _scaled_series
from qdepipe.models.gate_qrc import GateQRC

OUT = "results/cheb"
NQ = 6                 # F = V(4) * m(4) * n(6) = 96, the matched-budget setting
SEEDS = [0, 1, 2, 3, 4]
RTOL = 1e-10           # relative singular-value cutoff for the effective rank
EPS_CLIP = 1e-6


def _to_pm1(x):
    """Clip into (-1, 1) so arcsin/arccos stay away from their divergent edges."""
    return float(np.clip(x, -1.0 + EPS_CLIP, 1.0 - EPS_CLIP))


class PolyEncodedQRC(GateQRC):
    """QRC-rich with a polynomial-friendly encoding; only the encoding differs."""

    def __init__(self, cfg, poly_mode="arcsin"):
        super().__init__(cfg)
        self.poly_mode = poly_mode
        self.name = f"QRC-rich({poly_mode})"

    def _encode_unitary(self, x):
        n, dev = self.cfg.n_qubits, self.device
        z = _to_pm1(x)
        if self.poly_mode == "arcsin":
            theta = 2.0 * math.asin(z)
            return Q.op_on(self._rx5 @ Q.ry(theta, dev), 0, n, dev)
        if self.poly_mode == "cheb":
            U = torch.eye(2 ** n, dtype=Q.CDTYPE, device=dev)
            ac = math.acos(z)
            for q in range(n):
                U = Q.op_on(Q.ry(2.0 * (q + 1) * ac, dev), q, n, dev) @ U
            return U
        raise ValueError(self.poly_mode)


def _model(key, seed):
    if key in ("depth", "width"):
        cfg = _rich_cfg(NQ, V=4, seed=seed, encoding=key,
                        r=(NQ if key == "width" else 1))
        return GateQRC(cfg)
    return PolyEncodedQRC(_rich_cfg(NQ, V=4, seed=seed), key)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    npts = NPOINTS.get(NQ, 1500)
    rows = []
    for system in ["henon"]:
        u, split = _scaled_series(system, npts)
        tr = np.arange(split.train.start, split.train.stop)
        for key in ["depth", "arcsin", "cheb"]:
            for seed in SEEDS:
                X = np.asarray(_model(key, seed).featurize(u))[tr]
                s = np.linalg.svd(X, compute_uv=False)
                eff = int((s > s[0] * RTOL).sum())
                nz = s[s > 0]
                cond = float(s[0] / nz[-1])
                rows.append({"system": system, "encoding": key, "seed": seed,
                             "n_features": X.shape[1], "effective_rank": eff,
                             "condition_number": cond})
                print(f"[rank] {system} {key:7s} seed{seed}: "
                      f"rank {eff}/{X.shape[1]}  cond {cond:.3e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rank.csv", index=False)
    print("\nmedian by encoding:")
    print(df.groupby("encoding")[["effective_rank", "condition_number"]].median())


if __name__ == "__main__":
    main()
