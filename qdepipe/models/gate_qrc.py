"""GateQRC — gate-based quantum reservoir (PyTorch, density-matrix capable).

A fresh re-implementation of the validated legacy reservoir (old
`unified/qrclab/qrc.py`), exposing the same `ReservoirModel.featurize` interface
as the classical models so it flows through the identical data-engineering
pipeline and notebook template. The circuit semantics are reproduced exactly so
the port can be validated against the prior project's verified numbers
(v4 depth r=1, 4 qubits, CNOT+RZ ring, PhaseDamping 0.1, Z readout → NRMSE ≈ 0.10).

Circuit (per timestep t, temporal multiplexing):
  * window = last `window` inputs ending at t.
  * for v = 1..V: run a FRESH circuit from |0…0> over the last
    max(1, int(len(window)·v/V)) window elements; measure all readout
    observables. The V sub-readouts are concatenated → the feature row.
  * encode each window element x:
      depth : r repeats on qubit 0 of [RY((1+0.3k)·s·x), RX(π/5)]
      width : RY((1+0.3q)·s·x + π/4) on qubits q = 0..min(r,n)-1
    where s = `encode_scale` (1.0 reproduces the legacy encoding; this is the
    knob the encoding/input-scaling axis turns).
  * fixed layer per element:
      cnot_rz : CNOT(i,(i+1)%n) ring, then RZ(J_i) on every qubit
      isingxx : IsingXX(J_i) on ring pairs (i,(i+1)%n)
    J ~ U(-J_strength/2, +J_strength/2).
  * channel (density only, every qubit after the fixed layer):
      dephasing | amplitude | none (pure state).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import _qops as Q
from .base import ReservoirModel


@dataclass
class GateQRCConfig:
    n_qubits: int = 4
    encoding: str = "depth"            # "depth" | "width"
    r: int = 1                          # re-uploads per window element
    coupling: str = "isingxx"          # "isingxx" | "cnot_rz"
    J_strength: float = 1.0
    channel: str = "none"              # "none" | "dephasing" | "amplitude"
    gamma: float = 0.1
    V: int = 4                          # virtual nodes (temporal multiplexing)
    window: int = 5
    readout: tuple = ("Z",)            # subset of ("Z","X","Y","ZZ")
    encode_scale: float = 1.0          # input-scaling knob (1.0 = legacy)
    seed: int = 0
    shots: int | None = None           # None = exact expectation (idealized upper
                                       # bound); int = finite-shot estimate per
                                       # observable (binomial noise; finite-shot realism)


# ---- canonical validated configs -------------------------------------------
def v4_config(mode="depth", r=1, **kw):
    """Legacy v4: density + PhaseDamping(0.1), CNOT+RZ ring. Archived ≈ 0.100."""
    return GateQRCConfig(n_qubits=4, encoding=mode, r=r, coupling="cnot_rz",
                         channel="dephasing", gamma=0.1, V=4, window=5, **kw)


def v5_config(n_qubits=8, **kw):
    """Legacy v5: pure state, CNOT+RZ — the inert/deterministic artifact."""
    return GateQRCConfig(n_qubits=n_qubits, encoding="depth", r=1,
                         coupling="cnot_rz", channel="none", V=4, window=5, **kw)


def v6_config(n_qubits=10, **kw):
    """Legacy v6: pure state, IsingXX ring, genuine disorder, Z readout."""
    kw.setdefault("seed", 42)
    return GateQRCConfig(n_qubits=n_qubits, encoding="depth", r=1,
                         coupling="isingxx", J_strength=1.0, channel="none",
                         V=4, window=5, readout=("Z",), **kw)


def rich_config(n_qubits=10, **kw):
    """Legacy 'rich': v6 + Z+X+Y+ZZ(ring) readout → F = V·4n."""
    kw.setdefault("seed", 42)
    return GateQRCConfig(n_qubits=n_qubits, encoding="depth", r=1,
                         coupling="isingxx", J_strength=1.0, channel="none",
                         V=4, window=5, readout=("Z", "X", "Y", "ZZ"), **kw)


class GateQRC(ReservoirModel):
    name = "GateQRC"

    def __init__(self, cfg: GateQRCConfig | None = None, device=None):
        self.cfg = cfg or GateQRCConfig()
        n = self.cfg.n_qubits
        # small matrices: CPU avoids GPU launch overhead; large n → GPU
        if device is None:
            device = torch.device("cuda") if (n > 6 and torch.cuda.is_available()) else torch.device("cpu")
        self.device = device

        rng = np.random.default_rng(self.cfg.seed)
        self.J = rng.uniform(-self.cfg.J_strength / 2, self.cfg.J_strength / 2, size=n)

        self.U_fixed = self._build_fixed_layer()
        self._kraus_embedded = None
        if self.cfg.channel in ("dephasing", "amplitude"):
            kr = (Q.dephasing_kraus if self.cfg.channel == "dephasing"
                  else Q.amplitude_damping_kraus)(self.cfg.gamma, device)
            self._kraus_embedded = [[Q.op_on(K, q, n, device) for K in kr] for q in range(n)]

        self._readout_ops = self._build_readout_ops()
        self._rx5 = Q.rx(math.pi / 5, device)
        # deterministic shot-noise RNG (offset from J-seed so shots != reservoir draw)
        self._shot_rng = (np.random.default_rng(self.cfg.seed + 10_007)
                          if self.cfg.shots else None)

    def _sample(self, expval: float) -> float:
        """Finite-shot estimate of a ±1-eigenvalue Pauli observable: ⟨O⟩=2p-1 with
        p=(1+⟨O⟩)/2, so a `shots`-measurement estimate is 2·Binomial(shots,p)/shots − 1.
        Exact for every readout here (Z,X,Y,ZZ are all ±1 Paulis)."""
        p = min(1.0, max(0.0, 0.5 * (1.0 + expval)))
        k = self._shot_rng.binomial(self.cfg.shots, p)
        return 2.0 * k / self.cfg.shots - 1.0

    # ---- builders -------------------------------------------------------
    def _build_fixed_layer(self):
        cfg, n, dev = self.cfg, self.cfg.n_qubits, self.device
        U = torch.eye(2 ** n, dtype=Q.CDTYPE, device=dev)
        if cfg.coupling == "cnot_rz":
            for i in range(n):
                U = Q.cnot_pair(i, (i + 1) % n, n, dev) @ U
            for i in range(n):
                U = Q.op_on(Q.rz(float(self.J[i]), dev), i, n, dev) @ U
        elif cfg.coupling == "isingxx":
            for i in range(n):
                U = Q.isingxx_pair(float(self.J[i]), i, (i + 1) % n, n, dev) @ U
        else:
            raise ValueError(f"unknown coupling: {cfg.coupling}")
        return U

    def _build_readout_ops(self):
        n, dev = self.cfg.n_qubits, self.device
        P = Q.paulis(dev)
        ops = []
        if "Z" in self.cfg.readout:
            ops += [Q.op_on(P["Z"], q, n, dev) for q in range(n)]
        if "X" in self.cfg.readout:
            ops += [Q.op_on(P["X"], q, n, dev) for q in range(n)]
        if "Y" in self.cfg.readout:
            ops += [Q.op_on(P["Y"], q, n, dev) for q in range(n)]
        if "ZZ" in self.cfg.readout:
            ops += [Q.zz_pair(i, (i + 1) % n, n, dev) for i in range(n)]
        return ops

    @property
    def n_features(self) -> int:
        return self.cfg.V * len(self._readout_ops)

    @property
    def memory_window(self) -> int:
        return self.cfg.window

    # ---- encoding -------------------------------------------------------
    def _encode_unitary(self, x):
        cfg, n, dev = self.cfg, self.cfg.n_qubits, self.device
        xs = cfg.encode_scale * float(x)
        if cfg.encoding == "depth":
            u2 = torch.eye(2, dtype=Q.CDTYPE, device=dev)
            for k in range(cfg.r):
                u2 = self._rx5 @ Q.ry((1.0 + 0.3 * k) * xs, dev) @ u2
            return Q.op_on(u2, 0, n, dev)
        if cfg.encoding == "width":
            U = torch.eye(2 ** n, dtype=Q.CDTYPE, device=dev)
            for q in range(min(cfg.r, n)):
                ang = (1.0 + 0.3 * q) * xs + math.pi / 4
                U = Q.op_on(Q.ry(ang, dev), q, n, dev) @ U
            return U
        raise ValueError(f"unknown encoding mode: {cfg.encoding}")

    # ---- circuit over one sub-window -----------------------------------
    def _run_circuit(self, sub_window):
        n, dev = self.cfg.n_qubits, self.device
        density = self.cfg.channel != "none"
        state = Q.zero_density(n, dev) if density else Q.zero_state(n, dev)
        for x in sub_window:
            U = self.U_fixed @ self._encode_unitary(x)
            if density:
                state = U @ state @ U.conj().T
                for kraus_q in self._kraus_embedded:
                    state = sum(E @ state @ E.conj().T for E in kraus_q)
            else:
                state = U @ state
        expect = Q.expectation_rho if density else Q.expectation_psi
        vals = [expect(state, op) for op in self._readout_ops]
        if self.cfg.shots:
            vals = [self._sample(v) for v in vals]
        return vals

    def featurize(self, series) -> np.ndarray:
        """Temporal-multiplexed feature matrix (T, V·n_readout), causally aligned."""
        s = np.asarray(series, dtype=float).ravel()
        cfg = self.cfg
        T = len(s)
        states = np.zeros((T, self.n_features))
        for t in range(T):
            lo = max(0, t - cfg.window + 1)
            full = s[lo:t + 1]
            feats = []
            for v in range(1, cfg.V + 1):
                sub_len = max(1, int(len(full) * v / cfg.V))
                feats.extend(self._run_circuit(full[-sub_len:]))
            states[t] = feats
        return states

    def featurize_mv(self, U) -> np.ndarray:
        """Multivariate feature matrix for a d-channel input U (T, d).

        Encoding scheme (the single, obvious concatenation used by the multivariate
        Lorenz pilot): the causal window of the last `window` timesteps --- each a
        d-vector --- is flattened in time-major order into one scalar sequence and
        encoded by the SAME sequential depth re-upload as the univariate path. The
        circuit, readout observable set, virtual-node multiplexing and feature count
        (V·n_readout) are all unchanged; only the driving sequence is the flattened
        multivariate window. (No new encoding scheme is introduced.)
        """
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[:, None]
        cfg, T = self.cfg, U.shape[0]
        states = np.zeros((T, self.n_features))
        for t in range(T):
            lo = max(0, t - cfg.window + 1)
            full = U[lo:t + 1].reshape(-1)        # time-major flatten of the window
            feats = []
            for v in range(1, cfg.V + 1):
                sub_len = max(1, int(len(full) * v / cfg.V))
                feats.extend(self._run_circuit(full[-sub_len:]))
            states[t] = feats
        return states
