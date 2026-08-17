"""Frequency-domain redundancy test (Part D) — does adding ZZ introduce content
the single-qubit observables do not already supply?

Tests the §5.5 redundancy claim directly. The read-out is a linear ridge, so the
relevant notion is **linear reachability**: a ZZ observable whose time series is
linearly spanned by the single-qubit (Z,X,Y) time series carries nothing a linear
read-out can extract from it, hence is redundant. Two complementary measures, both
averaged over the ZZ observables:

  (step 2) residual-power fraction = mean over ZZ observables of (1 - R^2) of the
      linear regression of each ZZ time series onto the single-qubit span. Low =>
      ZZ is linearly reachable from the single-qubit set (redundant). Threshold-free.
  (step 3) new-frequency fraction = mean over ZZ observables of the fraction of
      each ZZ observable's power spectrum lying at frequencies OUTSIDE the
      single-qubit set's frequency support. Low => ZZ excites the same frequencies
      (redundant). Depends on a support threshold (reported at several values).

The two can diverge (same frequencies but linearly unreachable combination, or
vice versa); they are reported separately, never averaged.
"""
from __future__ import annotations

import numpy as np

from qdepipe.models.gate_qrc import GateQRC
from qdepipe import concentration as C


def observable_timeseries(system, encode_scale, n_qubits=6, n_points=1500, seed=0, store=None):
    """Drive the fixed n-qubit QRC-rich circuit (V=1, so each column is one
    observable's expectation) and split the readout into single-qubit (Z,X,Y) and
    two-qubit (ZZ) time-series blocks. Returns (S (T, 3n), ZZ (T, n)).

    `store`: optional, DEFAULT-OFF FeatureStore (store=None -> byte-identical). The
    V=1 config keys distinct from the V=4 paths (verified), so no collision.
    """
    from qdepipe.experiment import ExperimentConfig
    from qdepipe.feature_store import maybe_cached_featurize
    u, _ = C._scaled_series(system, n_points)
    # Key context mirrors _scaled_series (minmax/train/60-20-20).
    ecfg = ExperimentConfig(system=system, n_points=n_points, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2))
    q = GateQRC(C._rich_cfg(n_qubits, V=1, encode_scale=encode_scale, seed=seed))
    feats = maybe_cached_featurize(store, q, u, ecfg)
    n = n_qubits
    return feats[:, :3 * n], feats[:, 3 * n:4 * n]


def _psd(x):
    return np.abs(np.fft.rfft(x - np.mean(x))) ** 2


def residual_power_fraction(S, ZZ):
    """Mean over ZZ observables of (1 - R^2) for the linear regression of each ZZ
    time series onto the single-qubit span. Threshold-free."""
    Sc = S - S.mean(axis=0)
    fr = []
    for j in range(ZZ.shape[1]):
        z = ZZ[:, j] - ZZ[:, j].mean()
        beta, *_ = np.linalg.lstsq(Sc, z, rcond=None)
        r = z - Sc @ beta
        denom = float(np.sum(z ** 2))
        fr.append(float(np.sum(r ** 2) / denom) if denom > 0 else 0.0)
    return float(np.mean(fr))


def new_frequency_fraction(S, ZZ, support_threshold=0.01):
    """Mean over ZZ observables of the fraction of each ZZ observable's spectral
    power outside the single-qubit set's frequency support (bins where the summed
    single-qubit power exceeds `support_threshold` x its peak)."""
    Spow = np.sum([_psd(S[:, i]) for i in range(S.shape[1])], axis=0)
    support = Spow > support_threshold * Spow.max()
    fr = []
    for j in range(ZZ.shape[1]):
        zp = _psd(ZZ[:, j])
        tot = float(zp.sum())
        fr.append(float(zp[~support].sum() / tot) if tot > 0 else 0.0)
    return float(np.mean(fr))
