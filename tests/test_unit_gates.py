"""Pre-compute unit gates for the verification program (plan Tracks A-F).

Run before any sweep: each check guards an assumption a track's conclusion
rests on. Standalone-runnable: `python tests/test_unit_gates.py`.
"""
from __future__ import annotations

import numpy as np
import torch

from qdepipe.models.gate_qrc import GateQRC, GateQRCConfig
from qdepipe.models import _qops as Q
from qdepipe.pipeline.postprocess import LeakyMemory
from qdepipe.entanglement_probe import half_chain_entropy, reservoir_entropy_series


def _cfg(n=2, J=0.0, encoding="width", r=None, seed=0):
    return GateQRCConfig(n_qubits=n, encoding=encoding, r=(n if r is None else r),
                         coupling="isingxx", J_strength=J, channel="none",
                         V=4, window=5, readout=("Z", "X", "Y", "ZZ"), seed=seed)


# ---------------------------------------------------------------------------
def test_j0_fixed_layer_is_identity():
    """J_strength=0 => IsingXX(0) on every pair => fixed layer == identity."""
    q = GateQRC(_cfg(n=4, J=0.0))
    eye = torch.eye(2 ** 4, dtype=Q.CDTYPE, device=q.device)
    assert torch.allclose(q.U_fixed, eye, atol=1e-12), "J=0 fixed layer must be identity"


def test_j0_width_state_is_product():
    """width encoding on all qubits + J=0 => product state => S(rho_A) == 0."""
    q = GateQRC(_cfg(n=4, J=0.0))
    rng = np.random.default_rng(0)
    series = rng.uniform(0, 1, 40)
    S = reservoir_entropy_series(q, series, max_t=25)
    assert np.all(S < 1e-10), f"J=0 width state must be product; max S={S.max():.3e}"


def test_j1_state_is_entangled():
    """Default J=1 must actually generate entanglement (else 'ablation' is vacuous)."""
    q = GateQRC(_cfg(n=4, J=1.0))
    rng = np.random.default_rng(0)
    series = rng.uniform(0, 1, 40)
    S = reservoir_entropy_series(q, series, max_t=25)
    assert S.max() > 0.05, f"J=1 should entangle; max S={S.max():.3e} bits"


def test_entropy_bell_state():
    """Analytic anchor: Bell state has exactly 1 bit of half-chain entropy."""
    psi = torch.zeros(4, dtype=Q.CDTYPE)
    psi[0] = psi[3] = 1 / np.sqrt(2)
    S = half_chain_entropy(psi, 2)
    assert abs(S - 1.0) < 1e-10, f"Bell state entropy {S} != 1"


def test_leaky_eps1_is_identity():
    """LeakyMemory(eps=1): r_t = m_t exactly."""
    rng = np.random.default_rng(1)
    m = rng.normal(size=(50, 7))
    out = LeakyMemory(eps=1.0).fit(m).transform(m)
    assert np.allclose(out, m), "LeakyMemory(eps=1) must be the identity"


def test_leaky_is_causal():
    """Row t must not depend on rows > t (no future leakage through the filter)."""
    rng = np.random.default_rng(2)
    m = rng.normal(size=(30, 3))
    a = LeakyMemory(eps=0.3).fit(m).transform(m)
    m2 = m.copy()
    m2[20:] += 100.0                      # perturb only the future
    b = LeakyMemory(eps=0.3).fit(m2).transform(m2)
    assert np.allclose(a[:20], b[:20]), "leaky filter leaked future into past"


def test_ic_draws_reproducible():
    """IC draws from a fixed-seed RNG are identical across calls."""
    def draw():
        r = np.random.default_rng(7)
        return (r.uniform(-0.5, 0.5, 20), r.uniform(-0.3, 0.3, 20),
                r.normal(0, 2.0, (20, 3)))
    a, b = draw(), draw()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_cache_byte_identity_new_config():
    """Gate-(b) extension: width/J=0 (a NEW config family) cache-on == cache-off."""
    import tempfile
    from qdepipe.feature_store import FeatureStore, maybe_cached_featurize
    from qdepipe.experiment import ExperimentConfig
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 1, 60)
    cfg = ExperimentConfig(system="henon", n_points=60)
    raw = maybe_cached_featurize(None, GateQRC(_cfg(n=4, J=0.0)), u, cfg)
    with tempfile.TemporaryDirectory() as d:
        store = FeatureStore(root=d)
        miss = maybe_cached_featurize(store, GateQRC(_cfg(n=4, J=0.0)), u, cfg)
        hit = maybe_cached_featurize(store, GateQRC(_cfg(n=4, J=0.0)), u, cfg)
    assert miss.tobytes() == raw.tobytes(), "cache MISS diverges from raw"
    assert hit.tobytes() == raw.tobytes(), "cache HIT diverges from raw"


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} unit gates passed")
    sys.exit(1 if failed else 0)
