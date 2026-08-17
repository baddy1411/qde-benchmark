"""Gate (b) for path 6 (matched_budget_shots) — last and strictest.

Unlike paths 1-5 (which proved scale-independent MECHANISM transparency on tiny
proxies), this runs at the REAL n_points=1500 across all three shot counts,
because the claim is specifically that THESE published numbers
(exact=1.111e-3 / 8192->0.499 / 1024->0.800 / 18x margin) are cache-transparent.
Two distinct checks:

  (a) byte-identical cache off vs on  -> the cache does not move the number;
  (b) cache-off reproduces the committed CSV the thesis cites -> the number the
      cache doesn't move is the same number that was published.

Marked `slow` (excluded from the default suite); run with `pytest -m slow`.
"""
from __future__ import annotations

import tempfile

import pandas as pd
import pytest

from qdepipe.concentration import matched_budget_shots
from qdepipe.feature_store import FeatureStore

SEEDS = (0, 1, 2, 3, 4)   # the seeds behind the committed 5-seed means


@pytest.mark.slow
def test_gate_b_matched_budget_real_n1500():
    quiet = lambda *a, **k: None
    off = matched_budget_shots(seeds=SEEDS, log=quiet, store=None)

    store = FeatureStore(root=tempfile.mkdtemp(), namespace="mb")
    on = matched_budget_shots(seeds=SEEDS, log=quiet, store=store)

    # (a) the cache does not move the published numbers
    assert off == on, "cache moved the finite-shot headline (gate b violation)"

    # (b) cache-off reproduces the committed CSV the thesis cites
    committed = pd.read_csv("results/concentration/finite_shot_budget.csv")
    committed["shots"] = committed["shots"].astype(str)
    cmap = {r["shots"]: r for _, r in committed.iterrows()}
    for r in off:
        cm = cmap[str(r["shots"])]
        assert abs(r["NRMSE_quantum"] - float(cm["NRMSE_quantum"])) < 1e-9, r["shots"]
        assert abs(r["margin_ratio_esn_over_qrc"] - float(cm["margin_ratio_esn_over_qrc"])) < 1e-6, r["shots"]

    # within one run, 3 shots x 5 seeds = 15 distinct keys -> no intra-run reuse
    assert store.stats["misses"] == 15 and store.stats["hits"] == 0
