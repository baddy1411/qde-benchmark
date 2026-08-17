"""Fairness gate: executable comparison contracts (axis: research design).

Before two runs may be statistically compared (DM, MCS, ratio tables), the
gate PROVES the comparison is fair and raises otherwise. This turns the
fairness rules of the research-design chapter from prose into a blocking
check: an unfair comparison does not produce a wrong number, it produces no
number.

Contract checked across all runs entering one comparison:
  - identical dataset identity (same system, generation config, seed policy)
  - identical split identity (ranges, horizon, washout)
  - leakage-safe scaler scope ('train') on every run
  - identical prediction horizon
  - matched measured feature count F (within declared tolerance, default 0)
  - identical seed set (when seed_set is declared)
  - identical tuning budget (when declared)

Usage:
    from qdepipe.fairness_gate import FairnessViolation, assert_fair_comparison
    assert_fair_comparison([run_meta_a, run_meta_b])   # raises or passes

`run_meta` is a plain dict; missing OPTIONAL keys are skipped, but the five
REQUIRED keys must be present -- absence is itself a violation, because a
comparison whose fairness cannot be established must not run.
"""
from __future__ import annotations

REQUIRED = ("dataset_id", "split_id", "scaler_scope", "horizon", "n_features")
OPTIONAL_SAME = ("seed_set", "tuning_budget")


class FairnessViolation(RuntimeError):
    """Raised when a comparison would violate the fairness contract."""


def _same(values):
    return all(v == values[0] for v in values[1:])


def assert_fair_comparison(runs, f_tolerance: int = 0):
    """Validate that the runs form a fair comparison group.

    runs: sequence of dicts with the REQUIRED keys (and optionally
    OPTIONAL_SAME keys). Raises FairnessViolation with a precise reason.
    Returns True when the contract holds, so it can be used in asserts.
    """
    if len(runs) < 2:
        raise FairnessViolation("a comparison needs at least two runs")

    for i, r in enumerate(runs):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise FairnessViolation(
                f"run {i}: fairness cannot be established, missing {missing}")

    for key in ("dataset_id", "split_id", "horizon"):
        vals = [r[key] for r in runs]
        if not _same(vals):
            raise FairnessViolation(f"unequal {key}: {vals}")

    scopes = [r["scaler_scope"] for r in runs]
    bad = [s for s in scopes if s != "train"]
    if bad:
        raise FairnessViolation(
            f"leakage-unsafe scaler scope in comparison: {scopes} "
            f"(every run must use 'train')")

    fs = [int(r["n_features"]) for r in runs]
    if max(fs) - min(fs) > f_tolerance:
        raise FairnessViolation(
            f"unmatched measured feature counts: {fs} "
            f"(tolerance {f_tolerance})")

    for key in OPTIONAL_SAME:
        declared = [r[key] for r in runs if key in r]
        if declared and (len(declared) != len(runs) or not _same(declared)):
            raise FairnessViolation(
                f"unequal or partially-declared {key}: "
                f"{[r.get(key, '<undeclared>') for r in runs]}")

    return True
