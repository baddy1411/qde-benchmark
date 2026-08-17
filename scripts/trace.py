#!/usr/bin/env python3
"""Trace one reported number back to what produced it.

Answers the examiner's question "where did this number come from?" in one command,
and is honest about where the chain is strong and where it is not.

Two provenance layers exist, and they cover different things:

  REGISTRY   results_registry/*.parquet -- 360 scoreboard runs in four joinable
             tables. A number found here resolves to its model, seed, dataset
             (system parameters, initial condition, content hash) and split
             (exact ranges, scaler fitting range, content hash).

  ARTIFACTS  results/**.csv -- every other study: the scaling family, the
             concentration and finite-shot work, the IC study, entanglement, the
             rescue suite. These are outside the queryable layer, so the tool
             locates the committed CSV, the row, and the column instead.

Run:  .venv/bin/python scripts/trace.py 0.02003424617616511
      .venv/bin/python scripts/trace.py 1.11e-3 --tol 1e-3
      .venv/bin/python scripts/trace.py --run <run_id>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

W = 78


def rule(ch="-"):
    print("  " + ch * (W - 4))


def field(k, v):
    print(f"    {k:<26s} {v}")


def show_registry_hit(run_id):
    from qdepipe.registry_io import read_table
    runs, metrics = read_table("runs"), read_table("metrics")
    splits, datasets = read_table("splits"), read_table("datasets")
    r = runs[runs.run_id == run_id].iloc[0]
    m = metrics[metrics.run_id == run_id].iloc[0]
    s = splits[splits.split_id == r.split_id].iloc[0]
    d = datasets[datasets.dataset_id == r.dataset_id].iloc[0]

    print(f"\n  FOUND IN THE RUN REGISTRY\n")
    rule()
    print("  THE RUN")
    field("run_id", r.run_id)
    field("model", f"{r.model}  (family: {r.model_family})")
    field("seed", r.seed)
    field("configuration_id", r.configuration_id)
    field("forecast_mode", r.forecast_mode)
    field("status", r.status)
    rule()
    print("  THE DATA IT USED")
    field("dataset_id", d.dataset_id)
    field("system", d.system)
    field("system_parameters", d.system_parameters)
    field("initial_condition", d.initial_condition)
    field("random_seed", d.random_seed)
    field("sample_count", d.sample_count)
    field("transient_removed", d.transient_removed)
    field("dataset_hash", d.dataset_hash)
    rule()
    print("  HOW IT WAS SPLIT")
    field("split_id", s.split_id)
    field("train", f"{s.train_start} - {s.train_end}")
    field("validation", f"{s.validation_start} - {s.validation_end}")
    field("test", f"{s.test_start} - {s.test_end}")
    field("scaler fitted on", s.scaler_fit_range)
    field("window / horizon", f"{s.window_length} / {s.prediction_horizon}")
    field("split_hash", s.split_hash)
    rule()
    print("  WHAT WAS MEASURED")
    for k in ("nrmse", "mae", "valid_prediction_time", "feature_count", "feature_rank"):
        if k in m and pd.notna(m[k]):
            field(k, m[k])
    rule()
    print("  WHAT THIS LAYER DOES NOT RECORD")
    field("provenance_source", f"{r.provenance_source}  <- lineage rebuilt from artifacts,")
    field("", "   not captured while the run executed")
    field("git_commit", f"{r.git_commit}  <- not recorded per run")
    field("environment_id", f"{r.environment_id}  <- not recorded per run")
    print()
    print("  The independent evidence that these records are faithful is the full")
    print("  re-execution: every artifact regenerated from code and data alone.")


def _rel(series, value):
    """Relative error against the target. RELATIVE only, deliberately.

    An absolute tolerance is wrong for this job: with --tol 1e-2, tracing 0.00111
    matches 0.000208, and the tool then prints a complete, confident, WRONG
    lineage. Being unable to find a number is a recoverable answer; naming the
    wrong run for it is not.
    """
    return (series - value).abs() / max(abs(value), 1e-300)


def search_registry(value, tol):
    from qdepipe.registry_io import read_table
    metrics = read_table("metrics")
    num = metrics.select_dtypes("number")
    hits = []
    for col in num.columns:
        rel = _rel(num[col], value)
        for i in num.index[rel <= tol]:
            hits.append((float(rel[i]), metrics.loc[i, "run_id"], col, num.loc[i, col]))
    return sorted(hits, key=lambda h: h[0])


def search_artifacts(value, tol, limit=6):
    hits = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.csv"), recursive=True)):
        try:
            df = pd.read_csv(path, float_precision="round_trip")
        except Exception:
            continue
        num = df.select_dtypes("number")
        if num.empty:
            continue
        for col in num.columns:
            rel = _rel(num[col], value)
            for i in num.index[rel <= tol][:3]:
                ctx = {c: df.loc[i, c] for c in df.columns
                       if not pd.api.types.is_numeric_dtype(df[c])}
                hits.append((float(rel[i]), os.path.relpath(path, ROOT), i + 2, col,
                             num.loc[i, col], ctx))
        if len(hits) >= limit * 3:
            break
    return sorted(hits, key=lambda h: h[0])[:limit]


def match_quality(rel):
    if rel == 0.0:
        return "EXACT match"
    if rel <= 1e-12:
        return f"exact to double precision (relative error {rel:.1e})"
    return (f"APPROXIMATE match, relative error {rel:.2e} -- this is NOT the same "
            f"number.\n            Re-run with a tighter --tol to confirm identity.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("value", nargs="?", type=float, help="the number to trace")
    ap.add_argument("--run", help="trace a run_id directly")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="absolute OR relative match tolerance (default 1e-9)")
    args = ap.parse_args()

    print()
    print("=" * W)
    print("  PROVENANCE TRACE")
    print("=" * W)

    if args.run:
        show_registry_hit(args.run)
        return

    if args.value is None:
        ap.error("give a number to trace, or --run <run_id>")

    print(f"\n  tracing: {args.value!r}   (tolerance {args.tol:g})")

    reg = search_registry(args.value, args.tol)
    if reg:
        rel, run_id, col, found = reg[0]
        print(f"  matched registry column '{col}' = {found!r}")
        print(f"  match quality: {match_quality(rel)}")
        show_registry_hit(run_id)
        if len(reg) > 1:
            print(f"\n  ({len(reg)} registry rows within tolerance; showing the closest)")
        return

    print("\n  Not in the run registry -- that layer covers the 360 scoreboard runs.")
    print("  Searching the committed result artifacts instead.\n")
    art = search_artifacts(args.value, args.tol)
    if not art:
        print("  NOT FOUND. Widen the tolerance, e.g. --tol 1e-4.")
        print("  If it is a configuration constant rather than a measurement, it")
        print("  traces to a source file -- see docs/MANIFEST.csv.")
        return

    print("  FOUND IN A COMMITTED ARTIFACT\n")
    for rel, path, row, col, found, ctx in art[:4]:
        rule()
        field("file", path)
        field("row (1-based, with header)", row)
        field("column", col)
        field("value", repr(found))
        field("match quality", match_quality(rel))
        for k, v in list(ctx.items())[:6]:
            field(f"  {k}", v)
    rule()
    print("\n  These studies sit outside the queryable registry by design; the CSV is")
    print("  the artifact of record, it is committed, and the script that writes it")
    print("  is named in docs/MANIFEST.csv alongside the claim.")


if __name__ == "__main__":
    main()
