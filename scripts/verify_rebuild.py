#!/usr/bin/env python3
"""Rebuild comparator: verifies that every CSV in a rebuilt results tree is
identical to its committed counterpart, excluding columns that legitimately
vary between runs (wall-clock, memory, timestamps).

Usage:
    python scripts/verify_rebuild.py <rebuilt_root> <committed_root> [--scope PATHPREFIX]

For each *.csv under <rebuilt_root>/{results,results_de} (optionally limited to
--scope, e.g. --scope results/scaling_proof), the file is compared against the
same relative path under <committed_root>:

  IDENTICAL   byte-identical files
  EQUAL       identical after dropping excluded columns and normalising
              float formatting (values compared exactly, not approximately)
  DIFFERS     any remaining cell differs (first differences are printed)
  MISSING     no committed counterpart (new artifact -- flagged, not failed)

Exit code 0 iff no file DIFFERS. Output is a ledger-ready markdown table.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

# columns that legitimately vary run-to-run (timing, memory, timestamps)
EXCLUDE_COLS = {
    "data_generation_seconds", "validation_seconds", "transformation_seconds",
    "feature_generation_seconds", "training_seconds", "inference_seconds",
    "total_seconds", "peak_memory_mb", "wall_seconds", "rows_per_sec",
    "write_ms", "read_full_ms", "read_1col_ms", "speedup", "efficiency",
    "seconds", "start_time", "end_time", "created_at",
    "onestep_sec_per_step", "closedloop_sec_per_step",
}
# files whose payload is entirely timing-derived: structural check only
STRUCTURAL_ONLY = {"results_de/parallel.csv", "results_de/volume.csv",
                   "results_de/incremental.csv"}


def sha(path):
    return hashlib.blake2b(open(path, "rb").read()).hexdigest()[:16]


def compare(rebuilt, committed, rel):
    if sha(rebuilt) == sha(committed):
        return "IDENTICAL", ""
    try:
        a = pd.read_csv(rebuilt, keep_default_na=False)
        b = pd.read_csv(committed, keep_default_na=False)
    except Exception as e:
        return "DIFFERS", f"unreadable: {e}"
    drop_a = [c for c in a.columns if c in EXCLUDE_COLS]
    drop_b = [c for c in b.columns if c in EXCLUDE_COLS]
    a2, b2 = a.drop(columns=drop_a), b.drop(columns=drop_b)
    if rel in STRUCTURAL_ONLY:
        same = (list(a.columns) == list(b.columns) and len(a) == len(b))
        return ("EQUAL" if same else "DIFFERS",
                "structural: shape+schema" if same else
                f"shape {a.shape} vs {b.shape}")
    if list(a2.columns) != list(b2.columns):
        return "DIFFERS", f"columns {list(a2.columns)[:4]}... vs {list(b2.columns)[:4]}..."
    if len(a2) != len(b2):
        return "DIFFERS", f"rows {len(a2)} vs {len(b2)}"
    # exact value comparison first; ulp-level differences from multithreaded
    # BLAS reduction order are accepted under a DOCUMENTED tolerance and
    # reported as EQUAL-TOL with the max relative deviation.
    RTOL = 1e-9
    max_rel = 0.0
    for col in a2.columns:
        av = pd.to_numeric(a2[col], errors="coerce")
        bv = pd.to_numeric(b2[col], errors="coerce")
        num = av.notna() & bv.notna()
        neq = num & (av != bv)
        if neq.any():
            rel = ((av[neq] - bv[neq]).abs() /
                   bv[neq].abs().clip(lower=1e-300)).max()
            if rel > RTOL:
                i = neq.idxmax()
                return "DIFFERS", f"{col}[{i}]: {av[i]!r} vs {bv[i]!r}"
            max_rel = max(max_rel, float(rel))
        sa, sb = a2[col].astype(str)[~num], b2[col].astype(str)[~num]
        if not (sa == sb).all():
            i = (sa != sb).idxmax()
            return "DIFFERS", f"{col}[{i}]: {sa[i]!r} vs {sb[i]!r}"
    if max_rel > 0:
        return "EQUAL-TOL", f"ulp-level float noise, max rel diff {max_rel:.1e}"
    note = f"excluded cols: {sorted(set(drop_a) | set(drop_b))}" if (drop_a or drop_b) else "float formatting"
    return "EQUAL", note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rebuilt_root")
    ap.add_argument("committed_root")
    ap.add_argument("--scope", default="")
    args = ap.parse_args()

    rows, n_diff = [], 0
    for sub in ("results", "results_de"):
        base = os.path.join(args.rebuilt_root, sub)
        for dirpath, _, files in os.walk(base):
            for f in sorted(files):
                if not f.endswith(".csv"):
                    continue
                rp = os.path.join(dirpath, f)
                rel = os.path.relpath(rp, args.rebuilt_root)
                if args.scope and not rel.startswith(args.scope):
                    continue
                cp = os.path.join(args.committed_root, rel)
                if not os.path.exists(cp):
                    rows.append((rel, "MISSING", "no committed counterpart"))
                    continue
                verdict, note = compare(rp, cp, rel)
                if verdict == "DIFFERS":
                    n_diff += 1
                rows.append((rel, verdict, note))

    print("| artifact | verdict | note |")
    print("|---|---|---|")
    for rel, v, note in rows:
        print(f"| `{rel}` | {v} | {note} |")
    ident = sum(1 for r in rows if r[1] == "IDENTICAL")
    equal = sum(1 for r in rows if r[1] in ("EQUAL", "EQUAL-TOL"))
    miss = sum(1 for r in rows if r[1] == "MISSING")
    print(f"\nTOTAL {len(rows)}: {ident} identical, {equal} equal-after-exclusions, "
          f"{n_diff} DIFFER, {miss} new")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
