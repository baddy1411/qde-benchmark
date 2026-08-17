#!/usr/bin/env python3
"""Navigate from an experiment to the script that ran it and the results it produced.

For the question "show me your <X> experiment". The index is not hand-written.
It was extracted from the thesis results chapter itself, from the \\artifact{}
footnote the thesis attaches to every claim, and committed here as
docs/experiment_index.json. So an experiment can only appear in this list if the
thesis cites an artifact for it.

  python scripts/experiments.py              list every experiment
  python scripts/experiments.py shot         show the finite-shot one
  python scripts/experiments.py 4            show experiment 4
  python scripts/experiments.py entangle --full    print whole CSVs

To rebuild the index from a LaTeX source tree, run with --from-chapter <path>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "docs", "experiment_index.json")
W = 96

# Which script produces a given results directory or file. The thesis footnotes
# name the artifact, not the producer, so this is the one hand-maintained part.
PRODUCER = {
    "results/scaling_proof": "experiments/run_scaling_proof.py   (n=12 arm: experiments/run_n12_attempt.py)",
    "results/entanglement": "experiments/run_entanglement.py",
    "results/concentration": "experiments/concentration_run.py   (cross-system arm: scripts/noise_arm_cross_system.py)",
    "results/followup": "experiments/run_followup_tricks.py",
    "results/cheb": "experiments/run_cheb_encoding.py   (rank diagnostic: experiments/run_cheb_rank.py)",
    "results/rfqrc": "experiments/run_rfqrc.py   (controls: experiments/run_rfqrc_mv_control.py)",
    "results/rfqrc_cheb": "experiments/run_rfqrc_cheb.py",
    "results/dissipative": "experiments/run_dissipative_qrc.py   (tuned control: experiments/run_mc_tuned_esn.py)",
    "results/ic_robustness": "experiments/run_ic_study.py",
    "results/lorenz96": "experiments/run_lorenz96.py",
    "results/leaky": "experiments/run_leaky.py",
    "results/cross": "experiments/cross_system.py   (quantum arms: experiments/quantum_cross.py)",
    "results/significance": "experiments/cross_system.py",
    "results/pretrained": "experiments/run_pretrained_zeroshot.py",
    "results/model_scoreboard_all_runs.csv": "experiments/synthesize.py",
    "results/baseline_leaderboard.csv": "experiments/run_catalogue_ext.py",
    "results/adv_A2_matched.csv": "experiments/experiments_advanced.py",
    "results/adv_B_zz_ablation.csv": "experiments/experiments_advanced.py",
    "results/adv_D_climate.csv": "experiments/experiments_advanced.py",
    "results/headline_20seed.csv": "experiments/run_headline_20seed.py",
    "results/esn_budget_tune.csv": "experiments/run_esn_budget_tune.py",
}


def producer_for(path):
    for key in sorted(PRODUCER, key=len, reverse=True):
        if path == key or path.startswith(key.rstrip("/") + "/"):
            return PRODUCER[key]
    return "(see docs/MANIFEST.csv)"


def artifact_args(src):
    """Pull every \\artifact{...} argument, honouring ESCAPED braces.

    A plain \\artifact\\{([^}]+)\\} is wrong here: the thesis writes brace
    alternations inside the argument as \\{scores,dm,trend\\}, so the regex stops
    at the first escaped closing brace and silently drops the ".csv" -- which then
    shows up as a missing artifact. Scan with a depth counter instead, treating a
    backslash-escaped brace as literal text rather than nesting.
    """
    out, i, tag = [], 0, "\\artifact{"
    while (j := src.find(tag, i)) != -1:
        k, depth, buf = j + len(tag), 1, []
        while k < len(src) and depth:
            c = src[k]
            if c == "\\" and k + 1 < len(src) and src[k + 1] in "{}_":
                buf.append(src[k + 1]); k += 2; continue   # escaped brace or underscore
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if not depth:
                    break
            buf.append(c); k += 1
        out.append("".join(buf))
        i = k + 1
    return out


def expand(a):
    """results/x/{p,q}.csv -> [results/x/p.csv, results/x/q.csv]"""
    a = a.replace("\\allowbreak{}", "").replace("\\", "").strip()
    m = re.search(r"\{([^}]*)\}", a)
    if not m:
        return [a]
    return [a[:m.start()] + part.strip() + a[m.end():] for part in m.group(1).split(",")]


def load_index():
    """The committed index, extracted from the thesis by parse_chapter()."""
    with open(INDEX, encoding="utf-8") as fh:
        return json.load(fh)


def parse_chapter(chapter_path):
    src = open(chapter_path, encoding="utf-8").read()
    parts = re.split(r"\\section\{(.+?)\}", src)
    out = []
    for i in range(1, len(parts), 2):
        title, body = parts[i], parts[i + 1]
        label = (re.search(r"\\label\{(sec:[^}]+)\}", body) or [None, ""])[1]
        arts, seen = [], set()
        for a in artifact_args(body):
            for e in expand(a):
                if e.startswith("results") and e not in seen:
                    seen.add(e); arts.append(e)
        clean = re.sub(r"\\[A-Za-z]+\{?|\}", "", title).strip()
        out.append({"title": clean, "label": label, "artifacts": arts})
    return out


def show(exp, full=False):
    print()
    print("=" * W)
    print(f"  {exp['title']}")
    print("=" * W)
    print(f"  thesis section   {exp['label']}   (results chapter)")

    prods = []
    for a in exp["artifacts"]:
        p = producer_for(a)
        if p not in prods:
            prods.append(p)
    print(f"  produced by      {prods[0] if prods else '(unknown)'}")
    for p in prods[1:]:
        print(f"{'':19s}{p}")

    if not exp["artifacts"]:
        print("\n  No artifact footnote in this section — the numbers are configuration")
        print("  constants or come from a figure. See docs/MANIFEST.csv.")
        return

    print(f"\n  ARTIFACTS")
    for a in exp["artifacts"]:
        full_path = os.path.join(ROOT, a)
        ok = os.path.exists(full_path)
        size = f"{os.path.getsize(full_path):,} bytes" if ok else "NOT FOUND"
        print(f"    {'[ok]' if ok else '[--]'} {a:<58s} {size}")

    for a in exp["artifacts"]:
        full_path = os.path.join(ROOT, a)
        if not (os.path.exists(full_path) and a.endswith(".csv")):
            continue
        try:
            df = pd.read_csv(full_path, float_precision="round_trip")
        except Exception as e:
            print(f"\n  {a}: could not read ({e})")
            continue
        print()
        print("  " + "-" * (W - 4))
        print(f"  {a}   ({len(df)} rows x {len(df.columns)} cols)")
        print("  " + "-" * (W - 4))
        with pd.option_context("display.width", W, "display.max_columns", 12,
                               "display.max_colwidth", 22):
            body = df.to_string(index=False) if full else df.head(12).to_string(index=False)
        for line in body.splitlines():
            print("    " + line[:W - 6])
        if not full and len(df) > 12:
            print(f"    ... {len(df) - 12} more rows   (--full to print all)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="number, or part of an experiment name")
    ap.add_argument("--full", action="store_true", help="print whole CSVs, not the head")
    ap.add_argument("--from-chapter", metavar="PATH",
                    help="re-extract the index from a LaTeX results chapter and rewrite "
                         "docs/experiment_index.json")
    args = ap.parse_args()

    if args.from_chapter:
        exps = parse_chapter(args.from_chapter)
        with open(INDEX, "w", encoding="utf-8") as fh:
            json.dump(exps, fh, indent=1)
        print(f"wrote {INDEX}: {len(exps)} experiments")
        return

    exps = load_index()

    if not args.query:
        print()
        print("=" * W)
        print("  EXPERIMENTS IN THE RESULTS CHAPTER")
        print("=" * W)
        print(f"  {'#':>3s}  {'EXPERIMENT':<46s} {'ARTIFACTS':>9s}   PRODUCED BY")
        print("  " + "-" * (W - 4))
        for i, e in enumerate(exps, 1):
            prod = producer_for(e["artifacts"][0]).split("(")[0].strip() if e["artifacts"] else "-"
            print(f"  {i:>3d}  {e['title'][:46]:<46s} {len(e['artifacts']):>9d}   {prod}")
        print()
        print("  Show one:  python scripts/experiments.py <number or keyword>")
        print("  Example:   python scripts/experiments.py shot")
        return

    if args.query.isdigit():
        i = int(args.query)
        if not 1 <= i <= len(exps):
            sys.exit(f"no experiment {i}; there are {len(exps)}")
        show(exps[i - 1], args.full)
        return

    q = args.query.lower()
    hits = [e for e in exps if q in e["title"].lower() or q in e["label"].lower()]
    if not hits:
        sys.exit(f"nothing matches {args.query!r}; run without arguments to list them")
    for e in hits:
        show(e, args.full)


if __name__ == "__main__":
    main()
