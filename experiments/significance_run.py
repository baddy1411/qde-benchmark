#!/usr/bin/env python3
"""Significance analysis for the headline comparisons.

The benchmark reports mean NRMSE; this script attaches **statistical verdicts** to
the model differences via Diebold–Mariano (pairwise) and the Model Confidence Set
(which models are indistinguishable from the best).

Design rule for a valid test: every model in a comparison runs through ONE shared,
declared pipeline (identical data, split, scaling, lookback, horizon) so the
per-step forecast errors are aligned on the same test timestamps. Only the model —
and each quantum model's own best encoding — changes. We persist the raw per-seed
test forecasts (so the tests are reproducible from disk), run DM/MCS **per seed**,
then aggregate across seeds (the convention chosen in GAPS_PLAN §4-A).

Outputs (per system):
  results/forecasts/{system}/{model}_seed{n}.csv   raw aligned (t, y_true, y_pred)
  results/significance/{system}_DM_matrix.csv       pairwise mean-DM / p (square)
  results/significance/{system}_DM_pairs.csv        long form incl. frac-significant
  results/significance/{system}_MCS.csv             per-model MCS p-value + membership
  results/SIGNIFICANCE.md                            auto narrative with verdicts

Usage:
  ../venv/bin/python significance_run.py                 # henon, seeds 0..4
  ../venv/bin/python significance_run.py --system henon --seeds 5
  ../venv/bin/python significance_run.py --quick
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.registry import build_forecaster
from qdepipe.models import ESN, ESNConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe import significance as sig

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# best encode_scale per quantum model (from experiments_advanced.py, Experiment A)
BEST_ENCODE = {"qrc_v4": 6.0, "qrc_v6": 1.0, "qrc_rich": 0.5}
# QRC-rich's feature count, for the size-matched ESN
F_RICH = 96


def _log(m): print(m, flush=True)


def shared_config(system: str = "henon") -> ExperimentConfig:
    """The single declared pipeline all models share for the significance test.

    n_points=1500 keeps the quantum models tractable; lookback/horizon fixed so the
    test timestamps line up across every model.
    """
    return ExperimentConfig(n_points=1500, scaler="minmax", scaler_scope="train",
                            split_fracs=(0.6, 0.2, 0.2), washout=100, horizon=1,
                            lookback=5, alpha=1e-6)


def model_specs():
    """label -> (build(seed, cfg) -> Forecaster, per-model cfg overrides).

    The size-matched ESN (units = F_RICH) is the fair classical counterpart to
    QRC-rich; the quantum models carry their own best encoding.
    """
    specs = {}

    def reg(key, **ov):
        specs[key] = (lambda s, c, k=key: build_forecaster(k, s, c), ov)

    reg("ngrc")
    reg("esn")
    reg("elm")
    reg("random_forest")
    reg("qrc_v4", encode_scale=BEST_ENCODE["qrc_v4"])
    reg("qrc_v6", encode_scale=BEST_ENCODE["qrc_v6"])
    reg("qrc_rich", encode_scale=BEST_ENCODE["qrc_rich"])
    # size-matched ESN (F = QRC-rich's 96) — the matched-budget classical baseline
    specs[f"esn_matched_F{F_RICH}"] = (
        lambda s, c: ReservoirForecaster(ESN(ESNConfig(units=F_RICH, seed=s)),
                                         f"ESN(F={F_RICH})", external_window=True),
        {"windowing": "internal"})
    return specs


def _align(forecasts: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Trim every (y_true, y_pred) to the common test suffix (same final timestamp →
    the last L points share timestamps). Sanity-checks y_true agreement on overlap."""
    L = min(len(v[0]) for v in forecasts.values())
    out, ref = {}, None
    for m, (yt, yp) in forecasts.items():
        yt2, yp2 = np.asarray(yt)[-L:], np.asarray(yp)[-L:]
        out[m] = (yt2, yp2)
        if ref is None:
            ref = yt2
        elif not np.allclose(ref, yt2, atol=1e-8):
            # different targets ⇒ misaligned pipeline; fail loud rather than fake a test
            raise AssertionError(f"target mismatch for {m}: pipelines are not aligned")
    return out, L


def run_system(system: str, seeds, quick: bool):
    _log(f"[significance] system={system} seeds={list(seeds)}")
    base = shared_config(system)
    specs = model_specs()
    fdir = os.path.join(RESULTS, "forecasts", f"{system}_quantum")
    os.makedirs(fdir, exist_ok=True)

    # collect per-seed aligned forecasts
    per_seed = []        # list over seeds of {model: (yt, yp)}
    for s in seeds:
        fc_s = {}
        for label, (build, ov) in specs.items():
            cfg = replace(base, seed=s, **ov)
            r = run_experiment(build(s, cfg), cfg, keep_forecasts=True)
            fc_s[label] = (r.y_true, r.y_pred)
        aligned, L = _align(fc_s)
        for label, (yt, yp) in aligned.items():
            pd.DataFrame({"t": np.arange(L), "y_true": yt, "y_pred": yp}).to_csv(
                os.path.join(fdir, f"{label}_seed{s}.csv"), index=False)
        per_seed.append(aligned)
        _log(f"    seed {s}: {len(aligned)} models, {L} aligned test steps")

    names = list(specs)
    # ---- DM per seed, then aggregate -----------------------------------------
    n = len(names)
    dm_stats = np.full((len(seeds), n, n), np.nan)
    dm_ps = np.full((len(seeds), n, n), np.nan)
    for si, aligned in enumerate(per_seed):
        stat, pval, nm = sig.dm_matrix(aligned, h=1, loss="se")
        order = [nm.index(x) for x in names]
        dm_stats[si] = stat[np.ix_(order, order)]
        dm_ps[si] = pval[np.ix_(order, order)]
    mean_stat = np.nanmean(dm_stats, axis=0)
    median_p = np.nanmedian(dm_ps, axis=0)
    frac_sig = np.mean(dm_ps < 0.05, axis=0)

    # square matrix CSV: lower-tri = mean DM stat, upper-tri = median p
    sq = np.where(np.tril(np.ones((n, n)), -1).astype(bool), mean_stat, median_p)
    np.fill_diagonal(sq, 0.0)
    sdir = os.path.join(RESULTS, "significance")
    os.makedirs(sdir, exist_ok=True)
    pd.DataFrame(sq, index=names, columns=names).to_csv(
        os.path.join(sdir, f"{system}_quantum_DM_matrix.csv"))
    # long form
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows.append({"model_i": names[i], "model_j": names[j],
                         "mean_dm": mean_stat[i, j], "median_p": median_p[i, j],
                         "frac_seeds_sig": frac_sig[i, j]})
    pd.DataFrame(rows).to_csv(os.path.join(sdir, f"{system}_quantum_DM_pairs.csv"), index=False)

    # ---- MCS per seed, then aggregate ----------------------------------------
    in10 = {m: 0 for m in names}
    pvals = {m: [] for m in names}
    n_boot = 300 if quick else 1000
    for aligned in per_seed:
        losses = np.stack([(yp - yt) ** 2 for (yt, yp) in
                           [aligned[m] for m in names]], axis=0)
        res = sig.model_confidence_set(losses, names=names, alpha=(0.10, 0.25),
                                       n_boot=n_boot, block_length=20, seed=0)
        for m in names:
            pvals[m].append(res["mcs_pvalue"][m])
            if m in res["in_set"][0.10]:
                in10[m] += 1
    mcs_rows = [{"model": m, "mean_mcs_pvalue": float(np.mean(pvals[m])),
                 "frac_seeds_in_set_0.10": in10[m] / len(seeds)} for m in names]
    mcs_df = pd.DataFrame(mcs_rows).sort_values("mean_mcs_pvalue", ascending=False)
    mcs_df.to_csv(os.path.join(sdir, f"{system}_quantum_MCS.csv"), index=False)

    return names, mean_stat, median_p, frac_sig, mcs_df


def headline_verdicts(system, names, median_p, frac_sig):
    """The three forced pairs from the brief, with a plain-English verdict."""
    pairs = [("qrc_rich", f"esn_matched_F{F_RICH}"), ("qrc_rich", "ngrc"),
             ("qrc_rich", "qrc_v6")]
    out = []
    for a, b in pairs:
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            p, fr = median_p[i, j], frac_sig[i, j]
            verdict = ("significant" if p < 0.05 else
                       "NOT significant" if p > 0.10 else "borderline")
            out.append({"pair": f"{a} vs {b}", "median_p": p,
                        "frac_seeds_sig": fr, "verdict": verdict})
    return out


def write_report(results_by_system):
    L = ["# QDE — Significance Verdicts (auto-generated)\n",
         "*Diebold–Mariano (pairwise equal-accuracy) and Model Confidence Set, run "
         "per seed then aggregated. Squared-error loss, 1-step (HAC lag 0, HLN "
         "correction). p < 0.05 ⇒ the accuracy difference exceeds seed noise.*\n"]
    for system, (names, mean_stat, median_p, frac_sig, mcs_df) in results_by_system.items():
        L.append(f"## System: {system}\n")
        L.append("### Headline pairwise verdicts (Diebold–Mariano)\n")
        hv = pd.DataFrame(headline_verdicts(system, names, median_p, frac_sig))
        L.append(_md(hv) + "\n")
        L.append("### Model Confidence Set (α = 0.10)\n")
        L.append(_md(mcs_df) + "\n")
        winners = mcs_df[mcs_df["frac_seeds_in_set_0.10"] >= 0.5]["model"].tolist()
        L.append(f"**FINDING [{system}]:** the MCS (indistinguishable-from-best set) "
                 f"is essentially {{{', '.join(winners)}}}. Full matrices: "
                 f"`results/significance/{system}_DM_matrix.csv`, `..._MCS.csv`.\n")
    L.append("\n---\n*Regenerated by `significance_run.py` (`make significance`) from the "
             "persisted per-seed forecasts in `results/forecasts/`.*\n")
    with open(os.path.join(RESULTS, "SIGNIFICANCE.md"), "w") as f:
        f.write("\n".join(L))
    _log("    wrote results/SIGNIFICANCE.md")


def _md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".3g")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="henon")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 2 if args.quick else args.seeds
    seeds = tuple(range(n))
    os.makedirs(RESULTS, exist_ok=True)
    from qdepipe._provenance import write_versions
    write_versions(os.path.join(os.path.dirname(RESULTS), "versions.txt"))
    res = run_system(args.system, seeds, args.quick)
    write_report({args.system: res})
    _log("done — see results/SIGNIFICANCE.md")


if __name__ == "__main__":
    main()
