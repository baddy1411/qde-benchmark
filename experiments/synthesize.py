#!/usr/bin/env python3
"""synthesize.py — the cross-model synthesis phase (THESIS_PLAN §5–6).

The per-model notebooks each answer "how does *this* model respond to the
data-engineering axes". This script does the part no single notebook can: it
aggregates *across* all 23 models to produce the thesis deliverables —

    1. Baseline leaderboard      — every model at its sensible-default pipeline.
    2. RQ1  axis sensitivity     — per-model ratio swing for each axis, and
                                   whether the axis *ranking* is universal.
    3. RQ2  leakage study        — train-only vs global scaling (the artifact).
    4. RQ3  robust pipeline       — over the quantum models, the pipeline that is
                                   best by maximin / mean / variance of NRMSE.
    5. RQ3  matched-budget table — quantum vs best classical at comparable F.

Everything reuses the validated `qdepipe` machinery (registry + studies), so the
numbers here are produced by exactly the same code paths the notebooks use — the
synthesis is an aggregation, not a re-implementation.

Outputs land in ``results/`` as CSVs plus a human-readable ``SYNTHESIS.md`` that
strings the auto-generated FINDING lines together into a draft results section.

Usage
-----
    ../venv/bin/python synthesize.py              # full run (all models, 3 seeds)
    ../venv/bin/python synthesize.py --quick      # fast smoke: classical-only, 1 seed
    ../venv/bin/python synthesize.py --no-quantum # skip the compute-heavy QRCs
    ../venv/bin/python synthesize.py --seeds 5    # more seeds for tighter error bars

The quantum reservoirs re-simulate a circuit per feature row, so the full run is
compute-heavy; `--quick` exists to check the plumbing end-to-end in ~a minute.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment
from qdepipe.registry import MODELS, base_config, build_forecaster
from qdepipe import studies

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# Axes that have preset levels in studies.AXIS_LEVELS *and* can be swept here.
SWEEPABLE = [a for a in ("scaler", "scope", "split", "washout", "lookback",
                         "windowing", "horizon", "alpha", "postprocess", "encoding")]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(msg, flush=True)


def _save(df: pd.DataFrame, name: str):
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=False)
    _log(f"    wrote {os.path.relpath(path)}  ({len(df)} rows)")
    return path


def _md(df: pd.DataFrame, index=False, floatfmt=".4g") -> str:
    """Markdown table, falling back to plain text if `tabulate` isn't installed."""
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def _ratio_swing(agg_df: pd.DataFrame) -> float:
    """max/min of the per-level mean NRMSE — scale-free axis impact."""
    m = agg_df["mean"].to_numpy(dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return float("nan")
    lo = np.min(m)
    if lo <= 0:                       # NG-RC can hit ~1e-7; guard the divide
        lo = max(lo, 1e-12)
    return float(np.max(m) / lo)


# ---------------------------------------------------------------------------
# 1. baseline leaderboard
# ---------------------------------------------------------------------------
def baseline_leaderboard(keys, seeds) -> pd.DataFrame:
    _log("[1/5] baseline leaderboard")
    rows = []
    for k in keys:
        e = MODELS[k]
        t0 = time.time()
        df = studies.baseline(k, base_config(k), seeds=seeds)
        rows.append({
            "key": k, "model": e.name, "group": e.group,
            "nrmse_mean": df["nrmse"].mean(), "nrmse_std": df["nrmse"].std(ddof=0),
            "rmse_mean": df["rmse"].mean(), "mae_mean": df["mae"].mean(),
            "r2_mean": df["r2"].mean(),
            "n_features": int(df["n_features"].iloc[0]),
            "n_params": (None if pd.isna(df["n_params"].iloc[0])
                         else int(df["n_params"].iloc[0])),
        })
        _log(f"    {e.name:22s} NRMSE={rows[-1]['nrmse_mean']:.4g}"
             f" ({time.time()-t0:.1f}s)")
    out = pd.DataFrame(rows).sort_values("nrmse_mean").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# 2. RQ1 — per-model axis sensitivity + universality of the ranking
# ---------------------------------------------------------------------------
def axis_sensitivity(keys, seeds):
    _log("[2/5] RQ1 axis sensitivity (ratio swing per model x axis)")
    records = []
    for k in keys:
        e = MODELS[k]
        axes = [a for a in e.axes if a in SWEEPABLE]
        for a in axes:
            try:
                df = studies.axis_sweep(k, a, base_config(k), seeds=seeds)
                swing = _ratio_swing(studies.agg(df, "nrmse"))
            except Exception as ex:           # an axis a model can't honour
                _log(f"    [skip] {e.name}/{a}: {type(ex).__name__}: {ex}")
                swing = float("nan")
            records.append({"key": k, "model": e.name, "axis": a, "ratio_swing": swing})
        _log(f"    {e.name:22s} swept {len(axes)} axes")
    long = pd.DataFrame(records)
    # wide matrix model x axis for the universality check
    wide = long.pivot(index="model", columns="axis", values="ratio_swing")
    # rank axes *within* each model (1 = most impactful); then correlate rankings
    rank = wide.rank(axis=1, ascending=False)
    universality = rank.T.corr(method="spearman")   # model x model Spearman on axis ranks
    return long, wide, universality


# ---------------------------------------------------------------------------
# 3. RQ2 — leakage study (train-only vs global scaling)
# ---------------------------------------------------------------------------
def leakage(keys, seeds) -> pd.DataFrame:
    _log("[3/5] RQ2 leakage study (scope x split x alpha)")
    rows = []
    for k in keys:
        e = MODELS[k]
        df = studies.leakage_study(k, base_config(k), seeds=seeds)
        g = df.groupby("scope")["nrmse"].mean()
        train_v, global_v = g.get("train", np.nan), g.get("global", np.nan)
        rows.append({
            "key": k, "model": e.name, "group": e.group,
            "nrmse_train_scope": train_v, "nrmse_global_scope": global_v,
            "global_over_train": (global_v / train_v) if train_v else np.nan,
            "best_overall": df["nrmse"].min(), "worst_overall": df["nrmse"].max(),
            "artifact_ratio": (df["nrmse"].max() / df["nrmse"].min()
                               if df["nrmse"].min() else np.nan),
        })
        _log(f"    {e.name:22s} global/train={rows[-1]['global_over_train']:.3g}"
             f"  artifact={rows[-1]['artifact_ratio']:.3g}x")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. RQ3 — robust-pipeline search over the quantum models
# ---------------------------------------------------------------------------
def _candidate_pipelines(quick: bool):
    """A small, interpretable grid of leakage-safe candidate pipelines.

    All keep scope='train' (we already isolate leakage in RQ2); the search is
    over the *honest* pipeline choices a practitioner would actually tune.
    """
    scalers = ["minmax", "standard"]
    alphas = [1e-6, 1e-4]
    posts = [("none", []),
             ("leaky0.1", [studies.LeakyMemory(0.1)])]
    encs = [1.0] if quick else [1.0, 2.0]
    grid = []
    for sc in scalers:
        for a in alphas:
            for plabel, pp in posts:
                for ev in encs:
                    grid.append({
                        "label": f"{sc}|a={a:g}|{plabel}|e={ev:g}",
                        "ov": {"scaler": sc, "scaler_scope": "train",
                               "alpha": a, "postprocess": pp, "encode_scale": ev},
                    })
    return grid


def robust_pipeline(quantum_keys, seeds, quick) -> tuple[pd.DataFrame, str]:
    _log("[4/5] RQ3 robust-pipeline search (over quantum models)")
    cands = _candidate_pipelines(quick)
    rows = []
    for c in cands:
        per_model = {}
        for k in quantum_keys:
            vals = []
            for s in seeds:
                cfg = replace(base_config(k), seed=s, **c["ov"])
                r = run_experiment(build_forecaster(k, s, cfg), cfg)
                vals.append(r.nrmse)
            per_model[k] = float(np.mean(vals))
        v = np.array(list(per_model.values()), dtype=float)
        rows.append({
            "pipeline": c["label"],
            **{f"nrmse_{k}": per_model[k] for k in quantum_keys},
            "maximin": float(np.max(v)),   # best worst-case model (minimise)
            "mean": float(np.mean(v)),     # best average             (minimise)
            "variance": float(np.var(v)),  # most consistent          (minimise)
        })
        _log(f"    {c['label']:34s} maximin={rows[-1]['maximin']:.4g}"
             f" mean={rows[-1]['mean']:.4g}")
    df = pd.DataFrame(rows)
    # report all three criteria; the winner under each
    winners = {crit: df.loc[df[crit].idxmin(), "pipeline"]
               for crit in ("maximin", "mean", "variance")}
    verdict = ("FINDING [RQ3 robust pipeline]: "
               + "; ".join(f"{crit} -> '{p}'" for crit, p in winners.items()))
    if len(set(winners.values())) == 1:
        verdict += f"  (all three agree: '{next(iter(winners.values()))}')."
    else:
        verdict += "  (criteria disagree — pick the headline from the thesis framing)."
    return df.sort_values("maximin").reset_index(drop=True), verdict


# ---------------------------------------------------------------------------
# 5. RQ3 — matched feature-budget comparison (quantum vs best classical)
# ---------------------------------------------------------------------------
def matched_budget(quantum_keys, seeds) -> pd.DataFrame:
    """Compare each quantum model against an ESN sized to the *same* feature
    count F. ESN units=F gives F features directly; we read the quantum model's F
    from its baseline run and build a matched ESN, then score both identically.
    """
    _log("[5/5] RQ3 matched-budget comparison (quantum F vs ESN units=F)")
    from qdepipe.models import ESN, ESNConfig
    from qdepipe.forecasters import ReservoirForecaster
    rows = []
    for qk in quantum_keys:
        qcfg = base_config(qk)
        # quantum result + its feature count
        qvals, F = [], None
        for s in seeds:
            r = run_experiment(build_forecaster(qk, s, replace(qcfg, seed=s)), replace(qcfg, seed=s))
            qvals.append(r.nrmse)
            F = r.n_features
        # matched ESN at units=F, same pipeline (lookback/scaler/etc from qcfg)
        evals = []
        for s in seeds:
            ecfg = replace(qcfg, seed=s, windowing="internal")
            esn_fc = ReservoirForecaster(ESN(ESNConfig(units=int(F), seed=s)),
                                         f"ESN(F={F})", external_window=True)
            evals.append(run_experiment(esn_fc, ecfg).nrmse)
        qm, em = float(np.mean(qvals)), float(np.mean(evals))
        rows.append({
            "quantum_model": MODELS[qk].name, "F": int(F),
            "nrmse_quantum": qm, "nrmse_esn_matched": em,
            "quantum_better": qm < em,
            "ratio_esn_over_quantum": (em / qm) if qm else np.nan,
        })
        _log(f"    {MODELS[qk].name:12s} F={F:<4d} QRC={qm:.4g} vs ESN={em:.4g}"
             f"  {'QRC wins' if qm < em else 'ESN wins'}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def write_report(meta, lb, sens_long, sens_wide, universality,
                 leak_df, robust_df, robust_verdict, matched_df):
    lines = []
    A = lines.append
    A("# QDE — Cross-Model Synthesis (auto-generated)\n")
    A(f"*Generated by `synthesize.py` — seeds={meta['seeds']}, "
      f"models={meta['n_models']}{' (quick)' if meta['quick'] else ''}. "
      f"Numbers come from the same `qdepipe` code paths the notebooks use.*\n")

    A("## 1. Baseline leaderboard\n")
    A("Every model at its sensible-default pipeline, ranked by 1-step NRMSE "
      "(mean over seeds). Full table: `results/baseline_leaderboard.csv`.\n")
    top = lb.head(8)[["rank", "model", "group", "nrmse_mean", "nrmse_std", "n_features"]]
    A(_md(top) + "\n")
    best, worst = lb.iloc[0], lb.iloc[-1]
    A(f"**FINDING [baseline]:** best = **{best['model']}** "
      f"(NRMSE {best['nrmse_mean']:.4g}); worst = {worst['model']} "
      f"({worst['nrmse_mean']:.4g}); spread "
      f"{worst['nrmse_mean']/max(best['nrmse_mean'],1e-12):.3g}x across "
      f"{len(lb)} models.\n")

    A("## 2. RQ1 — which axes matter, and is the ranking universal?\n")
    A("Per-model **ratio swing** (max/min mean-NRMSE across an axis's levels). "
      "Full matrix: `results/rq1_axis_swing.csv`.\n")
    mean_swing = sens_long.groupby("axis")["ratio_swing"].mean().sort_values(ascending=False)
    A("Mean ratio swing across models (most impactful axis first):\n")
    A(_md(mean_swing.to_frame("mean_ratio_swing").reset_index(), floatfmt=".3g") + "\n")
    # universality: average off-diagonal Spearman of axis rankings
    u = universality.to_numpy(dtype=float)
    off = u[~np.eye(u.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    rho = float(np.mean(off)) if len(off) else float("nan")
    A(f"**FINDING [RQ1 universality]:** mean pairwise Spearman of the per-model "
      f"axis-impact ranking = **{rho:.2f}**. "
      + ("High (>0.6) — the axis ranking is broadly universal, so a single "
         "pipeline ordering transfers across models.\n" if rho > 0.6 else
         "Low/moderate — the most-impactful axis is **model-dependent**, so there "
         "is no single universal pipeline ordering (itself a finding).\n"))

    A("## 3. RQ2 — leakage (train-only vs global scaling)\n")
    A("`global_over_train` > 1 means global (leaky) scaling *looks* better than "
      "the honest train-only fit — the artifact. Full table: "
      "`results/rq2_leakage.csv`.\n")
    lk = leak_df[["model", "group", "nrmse_train_scope", "nrmse_global_scope",
                  "global_over_train", "artifact_ratio"]]
    A(_md(lk) + "\n")
    worst_art = leak_df.loc[leak_df["artifact_ratio"].idxmax()]
    A(f"**FINDING [RQ2]:** the largest leakage-driven swing is "
      f"**{worst_art['artifact_ratio']:.3g}x** ({worst_art['model']}), showing how "
      f"much an 'advantage' can be manufactured by scope+split+alpha choices "
      f"alone — exactly the artifact the thesis sets out to control.\n")

    if robust_df is not None:
        A("## 4. RQ3 — robust pipeline across the quantum models\n")
        A("Each candidate pipeline scored on every quantum model; we report all "
          "three selection criteria. Full table: `results/rq3_robust_pipeline.csv`.\n")
        A(_md(robust_df.head(6)) + "\n")
        A(f"**{robust_verdict}**\n")

    if matched_df is not None:
        A("## 5. RQ3 — matched feature-budget comparison\n")
        A("Each quantum model vs an ESN sized to the *same* feature count F, same "
          "pipeline. Full table: `results/rq3_matched_budget.csv`.\n")
        A(_md(matched_df) + "\n")
        nq = int(matched_df["quantum_better"].sum())
        A(f"**FINDING [RQ3 recommendation]:** at matched F, the quantum reservoir "
          f"beat the size-matched ESN in {nq}/{len(matched_df)} cases. "
          + ("The quantum feature map adds value at matched budget.\n" if nq else
             "The size-matched classical ESN matches or beats the QRCs — no "
             "quantum advantage survives a fair, matched-budget pipeline.\n"))

    A("\n---\n*This file is regenerated by `synthesize.py`; edit the script, not "
      "this file. CSVs in `results/` are the source of truth for the tables.*\n")

    path = os.path.join(RESULTS_DIR, "SYNTHESIS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    _log(f"    wrote {os.path.relpath(path)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke: classical-only, 1 seed, smaller robust grid")
    ap.add_argument("--no-quantum", action="store_true", help="skip the QRC models")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (default 3, or 1 with --quick)")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="restrict to these model keys (debug)")
    args = ap.parse_args(argv)

    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 3)
    seeds = tuple(range(n_seeds))

    quantum_keys = [k for k, e in MODELS.items() if e.group == "quantum"]
    keys = list(MODELS.keys())
    if args.only:
        keys = [k for k in args.only if k in MODELS]
    if args.quick or args.no_quantum:
        keys = [k for k in keys if k not in quantum_keys]
    q_in_play = [k for k in quantum_keys if not (args.quick or args.no_quantum)]
    if args.only:
        q_in_play = [k for k in q_in_play if k in keys]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log(f"qde synthesis — {len(keys)} models, seeds={seeds}"
         f"{' [quick]' if args.quick else ''}\n")
    t_start = time.time()

    lb = baseline_leaderboard(keys, seeds)
    _save(lb, "baseline_leaderboard.csv")

    sens_long, sens_wide, universality = axis_sensitivity(keys, seeds)
    _save(sens_long, "rq1_axis_swing.csv")
    _save(sens_wide.reset_index(), "rq1_axis_swing_matrix.csv")
    _save(universality.reset_index(), "rq1_universality_spearman.csv")

    leak_df = leakage(keys, seeds)
    _save(leak_df, "rq2_leakage.csv")

    robust_df = robust_verdict = matched_df = None
    if q_in_play:
        robust_df, robust_verdict = robust_pipeline(q_in_play, seeds, args.quick)
        _save(robust_df, "rq3_robust_pipeline.csv")
        matched_df = matched_budget(q_in_play, seeds)
        _save(matched_df, "rq3_matched_budget.csv")
    else:
        _log("[4/5] RQ3 robust pipeline  — skipped (no quantum models in play)")
        _log("[5/5] RQ3 matched budget   — skipped (no quantum models in play)")

    write_report(
        {"seeds": list(seeds), "n_models": len(keys), "quick": args.quick},
        lb, sens_long, sens_wide, universality, leak_df,
        robust_df, robust_verdict, matched_df)

    _log(f"\ndone in {time.time()-t_start:.1f}s — see results/SYNTHESIS.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
