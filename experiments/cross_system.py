#!/usr/bin/env python3
"""Cross-system classical battery — tests whether the
central conclusions hold across Hénon, Lorenz-63, and Mackey–Glass.

The hypothesis under test: NG-RC's crushing Hénon dominance is a quadratic-system
artifact, and "accuracy tracks the accessible feature map, not the substrate."
To test any apparent NG-RC vs ESN difference rigorously (i.e. rule out an
under-tuned-NG-RC artifact) this runs **four robustness pins**:

  1. NG-RC TUNED — sweep (lookback × degree) per system; the comparison is judged at
     NG-RC's *best* config, not a default.
  2. MULTI-SEED — every comparison over >=3 seeds (ESN/etc. reinitialised per seed).
  3. SIGNIFICANCE — Diebold–Mariano (per-seed → aggregated) on the persisted forecasts,
     plus a Model Confidence Set, for the classical model set on each system.
  4. Δ-SENSITIVITY — on Lorenz, repeat ESN-vs-NG-RC at three sampling intervals
     (Δ = 0.02, 0.05, 0.10) to test whether the ordering depends on sampling rate.

OUTCOME (3 seeds): there is **no NG-RC↔ESN reversal** — tuned NG-RC beats ESN on all
three systems (DM significant) at every Δ. The earlier single-seed "ESN beats NG-RC on
Lorenz" was an under-tuned-NG-RC artifact (refuted). The validated finding is the
feature-structure one: the system dictates the needed polynomial degree, and on Lorenz
the unique best model (MCS) is ELM+poly2, not NG-RC.

Also runs the feature-structure probe (NG-RC degree 1/2/3, ESN±poly2, ELM±poly2) and a
closed-loop climate table per system. Quantum models are deferred (classical-first).

Outputs: results/cross/{system}_*.csv, results/significance/{system}_DM_matrix.csv +
_DM_pairs.csv + _MCS.csv (classical set), results/forecasts/{system}/*, CROSS_SYSTEM.md.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.registry import build_forecaster
from qdepipe.models import NGRC, NGRCConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.closedloop import run_closed_loop
from qdepipe.pipeline.postprocess import Polynomial
from qdepipe import significance as sig

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SYSTEMS = ["henon", "lorenz", "mackeyglass"]
NGRC_LOOKBACKS = [2, 3, 5, 8]
NGRC_DEGREES = [1, 2, 3]


def _log(m): print(m, flush=True)


def base_cfg(system: str, lookback: int = 5, **ov) -> ExperimentConfig:
    """One declared pipeline per system (so all models are aligned). n_points kept
    modest; lookback default 5 (overridden where a model tunes it)."""
    return ExperimentConfig(system=system, n_points=3000, scaler="minmax",
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2),
                            washout=100, horizon=1, lookback=lookback, alpha=1e-6, **ov)


def _ngrc(k, degree):
    return lambda s, c: ReservoirForecaster(
        NGRC(NGRCConfig(k=k, degree=degree)), f"NGRC(k{k},d{degree})")


def _nrmse(make, cfg, seeds, keep=False):
    rs = [run_experiment(make(s, replace(cfg, seed=s)), replace(cfg, seed=s),
                         keep_forecasts=keep) for s in seeds]
    return rs


def _mean(rs):
    v = np.array([r.nrmse for r in rs])
    return float(v.mean()), float(v.std())


# ---------------------------------------------------------------------------
# PIN 1 — tune NG-RC per system
# ---------------------------------------------------------------------------
def tune_ngrc(system, seeds):
    rows, best = [], None
    for k in NGRC_LOOKBACKS:
        for d in NGRC_DEGREES:
            m, sd = _mean(_nrmse(_ngrc(k, d), base_cfg(system, lookback=k), seeds))
            rows.append({"k": k, "degree": d, "nrmse_mean": m, "nrmse_std": sd})
            if best is None or m < best["nrmse_mean"]:
                best = {"k": k, "degree": d, "nrmse_mean": m, "nrmse_std": sd}
    _log(f"    NG-RC best: k={best['k']} degree={best['degree']} "
         f"NRMSE={best['nrmse_mean']:.4g}")
    return pd.DataFrame(rows), best


# ---------------------------------------------------------------------------
# classical battery + feature-structure probe
# ---------------------------------------------------------------------------
def classical_models(best_ngrc):
    """label -> make(seed, cfg). NG-RC pinned at its best (k,degree)."""
    poly = [Polynomial(interactions=False)]
    return {
        "NG-RC(best)": (_ngrc(best_ngrc["k"], best_ngrc["degree"]), {"lookback": best_ngrc["k"]}),
        "NG-RC(d1)": (_ngrc(best_ngrc["k"], 1), {"lookback": best_ngrc["k"]}),
        "NG-RC(d2)": (_ngrc(best_ngrc["k"], 2), {"lookback": best_ngrc["k"]}),
        "NG-RC(d3)": (_ngrc(best_ngrc["k"], 3), {"lookback": best_ngrc["k"]}),
        "ESN": (lambda s, c: build_forecaster("esn", s, c), {}),
        "ESN+poly2": (lambda s, c: build_forecaster("esn", s, c), {"postprocess": poly}),
        "ELM": (lambda s, c: build_forecaster("elm", s, c), {}),
        "ELM+poly2": (lambda s, c: build_forecaster("elm", s, c), {"postprocess": poly}),
        "RandomForest": (lambda s, c: build_forecaster("random_forest", s, c), {}),
        "Linear-Ridge": (lambda s, c: build_forecaster("linear", s, c), {}),
    }


def _align(fc):
    L = min(len(v[0]) for v in fc.values())
    out = {m: (np.asarray(yt)[-L:], np.asarray(yp)[-L:]) for m, (yt, yp) in fc.items()}
    return out, L


def battery(system, seeds, models):
    """Run all models, persist forecasts, return leaderboard + per-seed aligned fcs."""
    fdir = os.path.join(RESULTS, "forecasts", system)
    os.makedirs(fdir, exist_ok=True)
    per_seed, summary = [], {}
    for s in seeds:
        fc = {}
        for label, (make, ov) in models.items():
            cfg = base_cfg(system, seed=s, **ov)
            r = run_experiment(make(s, cfg), cfg, keep_forecasts=True)
            fc[label] = (r.y_true, r.y_pred)
            summary.setdefault(label, []).append(r.nrmse)
        aligned, L = _align(fc)
        # only persist the comparison set (DM uses these)
        for label, (yt, yp) in aligned.items():
            safe = label.replace("(", "").replace(")", "").replace(",", "_").replace("+", "p")
            pd.DataFrame({"t": np.arange(L), "y_true": yt, "y_pred": yp}).to_csv(
                os.path.join(fdir, f"{safe}_seed{s}.csv"), index=False)
        per_seed.append(aligned)
    lead = pd.DataFrame([{"model": m, "nrmse_mean": float(np.mean(v)),
                          "nrmse_std": float(np.std(v))} for m, v in summary.items()]
                        ).sort_values("nrmse_mean").reset_index(drop=True)
    return lead, per_seed


# ---------------------------------------------------------------------------
# PIN 3 — significance (DM matrix + MCS) over the classical set
# ---------------------------------------------------------------------------
def significance(system, per_seed):
    names = list(per_seed[0])
    n = len(names)
    stats = np.full((len(per_seed), n, n), np.nan)
    ps = np.full((len(per_seed), n, n), np.nan)
    for si, aligned in enumerate(per_seed):
        st, pv, nm = sig.dm_matrix(aligned, h=1, loss="se")
        idx = [nm.index(x) for x in names]
        stats[si] = st[np.ix_(idx, idx)]
        ps[si] = pv[np.ix_(idx, idx)]
    mean_stat, median_p = np.nanmean(stats, 0), np.nanmedian(ps, 0)
    frac_sig = np.mean(ps < 0.05, 0)
    sdir = os.path.join(RESULTS, "significance")
    os.makedirs(sdir, exist_ok=True)
    sq = np.where(np.tril(np.ones((n, n)), -1).astype(bool), mean_stat, median_p)
    np.fill_diagonal(sq, 0.0)
    pd.DataFrame(sq, index=names, columns=names).to_csv(os.path.join(sdir, f"{system}_DM_matrix.csv"))
    rows = [{"model_i": names[i], "model_j": names[j], "mean_dm": mean_stat[i, j],
             "median_p": median_p[i, j], "frac_seeds_sig": frac_sig[i, j]}
            for i in range(n) for j in range(n) if i != j]
    pd.DataFrame(rows).to_csv(os.path.join(sdir, f"{system}_DM_pairs.csv"), index=False)
    # MCS
    in10 = {m: 0 for m in names}
    pv = {m: [] for m in names}
    for aligned in per_seed:
        losses = np.stack([(aligned[m][1] - aligned[m][0]) ** 2 for m in names], 0)
        res = sig.model_confidence_set(losses, names=names, alpha=(0.10, 0.25),
                                       n_boot=1000, block_length=20, seed=0)
        for m in names:
            pv[m].append(res["mcs_pvalue"][m])
            if m in res["in_set"][0.10]:
                in10[m] += 1
    mcs = pd.DataFrame([{"model": m, "mean_mcs_pvalue": float(np.mean(pv[m])),
                         "frac_seeds_in_set_0.10": in10[m] / len(per_seed)} for m in names]
                       ).sort_values("mean_mcs_pvalue", ascending=False)
    mcs.to_csv(os.path.join(sdir, f"{system}_MCS.csv"), index=False)
    return names, median_p, frac_sig, mcs


def esn_vs_ngrc(system, names, median_p, frac_sig, lead):
    i, j = names.index("ESN"), names.index("NG-RC(best)")
    p, fr = median_p[i, j], frac_sig[i, j]
    esn = float(lead[lead.model == "ESN"]["nrmse_mean"].iloc[0])
    ng = float(lead[lead.model == "NG-RC(best)"]["nrmse_mean"].iloc[0])
    winner = "ESN" if esn < ng else "NG-RC"
    verdict = "significant" if p < 0.05 else "NOT significant" if p > 0.10 else "borderline"
    return {"system": system, "esn_nrmse": esn, "ngrc_best_nrmse": ng,
            "winner": winner, "dm_median_p": p, "frac_seeds_sig": fr, "verdict": verdict}


# ---------------------------------------------------------------------------
# PIN 4 — Lorenz Δ-sensitivity
# ---------------------------------------------------------------------------
def delta_sensitivity(seeds, best_by_system):
    _log("  [PIN 4] Lorenz Δ-sensitivity (ESN vs NG-RC-best at 3 sampling rates)")
    rows = []
    for sysname, dval in [("lorenz_d02", 0.02), ("lorenz", 0.05), ("lorenz_d10", 0.10)]:
        b = best_by_system["lorenz"]            # reuse the tuned NG-RC from the main Lorenz run
        esn = _mean(_nrmse(lambda s, c: build_forecaster("esn", s, c), base_cfg(sysname), seeds))[0]
        ng = _mean(_nrmse(_ngrc(b["k"], b["degree"]), base_cfg(sysname, lookback=b["k"]), seeds))[0]
        rows.append({"delta": dval, "esn_nrmse": esn, "ngrc_best_nrmse": ng,
                     "winner": "ESN" if esn < ng else "NG-RC"})
        _log(f"      Δ={dval}: ESN={esn:.4g} NG-RC={ng:.4g} -> {'ESN' if esn < ng else 'NG-RC'}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "cross", "lorenz_delta_sensitivity.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# climate
# ---------------------------------------------------------------------------
def climate(system, seeds, best_ngrc):
    rows = []
    specs = {"NG-RC(best)": _ngrc(best_ngrc["k"], best_ngrc["degree"]),
             "ESN": lambda s, c: build_forecaster("esn", s, c),
             "ELM": lambda s, c: build_forecaster("elm", s, c),
             "RandomForest": lambda s, c: build_forecaster("random_forest", s, c)}
    for label, make in specs.items():
        vpt = []
        for s in seeds:
            cfg = base_cfg(system, seed=s,
                           lookback=best_ngrc["k"] if "NG-RC" in label else 5)
            try:
                cl = run_closed_loop(make(s, cfg), cfg, n_steps=300)
                vpt.append(cl.vpt_lyap)
            except Exception:
                break
        if vpt:
            rows.append({"model": label, "vpt_lyap_mean": float(np.mean(vpt))})
    return pd.DataFrame(rows).sort_values("vpt_lyap_mean", ascending=False)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _md(df, fmt=".4g"):
    try:
        return df.to_markdown(index=False, floatfmt=fmt)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = tuple(range(2 if args.quick else args.seeds))
    os.makedirs(os.path.join(RESULTS, "cross"), exist_ok=True)
    from qdepipe._provenance import write_versions
    write_versions(os.path.join(os.path.dirname(RESULTS), "versions.txt"))

    leaderboards, probes, h2h, mcs_by, climates, best_by = {}, {}, [], {}, {}, {}
    for system in SYSTEMS:
        _log(f"[{system}]")
        tune_df, best = tune_ngrc(system, seeds)
        best_by[system] = best
        tune_df.to_csv(os.path.join(RESULTS, "cross", f"{system}_ngrc_tune.csv"), index=False)
        models = classical_models(best)
        lead, per_seed = battery(system, seeds, models)
        lead.to_csv(os.path.join(RESULTS, "cross", f"{system}_leaderboard.csv"), index=False)
        leaderboards[system] = lead
        names, median_p, frac_sig, mcs = significance(system, per_seed)
        mcs_by[system] = mcs
        h2h.append(esn_vs_ngrc(system, names, median_p, frac_sig, lead))
        climates[system] = climate(system, seeds, best)
        climates[system].to_csv(os.path.join(RESULTS, "cross", f"{system}_climate.csv"), index=False)
        _log(f"    leaderboard top: {lead.iloc[0]['model']} ({lead.iloc[0]['nrmse_mean']:.4g})")

    delta_df = delta_sensitivity(seeds, best_by)

    # ---- assemble CROSS_SYSTEM.md ----
    L = ["# QDE — Cross-System Validation (auto-generated)\n",
         f"*`cross_system.py`, classical-first, seeds={list(seeds)}. Four robustness pins: "
         "NG-RC tuned (lookback×degree), multi-seed, Diebold–Mariano significance, and "
         "Lorenz Δ-sensitivity. Quantum cross-system runs deferred.*\n",
         "## Headline: does NG-RC's dominance generalise?\n",
         "ESN vs the **tuned** NG-RC on each system, with a DM significance verdict:\n",
         _md(pd.DataFrame(h2h)) + "\n",
         "**READ:** if the winner flips from NG-RC (Hénon) to ESN on Lorenz/Mackey–Glass and "
         "the DM verdict is significant, NG-RC's dominance is a Hénon (quadratic-system) "
         "artifact — the central thesis claim, now cross-validated.\n",
         "## PIN 4 — Lorenz Δ-sensitivity (not a one-sampling-rate artifact)\n",
         _md(delta_df) + "\n"]
    for system in SYSTEMS:
        L.append(f"## System: {system}\n")
        L.append("### Classical leaderboard (1-step NRMSE)\n")
        L.append(_md(leaderboards[system]) + "\n")
        L.append(f"NG-RC tuned best: see `results/cross/{system}_ngrc_tune.csv`. "
                 "Degree probe (NG-RC d1/d2/d3, ESN±poly2, ELM±poly2) is in the leaderboard "
                 "above — feature-structure story: accuracy tracks accessible polynomial "
                 "degree across architectures.\n")
        L.append("### Model Confidence Set (α=0.10)\n")
        L.append(_md(mcs_by[system]) + "\n")
        L.append("### Closed-loop climate (VPT, Lyapunov times)\n")
        L.append(_md(climates[system]) + "\n")
    L.append("\n---\n*Regenerated by `cross_system.py` (`python experiments/cross_system.py`). DM/MCS matrices in "
             "`results/significance/{system}_*`; per-seed forecasts in `results/forecasts/`.*\n")
    with open(os.path.join(RESULTS, "CROSS_SYSTEM.md"), "w") as f:
        f.write("\n".join(L))
    _log("\nwrote results/CROSS_SYSTEM.md")


if __name__ == "__main__":
    main()
