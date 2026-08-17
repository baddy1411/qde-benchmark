#!/usr/bin/env python3
"""Quantum cross-system pass — QRC on Lorenz-63 + Mackey-Glass.

Tests whether the Hénon matched-budget story generalises: on Hénon a size-matched
ESN beat QRC-v4/v6, only QRC-rich beat ESN(F), and even tuned QRC-rich lost to NG-RC
by ~700×; NO quantum model entered the MCS best-set. Does that hold on Lorenz /
Mackey-Glass — or does a quantum model finally enter a best-set (the one outcome that
would make the quantum side more than a foil)?

All quantum results here are EXACT-EXPECTATION statevector → an *idealized upper bound*
on quantum performance (finite-shot results are reported separately).

Rigor (carried from prior passes):
  * identical leakage-safe pipeline; quantum + classical on the SAME data path.
  * matched feature budget: each QRC (F) vs an ESN with units=F.
  * 5 seeds; per-seed raw forecasts persisted for DM/MCS + plots.
  * per-system ENCODING FAIRNESS: sweep encode_scale×scaler per model per system, judge
    the matched comparison at each model's per-system best (not Hénon-optimal, not default).
  * all models aligned on ONE pipeline (n_points=1500, the quantum-feasible size) so the
    joint DM/MCS over classical+quantum is valid; classical comparators are re-run here.
  * per-system checkpoint: Lorenz fully → commit → MG fully → commit.

Outputs per system: results/cross/{sys}_qrc_encoding.csv, {sys}_matched_budget.csv,
results/significance/{sys}_quantum_DM.csv, {sys}_MCS_joint.csv, climate appended to
results/cross/{sys}_climate.csv; results/QUANTUM_CROSS.md (incremental).
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.registry import build_forecaster
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.closedloop import run_closed_loop
from qdepipe.pipeline.postprocess import Polynomial
from qdepipe.data import lyapunov_step, generate
from qdepipe import significance as sig

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NEW_SYSTEMS = ["lorenz", "mackeyglass"]
QUANTUM = {"qrc_v4": 16, "qrc_v6": 24, "qrc_rich": 96}      # model -> F
ENCODE_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SCALERS = ["minmax", "standard"]
NGRC_BEST = {"lorenz": (8, 3), "mackeyglass": (8, 3)}        # (k, degree) from classical pass
NPTS = 1500
TITLES = {"lorenz": "Lorenz-63", "mackeyglass": "Mackey–Glass"}


def _log(m): print(m, flush=True)


def cfg_for(system, scaler="minmax", **ov):
    return ExperimentConfig(system=system, n_points=NPTS, scaler=scaler,
                            scaler_scope="train", split_fracs=(0.6, 0.2, 0.2),
                            washout=100, horizon=1, lookback=5, alpha=1e-6, **ov)


def _ngrc(k, d):
    return lambda s, c: ReservoirForecaster(NGRC(NGRCConfig(k=k, degree=d)), f"NG-RC(k{k}d{d})")


def _esn_units(F):
    return lambda s, c: ReservoirForecaster(ESN(ESNConfig(units=int(F), seed=s)),
                                            f"ESN(F={F})", external_window=True)


def _runs(make, cfg, seeds, keep=False):
    return [run_experiment(make(s, replace(cfg, seed=s)), replace(cfg, seed=s),
                           keep_forecasts=keep) for s in seeds]


def _mean(rs):
    v = np.array([r.nrmse for r in rs])
    return float(v.mean()), float(v.std())


# ---------------------------------------------------------------------------
# encoding fairness sweep (per system, per quantum model)
# ---------------------------------------------------------------------------
def encode_sweep(system, seeds, store=None):
    # `store` (DEFAULT-OFF) caches only the QUANTUM forecasters' featurize; the
    # classical battery is left uncached on purpose (its featurize is cheap, so the
    # cache overhead would exceed the saving).
    rows, best = [], {}
    for qk in QUANTUM:
        grid = []
        for sc in SCALERS:
            for e in ENCODE_GRID:
                m, sd = _mean(_runs(lambda s, c: build_forecaster(qk, s, c, store=store),
                                    cfg_for(system, scaler=sc, encode_scale=e), seeds))
                grid.append({"model": qk, "scaler": sc, "encode_scale": e,
                             "nrmse_mean": m, "nrmse_std": sd})
        g = pd.DataFrame(grid)
        b = g.loc[g["nrmse_mean"].idxmin()]
        d = g[(g.scaler == "minmax") & (np.isclose(g.encode_scale, 1.0))]["nrmse_mean"].iloc[0]
        best[qk] = {"scaler": b["scaler"], "encode_scale": float(b["encode_scale"]),
                    "nrmse": float(b["nrmse_mean"]), "default": float(d)}
        rows.extend(grid)
        _log(f"    {qk:9s} default={d:.4g} -> best={b['nrmse_mean']:.4g} "
             f"@ {b['scaler']} e={b['encode_scale']:g}  ({d/max(b['nrmse_mean'],1e-12):.2g}x)")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "cross", f"{system}_qrc_encoding.csv"), index=False)
    return best


# ---------------------------------------------------------------------------
# matched-budget at tuned best + persist forecasts for the comparison set
# ---------------------------------------------------------------------------
def matched_budget(system, seeds, best, store=None):
    # `store` (DEFAULT-OFF) is threaded into the QUANTUM comp entries only; the
    # classical entries (NG-RC/ESN/ELM/RandomForest) stay uncached by design.
    fdir = os.path.join(RESULTS, "forecasts", system)
    os.makedirs(fdir, exist_ok=True)
    k, d = NGRC_BEST[system]
    rows = []
    # comparison set whose forecasts we persist (aligned, n_points=1500)
    comp = {}     # label -> (make, cfg_overrides)
    comp[f"NG-RC(k{k}d{d})"] = (_ngrc(k, d), {})
    comp["ESN"] = (lambda s, c: build_forecaster("esn", s, c), {})
    comp["ELM"] = (lambda s, c: build_forecaster("elm", s, c), {})
    comp["ELM+poly2"] = (lambda s, c: build_forecaster("elm", s, c),
                         {"postprocess": [Polynomial(interactions=False)]})
    comp["RandomForest"] = (lambda s, c: build_forecaster("random_forest", s, c), {})
    for qk, F in QUANTUM.items():
        bsc, be = best[qk]["scaler"], best[qk]["encode_scale"]
        comp[qk] = (lambda s, c, k=qk: build_forecaster(k, s, c, store=store),
                    {"scaler": bsc, "encode_scale": be})
        comp[f"ESN(F={F})"] = (_esn_units(F), {"scaler": bsc, "windowing": "internal"})

    # run + persist per seed
    per_seed = []
    ngrc_ref = None
    for s in seeds:
        fc = {}
        for label, (make, ov) in comp.items():
            cfg = cfg_for(system, seed=s, **ov)
            r = run_experiment(make(s, cfg), cfg, keep_forecasts=True)
            fc[label] = (r.y_true, r.y_pred)
        per_seed.append(fc)
    # align by common suffix
    Lmin = min(len(v[0]) for fc in per_seed for v in fc.values())
    aligned_seeds = [{m: (np.asarray(yt)[-Lmin:], np.asarray(yp)[-Lmin:])
                      for m, (yt, yp) in fc.items()} for fc in per_seed]
    for si, s in enumerate(seeds):
        for label, (yt, yp) in aligned_seeds[si].items():
            safe = label.replace("(", "").replace(")", "").replace(",", "_").replace("+", "p").replace("=", "")
            pd.DataFrame({"t": np.arange(Lmin), "y_true": yt, "y_pred": yp}).to_csv(
                os.path.join(fdir, f"{safe}_seed{s}.csv"), index=False)

    # matched-budget table
    def meannr(label):
        return float(np.mean([np.sqrt(np.mean((a[label][1]-a[label][0])**2)) /
                              (np.std(a[label][0]) or 1) for a in aligned_seeds]))
    ng_label = f"NG-RC(k{k}d{d})"
    ng_ref = meannr(ng_label)
    for qk, F in QUANTUM.items():
        qn = meannr(qk)
        en = meannr(f"ESN(F={F})")
        rows.append({"model": qk, "F": F, "NRMSE_quantum_best": qn,
                     "NRMSE_ESN_F": en, "NRMSE_NGRC_ref": ng_ref,
                     "encode_scale": best[qk]["encode_scale"], "scaler": best[qk]["scaler"],
                     "beats_ESN": bool(qn < en)})
        _log(f"    {qk:9s} F={F:<3d} QRC={qn:.4g}  ESN(F)={en:.4g}  NG-RC={ng_ref:.4g}"
             f"  -> {'QRC' if qn < en else 'ESN'} wins")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "cross", f"{system}_matched_budget.csv"), index=False)
    return aligned_seeds, ng_label


# ---------------------------------------------------------------------------
# DM on matched pairs (seed-fraction-significant, NOT averaged p)
# ---------------------------------------------------------------------------
def quantum_dm(system, aligned_seeds, ng_label):
    pairs = [("qrc_rich", "ESN(F=96)"), ("qrc_rich", ng_label), ("qrc_v6", "ESN(F=24)")]
    rows = []
    for a, b in pairs:
        stats, ps = [], []
        for fc in aligned_seeds:
            e1 = fc[a][1] - fc[a][0]
            e2 = fc[b][1] - fc[b][0]
            st, p = sig.diebold_mariano(e1, e2, h=1, loss="se")
            stats.append(st); ps.append(p)
        ps = np.array(ps); stats = np.array(stats)
        rows.append({"pair": f"{a} vs {b}",
                     "frac_sig_0.10": float(np.mean(ps < 0.10)),
                     "frac_sig_0.05": float(np.mean(ps < 0.05)),
                     "median_dm": float(np.median(stats)),
                     "dm_min": float(np.min(stats)), "dm_max": float(np.max(stats)),
                     "median_p": float(np.median(ps))})
        _log(f"    DM {a} vs {b}: frac_sig@0.05={np.mean(ps<0.05):.2f} "
             f"median_DM={np.median(stats):.3g}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "significance", f"{system}_quantum_DM.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# joint MCS over classical + quantum (aligned, same pipeline)
# ---------------------------------------------------------------------------
def joint_dm_matrix(system, aligned_seeds):
    """Full pairwise DM over the joint (classical+quantum) aligned set, per-seed then
    aggregated → {system}_DM_pairs_joint.csv (long form for the heatmap figure)."""
    names = list(aligned_seeds[0])
    n = len(names)
    stats = np.full((len(aligned_seeds), n, n), np.nan)
    ps = np.full((len(aligned_seeds), n, n), np.nan)
    for si, fc in enumerate(aligned_seeds):
        st, pv, nm = sig.dm_matrix(fc, h=1, loss="se")
        idx = [nm.index(x) for x in names]
        stats[si] = st[np.ix_(idx, idx)]
        ps[si] = pv[np.ix_(idx, idx)]
    mean_stat, median_p = np.nanmean(stats, 0), np.nanmedian(ps, 0)
    frac_sig = np.mean(ps < 0.05, 0)
    rows = [{"model_i": names[i], "model_j": names[j], "mean_dm": mean_stat[i, j],
             "median_p": median_p[i, j], "frac_seeds_sig": frac_sig[i, j]}
            for i in range(n) for j in range(n) if i != j]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "significance",
                                           f"{system}_DM_pairs_joint.csv"), index=False)


def joint_mcs(system, aligned_seeds):
    names = list(aligned_seeds[0])
    in10 = {m: 0 for m in names}
    pv = {m: [] for m in names}
    for fc in aligned_seeds:
        losses = np.stack([(fc[m][1] - fc[m][0]) ** 2 for m in names], 0)
        res = sig.model_confidence_set(losses, names=names, alpha=(0.10, 0.25),
                                       n_boot=1000, block_length=20, seed=0)
        for m in names:
            pv[m].append(res["mcs_pvalue"][m])
            if m in res["in_set"][0.10]:
                in10[m] += 1
    df = pd.DataFrame([{"model": m, "mean_mcs_pvalue": float(np.mean(pv[m])),
                        "frac_seeds_in_set_0.10": in10[m] / len(aligned_seeds),
                        "is_quantum": m in QUANTUM} for m in names]
                      ).sort_values("mean_mcs_pvalue", ascending=False)
    df.to_csv(os.path.join(RESULTS, "significance", f"{system}_MCS_joint.csv"), index=False)
    best_set = df[df["frac_seeds_in_set_0.10"] >= 0.5]["model"].tolist()
    q_in = [m for m in best_set if m in QUANTUM]
    return df, best_set, q_in


# ---------------------------------------------------------------------------
# climate (append quantum VPT to existing per-system climate CSV)
# ---------------------------------------------------------------------------
def quantum_climate(system, seeds, best):
    le = lyapunov_step(system, generate(system, NPTS))
    _log(f"    climate: {system} Lyapunov/step = {le:.5f}")
    rows = []
    for qk in QUANTUM:
        bsc, be = best[qk]["scaler"], best[qk]["encode_scale"]
        vpt, spec, wass, div = [], [], [], 0
        for s in seeds:
            cfg = cfg_for(system, seed=s, scaler=bsc, encode_scale=be)
            try:
                cl = run_closed_loop(build_forecaster(qk, s, cfg), cfg, n_steps=300)
                vpt.append(cl.vpt_lyap); spec.append(cl.spectral_mse)
                wass.append(cl.wasserstein); div += int(cl.diverged)
            except Exception as ex:
                _log(f"      [skip] {qk}: {type(ex).__name__}"); break
        if vpt:
            rows.append({"model": qk, "vpt_lyap_mean": float(np.mean(vpt)),
                         "spectral_mse_mean": float(np.mean(spec)),
                         "wasserstein_mean": float(np.mean(wass)), "diverged_runs": div})
    qdf = pd.DataFrame(rows)
    qdf.to_csv(os.path.join(RESULTS, "cross", f"{system}_quantum_climate.csv"), index=False)
    # append (model, vpt_lyap_mean) to the existing climate CSV so fig3 includes QRC
    cpath = os.path.join(RESULTS, "cross", f"{system}_climate.csv")
    base = pd.read_csv(cpath)
    base = base[~base["model"].isin(QUANTUM)]                # idempotent: drop prior quantum rows
    merged = pd.concat([base, qdf[["model", "vpt_lyap_mean"]]], ignore_index=True)
    merged.sort_values("vpt_lyap_mean", ascending=False).to_csv(cpath, index=False)
    return qdf


# ---------------------------------------------------------------------------
# report (incremental)
# ---------------------------------------------------------------------------
def system_section(system, best, mb_df, dm_df, mcs_df, best_set, q_in, climate_df):
    L = [f"## System: {TITLES.get(system, system)}\n",
         "*Quantum results are exact-expectation (idealized upper bound on quantum "
         "performance; finite-shot results are reported separately).*\n",
         "### Encoding fairness (per-system best)\n"]
    et = pd.DataFrame([{"model": k, "default_nrmse": v["default"], "best_nrmse": v["nrmse"],
                        "best_scaler": v["scaler"], "best_encode_scale": v["encode_scale"]}
                       for k, v in best.items()])
    L.append(_md(et) + "\n")
    L.append("### Matched feature-budget (QRC at tuned best vs ESN(F), NG-RC reference)\n")
    L.append(_md(mb_df) + "\n")
    L.append("### Diebold–Mariano on matched pairs (5 seeds; fraction significant)\n")
    L.append(_md(dm_df) + "\n")
    L.append("### Joint Model Confidence Set (classical + quantum, α=0.10)\n")
    L.append(_md(mcs_df.head(6)) + "\n")
    L.append("### Closed-loop climate (quantum; appended to classical VPT)\n")
    L.append(_md(climate_df) + "\n")
    anyq = "YES" if q_in else "no"
    beats = mb_df[mb_df["beats_ESN"]]["model"].tolist()
    L.append(f"**VERDICT [{system}]:** quantum models beating their matched ESN(F): "
             f"{beats or 'none'}. Quantum in the MCS best-set: **{anyq}** "
             f"({q_in or '—'}). Best-set = {{{', '.join(best_set)}}}. "
             + ("**SHIFT vs Hénon — a quantum model entered a best-set; surfaced as a finding.**"
                if q_in else "Hénon pattern holds: classical owns the best-set.") + "\n")
    return "\n".join(L)


def _md(df, fmt=".4g"):
    try:
        return df.to_markdown(index=False, floatfmt=fmt)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(sections, done):
    head = ["# QDE — Quantum Cross-System Validation (auto-generated)\n",
            f"*`quantum_cross.py`, 5 seeds, exact-expectation statevector (idealized upper "
            f"bound). Systems completed: {', '.join(done)}. All models aligned on one "
            f"n_points={NPTS} pipeline so the joint DM/MCS is valid.*\n",
            "**Question:** does the Hénon matched-budget pattern (only QRC-rich beats ESN(F); "
            "no quantum in the MCS best-set; NG-RC dominates) hold on Lorenz / Mackey-Glass?\n"]
    with open(os.path.join(RESULTS, "QUANTUM_CROSS.md"), "w") as f:
        f.write("\n".join(head + sections))
    _log("    wrote results/QUANTUM_CROSS.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--systems", nargs="+", default=NEW_SYSTEMS)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    os.makedirs(os.path.join(RESULTS, "cross"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "significance"), exist_ok=True)
    from qdepipe._provenance import write_versions
    write_versions(os.path.join(os.path.dirname(RESULTS), "versions.txt"))

    sections, done = [], []
    t_start = time.time()
    for system in args.systems:
        _log(f"[{system}] encoding fairness sweep")
        t0 = time.time()
        best = encode_sweep(system, seeds)
        _log(f"[{system}] matched budget + forecasts")
        aligned, ng_label = matched_budget(system, seeds, best)
        if system == args.systems[0]:
            _log(f"  >>> EARLY SIGNAL: first system encode+matched took "
                 f"{time.time()-t0:.0f}s wall-clock <<<")
        dm_df = quantum_dm(system, aligned, ng_label)
        joint_dm_matrix(system, aligned)
        mcs_df, best_set, q_in = joint_mcs(system, aligned)
        climate_df = quantum_climate(system, seeds, best)
        mb_df = pd.read_csv(os.path.join(RESULTS, "cross", f"{system}_matched_budget.csv"))
        sections.append(system_section(system, best, mb_df, dm_df, mcs_df, best_set, q_in, climate_df))
        done.append(system)
        write_report(sections, done)        # CHECKPOINT: commit after each system
        _log(f"[{system}] DONE ({time.time()-t0:.0f}s). Quantum-in-best-set: {q_in or 'none'}")
    _log(f"\nall systems done in {time.time()-t_start:.0f}s — see results/QUANTUM_CROSS.md")


if __name__ == "__main__":
    main()
