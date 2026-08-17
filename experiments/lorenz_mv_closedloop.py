#!/usr/bin/env python3
"""Closed-loop (free-running) multivariate Lorenz-63 pilot (Part E-2).

The one-step multivariate pilot (Part E) was quadratic-friendly, so classical
degree-2 features won trivially and it did not reach the dynamics-reproduction
regime where Fellner et al. (2026) report quantum resources mattering. This probes
that regime: every model is anchored at the test start and free-runs --- fed its own
predicted (x,y,z) back as the next input --- for a fixed horizon. Divergence-aware
metrics (VPT in Lyapunov times, divergence count) are primary; no free-running NRMSE
headline; no Diebold--Mariano on divergent series. Preliminary, single-encoding.

SYMMETRIC FEEDBACK LOOP (the thing that must be right): the loop body
`closed_loop_mv` is identical for every model --- same anchor (test start), same
horizon, same freeze-on-non-finite, same divergence threshold (10x true range),
same VPT threshold (0.4). The ONLY per-model difference is the featuriser used to
turn the running history into the current feature row, which is the model's own
nature; the read-out wiring (last feature row -> shared trained multivariate ridge
W -> next 3-vector -> append) is the same for all. Reservoirs (ESN) re-featurise the
full history (unbounded memory); finite-memory models (QRC window=5, NG-RC/ELM
delay k) featurise the last-k suffix that reproduces the current feature row ---
exactly the univariate `onestep_predictor` convention.

encode_scale per QRC is REUSED from the one-step pilot's tuned values
(lorenz_mv_matched.csv); a closed-loop-specific re-tune is noted as future work, so
if QRC underperforms it may be partly an encoding-selection artifact (stated, not
hidden). n_points=1500, 5 seeds, horizon=400 steps (univariate-climate convention).

Outputs: results/cross/lorenz_mv_cl_vpt.csv, lorenz_mv_cl_climate.csv;
results/trajectories/lorenz_mv/{model}_seed{n}.csv. STOPS at the checkpoint.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

import lorenz_mv_pilot as P
from qdepipe.models import ESN, ESNConfig
from qdepipe.models.gate_qrc import GateQRC
from qdepipe.readout import ridge_fit, ridge_predict
from qdepipe import metrics
from qdepipe.data import lorenz as _lorenz

RESULTS = P.RESULTS
SEEDS = tuple(range(10))          # 10 seeds: resolve the overlapping 5-seed VPT ranges
N_STEPS = 400
VPT_THRESHOLD = 0.4
ALPHA, WASHOUT, HORIZON = 1e-6, 100, 1
# Lorenz per-step Lyapunov exponent the code holds (same as univariate climate)
LE_STEP = _lorenz.lyapunov_step(stride=5)          # 0.9056 * 0.01 * 5 = 0.04528


def _log(m): print(m, flush=True)


def _train_W(feats, U):
    X, Y = feats[:len(feats) - HORIZON], U[HORIZON:]
    tr = slice(WASHOUT, len(X))                    # fit on everything up to test in the loop
    return ridge_fit(X[tr], Y[tr], alpha=ALPHA, bias=True)


# ---- per-model (train-feature, step) builders; identical loop wiring -------
def build_model(label, U, split, enc, ng):
    """Return (W, step_fn). step_fn(history (t,3), W) -> next (3,) via the model's
    featuriser on the appropriate suffix + the shared ridge read-out W."""
    tr_end = split.train.stop                       # train W on [washout, train.stop)
    Utr = U[:tr_end]

    if label.startswith("qrc_"):
        qrc = GateQRC(P._qrc_cfg(label, enc[label], 0))     # seed only sets reservoir draw
        win = qrc.cfg.window
        W = _train_W(qrc.featurize_mv(Utr), Utr)
        def step(h, W=W, qrc=qrc, win=win):
            suf = h[-win:] if len(h) >= win else h
            f = qrc.featurize_mv(suf)[-1]
            return ridge_predict(f[None], W, bias=True).ravel()
        return W, step

    if label.startswith("ESN(F="):
        F = int(label.split("F=")[1].rstrip(")"))
        esn = ESN(ESNConfig(units=F, seed=0))
        W = _train_W(esn.featurize(Utr), Utr)
        def step(h, W=W, esn=esn):                  # ESN: unbounded memory -> full history
            f = esn.featurize(np.asarray(h))[-1]
            return ridge_predict(f[None], W, bias=True).ravel()
        return W, step

    if label == "NG-RC":
        k, deg = ng["k"], ng["degree"]
        poly = PolynomialFeatures(degree=deg, include_bias=False).fit(P._mv_delay(Utr, k))
        W = _train_W(poly.transform(P._mv_delay(Utr, k)), Utr)
        def step(h, W=W, k=k, poly=poly):
            d = P._mv_delay(np.asarray(h)[-k:], k)[-1]
            return ridge_predict(poly.transform(d[None]), W, bias=True).ravel()
        return W, step

    if label in ("ELM", "ELM+poly2"):
        units, p2, lb = 300, (label == "ELM+poly2"), 3
        rng = np.random.default_rng(7)
        Win = rng.uniform(-1, 1, size=(units, lb * 3)); b = rng.uniform(-1, 1, size=units)
        def feats_of(X):
            H = np.tanh(P._mv_delay(X, lb) @ Win.T + b)
            return np.concatenate([H, H ** 2], axis=1) if p2 else H
        W = _train_W(feats_of(Utr), Utr)
        def step(h, W=W, lb=lb, Win=Win, b=b, p2=p2):
            d = P._mv_delay(np.asarray(h)[-lb:], lb)[-1]
            H = np.tanh(d @ Win.T + b)
            f = np.concatenate([H, H ** 2]) if p2 else H
            return ridge_predict(f[None], W, bias=True).ravel()
        return W, step
    raise ValueError(label)


# ---- the ONE shared, symmetric free-running loop ---------------------------
def closed_loop_mv(step, U, split, n_steps=N_STEPS):
    anchor = split.test.start
    n = min(n_steps, len(U) - anchor)
    history = list(U[:anchor])                      # list of 3-vectors
    true = U[anchor:anchor + n]
    preds = []
    for _ in range(n):
        p = step(np.asarray(history))
        if not np.all(np.isfinite(p)):
            p = history[-1]                         # freeze on blow-up (univariate convention)
        preds.append(p); history.append(p)
    preds = np.asarray(preds)
    vpt = metrics.valid_prediction_time_mv(true, preds, VPT_THRESHOLD)
    diverged = bool(np.max(np.abs(preds)) > 10 * (np.max(np.abs(true)) + 1e-9))
    spec = float(np.mean([metrics.spectral_mse(true[:, c], preds[:, c]) for c in range(3)]))
    wass = float(np.mean([metrics.wasserstein1(true[:, c], preds[:, c]) for c in range(3)]))
    return {"vpt_steps": vpt, "vpt_lyap": vpt * LE_STEP, "diverged": diverged,
            "spectral_mse": spec, "wasserstein": wass}, true, preds


def main():
    os.makedirs(os.path.join(RESULTS, "cross"), exist_ok=True)
    tdir = os.path.join(RESULTS, "trajectories", "lorenz_mv")
    os.makedirs(tdir, exist_ok=True)
    from qdepipe._provenance import write_versions
    write_versions(os.path.join(os.path.dirname(RESULTS), "versions.txt"))

    U, split = P.scale_split(P.lorenz_state())
    # reuse one-step pilot's tuned encodings + NG-RC config
    mt = pd.read_csv(os.path.join(RESULTS, "cross", "lorenz_mv_matched.csv"))
    enc = {r.model: float(r.encode_scale) for r in mt.itertuples() if str(r.model).startswith("qrc_")}
    ngrow = mt[mt.model == "NG-RC"].iloc[0]
    ng = {"k": int(ngrow.ngrc_k), "degree": int(ngrow.ngrc_degree)}
    _log(f"closed-loop MV Lorenz: anchor={split.test.start}, horizon={N_STEPS}, "
         f"LE/step={LE_STEP:.4f}, enc={enc}, NG-RC k{ng['k']}d{ng['degree']}")

    models = ["NG-RC", "ELM", "ELM+poly2", "ESN(F=16)", "ESN(F=24)", "ESN(F=96)",
              "qrc_v4", "qrc_v6", "qrc_rich"]
    rows = []
    vpt_by, div_by = {}, {}        # per-model per-seed VPT and divergence (for significance)
    for m in models:
        vpts, divs, specs, wasss = [], 0, [], []
        for s in SEEDS:
            # seed varies the reservoir draw (ESN/QRC); deterministic models ignore it
            if m.startswith("qrc_"):
                qrc = GateQRC(P._qrc_cfg(m, enc[m], s)); win = qrc.cfg.window
                W = _train_W(qrc.featurize_mv(U[:split.train.stop]), U[:split.train.stop])
                step = lambda h, W=W, qrc=qrc, win=win: ridge_predict(
                    qrc.featurize_mv(h[-win:] if len(h) >= win else h)[-1][None], W, bias=True).ravel()
            elif m.startswith("ESN(F="):
                F = int(m.split("F=")[1].rstrip(")")); esn = ESN(ESNConfig(units=F, seed=s))
                W = _train_W(esn.featurize(U[:split.train.stop]), U[:split.train.stop])
                step = lambda h, W=W, esn=esn: ridge_predict(
                    esn.featurize(np.asarray(h))[-1][None], W, bias=True).ravel()
            else:
                W, step = build_model(m, U, split, enc, ng)   # deterministic; seed-independent
            res, true, preds = closed_loop_mv(step, U, split)
            vpts.append(res["vpt_lyap"]); divs += int(res["diverged"])
            div_by.setdefault(m, []).append(int(res["diverged"]))
            specs.append(res["spectral_mse"]); wasss.append(res["wasserstein"])
            safe = m.replace("(", "").replace(")", "").replace("=", "").replace("+", "p").replace("-", "")
            pd.DataFrame({"t": np.arange(len(true)), "tx": true[:, 0], "ty": true[:, 1], "tz": true[:, 2],
                          "px": preds[:, 0], "py": preds[:, 1], "pz": preds[:, 2]}).to_csv(
                os.path.join(tdir, f"{safe}_seed{s}.csv"), index=False)
        vpts = np.array(vpts); vpt_by[m] = vpts
        rows.append({"model": m, "vpt_lyap_median": float(np.median(vpts)),
                     "vpt_lyap_min": float(vpts.min()), "vpt_lyap_max": float(vpts.max()),
                     "diverged_count": divs, "n_seeds": len(SEEDS),
                     "spectral_mse_mean": float(np.mean(specs)),
                     "wasserstein_mean": float(np.mean(wasss)),
                     "is_quantum": m.startswith("qrc_")})
        _log(f"    {m:11s} VPT_lyap median={np.median(vpts):.3g} "
             f"[{vpts.min():.3g},{vpts.max():.3g}]  diverged={divs}/{len(SEEDS)}  "
             f"spec={np.mean(specs):.3g} W1={np.mean(wasss):.3g}")
    tbl = pd.DataFrame(rows).sort_values("vpt_lyap_median", ascending=False).reset_index(drop=True)
    tbl.to_csv(os.path.join(RESULTS, "cross", "lorenz_mv_cl_vpt.csv"), index=False)

    # ---- significance on the matched pairs (QRC vs same-F ESN; both stochastic) --
    from scipy.stats import mannwhitneyu, fisher_exact
    n = len(SEEDS)
    srows = []
    for q, e in [("qrc_rich", "ESN(F=96)"), ("qrc_v6", "ESN(F=24)")]:
        vq, ve = vpt_by[q], vpt_by[e]
        # VPT accuracy: one-sided Mann-Whitney, H1: QRC VPT > matched-ESN VPT
        try:
            U, pacc = mannwhitneyu(vq, ve, alternative="greater")
        except ValueError:
            U, pacc = np.nan, np.nan
        # stability: Fisher exact on the 2x2 [[not-div, div]] for QRC vs ESN
        qd, ed = int(sum(div_by[q])), int(sum(div_by[e]))
        _, pdiv = fisher_exact([[n - qd, qd], [n - ed, ed]])
        srows.append({"pair": f"{q} vs {e}", "qrc_vpt_median": float(np.median(vq)),
                      "esn_vpt_median": float(np.median(ve)),
                      "mannwhitney_U": float(U), "mannwhitney_p_greater": float(pacc),
                      "vpt_sig_0.05": bool(pacc < 0.05) if np.isfinite(pacc) else False,
                      "qrc_diverged": qd, "esn_diverged": ed, "n_seeds": n,
                      "fisher_p_divergence": float(pdiv), "div_sig_0.05": bool(pdiv < 0.05)})
    sig_df = pd.DataFrame(srows)
    sig_df.to_csv(os.path.join(RESULTS, "significance", "lorenz_mv_cl_significance.csv"), index=False)

    _log("\n================ CHECKPOINT (closed-loop) ================")
    _log("VPT table (Lyapunov times) + divergence counts:\n" + tbl.to_string(index=False))
    _log("\nmatched-pair significance (Mann-Whitney VPT one-sided; Fisher divergence):\n"
         + sig_df.to_string(index=False))
    # matched-pair read
    d = tbl.set_index("model")
    _log("\nmatched-pair VPT (QRC vs ESN at same F):")
    for q, e in [("qrc_v4", "ESN(F=16)"), ("qrc_v6", "ESN(F=24)"), ("qrc_rich", "ESN(F=96)")]:
        _log(f"  {q} median={d.loc[q,'vpt_lyap_median']:.3g} (div {int(d.loc[q,'diverged_count'])}/{len(SEEDS)}) "
             f"vs {e} median={d.loc[e,'vpt_lyap_median']:.3g} (div {int(d.loc[e,'diverged_count'])}/{len(SEEDS)})")
    _log("done — see results/cross/lorenz_mv_cl_vpt.csv")


if __name__ == "__main__":
    main()
