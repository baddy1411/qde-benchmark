#!/usr/bin/env python3
"""Scoped multivariate Lorenz-63 pilot (Part E).

Probes whether the univariate conclusion (a feature-matched classical reservoir,
and NG-RC, beat QRC; no quantum advantage) survives in the MULTIVARIATE regime
where Fellner et al. (2026) report quantum resources mattering. Hard-capped:
Lorenz-63 only, ONE encoding scheme, 5 seeds, n_points=1500 (quantum-joint), the
established DM/MCS machinery. A preliminary probe, not a co-equal result.

STRICT SYMMETRY (the thing that must be right): every model receives the same
scaled (T,3) Lorenz state and forecasts the same next (T,3) state via a multivariate
ridge read-out. QRC and its matched ESN both do internal windowing on the identical
input at the identical feature count F. NG-RC and ELM+poly2 are the strong classical
references (explicit multivariate delay + polynomial features), F reported honestly.

Encoding: the single 'obvious concatenation' scheme (GateQRC.featurize_mv) — the
window of the last `window` 3-vectors flattened time-major and fed through the
unchanged depth re-upload. encode_scale swept per QRC as in the univariate
matched-budget comparison; one scheme only (multi-encoding mixing-capacity analysis
is Fellner's, named as future work).

Outputs: results/cross/lorenz_mv_encoding.csv, lorenz_mv_matched.csv;
results/significance/lorenz_mv_DM.csv, lorenz_mv_MCS.csv;
results/forecasts/lorenz_mv/{model}_seed{n}.csv.

STOPS at the checkpoint (prints the matched-comparison setup + tables). No prose.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from qdepipe.data import generate_lorenz
from qdepipe.pipeline import temporal_split
from qdepipe.models import ESN, ESNConfig
from qdepipe.models.gate_qrc import GateQRC, GateQRCConfig
from qdepipe.readout import ridge_fit, ridge_predict
from qdepipe import metrics, significance as sig

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NPTS, SEEDS = 1500, tuple(range(5))
ENC_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ALPHA, WASHOUT = 1e-6, 100
QRC_F = {"qrc_v4": 16, "qrc_v6": 24, "qrc_rich": 96}


def _log(m): print(m, flush=True)


# ---- data: full (T,3) Lorenz state, train-only per-component min-max ---------
def lorenz_state():
    _, state = generate_lorenz(NPTS)              # (NPTS, 3) = (x,y,z)
    return np.asarray(state, dtype=float)


def scale_split(state):
    split = temporal_split(len(state), (0.6, 0.2, 0.2), WASHOUT)
    sc = MinMaxScaler().fit(state[split.train_fit])   # leakage-safe: fit on train
    return sc.transform(state), split


# ---- featurizers (all map U=(T,3) -> (T,F)) --------------------------------
def _qrc_cfg(key, encode_scale, seed):
    if key == "qrc_v4":
        return GateQRCConfig(n_qubits=4, encoding="depth", r=1, coupling="cnot_rz",
                             channel="dephasing", gamma=0.1, V=4, window=5,
                             readout=("Z",), encode_scale=encode_scale, seed=seed)
    if key == "qrc_v6":
        return GateQRCConfig(n_qubits=6, encoding="depth", r=1, coupling="isingxx",
                             J_strength=1.0, channel="none", V=4, window=5,
                             readout=("Z",), encode_scale=encode_scale, seed=seed)
    return GateQRCConfig(n_qubits=6, encoding="depth", r=1, coupling="isingxx",
                         J_strength=1.0, channel="none", V=4, window=5,
                         readout=("Z", "X", "Y", "ZZ"), encode_scale=encode_scale, seed=seed)


def feat_qrc(key, U, encode_scale, seed):
    return GateQRC(_qrc_cfg(key, encode_scale, seed)).featurize_mv(U)


def feat_esn(U, units, seed):
    return ESN(ESNConfig(units=units, seed=seed)).featurize(U)   # (T, units)


def _mv_delay(U, k):
    T, d = U.shape
    out = np.zeros((T, k * d))
    for j in range(k):
        sh = np.roll(U, j, axis=0); sh[:j] = U[0]
        out[:, j * d:(j + 1) * d] = sh
    return out


def feat_ngrc(U, k, degree):
    lin = _mv_delay(U, k)
    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(lin)


def feat_elm(U, units, seed, lookback=3, poly2=False):
    win = _mv_delay(U, lookback)                  # (T, lookback*3)
    rng = np.random.default_rng(seed + 7)
    W = rng.uniform(-1, 1, size=(units, win.shape[1]))
    b = rng.uniform(-1, 1, size=units)
    H = np.tanh(win @ W.T + b)                     # (T, units)
    return np.concatenate([H, H ** 2], axis=1) if poly2 else H


# ---- fit/predict (multivariate ridge; reservoir-style alignment) -----------
def fit_predict(feats, U, split, horizon=1):
    X, Y = feats[:len(feats) - horizon], U[horizon:]
    a, b, w = split.train.stop, split.test.start, split.washout
    tr, te = slice(w, a - horizon), slice(b, len(X))
    W = ridge_fit(X[tr], Y[tr], alpha=ALPHA, bias=True)
    return Y[te], ridge_predict(X[te], W, bias=True)   # (n_test,3) true, pred


# ---- encoding sweep (per QRC, pick best aggregate NRMSE over seeds) ---------
def encode_sweep(U, split):
    _log("[1] QRC encoding sweep (multivariate)")
    rows, best = [], {}
    for key in QRC_F:
        grid = []
        for e in ENC_GRID:
            aggs = []
            for s in SEEDS:
                yt, yp = fit_predict(feat_qrc(key, U, e, s), U, split)
                aggs.append(metrics.nrmse_vector(yt, yp)[0])
            grid.append((e, float(np.mean(aggs))))
            rows.append({"model": key, "encode_scale": e, "nrmse_agg_mean": float(np.mean(aggs))})
        be = min(grid, key=lambda t: t[1])
        best[key] = be[0]
        _log(f"    {key:9s} best encode_scale={be[0]:g}  NRMSE_agg={be[1]:.4g}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "cross", "lorenz_mv_encoding.csv"), index=False)
    return best


# ---- NG-RC degree/k sweep (best aggregate NRMSE) ---------------------------
def ngrc_best(U, split):
    best = None
    for k in (2, 3):
        for d in (1, 2, 3):
            f = feat_ngrc(U, k, d)
            yt, yp = fit_predict(f, U, split)
            agg = metrics.nrmse_vector(yt, yp)[0]
            if best is None or agg < best["nrmse"]:
                best = {"k": k, "degree": d, "F": f.shape[1], "nrmse": agg}
    _log(f"[2] NG-RC best: k={best['k']} degree={best['degree']} F={best['F']} "
         f"NRMSE_agg={best['nrmse']:.4g}")
    return best


# ---- full model set, 5 seeds, persist forecasts, NRMSE table ---------------
def run_models(U, split, enc_best, ng):
    fdir = os.path.join(RESULTS, "forecasts", "lorenz_mv")
    os.makedirs(fdir, exist_ok=True)
    # label -> featurizer(seed) -> feats ; plus the matched-F bookkeeping
    def make(label):
        if label.startswith("qrc_"):
            F = QRC_F[label]; e = enc_best[label]
            return (lambda s: feat_qrc(label, U, e, s)), F, {"encode_scale": e}
        if label.startswith("ESN(F="):
            F = int(label.split("F=")[1].rstrip(")"))
            return (lambda s: feat_esn(U, F, s)), F, {}
        if label == "NG-RC":
            return (lambda s: feat_ngrc(U, ng["k"], ng["degree"])), ng["F"], {"k": ng["k"], "degree": ng["degree"]}
        if label == "ELM":
            return (lambda s: feat_elm(U, 300, s)), 300, {}
        if label == "ELM+poly2":
            return (lambda s: feat_elm(U, 300, s, poly2=True)), 600, {}
        raise ValueError(label)

    models = ["NG-RC", "ELM", "ELM+poly2",
              "ESN(F=16)", "ESN(F=24)", "ESN(F=96)", "qrc_v4", "qrc_v6", "qrc_rich"]
    per_seed, rows = {m: [] for m in models}, []
    for m in models:
        feat_fn, F, meta = make(m)
        aggs, comps = [], []
        for s in SEEDS:
            yt, yp = fit_predict(feat_fn(s), U, split)
            agg, per = metrics.nrmse_vector(yt, yp)
            aggs.append(agg); comps.append(per)
            per_seed[m].append((yt, yp))
            safe = m.replace("(", "").replace(")", "").replace("=", "").replace("+", "p").replace("-", "")
            pd.DataFrame({"t": np.arange(len(yt)), "true_x": yt[:, 0], "true_y": yt[:, 1],
                          "true_z": yt[:, 2], "pred_x": yp[:, 0], "pred_y": yp[:, 1],
                          "pred_z": yp[:, 2]}).to_csv(os.path.join(fdir, f"{safe}_seed{s}.csv"), index=False)
        comps = np.array(comps)
        rows.append({"model": m, "F": F, "encode_scale": meta.get("encode_scale", ""),
                     "ngrc_k": meta.get("k", ""), "ngrc_degree": meta.get("degree", ""),
                     "NRMSE_agg_mean": float(np.mean(aggs)), "NRMSE_agg_std": float(np.std(aggs)),
                     "NRMSE_x": float(comps[:, 0].mean()), "NRMSE_y": float(comps[:, 1].mean()),
                     "NRMSE_z": float(comps[:, 2].mean())})
        _log(f"    {m:11s} F={F:<5d} NRMSE_agg={np.mean(aggs):.4g} "
             f"(x={comps[:,0].mean():.3g} y={comps[:,1].mean():.3g} z={comps[:,2].mean():.3g})")
    tbl = pd.DataFrame(rows).sort_values("NRMSE_agg_mean").reset_index(drop=True)
    tbl.to_csv(os.path.join(RESULTS, "cross", "lorenz_mv_matched.csv"), index=False)
    return models, per_seed, tbl


# ---- multivariate DM + MCS (loss = squared Euclidean residual per step) ----
def significance_mv(models, per_seed, ng_label):
    # DM headline pairs (error series = per-step residual norm; DM squares it -> SSE loss)
    def err(m, si):
        yt, yp = per_seed[m][si]
        return np.linalg.norm(yp - yt, axis=1)
    pairs = [("qrc_rich", "ESN(F=96)"), ("qrc_rich", "NG-RC"), ("qrc_v6", "ESN(F=24)")]
    drows = []
    for a, b in pairs:
        st, ps = [], []
        for si in range(len(SEEDS)):
            s_, p_ = sig.diebold_mariano(err(a, si), err(b, si))
            st.append(s_); ps.append(p_)
        st, ps = np.array(st), np.array(ps)
        drows.append({"pair": f"{a} vs {b}", "frac_sig_0.10": float(np.mean(ps < 0.10)),
                      "frac_sig_0.05": float(np.mean(ps < 0.05)), "median_dm": float(np.median(st)),
                      "dm_min": float(st.min()), "dm_max": float(st.max()), "median_p": float(np.median(ps))})
        _log(f"    DM {a} vs {b}: frac_sig@0.05={np.mean(ps<0.05):.2f} median_DM={np.median(st):.3g}")
    pd.DataFrame(drows).to_csv(os.path.join(RESULTS, "significance", "lorenz_mv_DM.csv"), index=False)

    # MCS over all models: loss[m,t] = ||resid||^2, per seed -> aggregate in-set fraction
    in10 = {m: 0 for m in models}; pv = {m: [] for m in models}
    for si in range(len(SEEDS)):
        L = min(len(per_seed[m][si][0]) for m in models)
        losses = np.stack([np.sum((per_seed[m][si][1][-L:] - per_seed[m][si][0][-L:]) ** 2, axis=1)
                           for m in models], axis=0)
        res = sig.model_confidence_set(losses, names=models, alpha=(0.10, 0.25),
                                       n_boot=1000, block_length=20, seed=0)
        for m in models:
            pv[m].append(res["mcs_pvalue"][m])
            if m in res["in_set"][0.10]:
                in10[m] += 1
    mcs = pd.DataFrame([{"model": m, "mean_mcs_pvalue": float(np.mean(pv[m])),
                         "frac_seeds_in_set_0.10": in10[m] / len(SEEDS),
                         "is_quantum": m.startswith("qrc_")} for m in models]
                       ).sort_values("mean_mcs_pvalue", ascending=False)
    mcs.to_csv(os.path.join(RESULTS, "significance", "lorenz_mv_MCS.csv"), index=False)
    return pd.DataFrame(drows), mcs


def main():
    os.makedirs(os.path.join(RESULTS, "cross"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "significance"), exist_ok=True)
    from qdepipe._provenance import write_versions
    write_versions(os.path.join(os.path.dirname(RESULTS), "versions.txt"))
    U, split = scale_split(lorenz_state())
    _log(f"multivariate Lorenz pilot: U shape={U.shape}, n_test={split.test.stop-split.test.start}, "
         f"seeds={list(SEEDS)}")
    enc_best = encode_sweep(U, split)
    ng = ngrc_best(U, split)
    _log("[3] full model set, 5 seeds")
    models, per_seed, tbl = run_models(U, split, enc_best, ng)
    _log("[4] significance")
    dm, mcs = significance_mv(models, per_seed, "NG-RC")

    _log("\n================ CHECKPOINT ================")
    _log("matched-comparison setup: every model input = scaled (x,y,z) state; "
         "target = next (x,y,z); multivariate ridge read-out.")
    _log("\nNRMSE table (aggregate + per-component):\n" + tbl.to_string(index=False))
    _log("\nMCS best-set (frac_in_set>=0.5):")
    bestset = mcs[mcs["frac_seeds_in_set_0.10"] >= 0.5]["model"].tolist()
    _log(f"  {bestset}  | quantum in set: {[m for m in bestset if m.startswith('qrc_')] or 'none'}")
    _log("done — see results/cross/lorenz_mv_matched.csv, results/significance/lorenz_mv_*.csv")


if __name__ == "__main__":
    main()
