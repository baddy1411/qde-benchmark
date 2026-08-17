"""Figure generation for the validated classical cross-system results.

Every figure reads ONLY from committed CSVs under results/ — no hardcoded numbers,
no fabricated values. Quantum-dependent figures are scaffolded as explicitly-pending
(they draw the classical layer and a labelled placeholder, never fake QRC points).

Outputs vector PDF + PNG to results/figures/, colorblind-safe palette, readable fonts.

Run via `make plots` or `python -m qdepipe.plots`.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
SYSTEMS = ["henon", "lorenz", "mackeyglass"]
TITLES = {"henon": "Hénon", "lorenz": "Lorenz-63", "mackeyglass": "Mackey–Glass"}

# Okabe–Ito colorblind-safe palette
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.dpi": 120, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

_manifest = []   # (figure_file, source_csvs, one-line description)


def _save(fig, name, sources, desc):
    os.makedirs(FIG, exist_ok=True)
    # strip embedded timestamps so re-runs are byte-identical (deterministic figures)
    meta = {"pdf": {"CreationDate": None}, "png": {"Software": None}}
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), metadata=meta[ext])
    plt.close(fig)
    _manifest.append((name, sources, desc))
    print(f"  wrote figures/{name}.pdf|png  <- {', '.join(sources)}")


def _csv(*parts):
    return os.path.join(RES, *parts)


# ---------------------------------------------------------------------------
# PRIORITY 1 — NG-RC degree-sweep (the thesis-in-one-figure)
# ---------------------------------------------------------------------------
def fig_degree_sweep():
    sources = [f"cross/{s}_ngrc_tune.csv" for s in SYSTEMS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, system in zip(axes, SYSTEMS):
        df = pd.read_csv(_csv("cross", f"{system}_ngrc_tune.csv"))
        ks = sorted(df["k"].unique())
        best = df.loc[df["nrmse_mean"].idxmin()]
        for i, k in enumerate(ks):
            d = df[df["k"] == k].sort_values("degree")
            is_best_k = (k == best["k"])
            ax.plot(d["degree"], d["nrmse_mean"], marker="o",
                    color=CB[i % len(CB)], lw=2.2 if is_best_k else 1.2,
                    alpha=1.0 if is_best_k else 0.5,
                    label=f"lookback k={k}" + (" (best)" if is_best_k else ""))
        ax.scatter([best["degree"]], [best["nrmse_mean"]], s=160, facecolors="none",
                   edgecolors="red", linewidths=2, zorder=5)
        ax.annotate(f"best: k={int(best['k'])}, d={int(best['degree'])}",
                    (best["degree"], best["nrmse_mean"]), textcoords="offset points",
                    xytext=(6, 10), fontsize=9, color="red")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("NG-RC polynomial degree")
        ax.set_title(TITLES[system])
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("1-step NRMSE (log scale)")
    fig.suptitle("NG-RC accuracy snaps to the polynomial degree each system requires",
                 fontsize=13, y=1.02)
    _save(fig, "fig1_ngrc_degree_sweep", sources,
          "NRMSE vs NG-RC degree per system; shows each system's required degree.")


# ---------------------------------------------------------------------------
# PRIORITY 2 — DM significance heatmap (proof-of-point)
# ---------------------------------------------------------------------------
def _dm_pairs_path(system):
    """Prefer the joint (classical+quantum) DM pairs when present, else classical."""
    joint = _csv("significance", f"{system}_DM_pairs_joint.csv")
    return joint if os.path.exists(joint) else _csv("significance", f"{system}_DM_pairs.csv")


def fig_dm_heatmap():
    sources = [os.path.relpath(_dm_pairs_path(s), RES) for s in SYSTEMS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    # 0 = indistinguishable (p>=0.10), 1 = sig at 0.10, 2 = sig at 0.05
    cmap = ListedColormap(["#DDDDDD", "#E69F00", "#0072B2"])
    for ax, system in zip(axes, SYSTEMS):
        pairs = pd.read_csv(_dm_pairs_path(system))
        names = sorted(set(pairs["model_i"]) | set(pairs["model_j"]))
        idx = {m: i for i, m in enumerate(names)}
        n = len(names)
        P = np.full((n, n), np.nan)
        for _, r in pairs.iterrows():
            P[idx[r["model_i"]], idx[r["model_j"]]] = r["median_p"]
        cls = np.zeros((n, n))
        cls[P >= 0.10] = 0
        cls[(P < 0.10) & (P >= 0.05)] = 1
        cls[P < 0.05] = 2
        np.fill_diagonal(cls, np.nan)
        ax.imshow(np.ma.masked_invalid(cls), cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        # highlight quantum models (red, bold) so the eye finds them in the matrix
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            if "qrc" in tl.get_text().lower():
                tl.set_color("#D55E00"); tl.set_fontweight("bold")
        ax.set_title(TITLES[system])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#DDDDDD", "#E69F00", "#0072B2"]]
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles, ["indistinguishable (p≥0.10)", "sig. at 0.10", "sig. at 0.05"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Pairwise Diebold–Mariano significance (summarised over seeds; "
                 "formal per-seed tests in the scaling family)", y=1.02)
    _save(fig, "fig2_dm_heatmap", sources,
          "DM p-value class per model pair per system; confirms MCS verdicts.")


# ---------------------------------------------------------------------------
# PRIORITY 3 — VPT / climate (1-step skill != attractor reproduction)
# ---------------------------------------------------------------------------
def fig_vpt():
    sources = [f"cross/{s}_climate.csv" for s in SYSTEMS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    def _color(m):
        if "qrc" in m.lower():
            return "#009E73"          # quantum
        if "NG-RC" in m:
            return "#D55E00"          # NG-RC
        return "#0072B2"              # other classical
    for ax, system in zip(axes, SYSTEMS):
        df = pd.read_csv(_csv("cross", f"{system}_climate.csv")).sort_values("vpt_lyap_mean")
        ax.barh(df["model"], df["vpt_lyap_mean"], color=[_color(m) for m in df["model"]])
        ax.set_xlabel("VPT (Lyapunov times)")
        ax.set_title(TITLES[system])
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("model")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#D55E00", "#009E73", "#0072B2"]]
    fig.subplots_adjust(bottom=0.22)
    fig.legend(handles, ["NG-RC", "quantum (QRC)", "other classical"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Closed-loop validity time — best 1-step model is NOT always the most "
                 "stable (see Lorenz: NG-RC worst VPT)", y=1.02, fontsize=12)
    _save(fig, "fig3_climate_vpt", sources,
          "VPT in Lyapunov times per model per system; 1-step vs climate divergence.")


# ---------------------------------------------------------------------------
# Seed-distribution (box/violin) from per-seed forecasts
# ---------------------------------------------------------------------------
def _per_seed_nrmse(system):
    """Recompute per-seed NRMSE from the committed forecast CSVs (no summary reuse)."""
    fdir = _csv("forecasts", system)
    if not os.path.isdir(fdir):
        return {}
    out = {}
    for f in os.listdir(fdir):
        if not f.endswith(".csv"):
            continue
        model = f.rsplit("_seed", 1)[0]
        d = pd.read_csv(os.path.join(fdir, f))
        err = d["y_pred"].to_numpy() - d["y_true"].to_numpy()
        denom = np.std(d["y_true"].to_numpy())
        nrmse = np.sqrt(np.mean(err ** 2)) / (denom if denom > 0 else 1.0)
        out.setdefault(model, []).append(nrmse)
    return out


def fig_seed_distribution():
    sources = [f"forecasts/{s}/*.csv" for s in SYSTEMS]
    # ~17 models per system: a tall 3-row layout (full width, generous height per
    # panel) keeps every model label legible at print size.
    fig, axes = plt.subplots(3, 1, figsize=(9, 13))
    drew = False
    for ax, system in zip(axes, SYSTEMS):
        data = _per_seed_nrmse(system)
        if not data:
            ax.set_visible(False); continue
        models = sorted(data, key=lambda m: np.mean(data[m]))
        vals = [data[m] for m in models]
        ax.boxplot(vals, vert=False, showmeans=True, widths=0.6)
        ax.set_yticks(range(1, len(models) + 1))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("NRMSE per seed (log)")
        ax.set_title(TITLES[system], fontsize=11)
        ax.margins(y=0.02)
        drew = True
    fig.suptitle("Per-seed NRMSE distribution (from committed forecasts) — replaces mean±std",
                 y=1.005, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    if drew:
        _save(fig, "fig4_seed_distribution", sources,
              "Box plot of per-seed NRMSE per model per system, from raw forecasts.")
    else:
        plt.close(fig)
        print("  [skip] fig4_seed_distribution: no forecast CSVs found")


# ---------------------------------------------------------------------------
# Matched-budget scaffold (classical line + NG-RC ref; QRC pending placeholder)
# ---------------------------------------------------------------------------
def _matched_panel(ax, system):
    """Plot one matched-budget panel from committed CSVs. Returns source list, or
    None if no data exists (caller scaffolds)."""
    if system == "henon":
        p = _csv("adv_A2_matched.csv")            # Hénon, QRC at tuned best
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
        F, q, e = d["F"], d["nrmse_quantum"], d["nrmse_esn_matched"]
        ng = float(d["nrmse_ngrc_ref"].iloc[0])
        src = ["adv_A2_matched.csv"]
    else:
        p = _csv("cross", f"{system}_matched_budget.csv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
        F, q, e = d["F"], d["NRMSE_quantum_best"], d["NRMSE_ESN_F"]
        ng = float(d["NRMSE_NGRC_ref"].iloc[0])
        src = [f"cross/{system}_matched_budget.csv"]
    order = np.argsort(F.to_numpy())
    Fo = F.to_numpy()[order]
    ax.plot(Fo, e.to_numpy()[order], "-s", color="#0072B2", label="ESN (units=F)")
    ax.scatter(F, q, marker="o", s=80, color="#009E73", zorder=5, label="QRC (tuned best)")
    for fi, qi, mi in zip(F, q, d["model"] if "model" in d else d["quantum_model"]):
        ax.annotate(str(mi).replace("qrc_", ""), (fi, qi), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="#009E73")
    ax.axhline(ng, ls="--", color="#D55E00", label="NG-RC reference")
    ax.set_yscale("log"); ax.set_xlabel("feature count F"); ax.set_title(TITLES.get(system, "Hénon"))
    ax.legend(fontsize=8)
    return src


def fig_matched_budget():
    """NRMSE vs F per system: ESN(F) line, real tuned-QRC points, NG-RC reference.
    All values from committed CSVs; a system with no matched-budget CSV is scaffolded
    as explicitly pending (never fabricated)."""
    systems = ["henon", "lorenz", "mackeyglass"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    sources = []
    for ax, system in zip(axes, systems):
        src = _matched_panel(ax, system)
        if src is None:
            ax.text(0.5, 0.5, f"{TITLES.get(system, 'Hénon')}\nmatched-budget PENDING\n"
                    "(quantum pass not yet run)", transform=ax.transAxes, ha="center",
                    va="center", fontsize=10, style="italic", color="grey")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            sources += src
    axes[0].set_ylabel("1-step NRMSE (log)")
    fig.suptitle("Matched feature-budget per system — does any QRC beat ESN(F) or NG-RC? "
                 "(exact-expectation: idealised quantum reference)", y=1.03, fontsize=12)
    _save(fig, "fig5_matched_budget", sources + ["baseline_leaderboard.csv"],
          "NRMSE vs F per system: ESN(F) line, tuned-QRC points, NG-RC ref (committed).")


# ---------------------------------------------------------------------------
# Forecast overlay (top classical models, per system)
# ---------------------------------------------------------------------------
def fig_forecast_overlay():
    sources = [f"forecasts/{s}/*.csv" for s in SYSTEMS]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    drew = False
    for ax, system in zip(axes, SYSTEMS):
        fdir = _csv("forecasts", system)
        if not os.path.isdir(fdir):
            ax.set_visible(False); continue
        # pick the best-mean model from the leaderboard if available
        lb_path = _csv("cross", f"{system}_leaderboard.csv")
        files = [f for f in os.listdir(fdir) if f.endswith("_seed0.csv")]
        if not files:
            ax.set_visible(False); continue
        best_model = None
        if os.path.exists(lb_path):
            lb = pd.read_csv(lb_path)
            cand = lb.iloc[0]["model"].replace("(", "").replace(")", "").replace(",", "_").replace("+", "p")
            match = [f for f in files if f.startswith(cand + "_seed0")]
            if match:
                best_model = match[0]
        f0 = best_model or files[0]
        d = pd.read_csv(os.path.join(fdir, f0))
        m = min(200, len(d))
        ax.plot(d["y_true"][:m].to_numpy(), color="black", lw=1.3, label="true")
        ax.plot(d["y_pred"][:m].to_numpy(), color="#E69F00", lw=1.2, ls="--",
                label=f"prediction ({f0.rsplit('_seed', 1)[0]})")
        ax.set_title(f"{TITLES[system]} — 1-step forecast overlay (test, first {m})")
        ax.set_xlabel("test step"); ax.set_ylabel("x (scaled)")
        ax.legend(fontsize=9)
        drew = True
    fig.tight_layout()
    if drew:
        _save(fig, "fig6_forecast_overlay", sources,
              "True vs 1-step prediction (best classical model) per system, from forecasts.")
    else:
        plt.close(fig)
        print("  [skip] fig6_forecast_overlay: no forecast CSVs")


# ---------------------------------------------------------------------------
# NOVEL FIGURE — concentration: ZZ variance collapse vs vanishing ZZ benefit
# ---------------------------------------------------------------------------
def fig_concentration():
    src = "concentration/scaling.csv"
    path = _csv("concentration", "scaling.csv")
    if not os.path.exists(path):
        print("  [skip] fig7_concentration: scaling.csv missing (run make shots)")
        return
    sc = pd.read_csv(path)
    vsrc = "concentration/scaling_variance_5seed.csv"
    bsrc = "concentration/scaling_benefit_5seed.csv"
    vpath = _csv("concentration", "scaling_variance_5seed.csv")
    bpath = _csv("concentration", "scaling_benefit_5seed.csv")
    vdf = pd.read_csv(vpath) if os.path.exists(vpath) else None
    bdf = pd.read_csv(bpath) if os.path.exists(bpath) else None
    systems = list(sc["system"].unique())
    fig, axes = plt.subplots(2, len(systems), figsize=(6.5 * len(systems), 8),
                             squeeze=False, layout="constrained")
    for ci, system in enumerate(systems):
        d = sc[sc.system == system]
        # variance panel: FIVE seeds, error bars = ±1 std across seeds. The point is that
        # the 2-local/1-local separation sits within seed noise — there is NO robust trend
        # with qubit count (the 2-seed monotonic 'decline' was a small-sample artifact).
        ax = axes[0][ci]
        if vdf is not None and (vdf.system == system).any():
            dv = vdf[vdf.system == system].sort_values("n_qubits")
            nv = dv["n_qubits"].tolist()
            ax.errorbar(nv, dv["var_1local_mean"], yerr=dv["var_1local_std"], fmt="-o",
                        color="#0072B2", lw=2, capsize=5, label="1-local (Z,X,Y)")
            ax.errorbar(nv, dv["var_2local_mean"], yerr=dv["var_2local_std"], fmt="-s",
                        color="#D55E00", lw=2, capsize=5, label="2-local (ZZ)")
            ax.set_ylim(bottom=0); ax.set_xticks(nv)
            ax.set_ylabel("variance of ⟨o⟩" if ci == 0 else "")
            ax.set_title(f"{TITLES.get(system, system)} — observable variance (5 seeds)")
            ax.legend(fontsize=9)
        ax.set_xlabel("n_qubits")
        # benefit panel: 5-seed with ±1-std error bars for n∈{4,6,8}; n=10 (Hénon
        # only) stays at its original two seeds and is marked distinctly. The error
        # bars make the growing-but-noisy MG trend visible (std 0.03→0.09→0.18).
        ns = sorted(d["n_qubits"].unique())
        ax2 = axes[1][ci]
        b5 = bdf[bdf.system == system].set_index("n_qubits") if bdf is not None else None
        x5, y5, e5, x2, y2 = [], [], [], [], []
        for n in ns:
            if b5 is not None and n in b5.index:
                x5.append(n); y5.append(float(b5.loc[n, "benefit_5seed_mean"]))
                e5.append(float(b5.loc[n, "benefit_5seed_std"]))
            else:  # n=10: 2-seed, from scaling.csv
                zxy = float(d[(d.n_qubits == n) & (d.readout_set == "Z+X+Y")]["NRMSE"].iloc[0])
                full = float(d[(d.n_qubits == n) & (d.readout_set == "Z+X+Y+ZZ")]["NRMSE"].iloc[0])
                x2.append(n); y2.append(zxy / full)
        if x5:
            ax2.errorbar(x5, y5, yerr=e5, fmt="-D", color="#009E73", lw=2, capsize=5,
                         label="ZZ benefit (5 seeds, ±1 std)")
        if x2:
            ax2.plot(x2, y2, "x", color="#888888", ms=10, mew=2,
                     label="n=10 (2 seeds; excluded from variance)")
        ax2.axhline(1.0, ls=":", color="grey", label="ZZ buys nothing (ratio = 1)")
        ax2.set_xticks(ns); ax2.set_xlabel("n_qubits")
        lo = min([y5[i] - e5[i] for i in range(len(y5))] + y2 + [0.98]) if (y5 or y2) else 0.98
        hi = max([y5[i] + e5[i] for i in range(len(y5))] + y2 + [1.02]) if (y5 or y2) else 1.25
        ax2.set_ylim(lo * 0.99, hi * 1.02)
        ax2.set_ylabel("ZZ benefit ratio" if ci == 0 else "")
        ax2.set_title(f"{TITLES.get(system, system)} — does ZZ still help?")
        ax2.legend(fontsize=8)
    fig.suptitle("Mechanism check: observable variance (top, no robust trend) and "
                 "ZZ benefit ratio (bottom; ratio = NRMSE(Z+X+Y) / NRMSE(+ZZ))")
    _save(fig, "fig7_concentration", [src, vsrc, bsrc],
          "5-seed observable variance (top) and ZZ NRMSE-benefit (bottom), both ±1-std error "
          "bars; no robust variance trend; MG benefit grows but noisy at high n; n=10 stays 2-seed.")


def fig_finite_shots():
    src = "concentration/finite_shots.csv"
    path = _csv("concentration", "finite_shots.csv")
    if not os.path.exists(path):
        print("  [skip] fig8_finite_shots: finite_shots.csv missing")
        return
    fs = pd.read_csv(path)
    # x-axis: shots as ordered categories exact > 8192 > 1024
    order = {"exact": 0, "8192": 1, "1024": 2}
    fs["xo"] = fs["shots"].astype(str).map(order)
    ns = sorted(fs["n_qubits"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.4), squeeze=False)
    for ci, n in enumerate(ns):
        ax = axes[0][ci]
        for rd, color, mk in (("Z+X+Y", "#0072B2", "o"), ("Z+X+Y+ZZ", "#D55E00", "s")):
            d = fs[(fs.n_qubits == n) & (fs.readout_set == rd)].sort_values("xo")
            ax.plot(d["xo"], d["NRMSE"], f"-{mk}", color=color, lw=2, label=rd)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["exact", "8192", "1024"])
        ax.set_xlabel("shots (← more realistic)")
        ax.set_yscale("log"); ax.set_title(f"n_qubits = {n}")
        ax.legend(fontsize=9)
    axes[0][0].set_ylabel("1-step NRMSE (log)")
    fig.suptitle("Finite-shot degradation: exact expectation is a ceiling; ZZ readout "
                 "degrades under shot noise", y=1.04, fontsize=12)
    fig.tight_layout()
    _save(fig, "fig8_finite_shots", [src],
          "NRMSE vs shots for Z+X+Y vs Z+X+Y+ZZ per n_qubits; finite-shot degradation.")


# ---------------------------------------------------------------------------
# Frequency-domain redundancy test (mechanism refinement; Part D)
# ---------------------------------------------------------------------------
def fig_frequency_redundancy():
    src = "concentration/frequency_redundancy.csv"
    path = _csv("concentration", "frequency_redundancy.csv")
    if not os.path.exists(path):
        print("  [skip] fig10_frequency: frequency_redundancy.csv missing")
        return
    df = pd.read_csv(path).set_index("system")
    syscol = {"henon": "#0072B2", "mackeyglass": "#D55E00"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # left: linear residual-power fraction (both ~0)
    sysn = [s for s in ("henon", "mackeyglass") if s in df.index]
    rp = [float(df.loc[s, "zz_residual_power_fraction"]) for s in sysn]
    axL.bar([TITLES.get(s, s) for s in sysn], rp, color=[syscol[s] for s in sysn])
    axL.set_yscale("log"); axL.set_ylabel("ZZ residual-power fraction $(1-R^2)$")
    axL.set_title("Linear reachability: ZZ is linearly spanned\nby single-qubit set on BOTH systems")
    for i, v in enumerate(rp):
        axL.annotate(f"{v:.1e}", (i, v), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=9)
    axL.set_ylim(top=max(rp) * 100)
    # right: new-frequency fraction vs support threshold
    thrs = [0.001, 0.01, 0.05]
    cols = [f"zz_new_frequency_fraction_thr{t}" for t in thrs]
    for s in sysn:
        axR.plot(thrs, [float(df.loc[s, c]) for c in cols], "-o", color=syscol[s],
                 label=TITLES.get(s, s))
    axR.set_xscale("log"); axR.set_xlabel("single-qubit support threshold")
    axR.set_ylabel("ZZ new-frequency fraction")
    axR.set_title("New-frequency content: MG $>$ Hénon\n(ordering robust across thresholds)")
    axR.legend(fontsize=9)
    fig.suptitle("Frequency-domain redundancy test: the linear measure finds ZZ redundant on both "
                 "systems;\nthe new-frequency measure separates them", y=1.05, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig10_frequency_redundancy", [src],
          "Linear residual-power (left, both ~0) vs new-frequency fraction (right, MG>Hénon).")


# ---------------------------------------------------------------------------
# Phase-space attractor reconstruction (the climate story made visual)
# ---------------------------------------------------------------------------
def fig_attractor():
    tdir = _csv("trajectories")
    if not os.path.isdir(tdir):
        print("  [skip] fig9_attractor: trajectories/ missing (run experiments/closedloop_traj.py)")
        return
    tau = {"henon": 1, "lorenz": 8, "mackeyglass": 12}
    models = ["true", "NG-RC", "ESN", "RandomForest"]
    systems = ["henon", "lorenz", "mackeyglass"]
    fig, axes = plt.subplots(len(systems), len(models),
                             figsize=(3.4 * len(models), 3.2 * len(systems)), squeeze=False)
    for ri, system in enumerate(systems):
        tp = os.path.join(tdir, f"{system}_true.csv")
        if not os.path.exists(tp):
            continue
        true = pd.read_csv(tp)["true"].to_numpy()
        t = tau[system]
        lo, hi = float(np.min(true)), float(np.max(true))
        span = hi - lo
        xl = (lo - 0.6 * span, hi + 0.6 * span)
        for ci, model in enumerate(models):
            ax = axes[ri][ci]
            col = "true" if model == "true" else "pred"
            p = os.path.join(tdir, f"{system}_{model}.csv")
            if not os.path.exists(p):
                ax.set_visible(False); continue
            x = pd.read_csv(p)[col].to_numpy()
            x = np.clip(x, xl[0], xl[1])              # keep blown-up runs from breaking scale
            a, b = x[:-t], x[t:]
            color = "#000000" if model == "true" else "#0072B2"
            diverged = (model != "true") and (np.max(np.abs(pd.read_csv(p)[col].to_numpy()))
                                              > 5 * max(abs(lo), abs(hi)))
            if diverged:
                color = "#D55E00"
            ax.plot(a, b, ".", ms=1.4, color=color, alpha=0.6)
            ax.set_xlim(xl); ax.set_ylim(xl)
            ax.set_xticks([]); ax.set_yticks([])
            ttl = (TITLES[system] + " — true") if model == "true" else model
            if diverged:
                ttl += "  (DIVERGED)"
            ax.set_title(ttl, fontsize=10, color=color if diverged else "black")
            if ci == 0:
                ax.set_ylabel(TITLES[system], fontsize=11)
    fig.suptitle("Free-running attractor reconstruction (delay-embedded): NG-RC leaves the "
                 "Lorenz / Mackey-Glass attractor despite top 1-step skill — "
                 "1-step accuracy ≠ learned dynamics", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "fig9_attractor",
          [f"trajectories/{s}_*.csv" for s in systems],
          "Delay-embedded free-run attractor: true vs each model's autonomous rollout.")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
def write_index():
    lines = ["# QDE — Figure Index (auto-generated)\n",
             "*Generated by `qdepipe/plots.py` (`make plots`). Every figure reads only "
             "from committed CSVs listed below — no hardcoded or fabricated values. "
             "Quantum-dependent panels are explicitly marked pending.*\n",
             "| Figure | Source CSV(s) | What it shows |",
             "|---|---|---|"]
    for name, sources, desc in _manifest:
        lines.append(f"| `figures/{name}.pdf` (+png) | {', '.join(f'`{s}`' for s in sources)} | {desc} |")
    lines.append("\n## Notes\n- **fig5 (matched budget)** now carries real tuned-QRC points "
                 "per system (exact-expectation idealised reference); NG-RC reference shown.\n"
                 "- **fig7 (mechanism)** variance panel uses comparable-n_points rows (n≤8); the "
                 "n=10 variance is excluded as a reduced-n_points confound (NRMSE panel keeps n=10).\n"
                 "- **fig9 (attractor)** uses closed-loop free-running trajectories generated by "
                 "`experiments/closedloop_traj.py` → `results/trajectories/*.csv`; diverged runs clipped to the "
                 "true attractor's range ±0.6·span for display and labelled DIVERGED.\n"
                 "- All quantum results are exact-expectation (idealised reference); finite-shot "
                 "realism is fig8.\n")
    with open(os.path.join(RES, "FIGURES.md"), "w") as f:
        f.write("\n".join(lines))
    print("  wrote results/FIGURES.md")



# ---------------------------------------------------------------------------
# fig15 — exemplary input series (main text; the reader should meet the data)
# ---------------------------------------------------------------------------
def fig_input_series():
    """The three benchmark signals themselves, as the pipeline delivers them.

    Added to the main text so a reader sees the data before any metric. Reads the
    committed target trajectories, which are the same series the forecasting
    experiments consume.
    """
    sources = [f"trajectories/{s}_true.csv" for s in SYSTEMS]
    WIN = {"henon": 200, "lorenz": 400, "mackeyglass": 400}
    NOTE = {
        "henon": "discrete map — the update rule is an exact low-degree polynomial",
        "lorenz": "continuous flow, sampled — x-component of the three-dimensional state",
        "mackeyglass": "delay system — the state depends on values many steps back",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.6))
    for ax, system in zip(axes, SYSTEMS):
        d = pd.read_csv(_csv("trajectories", f"{system}_true.csv"))
        y = d["true"].to_numpy()[: WIN[system]]
        ax.plot(np.arange(len(y)), y, color=CB[0], lw=1.0)
        ax.set_xlim(0, len(y))
        ax.set_ylabel("scaled value", fontsize=10)
        # title carries the annotation, so nothing is written over the signal
        ax.set_title(f"{TITLES[system]} — {NOTE[system]}", fontsize=11, loc="left", pad=6)
    axes[-1].set_xlabel("time step (after washout)")
    fig.tight_layout()
    _save(fig, "fig15_input_series", sources,
          "Exemplary time series of each benchmark system: the signals every "
          "model receives, before any metric is computed.")


# ---------------------------------------------------------------------------
# fig16 — exemplary forecasts from the various models (main text)
# ---------------------------------------------------------------------------
def fig_model_forecasts():
    """True vs one-step forecast for one model of each family, per system.

    fig6 shows only the best model per system; this shows the comparison the
    thesis actually argues about — the explicit-polynomial classical reference,
    a recurrent classical reservoir at the matched feature count, and the
    quantum reservoir.

    Provenance guard: results/forecasts/ contains runs from two evaluation
    passes with different test-segment lengths (299 and 599 rows). Only models
    scored on an IDENTICAL test segment may be drawn on shared axes, so this
    function asserts that the chosen models' y_true vectors are equal before
    plotting, and raises rather than producing a misleading figure.
    """
    STEMS = {
        "henon": [("NG-RC", "ngrc", CB[7]),
                  ("ESN (F=96)", "esn_matched_F96", CB[0]),
                  ("QRC-rich", "qrc_rich", CB[3])],
        "lorenz": [("NG-RC", "NG-RCk8d3", CB[7]),
                   ("ESN (F=96)", "ESNF96", CB[0]),
                   ("QRC-rich", "qrc_rich", CB[3])],
        "mackeyglass": [("NG-RC", "NG-RCk8d3", CB[7]),
                        ("ESN (F=96)", "ESNF96", CB[0]),
                        ("QRC-rich", "qrc_rich", CB[3])],
    }
    sources = [f"forecasts/{s}/*_seed0.csv" for s in SYSTEMS]
    WIN = 120
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4),
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    for cj, system in enumerate(SYSTEMS):
        top, bot = axes[0][cj], axes[1][cj]
        series = []
        for label, stem, color in STEMS[system]:
            d = pd.read_csv(_csv("forecasts", system, f"{stem}_seed0.csv"))
            series.append((label, color, d))
        # identical-test-segment guard
        ref = series[0][2]["y_true"].to_numpy()
        for label, _, d in series[1:]:
            other = d["y_true"].to_numpy()
            if other.shape != ref.shape or not np.allclose(other, ref, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"fig16: {system} model '{label}' was scored on a different test "
                    f"segment (n={other.shape[0]} vs {ref.shape[0]}); refusing to plot "
                    "models from different evaluation passes on shared axes")
        t = series[0][2]["t"].to_numpy()[:WIN]
        top.plot(t, ref[:WIN], color="#BFC6D4", lw=3.4, label="true", zorder=1, solid_capstyle="round")
        for label, color, d in series:
            top.plot(t, d["y_pred"].to_numpy()[:WIN], color=color, lw=1.1, ls="--",
                     label=label, zorder=3)
            err = np.abs(d["y_true"].to_numpy() - d["y_pred"].to_numpy())[:WIN]
            bot.plot(t, np.maximum(err, 1e-12), color=color, lw=1.1, label=label)
        top.set_title(f"{TITLES[system]}   (test segment: {ref.shape[0]} steps, seed 0)",
                      fontsize=11)
        bot.set_yscale("log")
        bot.set_xlabel("test-set time step")
        if cj == 0:
            top.set_ylabel("scaled value")
            bot.set_ylabel("absolute one-step error (log)")
            top.legend(fontsize=8.5, ncol=2, loc="lower left", framealpha=0.9)
    fig.suptitle("Exemplary one-step forecasts (top) and their errors (bottom). "
                 "Every model tracks the signal; the errors differ by orders of "
                 "magnitude.", y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig16_model_forecasts", sources,
          "True vs one-step forecast (top) and absolute error on a log axis "
          "(bottom) for NG-RC, a feature-matched ESN and QRC-rich per system, "
          "all scored on an identical test segment.")


def main():
    os.makedirs(FIG, exist_ok=True)
    print("generating figures (committed CSVs only):")
    fig_degree_sweep()        # priority 1
    fig_dm_heatmap()          # priority 2
    fig_vpt()                 # priority 3
    fig_seed_distribution()
    fig_matched_budget()
    fig_forecast_overlay()
    fig_input_series()      # main-text: the data itself
    fig_model_forecasts()   # main-text: forecasts from the various models
    fig_concentration()       # observable-concentration mechanism figure
    fig_finite_shots()        # finite-shot degradation
    fig_frequency_redundancy()  # frequency-domain redundancy refinement (Part D)
    fig_attractor()           # phase-space attractor reconstruction (climate, visual)
    fig_scaling_proof()       # qubit-scaling proof
    fig_entanglement()        # dynamics-level ablation + entropy
    fig_ic_robustness()       # 20 initial conditions per system
    fig_leaky()               # RF-QRC leaky variant
    write_index()
    print("done — see results/FIGURES.md")



# ---------------------------------------------------------------------------
# fig11 — qubit-scaling proof: NRMSE vs n, all contenders
# ---------------------------------------------------------------------------
def fig_scaling_proof():
    sc = pd.read_csv(_csv("scaling_proof", "scores.csv"), keep_default_na=False)
    sc["nrmse"] = sc["nrmse"].astype(float)
    sc = sc.drop_duplicates(subset=["system", "n_qubits_eq", "model", "seed"], keep="last")
    MODELS = [("qrc", "QRC-rich (F=16n)", CB[3], "o"),
              ("elmF", "ELM(F) — random-feature control", CB[2], "s"),
              ("esnF", "ESN(F) — matched weak baseline", CB[0], "^"),
              ("ngrc", "NG-RC yardstick", CB[7], "D")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, system in zip(axes, SYSTEMS):
        d = sc[sc.system == system]
        ns = sorted(d.n_qubits_eq.unique())
        for key, label, color, marker in MODELS:
            med = [d[(d.model == key) & (d.n_qubits_eq == n)].nrmse.median() for n in ns]
            q1 = [d[(d.model == key) & (d.n_qubits_eq == n)].nrmse.quantile(.25) for n in ns]
            q3 = [d[(d.model == key) & (d.n_qubits_eq == n)].nrmse.quantile(.75) for n in ns]
            ax.plot(ns, med, marker=marker, color=color, label=label, lw=1.6, ms=5)
            ax.fill_between(ns, q1, q3, color=color, alpha=0.18, lw=0)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xlabel("qubit count n  (feature budget F = 16n)")
        ax.set_title(TITLES[system])
    axes[0].set_ylabel("one-step NRMSE (median, IQR band)")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("The qubit-scaling proof: QRC is flat in n and never beats the "
                 "matched random-feature control", y=1.04, fontsize=12)
    _save(fig, "fig11_scaling_proof", ["scaling_proof/scores.csv"],
          "NRMSE vs qubit count for QRC vs matched ESN/ELM and NG-RC; QRC flat, "
          "ELM(F) wins at every n (105/105 DM significant).")


# ---------------------------------------------------------------------------
# fig12 — entanglement ablation: NRMSE vs J with entropy overlay
# ---------------------------------------------------------------------------
def fig_entanglement():
    sc = pd.read_csv(_csv("entanglement", "scores.csv"), keep_default_na=False)
    sc["nrmse"] = sc["nrmse"].astype(float); sc["J"] = sc["J"].astype(float)
    sc = sc.drop_duplicates(subset=["system", "encoding", "n", "J", "seed"], keep="last")
    en = pd.read_csv(_csv("entanglement", "entropy.csv"), keep_default_na=False)
    en["J"] = en["J"].astype(float)
    systems = ["henon", "mackeyglass"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, system in zip(axes, systems):
        ax2 = ax.twinx()
        for enc, color in [("depth", CB[3]), ("width", CB[0])]:
            d = sc[(sc.system == system) & (sc.encoding == enc) & (sc.n == 6)]
            js = sorted(d.J.unique())
            med = [d[d.J == j].nrmse.median() for j in js]
            ax.plot(js, med, marker="o", color=color, lw=1.6, ms=5,
                    label=f"{enc} encoding — NRMSE")
            e = en[(en.system == system) & (en.encoding == enc) & (en.n == 6)]
            e = e.sort_values("J")
            ax2.plot(e.J, e.S_mean, ls="--", marker="x", color=color, alpha=0.55,
                     label=f"{enc} — S(ρ_A)")
        ax.set_yscale("log"); ax.set_xlabel("coupling strength J  (J=0 ⇒ separable)")
        ax.set_title(TITLES.get(system, system))
        ax2.set_ylabel("half-chain entropy S(ρ_A) [bits]", fontsize=9)
        ax2.set_ylim(0, 3.05); ax2.grid(False)
    axes[0].set_ylabel("one-step NRMSE (median, n=6)")
    # dedup legend across the twin axes; place below the panels, clear of data
    seen, H, L = set(), [], []
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle("Dynamics-level entanglement ablation: entanglement (dashed) grows with J; "
                 "accuracy gains saturate at ≤3.0×, encoding choice dominates", y=1.04)
    _save(fig, "fig12_entanglement_ablation",
          ["entanglement/scores.csv", "entanglement/entropy.csv"],
          "NRMSE vs J at n=6 (solid, log) with measured half-chain entropy (dashed); "
          "J=0 is the provably separable circuit.")


# ---------------------------------------------------------------------------
# fig13 — IC robustness: per-IC NRMSE distributions + divergence
# ---------------------------------------------------------------------------
def fig_ic_robustness():
    one = pd.read_csv(_csv("ic_robustness", "onestep.csv"), keep_default_na=False)
    one["nrmse"] = one["nrmse"].astype(float)
    one = one.drop_duplicates(subset=["system", "ic_index", "model", "seed"], keep="last")
    clo = pd.read_csv(_csv("ic_robustness", "closedloop.csv"), keep_default_na=False)
    clo["diverged"] = clo["diverged"].astype(str) == "True"
    MODELS = [("ngrc", "NG-RC", CB[7]), ("elm300", "ELM", CB[2]),
              ("esn300", "ESN", CB[0]), ("qrc_rich", "QRC-rich", CB[3])]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for ax, system in zip(axes[:3], SYSTEMS):
        d = one[one.system == system]
        data, labels, colors = [], [], []
        for key, label, color in MODELS:
            per_ic = d[d.model == key].groupby("ic_index").nrmse.median()
            data.append(np.log10(per_ic.values)); labels.append(label); colors.append(color)
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(TITLES[system])
        if ax is axes[0]:
            ax.set_ylabel("log10 NRMSE across 20 ICs")
    axd = axes[3]
    width = 0.25
    for i, system in enumerate(SYSTEMS):
        d = clo[clo.system == system]
        fr = [d[d.model == k].diverged.mean() for k, _, _ in MODELS]
        axd.bar(np.arange(len(MODELS)) + (i - 1) * width, fr, width,
                label=TITLES[system], color=CB[i], alpha=0.85)
    axd.set_xticks(range(len(MODELS)))
    axd.set_xticklabels([l for _, l, _ in MODELS], fontsize=8)
    axd.set_ylabel("closed-loop divergence fraction"); axd.set_ylim(0, 1.05)
    axd.legend(fontsize=7.5); axd.set_title("stability across ICs")
    fig.suptitle("Initial-condition robustness: QRC best on 0/60 ICs (violins), "
                 "but bounded observables divergence-stable (bars)", y=1.05)
    _save(fig, "fig13_ic_robustness",
          ["ic_robustness/onestep.csv", "ic_robustness/closedloop.csv"],
          "Per-IC NRMSE distributions (20 ICs/system) and closed-loop divergence "
          "fractions; ranking is IC-invariant.")


# ---------------------------------------------------------------------------
# fig14 — leaky-memory sweep: exact + finite-shot arms
# ---------------------------------------------------------------------------
def fig_leaky():
    sc = pd.read_csv(_csv("leaky", "scores.csv"), keep_default_na=False)
    sc["nrmse"] = sc["nrmse"].astype(float); sc["eps"] = sc["eps"].astype(float)
    sc = sc.drop_duplicates(subset=["system", "model", "seed", "eps", "shots"], keep="last")
    ex = sc[sc.shots == "None"]
    MODELS = [("qrc_rich", "QRC-rich(n=6)", CB[3]), ("elm_matched", "ELM(F=96)", CB[2]),
              ("esn_matched", "ESN(F=96)", CB[0]), ("ngrc", "NG-RC", CB[7])]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, system in zip(axes[:3], SYSTEMS):
        d = ex[ex.system == system]
        for key, label, color in MODELS:
            g = d[d.model == key].groupby("eps").nrmse.median().sort_index()
            ax.plot(g.index, g.values, marker="o", color=color, lw=1.6, ms=4, label=label)
        ax.set_yscale("log"); ax.set_xlabel("leak ε   (ε=1 ⇒ no leak)")
        ax.set_title(TITLES[system])
    axes[0].set_ylabel("one-step NRMSE (median)")
    axes[0].legend(fontsize=7.5)
    axs = axes[3]
    sh = sc[(sc.system == "henon") & (sc.model == "qrc_rich") & (sc.shots != "None")]
    for shots, color in [("8192", CB[5]), ("1024", CB[4])]:
        g = sh[sh.shots == shots].groupby("eps").nrmse.median().sort_index()
        axs.plot(g.index, g.values, marker="s", color=color, lw=1.6, ms=4,
                 label=f"{shots} shots")
    g = ex[(ex.system == "henon") & (ex.model == "qrc_rich")].groupby("eps").nrmse.median()
    axs.plot(g.sort_index().index, g.sort_index().values, marker="o", color=CB[3],
             lw=1.6, ms=4, label="exact")
    axs.set_yscale("log"); axs.set_xlabel("leak ε"); axs.set_title("Hénon QRC under shots")
    axs.legend(fontsize=7.5)
    fig.suptitle("RF-QRC-style leaky integration harms every model monotonically "
                 "and does not rescue the finite-shot collapse", y=1.05)
    _save(fig, "fig14_leaky", ["leaky/scores.csv"],
          "NRMSE vs leak ε for quantum and classical models (exact) and the "
          "finite-shot arm; ε=1 (no leak) is optimal everywhere.")

if __name__ == "__main__":
    main()
