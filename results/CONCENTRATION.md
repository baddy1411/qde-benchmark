# QDE — Exponential-Concentration Analysis (auto-generated)

*`concentration_run.py` (`make shots`). The mechanism behind the ZZ-ablation null (Hénon: Z→Z+X+Y = 59×, +ZZ = 1.03×). Exact-expectation results are an **idealized upper bound**; finite-shot runs show realism makes it worse, not better.*

## Does 2-local (ZZ) variance collapse faster than 1-local?

*The variance-ratio trend is read over the comparable-`n_points` rows (n≤8 at n_points≥1200). The n=10 row used reduced n_points (600) for runtime; the spot-check below shows small samples **inflate** the 2-local variance estimate, so n=10's variance is not comparable and is excluded from the trend (its NRMSE row remains valid).*

### henon

|   n_qubits |   var_1local |   var_2local |   var_ratio_2over1 |   NRMSE_ZXY |   NRMSE_ZXYZZ |   zz_benefit_ratio |
|-----------:|-------------:|-------------:|-------------------:|------------:|--------------:|-------------------:|
|          4 |     0.00785  |     0.007593 |             0.9674 |    0.004568 |      0.004418 |              1.034 |
|          6 |     0.008359 |     0.006581 |             0.7873 |    0.004545 |      0.004478 |              1.015 |
|          8 |     0.007915 |     0.006013 |             0.7596 |    0.004816 |      0.004683 |              1.029 |
|         10 |     0.005275 |     0.01085  |             2.056  |    0.004611 |      0.004475 |              1.03  |

**FINDING [henon, n≤8]:** 2-local/1-local variance ratio goes 0.967 (n=4) → 0.76 (n=8) — ZZ variance **declines relative to** single-qubit (mild concentration). But the ZZ NRMSE-benefit ratio goes 1.03 → 1.03 (stays ≈1 — ZZ near-useless at every n). So the forecasting benefit of ZZ does **not** track its (mild) variance concentration here.

### mackeyglass

|   n_qubits |   var_1local |   var_2local |   var_ratio_2over1 |   NRMSE_ZXY |   NRMSE_ZXYZZ |   zz_benefit_ratio |
|-----------:|-------------:|-------------:|-------------------:|------------:|--------------:|-------------------:|
|          4 |      0.0119  |      0.02691 |              2.261 |    0.004995 |      0.004828 |              1.035 |
|          6 |      0.01533 |      0.02405 |              1.569 |    0.005367 |      0.00457  |              1.174 |
|          8 |      0.01442 |      0.0209  |              1.449 |    0.005017 |      0.004129 |              1.215 |

**FINDING [mackeyglass, n≤8]:** 2-local/1-local variance ratio goes 2.26 (n=4) → 1.45 (n=8) — ZZ variance **declines relative to** single-qubit (mild concentration). But the ZZ NRMSE-benefit ratio goes 1.03 → 1.22 (**grows** — ZZ helps *more* at larger n, the **opposite** of what concentration predicts). So the forecasting benefit of ZZ does **not** track its (mild) variance concentration here.

## n=8 spot-check — reduced n_points inflates the variance estimate

|   n_qubits | n_points   |   mean_var_1local |   mean_var_2local |
|-----------:|:-----------|------------------:|------------------:|
|          8 | npts=1200  |          0.007047 |          0.007569 |
|          8 | npts=400   |          0.007444 |          0.01021  |

**Confound confirmed:** at n=8, cutting n_points 1200→400 changes the 2-local variance estimate 0.00757→0.0102 (1.35×). This is why the n=10 (n_points=600) variance is excluded from the trend.

## Verdict: does concentration explain the ZZ-ablation null?

**No — not cleanly.** Mild concentration is present in the *variance* (2-local declines relative to 1-local for n≤8 on both systems), but the *forecasting benefit* of ZZ does **not** track it: on Hénon ZZ is near-useless at every n (~1.03×, reproducing the ablation) with no n-trend, and on Mackey-Glass the ZZ benefit actually **grows** with n (up to ~1.2×) — the opposite of the concentration prediction. Since ZZ genuinely helps (and increasingly) on Mackey-Glass, the Hénon ablation null is better explained by **redundancy** — on the quadratic Hénon map the ZZ correlators are largely reachable from the X,Y single-qubit measurements plus the ridge readout, so they add nothing *there specifically*. The honest mechanism is redundancy-on-Hénon, not universal exponential concentration. (The variance does concentrate mildly; it just isn't what drives the forecasting null.)

## Finite-shot: does ZZ degrade faster than single-qubit?

| system   |   n_qubits | readout_set   | shots   |    NRMSE |   NRMSE_std |
|:---------|-----------:|:--------------|:--------|---------:|------------:|
| henon    |          4 | Z+X+Y         | exact   | 0.004542 |   5.344e-05 |
| henon    |          4 | Z+X+Y+ZZ      | exact   | 0.004417 |   1.811e-05 |
| henon    |          4 | Z+X+Y         | 8192    | 0.23     |   0.004104  |
| henon    |          4 | Z+X+Y+ZZ      | 8192    | 0.1751   |   0.003826  |
| henon    |          4 | Z+X+Y         | 1024    | 0.5457   |   0.0145    |
| henon    |          4 | Z+X+Y+ZZ      | 1024    | 0.4259   |   0.01025   |
| henon    |          6 | Z+X+Y         | exact   | 0.004569 |   6.325e-05 |
| henon    |          6 | Z+X+Y+ZZ      | exact   | 0.004461 |   7.649e-05 |
| henon    |          6 | Z+X+Y         | 8192    | 0.2281   |   0.008364  |
| henon    |          6 | Z+X+Y+ZZ      | 8192    | 0.1748   |   0.002356  |
| henon    |          6 | Z+X+Y         | 1024    | 0.5279   |   0.02427   |
| henon    |          6 | Z+X+Y+ZZ      | 1024    | 0.4275   |   0.003707  |
| henon    |          8 | Z+X+Y         | exact   | 0.004724 |   0.0001322 |
| henon    |          8 | Z+X+Y+ZZ      | exact   | 0.0046   |   0.0001516 |
| henon    |          8 | Z+X+Y         | 8192    | 0.2475   |   0.01844   |
| henon    |          8 | Z+X+Y+ZZ      | 8192    | 0.1884   |   0.008727  |
| henon    |          8 | Z+X+Y         | 1024    | 0.55     |   0.04112   |
| henon    |          8 | Z+X+Y+ZZ      | 1024    | 0.4372   |   0.01011   |

**FINDING [finite-shot, n=8]:** both readouts collapse by ~100× from exact to 1024 shots — Z+X+Y 0.00472→0.55, Z+X+Y+ZZ 0.0046→0.437. Notably the ZZ-augmented readout degrades *slightly less* (more features → more ridge averaging), i.e. ZZ does **not** degrade faster under shots — refuting that specific sub-prediction. Either way, exact expectation was an unreachable ceiling.

## Headline matched-budget under shots: QRC-rich vs ESN(F), Hénon

| system   | model    |   F | shots   |   NRMSE_quantum |   NRMSE_ESN_F | beats_ESN   |   margin_ratio_esn_over_qrc |
|:---------|:---------|----:|:--------|----------------:|--------------:|:------------|----------------------------:|
| henon    | QRC-rich |  96 | exact   |        0.001111 |       0.02003 | True        |                    18.03    |
| henon    | QRC-rich |  96 | 8192    |        0.4994   |       0.02003 | False       |                     0.04012 |
| henon    | QRC-rich |  96 | 1024    |        0.8005   |       0.02003 | False       |                     0.02503 |

**FINDING [headline under shots]:** QRC-rich NRMSE 0.00111 (exact) → 0.801 (1024 shots) vs ESN(F)=0.02. At 1024 shots QRC-rich **no longer beats** its matched ESN — the idealized-upper-bound caveat quantified: realism widens, never closes, the gap to classical.


---
*Regenerated by `concentration_run.py`. Figures: `make plots` (fig7 concentration, fig8 finite-shot). Source CSVs in `results/concentration/`.*
