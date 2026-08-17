# QDE — Advanced Experiments (auto-generated)

*`experiments_advanced.py` — seeds=[0, 1, 2, 3, 4]. Targets the four questions a committee asks past the baseline benchmark.*

## A. Fairness — each QRC at its best encoding

Sweeping the quantum encoding (angle scale x scaler); the default the main synthesis used was minmax / encode_scale=1.0. Full grid: `results/adv_A_encoding.csv`.

| model    |   default_nrmse |   best_nrmse | best_scaler   |   best_encode_scale |   improvement_x |
|:---------|----------------:|-------------:|:--------------|--------------------:|----------------:|
| qrc_v4   |        0.564    |     0.2975   | minmax        |                 6   |           1.895 |
| qrc_v6   |        0.2706   |     0.2706   | minmax        |                 1   |           1     |
| qrc_rich |        0.004431 |     0.001111 | minmax        |                 0.5 |           3.989 |

**FINDING [A]:** tuning the encoding changes the QRC error by up to **4x** — so the matched comparison must be made at each model's best operating point, which is what A2 below does.

## A2. Matched budget at best encoding

Each QRC at its best encoding vs an ESN with `units = F` (same pipeline), and NG-RC as the absolute classical reference. Full table: `results/adv_A2_matched.csv`.

| quantum_model   |   F | best_scaler   |   best_encode_scale |   nrmse_quantum |   nrmse_esn_matched |   nrmse_ngrc_ref | qrc_beats_matched_esn   |
|:----------------|----:|:--------------|--------------------:|----------------:|--------------------:|-----------------:|:------------------------|
| qrc_v4          |  16 | minmax        |                 6   |        0.2975   |             0.07484 |        1.587e-06 | False                   |
| qrc_v6          |  24 | minmax        |                 1   |        0.2706   |             0.05218 |        1.587e-06 | False                   |
| qrc_rich        |  96 | minmax        |                 0.5 |        0.001111 |             0.02003 |        1.587e-06 | True                    |

**FINDING [A2]:** even at its best encoding, the QRC beats the size-matched ESN in 1/3 cases; NG-RC remains the strongest model throughout.

## B. Causal — ZZ-correlator ablation

One fixed 6-qubit circuit; only the readout changes. (Z,X,Y) -> (Z,X,Y,ZZ) isolates ZZ. Full table: `results/adv_B_zz_ablation.csv`.

| readout   | has_ZZ   |   n_features |   nrmse_mean |   nrmse_std |
|:----------|:---------|-------------:|-------------:|------------:|
| Z         | False    |           24 |     0.2706   |   0.02986   |
| Z+X+Y     | False    |           72 |     0.004549 |   5.731e-05 |
| Z+X+Y+ZZ  | True     |           96 |     0.004431 |   6.958e-05 |

**FINDING [B]:** the rich readout's gain comes from the **single-qubit measurement basis, not the ZZ correlators.** Going Z -> Z+X+Y improves NRMSE **59.5x**, while adding the two-qubit ZZ terms on top of that contributes only **1.03x** — essentially nothing. This *refutes* the tempting 'quantum advantage = ZZ quadratic correlators' story: what helps is a richer set of single-qubit observables, which a classical model can match (see C).

## C. Control — quantum, or just quadratic features?

Classical reservoirs given explicit degree-2 features, vs QRC-rich. Full table: `results/adv_C_quadratic_control.csv`.

| model                        |   nrmse_mean |   nrmse_std |
|:-----------------------------|-------------:|------------:|
| NG-RC (explicit quadratic)   |    2.384e-07 |   0         |
| ELM + quadratic (poly2)      |    4.684e-05 |   2.623e-06 |
| ELM (linear readout)         |    0.0001432 |   1.302e-05 |
| QRC-rich (quantum quadratic) |    0.001111  |   4.291e-05 |
| ESN + quadratic (poly2)      |    0.004979  |   0.000271  |
| ESN (linear readout)         |    0.006725  |   0.0001771 |

**FINDING [C]:** the ranking is governed by *whether the feature space contains quadratic terms*, not by quantumness: NG-RC (explicit quadratic of lags) leads, QRC-rich and quadratic-augmented classical reservoirs cluster together, and linear-readout reservoirs trail. The quantum win is a feature-structure effect.

## D. Climate — attractor reproduction (closed loop)

Autonomous rollout; VPT in Lyapunov times (higher better), log-spectral MSE and Wasserstein-1 of the invariant density (lower better). Full table: `results/adv_D_climate.csv`.

| model         |   vpt_lyap_mean |   spectral_mse_mean |   wasserstein_mean |   diverged_runs |
|:--------------|----------------:|--------------------:|-------------------:|----------------:|
| ngrc          |          15.05  |              0.6319 |            0.0119  |               0 |
| elm           |           9.758 |              0.6388 |            0.01004 |               0 |
| random_forest |           8.643 |              0.9271 |            0.01308 |               0 |
| qrc_rich      |           3.317 |              0.6962 |            0.01836 |               0 |
| esn           |           3.206 |              0.6161 |            0.0157  |               0 |
| qrc_v6        |           2.764 |              4.742  |            2.962   |               2 |
| linear        |           0     |              6.831  |            0.2379  |               0 |

**FINDING [D]:** longest valid prediction time = **ngrc** (15.1 Lyapunov times). Closed-loop ranking is the chaos-specific check that 1-step NRMSE cannot provide.


---
*Regenerated by `experiments_advanced.py`; CSVs in `results/` are the source of truth.*
