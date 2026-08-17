# QDE — Quantum Cross-System Validation (auto-generated)

*`quantum_cross.py`, 5 seeds, exact-expectation statevector (idealized upper bound). Systems completed: lorenz, mackeyglass. All models aligned on one n_points=1500 pipeline so the joint DM/MCS is valid.*

**Question:** does the Hénon matched-budget pattern (only QRC-rich beats ESN(F); no quantum in the MCS best-set; NG-RC dominates) hold on Lorenz / Mackey-Glass?

## System: Lorenz-63

*Quantum results are exact-expectation (idealized upper bound on quantum performance; finite-shot results are reported separately).*

### Encoding fairness (per-system best)

| model    |   default_nrmse |   best_nrmse | best_scaler   |   best_encode_scale |
|:---------|----------------:|-------------:|:--------------|--------------------:|
| qrc_v4   |         0.09922 |     0.08143  | minmax        |                 2   |
| qrc_v6   |         0.1508  |     0.08649  | minmax        |                 2   |
| qrc_rich |         0.01614 |     0.008679 | minmax        |                 1.5 |

### Matched feature-budget (QRC at tuned best vs ESN(F), NG-RC reference)

| model    |   F |   NRMSE_quantum_best |   NRMSE_ESN_F |   NRMSE_NGRC_ref |   encode_scale | scaler   | beats_ESN   |
|:---------|----:|---------------------:|--------------:|-----------------:|---------------:|:---------|:------------|
| qrc_v4   |  16 |             0.08143  |       0.2378  |         0.001499 |            2   | minmax   | True        |
| qrc_v6   |  24 |             0.08649  |       0.1493  |         0.001499 |            2   | minmax   | True        |
| qrc_rich |  96 |             0.008679 |       0.09822 |         0.001499 |            1.5 | minmax   | True        |

### Diebold–Mariano on matched pairs (5 seeds; fraction significant)

| pair                    |   frac_sig_0.10 |   frac_sig_0.05 |   median_dm |   dm_min |   dm_max |   median_p |
|:------------------------|----------------:|----------------:|------------:|---------:|---------:|-----------:|
| qrc_rich vs ESN(F=96)   |               1 |               1 |     -12.42  |  -14.47  |   -10.76 |  7.974e-29 |
| qrc_rich vs NG-RC(k8d3) |               1 |               1 |       9.299 |    8.504 |    10.6  |  3.143e-18 |
| qrc_v6 vs ESN(F=24)     |               1 |               1 |      -8.752 |   -9.555 |    -3.37 |  1.589e-16 |

### Joint Model Confidence Set (classical + quantum, α=0.10)

| model        |   mean_mcs_pvalue |   frac_seeds_in_set_0.10 | is_quantum   |
|:-------------|------------------:|-------------------------:|:-------------|
| ELM+poly2    |                 1 |                        1 | False        |
| NG-RC(k8d3)  |                 0 |                        0 | False        |
| ESN          |                 0 |                        0 | False        |
| ELM          |                 0 |                        0 | False        |
| RandomForest |                 0 |                        0 | False        |
| qrc_v4       |                 0 |                        0 | True         |

### Closed-loop climate (quantum; appended to classical VPT)

| model    |   vpt_lyap_mean |   spectral_mse_mean |   wasserstein_mean |   diverged_runs |
|:---------|----------------:|--------------------:|-------------------:|----------------:|
| qrc_v4   |          0.1721 |              2.67   |             0.2443 |               0 |
| qrc_v6   |          0.1992 |              3.093  |             0.2865 |               0 |
| qrc_rich |          0.6339 |              0.4771 |             0.108  |               0 |

**VERDICT [lorenz]:** quantum models beating their matched ESN(F): ['qrc_v4', 'qrc_v6', 'qrc_rich']. Quantum in the MCS best-set: **no** (—). Best-set = {ELM+poly2}. Hénon pattern holds: classical owns the best-set.

## System: Mackey–Glass

*Quantum results are exact-expectation (idealized upper bound on quantum performance; finite-shot results are reported separately).*

### Encoding fairness (per-system best)

| model    |   default_nrmse |   best_nrmse | best_scaler   |   best_encode_scale |
|:---------|----------------:|-------------:|:--------------|--------------------:|
| qrc_v4   |        0.04127  |     0.03651  | minmax        |                 0.5 |
| qrc_v6   |        0.1099   |     0.07425  | minmax        |                 3   |
| qrc_rich |        0.004618 |     0.004618 | minmax        |                 1   |

### Matched feature-budget (QRC at tuned best vs ESN(F), NG-RC reference)

| model    |   F |   NRMSE_quantum_best |   NRMSE_ESN_F |   NRMSE_NGRC_ref |   encode_scale | scaler   | beats_ESN   |
|:---------|----:|---------------------:|--------------:|-----------------:|---------------:|:---------|:------------|
| qrc_v4   |  16 |             0.03651  |       0.04078 |         0.001326 |            0.5 | minmax   | True        |
| qrc_v6   |  24 |             0.07425  |       0.02305 |         0.001326 |            3   | minmax   | False       |
| qrc_rich |  96 |             0.004618 |       0.0101  |         0.001326 |            1   | minmax   | True        |

### Diebold–Mariano on matched pairs (5 seeds; fraction significant)

| pair                    |   frac_sig_0.10 |   frac_sig_0.05 |   median_dm |   dm_min |   dm_max |   median_p |
|:------------------------|----------------:|----------------:|------------:|---------:|---------:|-----------:|
| qrc_rich vs ESN(F=96)   |               1 |               1 |     -10.81  |  -11.23  |   -9.856 |  3.491e-23 |
| qrc_rich vs NG-RC(k8d3) |               1 |               1 |      11.51  |    9.942 |   12.15  |  1.274e-25 |
| qrc_v6 vs ESN(F=24)     |               1 |               1 |       7.364 |    7.014 |    7.453 |  1.761e-12 |

### Joint Model Confidence Set (classical + quantum, α=0.10)

| model        |   mean_mcs_pvalue |   frac_seeds_in_set_0.10 | is_quantum   |
|:-------------|------------------:|-------------------------:|:-------------|
| NG-RC(k8d3)  |            1      |                        1 | False        |
| ELM+poly2    |            0.2286 |                        1 | False        |
| ELM          |            0.0004 |                        0 | False        |
| ESN          |            0      |                        0 | False        |
| RandomForest |            0      |                        0 | False        |
| qrc_v4       |            0      |                        0 | True         |

### Closed-loop climate (quantum; appended to classical VPT)

| model    |   vpt_lyap_mean |   spectral_mse_mean |   wasserstein_mean |   diverged_runs |
|:---------|----------------:|--------------------:|-------------------:|----------------:|
| qrc_v4   |         0.04988 |             45.63   |            21.6    |               5 |
| qrc_v6   |         0.2064  |              1.24   |             0.375  |               0 |
| qrc_rich |         0.1617  |              0.4111 |             0.1668 |               0 |

**VERDICT [mackeyglass]:** quantum models beating their matched ESN(F): ['qrc_v4', 'qrc_rich']. Quantum in the MCS best-set: **no** (—). Best-set = {NG-RC(k8d3), ELM+poly2}. Hénon pattern holds: classical owns the best-set.
