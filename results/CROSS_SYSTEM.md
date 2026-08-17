# QDE — Cross-System Validation (auto-generated)

*`cross_system.py`, classical-first, seeds=[0, 1, 2]. Four robustness pins: NG-RC tuned (lookback×degree), multi-seed, Diebold–Mariano significance, and Lorenz Δ-sensitivity. Quantum cross-system runs deferred.*

## Headline: does NG-RC's dominance generalise?

ESN vs the **tuned** NG-RC on each system, with a DM significance verdict:

| system      |   esn_nrmse |   ngrc_best_nrmse | winner   |   dm_median_p |   frac_seeds_sig | verdict     |
|:------------|------------:|------------------:|:---------|--------------:|-----------------:|:------------|
| henon       |    0.006894 |         3.015e-07 | NG-RC    |     9.937e-67 |                1 | significant |
| lorenz      |    0.01798  |         0.001266  | NG-RC    |     2.275e-30 |                1 | significant |
| mackeyglass |    0.006111 |         0.001167  | NG-RC    |     2.444e-43 |                1 | significant |

**READ:** if the winner flips from NG-RC (Hénon) to ESN on Lorenz/Mackey–Glass and the DM verdict is significant, NG-RC's dominance is a Hénon (quadratic-system) artifact — the central thesis claim, now cross-validated.

## PIN 4 — Lorenz Δ-sensitivity (not a one-sampling-rate artifact)

|   delta |   esn_nrmse |   ngrc_best_nrmse | winner   |
|--------:|------------:|------------------:|:---------|
|    0.02 |    0.001428 |         0.0001696 | NG-RC    |
|    0.05 |    0.01798  |         0.001266  | NG-RC    |
|    0.1  |    0.08897  |         0.03015   | NG-RC    |

## System: henon

### Classical leaderboard (1-step NRMSE)

| model        |   nrmse_mean |   nrmse_std |
|:-------------|-------------:|------------:|
| NG-RC(best)  |    3.015e-07 |   0         |
| NG-RC(d2)    |    3.015e-07 |   0         |
| NG-RC(d3)    |    2.783e-06 |   0         |
| ELM+poly2    |    7.382e-05 |   2.597e-07 |
| ELM          |    0.0001742 |   1.306e-05 |
| ESN+poly2    |    0.005255  |   0.0002769 |
| ESN          |    0.006894  |   0.0001556 |
| RandomForest |    0.01461   |   9.373e-05 |
| Linear-Ridge |    0.8568    |   1.11e-16  |
| NG-RC(d1)    |    0.8813    |   0         |

NG-RC tuned best: see `results/cross/henon_ngrc_tune.csv`. Degree probe (NG-RC d1/d2/d3, ESN±poly2, ELM±poly2) is in the leaderboard above — feature-structure story: accuracy tracks accessible polynomial degree across architectures.

### Model Confidence Set (α=0.10)

| model        |   mean_mcs_pvalue |   frac_seeds_in_set_0.10 |
|:-------------|------------------:|-------------------------:|
| NG-RC(best)  |                 1 |                        1 |
| NG-RC(d2)    |                 1 |                        1 |
| NG-RC(d1)    |                 0 |                        0 |
| NG-RC(d3)    |                 0 |                        0 |
| ESN          |                 0 |                        0 |
| ESN+poly2    |                 0 |                        0 |
| ELM          |                 0 |                        0 |
| ELM+poly2    |                 0 |                        0 |
| RandomForest |                 0 |                        0 |
| Linear-Ridge |                 0 |                        0 |

### Closed-loop climate (VPT, Lyapunov times)

| model        |   vpt_lyap_mean |
|:-------------|----------------:|
| NG-RC(best)  |          12.76  |
| RandomForest |           7.132 |
| ELM          |           6.995 |
| ESN          |           4.938 |

## System: lorenz

### Classical leaderboard (1-step NRMSE)

| model        |   nrmse_mean |   nrmse_std |
|:-------------|-------------:|------------:|
| ELM+poly2    |    0.0002468 |   2.511e-05 |
| ELM          |    0.0005064 |   4.858e-05 |
| NG-RC(best)  |    0.001266  |   0         |
| NG-RC(d3)    |    0.001266  |   0         |
| ESN+poly2    |    0.008593  |   0.0007168 |
| ESN          |    0.01798   |   0.001355  |
| NG-RC(d1)    |    0.03932   |   0         |
| NG-RC(d2)    |    0.04042   |   0         |
| RandomForest |    0.04175   |   0.0002893 |
| Linear-Ridge |    0.04733   |   0         |

NG-RC tuned best: see `results/cross/lorenz_ngrc_tune.csv`. Degree probe (NG-RC d1/d2/d3, ESN±poly2, ELM±poly2) is in the leaderboard above — feature-structure story: accuracy tracks accessible polynomial degree across architectures.

### Model Confidence Set (α=0.10)

| model        |   mean_mcs_pvalue |   frac_seeds_in_set_0.10 |
|:-------------|------------------:|-------------------------:|
| ELM+poly2    |          1        |                        1 |
| NG-RC(best)  |          0.004    |                        0 |
| NG-RC(d3)    |          0.004    |                        0 |
| ELM          |          0.004    |                        0 |
| ESN+poly2    |          0.003333 |                        0 |
| NG-RC(d1)    |          0        |                        0 |
| NG-RC(d2)    |          0        |                        0 |
| ESN          |          0        |                        0 |
| RandomForest |          0        |                        0 |
| Linear-Ridge |          0        |                        0 |

### Closed-loop climate (VPT, Lyapunov times)

| model        |   vpt_lyap_mean |
|:-------------|----------------:|
| RandomForest |          1.313  |
| ESN          |          0.7698 |
| ELM          |          0.7547 |
| NG-RC(best)  |          0.3622 |

## System: mackeyglass

### Classical leaderboard (1-step NRMSE)

| model        |   nrmse_mean |   nrmse_std |
|:-------------|-------------:|------------:|
| NG-RC(best)  |     0.001167 |   0         |
| NG-RC(d3)    |     0.001167 |   0         |
| NG-RC(d2)    |     0.001546 |   2.168e-19 |
| ELM+poly2    |     0.001546 |   2.109e-05 |
| ELM          |     0.001721 |   1.983e-05 |
| NG-RC(d1)    |     0.002213 |   0         |
| Linear-Ridge |     0.002998 |   0         |
| ESN+poly2    |     0.004397 |   0.00043   |
| ESN          |     0.006111 |   0.0004657 |
| RandomForest |     0.03349  |   0.0001153 |

NG-RC tuned best: see `results/cross/mackeyglass_ngrc_tune.csv`. Degree probe (NG-RC d1/d2/d3, ESN±poly2, ELM±poly2) is in the leaderboard above — feature-structure story: accuracy tracks accessible polynomial degree across architectures.

### Model Confidence Set (α=0.10)

| model        |   mean_mcs_pvalue |   frac_seeds_in_set_0.10 |
|:-------------|------------------:|-------------------------:|
| NG-RC(best)  |          1        |                        1 |
| NG-RC(d3)    |          1        |                        1 |
| ELM+poly2    |          0.007333 |                        0 |
| NG-RC(d1)    |          0        |                        0 |
| NG-RC(d2)    |          0        |                        0 |
| ESN          |          0        |                        0 |
| ESN+poly2    |          0        |                        0 |
| ELM          |          0        |                        0 |
| RandomForest |          0        |                        0 |
| Linear-Ridge |          0        |                        0 |

### Closed-loop climate (VPT, Lyapunov times)

| model        |   vpt_lyap_mean |
|:-------------|----------------:|
| ESN          |          1.092  |
| ELM          |          0.3153 |
| NG-RC(best)  |          0.2064 |
| RandomForest |          0.0946 |


---
*Regenerated by `cross_system.py` (`python experiments/cross_system.py`). DM/MCS matrices in `results/significance/{system}_*`; per-seed forecasts in `results/forecasts/`.*
