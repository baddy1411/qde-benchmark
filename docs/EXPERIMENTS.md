# Experiment index

Every script in `experiments/`, what it runs, and what it writes. Run all of
them from the repository root:

```bash
python experiments/run_scaling_proof.py
```

Every sweep is resumable. If one is interrupted, re-run the same command and it
skips the cells that already have results.

For an interactive version of this table that also prints the result tables:

```bash
python scripts/experiments.py
python scripts/qde_ui.py --tui
```

## The core comparison

| Script | What it runs | Writes |
|---|---|---|
| `synthesize.py` | the baseline leaderboard: 24 models × 3 systems × 5 seeds on one common split | `results/model_scoreboard_all_runs.csv` |
| `experiments_advanced.py` | matched feature budget, ZZ read-out ablation, quadratic control, climate battery | `results/adv_*.csv` |
| `run_scaling_proof.py` | the headline family: quantum vs matched ELM and NG-RC at 4–10 qubits, three systems | `results/scaling_proof/` |
| `run_n12_attempt.py` | the exploratory 12-qubit arm on Hénon | `results/scaling_proof/` |
| `significance_run.py` | Diebold–Mariano and Model Confidence Set on the headline comparisons | `results/significance/` |
| `henon_joint_mcs.py` | joint Model Confidence Set on Hénon | `results/significance/` |
| `cross_system.py` | the classical battery across all three systems | `results/cross/`, `results/significance/` |
| `quantum_cross.py` | the quantum arms on Lorenz-63 and Mackey–Glass | `results/cross/` |

## Mechanism: what is actually doing the work

| Script | What it runs | Writes |
|---|---|---|
| `run_entanglement.py` | encoding sweep and the separable-circuit ablation, with entanglement entropy measured at every point | `results/entanglement/` |
| `concentration_run.py` | finite-shot degradation and the concentration analysis | `results/concentration/` |
| `frequency_run.py` | frequency-domain redundancy of the ZZ observable | `results/concentration/frequency_redundancy.csv` |
| `run_variance_5seed.py`, `run_benefit_5seed.py` | five-seed mechanism checks | `results/concentration/scaling_*_5seed.csv` |
| `run_leaky.py` | classical leaky integration applied to every model's features, plus a finite-shot arm | `results/leaky/` |

## The five rescue attempts

| Script | What it runs | Writes |
|---|---|---|
| `run_followup_tricks.py` | classical read-out augmentation and the virtual-node sweep | `results/followup/` |
| `run_cheb_encoding.py` | arcsin and Chebyshev-tower encodings, against a classical Chebyshev delay map | `results/cheb/` |
| `run_cheb_rank.py` | rank and conditioning diagnostic for those encodings | `results/cheb/rank.csv` |
| `run_rfqrc.py` | the recurrence-free architecture from the literature, with its classical twin | `results/rfqrc/` |
| `run_rfqrc_mv_control.py` | the multivariate control for that architecture | `results/rfqrc/` |
| `run_rfqrc_cheb.py` | recurrence-free plus Chebyshev encoding | `results/rfqrc_cheb/` |
| `run_dissipative_qrc.py` | the dissipative quantum-memory reservoir | `results/dissipative/` |
| `run_mc_tuned_esn.py` | the corrected memory-capacity control, equal tuning budget | `results/dissipative/mc_tuned_*.csv` |

## Robustness and scope

| Script | What it runs | Writes |
|---|---|---|
| `run_ic_study.py` | 20 initial conditions per system, one-step and autonomous | `results/ic_robustness/` |
| `run_climate_full.py`, `regen_mg_climate.py`, `closedloop_traj.py` | the autonomous rollout climate battery | `results/cross/*_climate*.csv`, `results/trajectories/` |
| `run_lorenz96.py` | the 20-dimensional system | `results/lorenz96/` |
| `lorenz_mv_pilot.py`, `lorenz_mv_closedloop.py` | the multivariate Lorenz-63 pilot | `results/cross/lorenz_mv_*.csv` |
| `run_qngrc_comparison.py` | a quantum NG-RC variant against classical NG-RC | `results/cross/qngrc_comparison.csv` |

## Context baselines

| Script | What it runs | Writes |
|---|---|---|
| `run_catalogue_ext.py` | the extended untuned catalogue: statistical, tree, and deep models | `results/catalogue_ext.csv` |
| `run_pretrained_zeroshot.py` | Chronos-Bolt zero-shot. **Downloads a checkpoint from Hugging Face** | `results/pretrained/` |
| `run_headline_20seed.py` | the 20-seed refresh of the headline table | `results/headline_20seed.csv` |
| `run_esn_budget_tune.py` | the ESN tuning grid, matched in configuration count to the quantum one | `results/esn_budget_tune.csv` |

## Data engineering (thesis chapter 8)

| Script | What it measures | Writes |
|---|---|---|
| `run_de_volume.py` | throughput and memory as the series grows to one million rows | `results_de/volume.csv` |
| `run_de_storage.py` | columnar versus row-oriented storage. The fastest script here, about a minute | `results_de/storage.csv` |
| `run_de_parallel.py` | speed-up and efficiency across worker processes | `results_de/parallel.csv` |
| `run_de_incremental.py` | how much re-running skips | `results_de/incremental.csv` |
| `run_de_reproducibility.py` | bit-identity of a full regeneration | `results_de/reproducibility.csv` |
| `run_de_failure.py` | eight injected fault classes, all expected to be caught | `results_de/failure.csv` |

## Support

| Script | What it does |
|---|---|
| `estimate_lyapunov.py` | estimates the Lyapunov exponents used to normalise valid prediction time. Provenance for the constants in `qdepipe/data/` |
| `../smoke_test.py` | the end-to-end sanity gate, at the repository root |

## Inspection tools (`scripts/`, these read results rather than producing them)

| Script | What it does |
|---|---|
| `demo.py` | guided walkthrough of the five things the pipeline guarantees |
| `qde_ui.py` | interactive terminal browser over experiments and results |
| `experiments.py` | list experiments, print their result tables |
| `trace.py` | trace a reported number back to its artifact and run |
| `extended_family.py` | pool every quantum-vs-classical test into one extended family |
| `noise_arm_cross_system.py` | the cross-system finite-shot arm |
| `verify_rebuild.py` | compare a rebuilt results tree against the committed one, byte for byte |
| `container_verify.py` | regenerate a scope inside the pinned container and diff |
| `build_results_workbook.py` | collect every run and comparison into one spreadsheet |
| `analysis/*.py` | derive the claim tables in `derived/` from the raw artifacts |
