# Results

Every number below is in a committed file in this repository. The file is named
next to it. To pull one up interactively:

```bash
python scripts/experiments.py            # list the experiments
python scripts/experiments.py scaling    # show one, with its result tables
python scripts/trace.py 0.02003424617616511          # find which artifact holds a value
```

---

## The headline: 26 matched comparisons, 26 classical wins

The confirmatory family is the qubit-scaling study. The quantum reservoir is run
at 4, 6, 8, 10 and 12 qubits on three chaotic systems, against two comparators:
a size-matched random-feature control (ELM) and the tuned polynomial reference
(NG-RC). That gives **26 conditions** and, counting one Diebold–Mariano test per
random seed inside each condition, **214 per-seed tests**.

| | |
|---|---|
| Per-seed tests where the quantum model is significantly worse | **213 of 214** |
| Surviving Holm–Bonferroni correction over the whole family | all 214 (largest adjusted *p* = 9.7e-5) |
| Conditions won by the classical model | **26 of 26** |
| Conditions won on every single seed | 25 of 26 |
| The exception | Hénon at 4 qubits against the ELM, where the classical model wins 9 of 10 seeds |
| Median error ratio, quantum vs ELM | 4.6× (bootstrap 95% CI [3.0, 5.2]) |
| Median error ratio, quantum vs NG-RC | 7.3× (CI [6.2, 9.4]) |

The quantum model never takes a majority of seeds in any condition.

Files: `results/scaling_proof/{scores,dm,trend}.csv`.

Why both counts are reported: seeds are repeats *within* a condition, not
independent experiments. They share the system, the qubit count, the trajectory
and the split. The per-condition count (26) is the honest one. The per-seed
count (214) is the detail inside each condition, and it is what the multiple-
testing correction is applied to.

## Adding qubits does not help

Median one-step NRMSE on Hénon, as the qubit count grows:

| Qubits | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Quantum reservoir | 4.4e-3 | 4.4e-3 | 4.5e-3 | 4.5e-3 | 4.6e-3 |
| Matched random projection (ELM) | 2.3e-3 | 9.6e-4 | 7.5e-4 | 1.1e-3 | 9.7e-4 |
| Tuned NG-RC | 1.6e-6 | 1.6e-6 | 2.2e-6 | 5.7e-6 | 5.7e-6 |
| Nominal quantum state dimension | 16 | 64 | 256 | 1024 | **4096** |

The state space grows by a factor of 256 across that row. The error does not
move. The read-out never sees the state space — it sees `d_read = V · m · n_q`
measured numbers, which grows linearly.

File: `results/scaling_proof/scores.csv`.

## What actually moves quantum accuracy

Each pipeline axis was swept in turn with everything else fixed, and scored by
its *ratio swing*: worst mean error over best, across that axis's levels.

| Axis | Ratio swing |
|---|---|
| **Scaler** (a classical preprocessing choice) | **6.77** |
| Forecast horizon | 3.44 |
| Post-processing filter | 3.25 |
| Look-back length | 2.88 |
| Ridge penalty | 2.62 |
| Input encoding | 1.98 |
| Split fractions | 1.04 |
| Washout | 1.00 |
| Scaler scope | 1.0001 |

The largest single influence on the accuracy of the quantum models is a
classical preprocessing choice with nothing quantum about it.

File: `results/rq1_axis_swing.csv`.

In the mechanism ablations the same pattern holds. Switching the input encoding
from depth to width improves the error by **17× to 77×**. Removing entanglement
entirely — verified separable, half-chain entropy 3.7e-11 bits — costs at most
**3.0×**, median 1.4×. Adding the two-qubit ZZ correlator to the read-out is
worth about **1.03×**, while classical products of the same single-qubit
observables are worth about **100×**.

Files: `results/entanglement/{scores,separable_dm,entropy}.csv`,
`results/adv_B_zz_ablation.csv`, `results/followup/scores.csv`.

## The idealised results do not survive real measurement

All quantum results above assume exact expectation values, which means
infinitely many measurement shots. Under finite sampling:

| System | Persistence NRMSE | Skill over persistence: exact | at 8192 shots | at 1024 shots |
|---|---|---|---|---|
| Hénon | 1.600 | 358× | 9.0× | 3.7× |
| Lorenz-63 | 0.287 | 20.3× | 3.2× | 1.7× |
| Mackey–Glass | 0.151 | 33.9× | 2.2× | 1.2× |

Skill is measured against persistence — predicting that tomorrow equals today —
because raw NRMSE is not comparable across these systems. The collapse is worst
exactly where the raw error looks best: Mackey–Glass at 1024 shots reads 0.128,
the smallest error in the table, yet beats copy-the-last-value by only 1.19×.

Files: `results/concentration/{finite_shot_budget,noise_arm_cross_system}.csv`.

## Five attempts to make a quantum model win

| Attempt | What happened |
|---|---|
| Classical read-out augmentation (products of the quantum features) | Improved the quantum model 100× on Hénon and closed almost all of the gap to the random-feature control — but the equally augmented control gained too, and NG-RC is still significantly better in 5 of 5 seeds |
| Arcsin and Chebyshev polynomial encodings | Backfired. 5–21× worse than plain depth encoding; lost all 120 tests. The classical Chebyshev map of the same function class became the best model in the entire study |
| Recurrence-free architecture from the literature | Strong on its home-ground multivariate task, and shot-robust — but a classical twin with the circuit replaced by a random projection matched it (8.06 vs 8.02 Lyapunov times) |
| Dissipative quantum-memory reservoir | Genuinely quantum temporal structure and a real dissipation optimum. Its apparent memory-capacity win (9.1 vs 4.8) reversed once the classical control was tuned for memory with an equal budget: 21.4 vs 6.7 |
| More virtual nodes (temporal multiplexing) | Same saturation as qubit count. V=16 beats V=4 by 2–34%, while the control at the same feature count improves ~5× |

Files: `results/followup/`, `results/cheb/`, `results/rfqrc/`,
`results/dissipative/`.

## One thing the quantum model does do better

In autonomous rollout, where a model feeds its own forecast back as input, the
quantum reservoir blows up less often. On Lorenz-63 the degree-three NG-RC — one
of the strongest one-step models — diverges on every initial condition, while
the quantum reservoir diverges on about 13%. On Mackey–Glass the ELM diverges on
about 98% against about 2%.

This is a stability property, not an accuracy one. It comes from the Pauli
observables being bounded to [-1, 1], so the features cannot run away. It does
not come with a longer valid prediction time or a lower error, and a stable
forecast that is wrong is not a successful forecast.

Files: `results/ic_robustness/closedloop.csv`, `results/adv_D_climate.csv`.

## Engineering results

| Experiment | Finding | File |
|---|---|---|
| Volume scaling | NG-RC reaches 1M rows at ~40,000 rows/s; quantum feature generation runs at ~811 rows/s, roughly 50× slower | `results_de/volume.csv` |
| Storage format | Columnar reads ~53× faster than row-oriented | `results_de/storage.csv` |
| Parallelism | 2.51× speed-up at 8 workers; efficiency falls from 0.79 to 0.31 because the numerical library already threads | `results_de/parallel.csv` |
| Incremental execution | Re-running skips completed cells | `results_de/incremental.csv` |
| Reproducibility | Bit-identical on the validated machine, table hash `dfe17a25da6d2a94` | `results_de/reproducibility.csv` |
| Fault containment | 8 of 8 injected fault classes caught before reaching a result file | `results_de/failure.csv` |

Narrative version: [DE_EXPERIMENT_RESULTS.md](DE_EXPERIMENT_RESULTS.md).

## What this does not show

- Only simulable qubit counts were reached (up to 12), on a CPU statevector
  simulator. Nothing here speaks to hardware at scale.
- Only the evaluated circuit families, encodings and observable sets. A
  different feature map could behave differently.
- Mostly univariate forecasting of chaotic systems. Other task classes are
  untested.
- Finite-shot sampling is modelled; hardware channel noise is not.

The conclusion is deliberately narrow: within this scope, what governs
performance is the structure of the features the read-out can reach, not whether
a quantum device produced them.
