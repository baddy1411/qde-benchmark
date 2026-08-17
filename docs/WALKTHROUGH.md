# Walkthrough

A tour of what this repository does, using only commands that read the committed
results. Nothing here recomputes anything, and nothing takes more than a few
seconds.

## Start with the browser

```bash
python scripts/qde_ui.py --tui
```

One screen: the 19 experiments, the 6 demo steps, and the tools. Arrow keys (or
`j`/`k`) to move, **Enter** to run, **q** to quit. The right pane shows what the
selected item is, which artifacts it produced, and the exact command it will
run — so nothing in the UI is a magic path you could not have typed yourself.

Every action runs as a separate process. If something raises, you see the
traceback and the browser is still there.

Prefer plain commands? `--list` prints the whole menu with the command for each
entry:

```bash
python scripts/qde_ui.py --list
```

## The demo: six steps, about two seconds

```bash
python scripts/demo.py            # all six
python scripts/demo.py --step 3   # just one
```

| Step | What it shows |
|---|---|
| 1 | The environment verifies its own arithmetic, and refuses to run when it is wrong |
| 2 | One experiment end to end, every pipeline stage explicit |
| 3 | Data leakage demonstrated live across nine configurations |
| 4 | Leakage machine-checked across every run: 360 of 360 |
| 5 | The feature cache cannot change a result, byte for byte |
| 6 | The same configuration run twice, identical to the last bit |

### Step 3 is the interesting one

It is the only step that *demonstrates a failure* rather than asserting a
safeguard, and the result is less obvious than you would expect:

```
      system       scaler      train-only  whole series    change
      henon        minmax        0.009143      0.009142      0.0%
      lorenz       standard      0.076038      0.028270     62.8%  <-- flattered
      lorenz       robust        0.095170      0.049407     48.1%  <-- flattered
      henon        robust        0.100127      0.113347    -13.2%
```

Same model, same data, same everything. The only change is whether the scaler
was fitted on the training segment or on the whole series. On Lorenz with
standard scaling, leaking buys a 63% better number for free.

Now look at the first row. On Hénon with min-max scaling — this thesis's own
configuration — the effect is under a tenth of a percent. And in two cells,
leaking makes the score *worse*.

That is the point. Leakage does not announce itself. It can flatter you
enormously, not at all, or backwards, depending on the system and the scaler. So
you cannot catch it by looking at your results and asking whether they seem too
good. It has to be structurally impossible, and then checked by machine. Step 4
is that check: 360 of 360 runs verified leakage-safe by a query over the run
registry.

### Step 1 has a story behind it

The guard in step 1 exists because cross-environment checking found a real
defect. A numerical library returned zero from a complex dot product without
raising an error. Every quantum expectation value became zero, every feature
matrix became all zeros, and the pipeline reported the score of predicting the
mean — on every system, at every shot budget. It looked like a finding.

The guard now catches that at import, and refuses to run.

## Finding an experiment

```bash
python scripts/experiments.py                 # list all 19
python scripts/experiments.py entangle        # by keyword
python scripts/experiments.py 4               # by number
python scripts/experiments.py leaky --full    # print the whole CSVs
```

Showing one gives you the thesis section it belongs to, the script that produced
it, every artifact it wrote with a file-exists check and byte size, and the
actual result table read from the committed CSV.

The index is not hand-written. It was extracted from the results chapter itself,
from the `\artifact{}` footnote the thesis attaches to every claim, and committed
as `docs/experiment_index.json`. All 49 artifact references across the 19
experiments resolve to files that exist.

| # | Experiment | Produced by |
|---|---|---|
| 1 | Baseline Leaderboard | `synthesize.py` |
| 2 | Pretrained Zero-Shot Baseline | `run_pretrained_zeroshot.py` |
| 3 | Matched Measured-Feature Comparison | `experiments_advanced.py` |
| 4 | Qubit Scaling | `run_scaling_proof.py` |
| 5 | Encoding Sensitivity | `run_entanglement.py` |
| 6 | Entanglement Ablation | `run_entanglement.py` |
| 7 | Observable / Read-out Ablation | `experiments_advanced.py` |
| 8 | Feature Redundancy | `concentration_run.py` |
| 9 | Read-out Augmentation, Virtual Nodes | `run_followup_tricks.py` |
| 10 | Polynomial-Structured Encodings | `run_cheb_encoding.py` |
| 11 | The Recurrence-Free Architecture | `run_rfqrc.py` |
| 12 | The Dissipative Quantum-Memory Reservoir | `run_dissipative_qrc.py` |
| 13 | Initial-Condition Robustness | `run_ic_study.py` |
| 14 | Closed-Loop, VPT, Attractor Fidelity | `run_ic_study.py` |
| 15 | Lorenz-96 | `run_lorenz96.py` |
| 16 | Multivariate Lorenz Pilot | `cross_system.py` |
| 17 | Finite-Shot Degradation | `concentration_run.py` |
| 18 | Leaky Integration | `run_leaky.py` |
| 19 | qNG-RC and NG-RC Degree Sensitivity | `cross_system.py` |

Showing a result is instant. Re-running the experiment behind it is twenty
minutes to hours, so the tool never offers to.

## Tracing one number back to its source

```bash
python scripts/trace.py 0.02003424617616511
```

Give it any number from the results. It searches two layers and tells you which
one answered.

**Layer 1, the run registry** — 360 scoreboard runs across four joinable Parquet
tables. A hit resolves the whole chain:

```
THE RUN            run_id, model, family, seed, configuration_id, status
THE DATA IT USED   system, system_parameters, random_seed, sample_count,
                   transient_removed, dataset_hash
HOW IT WAS SPLIT   train 0-900, validation 900-1200, test 1200-1500,
                   scaler fitted on: train, split_hash
WHAT WAS MEASURED  nrmse, mae, valid_prediction_time, feature_count, feature_rank
```

**Layer 2, the committed artifacts** — the scaling family, the concentration and
finite-shot studies, the IC study, entanglement and the rescue suite sit outside
the queryable layer by design. The tool then names the CSV, the row number and
the column, with the categorical context of that row.

### The limitation, stated up front

The tool prints this itself. The registry's lineage is **reconstructed** — it
was rebuilt from the artifacts after the fact, not captured while each run
executed. `git_commit` and `environment_id` are null for all 360 rows, recorded
as not-measured rather than filled in with the backfill's own values.

The independent evidence that these records are faithful is the full
re-execution: every artifact regenerated from code and data alone, on the
development machine and then again on a different platform.

### The same trail without the tool

1. Every number in the thesis carries a footnote naming its artifact:
   `\artifact{results/.../file.csv}`
2. That CSV is committed here.
3. `docs/MANIFEST.csv` maps each cited artifact to the script, configuration
   and checksum that produced it.

## Other quick commands

| Question | Command | Time |
|---|---|---|
| Does the test suite pass? | `make test` | ~15 s |
| Does the whole thing work? | `make verify` | ~2 min |
| Can you validate a config? | `qde validate <config.yaml>` | instant |
| What can the CLI do? | `qde list-tasks` | instant |
| Show one model end to end | `python smoke_test.py` | ~10 s |
| Every run and comparison in one spreadsheet | `python scripts/build_results_workbook.py` | ~5 s |

## About the NumPy warnings

Running anything outside the demo will print these. The demo filters them; the
test suite does not:

```
RuntimeWarning: divide by zero encountered in matmul
RuntimeWarning: overflow encountered in matmul
RuntimeWarning: invalid value encountered in matmul
```

They are cosmetic, and that was checked rather than assumed:

```python
import numpy as np, warnings
rng = np.random.default_rng(0)
W, x = rng.standard_normal((300, 300)), rng.standard_normal(300)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    y = W @ x
print(len(w), np.isfinite(y).all())     # -> 3 True
```

A bare matrix-vector product on random dense data raises all three, with a
finite result. It is NumPy 2.x on Apple Accelerate leaving floating-point status
flags set after a matmul. Nothing in the pipeline divides by zero — the ESN
feature matrix contains no infinities and no NaNs, with values inside
[-0.71, 0.78].

The demo filters that one message specifically, not warnings in general, so a
real numerical warning would still show up.
