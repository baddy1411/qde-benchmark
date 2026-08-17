# Architecture

How the pipeline is put together, and why it is built this way.

The short version: a negative result is only worth as much as the comparison
that produced it. So the comparison is made identical by construction rather
than by care, and every reported number is traceable back to the run that
produced it.

## One pipeline, twelve stages

Every model in this study — quantum or classical — runs through the same stages
in the same order:

1. system configuration (which system, how many points, which seed)
2. deterministic trajectory generation
3. data-quality validation, including a chaos check
4. temporal train / validation / test split
5. scaling, with statistics fitted on the training segment only
6. reservoir feature generation, optionally served from cache
7. ridge read-out fitting on the training segment
8. one-step or autonomous forecasting on the test segment
9. metric computation
10. structured result persistence
11. statistical analysis
12. figure and table generation

Stages 1–9 are identical for every model. Only the feature generator in stage 6
is swapped. That is what makes the comparison fair in practice and not merely in
principle: a difference in a result cannot come from a difference in data
handling, because there is only one data handler.

## Where the code lives

| Path | What is in it |
|---|---|
| `qdepipe/data/` | the four systems: Hénon, Lorenz-63, Mackey–Glass, Lorenz-96. Seeded integrators, no downloaded data |
| `qdepipe/pipeline/` | split, scaling, embedding, post-processing — the leakage-safe stages |
| `qdepipe/models/` | feature generators: `gate_qrc`, `esn`, `elm`, `ngrc`, `qngrc`, and the quantum primitives in `_qops` |
| `qdepipe/forecasters/` | the wrappers that turn a feature generator into a forecaster |
| `qdepipe/readout.py` | the one ridge read-out every model shares |
| `qdepipe/feature_store.py` | the content-addressed cache |
| `qdepipe/registry*.py` | the Parquet run registry and its identifiers |
| `qdepipe/significance.py` | Diebold–Mariano, Holm–Bonferroni, Model Confidence Set |
| `qdepipe/contracts.py` | schema validation for everything written to disk |
| `qdepipe/fairness_gate.py` | the check that a comparison is actually matched before it runs |

## The three data layers

- **Raw** — the generated trajectory.
- **Intermediate** — the split and scaled series.
- **Result** — the persisted metrics, forecasts and figures.

Only the expensive layer is persisted independently. Raw and intermediate series
regenerate from a seed in microseconds to milliseconds, so storing them would
cost disk and I/O for nothing. The layer that is genuinely expensive to
recompute is the quantum feature matrix, and that is what the cache holds.

## Leakage safety

The scaler is fitted on the training segment only. This matters more than it
sounds: fitting it on the full series leaks the test range into training, and
that exact mistake produced a **false quantum advantage** in this project's own
early work. The rule exists because it was learned the hard way.

Three things enforce it:

- `temporal_split` computes segment boundaries once, before any model runs.
- The fit/transform contract makes it impossible for a transform to see data the
  fit did not.
- The scaler scope is written into every run's identity in the registry, so
  "was this run leakage-safe?" is a query, not a claim.

The washout period is excluded from the training-fit scope for the same reason.

## The feature cache

Quantum feature generation is the expensive part of the whole study, so results
are cached. A cache that could change a result would be worse than no cache, so
the key is a BLAKE2b hash of everything that determines the feature matrix:

1. **Data identity** — system, sample count, scaler type, scaler-fitting scope,
   split fractions. All of these matter because a train-fitted scaler makes the
   scaled series depend on them.
2. **Model fingerprint** — class name plus the complete configuration: qubit
   count, encoding, coupling strengths, shots, reservoir seed.
3. **External window layout**, where it applies.
4. **A feature-contract version string**, bumped whenever the meaning of the
   featurisation code changes, so a stale entry is never served.

Dependency versions are deliberately *not* in the key. The guarantee that a
cache hit cannot change a result is instead a test: features must be
byte-identical with the cache on and off. That test is in the suite.

Full contract: the data-contracts
appendix of the thesis. The implementation is `qdepipe/feature_store.py`.

## The run registry

Four joinable Parquet tables — `dataset`, `split`, `run`, `metrics` — with a
declared schema each. Writes are schema-validated before they can land, atomic
(temp file then rename), and duplicate-protected on the table's logical key.
Unknown columns are rejected.

The registry is what makes `scripts/trace.py` possible: from a reported number
you can walk back to the run, its configuration, its dataset and the code
version that produced it.

One honesty feature worth naming. A registry row is *captured* when the run that
produced it wrote it, and *reconstructed* when it was rebuilt from committed
artifacts afterwards. Those are not equally strong evidence, so the table
records which applies in a `provenance_source` column rather than leaving the
distinction to prose. Fields a reconstruction cannot know — wall-clock times,
environment identity, the commit that was checked out — are left null on
reconstructed rows rather than being filled in with the backfill's own values.

## Self-checking numerics

Before the pipeline will run, it validates its own numerical primitives against
known answers. This is not decorative. A container environment was found in
which `torch.vdot` on complex vectors silently returned zero and raised no
error. Every quantum expectation value became zero, every feature matrix became
all zeros, and the results looked like a real finding: NRMSE ≈ 1.0 everywhere,
which is exactly the score you get by predicting the mean.

The existing unit tests did not catch it, because they inspected the reservoir
state or compared a feature matrix against itself, and an all-zero matrix passes
both. None of them compared a measured expectation against a known answer.

Now `qdepipe/models/_qops.py` probes the platform primitive at startup, prefers
the fast native path where it is provably correct, falls back to a safe
implementation where it is not, and refuses to run if neither works. Regeneration
on a machine that passes the probe stays bit-identical.

## Fairness enforcement

`qdepipe/fairness_gate.py` checks, before a comparison runs, that the arms
actually share what they are supposed to share: the same measured feature count,
the same test segment, the same sample size. A matched comparison that is not
matched fails loudly instead of producing a number.

## Statistical layer

- **Diebold–Mariano** with the Harvey–Leybourne–Newbold small-sample correction,
  run per seed on per-step forecast errors, with a Newey–West variance estimator.
- **Holm–Bonferroni** across the whole pre-specified family, because 214 tests
  invite flukes.
- **Model Confidence Set** for "which models are statistically indistinguishable
  from the best", which controls the set jointly instead of through independent
  pairwise decisions.

The exact estimators are in `qdepipe/significance.py`.

## Testing

131 tests in four groups:

- **Unit** — the statistical functions, the metrics, the numerical primitives,
  each against known answers.
- **Wiring** — an experiment script and the library must agree. These import the
  actual script, not a reimplementation of it.
- **Data quality** — schema, ranges, chaos checks on generated series.
- **Fault injection** — eight classes of corrupt input (missing column, NaN
  metric, duplicate run id, overlapping train/test split, and four more) are
  injected into throwaway fixtures. All eight must be caught.

Run them with `make test`, or the whole gate with `make verify`.
