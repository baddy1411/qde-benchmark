# Quantum vs Classical Reservoir Computing on Chaotic Time Series

A benchmark that asks one question and answers it carefully: **does a quantum
reservoir forecast chaotic systems better than a classical one, when both are
given the same number of features to work with?**

The short answer is no. Across 26 matched comparisons the classical model wins
every one. This repository holds the code that produced that answer, every
result file behind it, and the machinery that makes those files checkable.

Companion artifact for the M.Sc. Data Engineering thesis
*A Reproducible Data-Engineering Framework for Benchmarking Quantum and
Classical Reservoir Computing on Chaotic Time Series*
(Badrish Madapuji Srinivasan, Constructor University, 2026).

---

## Start here

```bash
git clone <this-repo> && cd qde-benchmark
python3 -m venv .venv && source .venv/bin/activate
make install
make verify
```

`make verify` runs an end-to-end pipeline check, the 131-test suite, and 8
fault-injection gates. It takes about two minutes and needs no quantum
simulation. If it passes, everything else in this repository will run.

Then, to see what the project actually does:

```bash
make demo        # a guided walkthrough of the five things the pipeline guarantees
make browse      # an interactive terminal browser over every experiment and its results
```

Neither computes anything. They read the committed result files and show you
what is in them.

---

## What the benchmark found

| | |
|---|---|
| Matched comparisons run | 26 conditions (214 per-seed statistical tests) |
| Won by the classical model | **26 of 26** |
| Per-seed tests where the quantum model is significantly worse | 213 of 214, all surviving Holm–Bonferroni correction |
| Typical size of the gap | 4.6× worse than a random-feature control, 7.3× worse than tuned NG-RC |
| Effect of adding qubits (4 → 12) | none; error is flat while the state space grows 16 → 4096 |
| Largest single influence on quantum accuracy | the **classical** input scaler, not any quantum knob |

Five separate attempts were made to make a quantum model win: classical read-out
augmentation, two theoretically motivated polynomial encodings, a faithful
reimplementation of the literature's recurrence-free architecture, and a
dissipative reservoir whose memory lives in the quantum state. Each was paired
with a structure-matched classical control. Every one either reached parity with
random features, was reproduced by its classical twin, or made things worse.

The most accurate model found anywhere in the study is classical.

Full numbers: [docs/RESULTS.md](docs/RESULTS.md).

---

## Layout

```
qde-benchmark/
├── qdepipe/          the library: one leakage-safe pipeline every model runs through
├── experiments/      one script per experiment family (41 scripts)
├── scripts/          tools for inspecting the results, not producing them
├── tests/            131 unit, wiring, data-quality and fault-injection tests
├── results/          522 committed result files + 32 figures — the evidence
├── results_de/       the six data-engineering experiments
├── derived/          claim tables computed from results/ by scripts/analysis/
├── docker/           pinned container for the cross-environment check
├── docs/             reproduction guide, architecture, walkthrough, manifest
└── smoke_test.py     the one-file sanity check
```

Run everything from the repository root. Scripts write into `results/` and
`results_de/` relative to the working directory.

### Where to look first

| If you want to | Read or run |
|---|---|
| understand the design | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| see the guarantees demonstrated live | `make demo` |
| find the experiment behind a claim | `python scripts/experiments.py` |
| trace a single number to its source | `python scripts/trace.py 0.02003424617616511` |
| reproduce a result | [docs/REPRODUCTION.md](docs/REPRODUCTION.md) |
| know what each script does | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| check a claim against the artifact | [docs/MANIFEST.csv](docs/MANIFEST.csv) |

---

## How the comparison is kept fair

The result is a negative one, so the comparison has to be beyond argument. Four
things are held equal by construction, not by good intentions.

**One pipeline.** Every model, quantum or classical, goes through the same
twelve stages: generation, validation, temporal split, scaling, feature
generation, ridge read-out, forecasting, metrics, persistence. Only the feature
generator is swapped. A difference in a result cannot come from a difference in
data handling.

**Matched feature count.** The quantum model measures `d_read = V · m · n_q`
values per timestep. Its classical opponents are then sized to produce exactly
that many features, so both read-outs train the same number of weights. What
differs is feature *quality*, not *quantity*.

**A random-feature control.** Alongside the tuned classical baselines sits an
ELM: the same number of features, produced by a frozen random projection with
nothing intelligent in it. If a quantum circuit cannot beat random numbers of
equal size, its structure is contributing nothing. This control decides most of
the study.

**No leakage.** The scaler is fitted on the training segment only. This is not
asserted in prose — it is recorded in every run's identity and checkable with
one query over the run registry. The rule exists because fitting the scaler on
the full series produced a *false quantum advantage* in this project's own early
work.

---

## How the numbers are made checkable

- **Deterministic** — seeded end to end. On the validated machine the benchmark
  reproduces bit-identically, verified by hash.
- **Content-addressed cache** — expensive quantum features are keyed on a hash
  of everything that determines them. A test proves features are byte-identical
  with the cache on and off, so a cache hit cannot change a result.
- **Traceable** — a Parquet run registry links every reported number back
  through run, configuration and dataset to the code version that produced it.
  `python scripts/trace.py <value>` walks that chain.
- **Self-checking** — the pipeline validates its own numerical primitives before
  it will run. This is not decorative: it caught a silent complex-arithmetic
  fault in a container that had been turning every quantum feature into zero.
- **Fault-contained** — eight classes of corrupt input are injected into
  throwaway fixtures; all eight are caught before they can reach a result file.

The container check in `docker/` re-runs a scope of the benchmark against a
different BLAS to separate what is portable (orderings, selections, verdicts —
all of them) from what is machine-specific (the last floating-point digits, and
on Hénon the absolute values, because iterating a chaotic map amplifies a
last-place difference into a different trajectory).

---

## Reproducing results

```bash
make baseline        # the leaderboard across 24 models and 3 systems
make scaling         # the qubit-scaling family: the headline 214-test result
make shots           # finite-shot degradation
make rescue          # the five attempts to make a quantum model win
make engineering     # the six data-engineering experiments
make reproduce       # all of it, end to end
```

Every sweep is resumable: re-running skips cells that already have results.

The quantum feature cache (~1.6 GB) is **not** included here. It is a pure
speed-up and nothing depends on it — every result regenerates without it, just
slower. With the cache the full run is about an hour; without it, uncached
quantum sweeps take days on a 10-core CPU. Classical families and all
data-engineering experiments finish in minutes either way. Per-family runtimes
and expected values: [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

---

## Requirements

Python 3.9 or newer. `make install` pulls the dependency ranges from
`pyproject.toml`; `make install-pinned` installs the exact versions the results
were produced with (`requirements.lock`). No GPU, no quantum hardware, no
network access at run time — every dataset is generated by a seeded integrator
in this repository.

One experiment is kept out of the default install:
`experiments/run_pretrained_zeroshot.py` downloads a published Chronos-Bolt
checkpoint from Hugging Face. It is a context baseline and nothing in the
headline conclusions depends on it. Add it with `make install-pretrained`.

## Data and licence

No private data, no credentials, no personal information. All four benchmark
systems (Hénon, Lorenz-63, Mackey–Glass, Lorenz-96) are generated by seeded
integrators in `qdepipe/data/`.

Code is MIT-licensed — see [LICENSE](LICENSE). Third-party components used as
dependencies rather than vendored are credited in
[docs/CREDITS.md](docs/CREDITS.md).
