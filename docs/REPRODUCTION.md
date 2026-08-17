# Reproduction guide

How to regenerate the results in this repository, what each family costs to run,
and what the output should look like.

Run every command from the repository root.

## Requirements

- macOS or Linux, Python 3.9 or newer. Validated on Python 3.9.6, Apple Silicon,
  10-core CPU, no GPU.
- About 4 GB free disk if you let the quantum feature cache build up.
  16 GB RAM recommended.
- No network access needed at run time. The one exception is
  `experiments/run_pretrained_zeroshot.py`, which downloads a published
  Chronos-Bolt checkpoint. It is not part of the default install; add it with
  `make install-pretrained`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
make install          # dependency ranges from pyproject.toml
```

To install the exact versions the committed results were produced with:

```bash
make install-pinned   # pip install -r requirements.lock
```

## First: check the install works (about 2 minutes)

```bash
make verify
```

That runs three things:

| Command | What it checks |
|---|---|
| `python smoke_test.py` | one classical model end to end through all pipeline stages |
| `python -m pytest tests/ -q` | 131 unit, wiring and data-quality tests |
| `python tests/test_unit_gates.py` | 8 fault-injection gates, all 8 expected to be caught |

If `make verify` passes, every other command in this guide will run.

## Reproduction commands

| # | Result family | Command | Runtime | Writes |
|---|---|---|---|---|
| 1 | Baseline leaderboard | `make baseline` | ~10 min | `results/baseline_leaderboard.csv`, `results/model_scoreboard_all_runs.csv` |
| 2 | Matched budget, ZZ ablation, climate | `make matched` | ~20 min | `results/adv_*.csv` |
| 3 | Significance (DM, MCS) | `make significance` | ~5 min | `results/significance/` |
| 4 | Qubit scaling: the 214-test family | `make scaling` | hours uncached, minutes cached | `results/scaling_proof/` |
| 5 | Finite-shot degradation | `make shots` | ~30 min | `results/concentration/` |
| 6 | Entanglement and leaky-integration ablations | `make mechanism` | ~40 min | `results/entanglement/`, `results/leaky/` |
| 7 | The five rescue experiments | `make rescue` | hours | `results/followup/`, `results/cheb/`, `results/rfqrc/`, `results/dissipative/` |
| 8 | Data-engineering experiments | `make engineering` | ~10 min | `results_de/` |
| 9 | Figures | `make plots` | ~2 min | `results/figures/` |
| — | Everything above | `make reproduce` | see note below | all of the above |

Runtimes are for the validated hardware. Every sweep is **resumable**: if one is
interrupted, re-run the same command and it skips the cells that already have
results.

The quantum feature cache is not shipped with this repository. It is a pure
speed-up — every result regenerates without it. With it, `make reproduce` is
about an hour. Without it, the uncached quantum sweeps (4, 6, 7) take days on a
10-core CPU. Everything classical, and all of `make engineering`, is minutes
either way.

The fastest single thing that produces a real result is
`python experiments/run_de_storage.py` — about one minute.

## What the output should look like

On the validated platform, deterministic and seeded results reproduce
**bit-identically**, verified by table hash `dfe17a25da6d2a94` in
`results_de/reproducibility.csv`.

Spot checks:

| Quantity | Expected |
|---|---|
| Hénon NG-RC one-step NRMSE | ≈ 2.4e-7 |
| QRC-rich, matched feature budget, Hénon | ≈ 1.11e-3 |
| Same model at 8192 shots | ≈ 0.499 |

Two caveats when comparing:

- Timing and memory columns vary with machine load. Compare orders of magnitude,
  not digits.
- On a different platform or BLAS, expect differences in the last floating-point
  digits. Compare to about six significant figures. On Hénon the absolute values
  can differ by a few percent, because iterating a chaotic map amplifies a
  last-place difference into a different trajectory. Orderings, model selections
  and statistical verdicts are unaffected — that is measured, not assumed, in
  `docs/CROSS_PLATFORM_REPRODUCTION.pdf`.

## Cross-environment check

```bash
make docker-build
make docker-test        # test suite inside the pinned image
make docker-verify      # regenerate a scope and diff against the committed files
```

The image uses OpenBLAS on Debian, while the committed artifacts were produced
on Apple silicon with Accelerate. So this measures portability across BLAS
implementations, not repeatability on one machine. See `docker/README.md` for
what was observed.

## Troubleshooting

- **A long sweep was interrupted.** Re-run the same command. The done-set resume
  skips completed cells.
- **`snakemake` install fails on Python 3.9.** It needs the 7.x line and
  `pulp<2.8`. Both come from the `orchestration` extra that `make install` uses.
- **Regenerated figures are not byte-identical.** Expected. Figure output is
  timestamp-stripped, so the difference is not metadata — it is font metrics and
  raster sizing, which depend on the matplotlib and font versions installed.
  The numbers plotted are read from the committed CSVs and do not change. The
  images committed here are the ones used in the thesis.
- **A test imports an experiment script and fails.** Run pytest from the
  repository root; `tests/conftest.py` puts `experiments/` on the path.

## Inventory

`docs/MANIFEST.csv` lists every thesis-cited artifact with its producing script,
configuration, checksum and expected runtime.
