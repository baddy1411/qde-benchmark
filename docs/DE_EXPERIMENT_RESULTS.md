# Data-Engineering Experiment Results (A–F)

Real numbers from the six DE experiments, run on the target hardware (MacBook Air,
Apple Silicon, 10 cores, CPU-only — no CUDA, confirmed). Every experiment writes a
CSV under `results_de/`; the scripts are `run_de_*.py`. These feed the Scalability,
Failure-Handling, and Reproducibility sections of the DE thesis chapter.

**Honesty note carried throughout:** all quantum numbers are CPU statevector costs.
Timing fields are the one thing that does *not* reproduce exactly run-to-run (system
load); every other measured quantity is deterministic or seeded.

---

## Experiment A — data-volume scaling (`results_de/volume.csv`)

Per-stage timing + peak memory as series length grows; each model sweeps ascending
sizes until a 90 s per-cell wall-clock budget or a memory limit stops it.

| Model | Largest n reached | Total time | Peak memory | Throughput |
|---|---|---|---|---|
| NG-RC | **1,000,000** | 24.8 s | 1495 MB | 40,411 rows/s |
| ELM | 500,000 | 14.0 s | 2627 MB | 35,767 rows/s |
| ESN | 500,000 | 16.5 s | 2817 MB | 30,284 rows/s |
| QRC-rich (n=6) | 5,000 | 6.2 s | 94 MB | **811 rows/s** |

**Findings:**
- **Quantum feature generation is the throughput ceiling, not data volume.** QRC-rich processes ~811 rows/s vs NG-RC's ~40,000 — a **~50× gap**. To featurize 1M rows, QRC would need ~20 minutes of CPU statevector simulation where NG-RC finishes in 25 s. This is the data-volume complement to the qubit-scaling proof: QRC is bottlenecked on both axes (accuracy flat in qubits, throughput ~50× behind classical).
- The classical bottleneck is **memory, not time** — ELM/ESN hit ~2.8 GB at 500k rows (the T×300 feature matrix), which is what caps them below 1M on a 16 GB machine, whereas NG-RC's narrow feature map (T×10) reaches 1M at 1.5 GB.

## Experiment B — storage format (`results_de/storage.csv`)

One representative result table (mixed dtypes) written CSV vs Parquet vs Parquet-zstd
vs NumPy `.npy`; 500k-row row shown.

| Format | File size | Write | Full read | **1-column read** | Compression vs CSV | Schema kept |
|---|---|---|---|---|---|---|
| CSV | 56.8 MB | 1092 ms | 201 ms | 93.8 ms | 1.0× | yes (re-inferred) |
| Parquet (snappy) | 20.4 MB | 83 ms | 59 ms | **1.8 ms** | 2.8× | yes (native) |
| Parquet (zstd) | 17.5 MB | 92 ms | 62 ms | 4.1 ms | **3.2×** | yes (native) |
| NumPy `.npy` | 27.3 MB | 5 ms | 1.7 ms | 1.6 ms | 2.1× | **no** (loses strings + column names) |

**Findings:**
- **Selected-column read is 52.7× faster in Parquet than CSV** (1.8 ms vs 93.8 ms) — the headline storage number. CSV must parse every column of every row to return one; Parquet reads only that column's pages.
- Parquet also writes **13× faster** and compresses **3.2×** (zstd), while preserving dtypes natively (CSV re-infers them on every read; `.npy` silently drops the string columns and headers entirely — which is exactly why `.npy` is right for the homogeneous feature cache but wrong for the mixed-dtype result registry).

## Experiment C — parallel execution (`results_de/parallel.csv`)

24 independent (system, seed) ESN cells at 20k points, via stdlib
`ProcessPoolExecutor` (not dask).

| Workers | Wall time | Speed-up | Efficiency | Failed |
|---|---|---|---|---|
| 1 | 9.24 s | 1.00× | 1.00 | 0 |
| 2 | 5.84 s | 1.58× | 0.79 | 0 |
| 4 | 4.03 s | 2.29× | 0.57 | 0 |
| 8 | 3.68 s | 2.51× | 0.31 | 0 |

**Findings:**
- Real speed-up to **2.51× at 8 workers, 0 failures** — but **efficiency falls from 0.79 (2 workers) to 0.31 (8)**. This is an honest, instructive result: NumPy's BLAS already multithreads each ESN's matrix ops, so adding worker processes on a 10-core chip **oversubscribes** the cores. The practical recommendation is **2–4 workers** (the efficiency sweet spot), or pinning `OMP_NUM_THREADS=1` per worker to make process-parallelism clean — a genuine data-engineering trade-off, not a free lunch.
- stdlib `concurrent.futures` was sufficient; dask's distributed scheduler would add operational complexity for zero benefit on this single-machine, independent-task workload.

## Experiment D — incremental execution (`results_de/incremental.csv`)

The consolidated `run_sweep` resume logic across five scenarios (12-cell grid).

| Scenario | Executed | Skipped | Failed |
|---|---|---|---|
| Clean rebuild | 12 | 0 | 0 |
| Cached rerun (no change) | **0** | 12 | 0 |
| Change one model config | **1** | 11 | 0 |
| Change one dataset | **1** | 12 | 0 |
| Run with mid-sweep failure | 11 | 0 | 1 |
| Resume after failure | **1** | 11 | 0 |

**Findings:**
- Incremental execution behaves exactly as a build system should: a no-change rerun recomputes **nothing**, changing one config or adding one dataset recomputes **exactly one cell**, and a resume after a mid-sweep failure recomputes **only the failed cell** — while the failure is isolated (the other 11 cells complete). No duplicate rows are ever created (registry dedup). This is the resume behaviour that was informally reimplemented six times across the verification program, now in one tested utility.

## Experiment E — reproducibility (`results_de/reproducibility.csv`)

A 3-model benchmark run twice in-process and once in a fresh subprocess; checksums
and metrics compared.

| Component | dataset hash | config id | NRMSE identical | Class |
|---|---|---|---|---|
| henon/ngrc/s0 | match | match | yes | deterministic |
| henon/esn/s0 | match | match | yes | stochastic-but-seeded |
| lorenz/esn/s1 | match | match | yes | stochastic-but-seeded |

Result-table hash **identical across all three runs** (`dfe17a25da6d2a94`), including
the fresh subprocess.

**Findings:**
- **Every component reproduced bit-identically**, including the stochastic ESN (whose random reservoir draw matches only because the seed is threaded end-to-end). The experiment is designed so that an *un*seeded randomness source would surface here as a `DIFFER` — its absence is the positive result. The only field deliberately excluded from the equality check is wall-clock time, which legitimately varies with system load.

## Experiment F — failure & data-quality recovery (`results_de/failure.csv`)

Eight fault classes injected into throwaway fixtures (never `results/`).

| Fault | Detected | Detection layer | Reached final results |
|---|---|---|---|
| Missing/unknown column | yes | registry schema | no |
| NaN metric value | yes | isfinite guard | no |
| Duplicate run id | yes | registry dedup | no |
| Overlapping train/test split | yes | split-order invariant | no |
| Invalid config (bad system) | yes | pydantic contract | no |
| Invalid config (n_points≤0) | yes | pydantic contract | no |
| Interrupted run (mid-sweep) | yes | sweep isolation + resume | no |
| Overwrite of a verified result | yes | guard_csv_write | no |

**Finding:** **8/8 faults detected, 0 reached final results.** Each is caught at the
layer responsible for it, before it could contaminate a result row — the hands-on
proof that the DE guards actually fire, with the overwrite-guard case being the
systemic fix for the one real filename-collision incident this project had.

---

## One-paragraph summary for the chapter

The pipeline scales cleanly on classical models (NG-RC to 1 M rows at 40 k rows/s)
and is bottlenecked on quantum feature generation by ~50× (811 rows/s), making
simulation cost — not data volume — the quantum ceiling. Parquet cuts selected-column
reads 53× and storage 3.2× over CSV. Process-parallelism gives a real but
sub-linear 2.5× speed-up that peaks in efficiency at 2–4 workers because BLAS
already multithreads each task. Incremental execution recomputes only what changed
and resumes only the failed cell after an isolated failure. The whole benchmark
reproduces bit-identically across processes, and all eight injected fault classes
are caught before reaching results. Every number traces to a CSV under `results_de/`.
