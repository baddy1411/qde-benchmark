# Rebuild Ledger

Workspace: a separate git worktree, detached at `1d26dfb`.
Interpreter: the main repo's pinned venv (`qde/.venv`, Python 3.9.6, requirements.lock).
Rule: `results/`, `results_de/` and the feature cache were emptied at setup;
every artifact in this tree is produced fresh, from scratch, in this tree.
Comparisons are against the committed artifacts in the main tree.

---

## G0 — Foundations, part one  ·  2026-07-27

| Check | Result |
|---|---|
| Test suite in rebuild tree | 105 passed / 1 failed / 1 deselected |
| The 1 failure | `test_backfilled_registry_leakage_audit`: FileNotFoundError on `results/model_scoreboard_all_runs.csv` — the file this rebuild deliberately deleted. Expected consequence of the clean slate, not a defect. Re-check after G1 rebuilds the scoreboard. |
| Unit gates | 8/8 passed |
| Trajectory determinism (main vs rebuild tree, blake2b of raw arrays, n=3000) | henon `0b5ca44a4c49badd` = match; lorenz `b1278459ffe311af` = match; mackeyglass `9647f985c22f2e9c` = match; lorenz96 `d7f9c0a0cceac1d9` = match |
| Lyapunov per-step spot values | henon 0.41148…, lorenz 0.04528, mackeyglass 0.00838 — consistent with committed conventions |

Verdict so far: ground floor sound. Pending in G0 step 2: `scripts/verify_rebuild.py`
(the comparator) and `scripts/analysis/*.py` (every derived thesis number as a script).

## G0 — Foundations, part two  ·  2026-07-27

Tools built on master (commit `85cf673`) and synced into this worktree
(code only; the cleared results trees stay cleared):

| Deliverable | Status |
|---|---|
| `scripts/verify_rebuild.py` (the comparator/referee) | written; exit-code gate; timing columns excluded; structural-only policy for timing-dominated DE files |
| `scripts/analysis/scaling_claims.py` | validated vs committed artifacts: 214 / 213 / 1, reversal henon n=4 vs elmF, max raw p 5.63e-5, max Holm 9.72e-5, core 210, ratios 4.6 [3.0,5.2] and 7.3 [6.2,9.4] — ALL match thesis |
| `scripts/analysis/ic_aggregate.py` | validated: 0/60 (committed winner-per-IC convention) + 0/180 fine-grained; henon W1 0.0132 vs 0.0136 match |
| `scripts/analysis/chronos_claims.py` | validated: 1.02 / 0.22 / 0.12, 41x vs linear ridge, 989 -> 5.6 — match |
| `scripts/analysis/rescue_claims.py` | validated: parity 1.037, V-gap 4.4x -> 23.3x, chebpoly 9.59e-8, twin 8.06 vs 8.02, shots +8%, MV collapse 8.01 -> 0.09, advocate 1.43e-8 — match |
| `qdepipe/fairness_gate.py` + 9 tests | blocking comparison contract; suite now 115 passed |

Findings (all resolved before closing G0):
1. **Thesis erratum found by the scripts**: polynomial-encoding section said
   "60 of 60" DM tests vs the ELM control; the artifact contains 30 (all
   significant losses) and 120 across all four comparators (all significant
   losses). Fixed in Ch.7; logged in the corrected-inconsistencies appendix.
2. IC tally convention aligned to the committed winner-per-IC definition
   (0/60); the finer 0/180 tally is reported alongside as a robustness check.
3. Advocate write-up overcounted DM pairs (15 stated, 13 computed, 0
   significant); MD corrected; the complete family regenerates in G5.

**G0 verdict: PASS.** Ground floor verified, referee and producers in place,
one thesis erratum caught and corrected before rebuild spending began.
Next: G1 (classical core) on approval.

## G1 — Classical core  ·  2026-07-27  ·  verdict: PASS

Chain: synthesize → cross_system → esn_budget_tune → headline_20seed →
significance_run → henon_joint_mcs → build_model_scoreboard (+ reruns below).
Fresh compute; empty feature cache; ~3.5 h total.

Comparator (final sweep of G1 scope): **154 IDENTICAL, 24 EQUAL/EQUAL-TOL,
0 unexplained DIFFERS.** Thesis-critical checks: baseline leaderboard 23
shared models max abs diff 0.00e+00; henon_quantum_DM.csv byte-identical;
joint MCS conclusion reproduced verbatim (best-set = NG-RC, quantum none);
deferred registry test now PASSES.

Findings (all root-caused, two fixed on master):
1. **Filename collision reproduced live** (the documented historical class):
   significance_run and cross_system both wrote significance/{system}_DM_*;
   output depended on execution order. FIXED: significance_run now writes
   *_quantum_* names (master 0bc1e29). Committed layout restored by ordered
   rerun; significance scope re-verified 0 DIFFER.
2. **Second collision instance in forecast dumps**: three writers shared
   forecasts/{system}/ at different lengths (299 vs 599). FIXED: quantum
   battery dumps namespaced to forecasts/{system}_quantum/ (master 70795b7).
   Diagnostic dumps only; nothing thesis-cited.
3. **Catalog drift, benign**: rebuilt leaderboard/rq* files carry one extra
   row/column because qNG-RC entered the registry after the committed run;
   the 23 shared models match with zero numeric difference.
4. **EQUAL-TOL policy**: ulp-level float noise (max ~1e-14 rel) on ESN-path
   aggregates from multithreaded BLAS reduction order; comparator now
   reports it explicitly under rtol 1e-9 (master c32539c).
5. Deferred to G2 by construction: cross climate files (quantum rows appended
   by quantum_cross), lorenz/mackeyglass forecast dumps, scaling_proof (in
   flight at sweep time).

Also in this window (main repo, outside the rebuild scope): catalogue
extension added on master — Theta, ElasticNet, ExtraTrees, KernelRidge in
the committed leaderboard frame, new artifact results/catalogue_ext.csv;
registry 28 models; LSTM et al. were already present.

## G2 — Quantum core  ·  2026-07-28  ·  verdict: PASS

Chain: run_scaling_proof → run_entanglement → concentration_run →
frequency_run → variance/benefit 5-seed → experiments_advanced →
quantum_cross. Entirely fresh statevector simulation from an empty feature
cache; 1048 logged cells; overnight.

Comparator (full sweep): **282 IDENTICAL, 37 EQUAL/EQUAL-TOL, 0 unexplained
DIFFERS.** Every headline quantum artifact byte-identical, including
scaling_proof/{scores(shared),dm,trend}, entanglement/{scores,entropy,
separable_dm}, adv_A2_matched, adv_B_zz_ablation, finite_shot files, the
quantum climate files and frequency_redundancy. adv_D_climate EQUAL-TOL at
2.4e-16 relative.

Decisive check — the thesis headline re-derived from the REBUILT artifacts:
family 214; 213 significantly worse / 1 better; reversal henon n=4 vs elmF;
max raw p 5.63e-5; max Holm 9.72e-5 (all significant); core 210; ratios
4.6 [3.0, 5.2] and 7.3 [6.2, 9.4]. Every number matches the thesis exactly.
The central result now exists twice, computed independently end to end.

Finding: the committed scaling_proof/scores.csv contains 60 legacy rows
(lorenz n=12, classical models only) from a superseded pre-guard code path
that the pinned code intentionally does not produce (n=12 is a henon-only
arm). Verified unused: committed dm.csv contains zero lorenz n=12 rows, so
no analysis ever consumed them. The 887 shared rows match with max |diff|
0.00e+00. Documented here; committed artifact left untouched.

G1 deferred items now converged: cross climate files complete with quantum
rows; lorenz/mackeyglass quantum forecast dumps written; quantum-in-best-set
"none" reproduced on both systems.

## G3 — Dynamics and robustness  ·  2026-07-28  ·  verdict: PASS

Chain: run_ic_study → run_lorenz96 → lorenz_mv_pilot → lorenz_mv_closedloop
→ closedloop_traj → run_climate_full. Fresh compute (~3 h; includes the
QRC-rich cells of the IC study and the MV pilot).

Comparator (cumulative sweep): **440 IDENTICAL, 41 EQUAL/EQUAL-TOL, 0 new
unexplained DIFFERS** — the 8 residual DIFFERS are exactly the previously
documented set (7 catalog-drift files, 1 scaling legacy-rows file).
G3 scope entirely clean: ic_robustness/{onestep,closedloop,summary},
lorenz96/{scores,dm,mcs,ngrc_probe}, lorenz_mv_cl_vpt, trajectories, and
climate_full all IDENTICAL (two files EQUAL-TOL at ~5e-16 relative).

Claims re-derived from the REBUILT artifacts: quantum wins 0 of 60 IC cells
(0 of 180 at per-seed granularity); henon W1 medians 0.0132 vs 0.0136 —
the disclosed 3% quantum fidelity cell reproduces exactly. The
climate_full VPT values match the committed VPT-only tables a third
independent time.

## G4–G7 — Realism, rescue suite, context baselines, engineering  ·  2026-07-27  ·  verdict: PASS (all four)

**G4 (realism):** leaky study rebuilt; finite-shot files were already
byte-identical from the G2 chain. Orphan-artifact finding: the thesis-cited
qngrc_comparison.csv had NO producer script in the repository; reconstructed
(experiments/run_qngrc_comparison.py, master b9bdffb) and verified to reproduce all 15
rows with relative difference 0.000e+00, then re-run here identically.

**G5 (rescue suite, second independent fresh pass):** followup, cheb, rfqrc
(all phases + MV twin), rfqrc_cheb, advocate. All artifacts IDENTICAL or
EQUAL-TOL. Claims re-derived from rebuilt tree: parity 1.037, V-gap
4.4x→23.3x, ChebPoly 9.59e-8, twin 8.06 vs 8.02, shots +8%, MV collapse
→0.09, advocate 1.43e-8. The advocate DM family regenerated COMPLETE
(60 rows vs the committed 52 with the documented resume gap): 0/15
significant vs the classical twin — the tie holds at full coverage; the
complete file adopted on master (closes G0 finding #3).

**G6 (Chronos):** both contexts rebuilt through the identical pipeline;
zeroshot claims reproduce exactly (1.02 / 0.22 / 0.12, 41x vs linear ridge,
989→5.6 spectral).

**G7 (engineering):** storage/incremental IDENTICAL-or-structural; fault
injection 8/8 contained, PASS; **reproducibility hash dfe17a25da6d2a94
reproduced IDENTICAL in a fresh subprocess** — the thesis's central
engineering claim survives its second life. Volume + parallel measured on a
quiet machine after the compute chains ended: parallel saturation signature
reproduced (2.32x @ 8 workers, efficiency 0.29; committed 2.51x/0.31 — same
BLAS-saturation diagnosis; timing-sensitive, structural-only per policy).
Process note: the initial queued G7b runner deadlocked against the
completion watcher (mutual pgrep pattern match); killed and run directly.

Cumulative comparator: **457 IDENTICAL, 47 EQUAL/EQUAL-TOL**; residual
DIFFERS = the 8 previously documented (7 catalog-drift, 1 legacy-rows) plus
qadv/dm.csv which is now superseded on master by the complete rebuild file.

## G8 — Derived statistics, figures, prose cross-check  ·  2026-07-27  ·  verdict: PASS

**Figures:** all 14 regenerated from the rebuilt tree; 12 byte-identical.
The two that differ are corrections, not regressions:
- fig11 (qubit-scaling proof): the committed version plotted the 60
  out-of-design legacy rows as dangling classical points at n=12 on the
  Lorenz panel; the rebuilt figure ends the Lorenz panel at n=10 as the
  pre-specified design does. Design-correct version adopted on master;
  thesis rebuilt with it.
- fig4 (seed distribution): committed version was built from era-mixed
  diagnostic forecast dumps; rebuilt version reflects the collision-fixed,
  fresh dumps. Adopted.

**Derived claims, all re-derived from the REBUILT artifacts across G2–G7:**
scaling family 214/213/1 with Holm 9.72e-5 and ratios 4.6/7.3 (CIs match);
IC 0/60 and the disclosed 0.0132-vs-0.0136 cell; rescue suite (parity 1.037,
twin 8.06 vs 8.02, ChebPoly 9.59e-8, MV collapse to 0.09, advocate 1.43e-8
with complete 0/15 DM); Chronos (1.02/0.22/0.12, 41x, 989→5.6). Every
number printed in the thesis reproduces from the rebuilt tree.

**Engineering magnitudes on the quiet-machine rerun:** throughput gap 51x
(thesis ~50x), storage 1-column speedup 55.7x (52.7x), compression 3.3x
(3.2x), parallel 2.32x@8/eff 0.29 (2.51x/0.31), hash dfe17a25da6d2a94
IDENTICAL, faults 8/8. Timing magnitudes concordant; identity-class claims
exact.

## PROGRAM VERDICT

Nine groups, 31 studies, ~520 artifacts: rebuilt from empty directories and
verified. Final tally: 457+ byte-identical, 47 equal within the documented
rtol, every residual difference root-caused and recorded above. Findings
fixed on master during the program: one thesis erratum (60-of-60), two
filename-collision hazards, one orphan artifact (producer reconstructed,
15/15 rows exact), one incomplete DM family (completed), two figures built
from out-of-design/era-mixed data (corrected). The thesis's every number now
exists twice, computed independently end to end.
