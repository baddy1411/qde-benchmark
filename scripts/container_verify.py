#!/usr/bin/env python3
"""Regenerate a scope of the benchmark inside the container and diff it against
the committed artifacts.

This answers a question the thesis explicitly leaves open (§10.5): the committed
results were produced on Apple silicon against Accelerate BLAS, and cross-platform
floating-point behaviour was never tested. Running the same code in a Linux
container against OpenBLAS is that test.

The expected outcome is NOT byte-identity. Different BLAS implementations reduce
in different orders, so the last bits of a dot product may differ, and chaotic
dynamics amplify that. What this measures is *how far apart* the two environments
land, per column, so the thesis can state a number rather than a hope.

Isolation follows the same pattern as the full re-execution already reported in
Chapter 8: `git archive HEAD` is extracted into a scratch tree, the container is
given that tree read-write, and the committed repository is never mounted. The
runners write next to their own __file__, so this is the only way to redirect
their output without editing them.

Usage:
    python scripts/container_verify.py --image qde-repro:local
    python scripts/container_verify.py --image qde-repro:local --scope classical
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Scopes ordered by cost. Only cheap, deterministic, classical families by
# default: the point is to characterise BLAS drift, not to re-run the quantum
# suite in a container.
SCOPES = {
    "smoke": (
        "one ESN end-to-end through the pipeline (~1 min)",
        ["python", "smoke_test.py"],
        [],
    ),
    "classical": (
        "classical cross-system battery, all three systems (~10-20 min)",
        ["python", "experiments/cross_system.py"],
        # Only what experiments/cross_system.py actually writes. Diffing a whole directory
        # counts files the run never touched as IDENTICAL, which is a false pass.
        [f"results/cross/{s}_{k}.csv"
         for s in ("henon", "lorenz", "mackeyglass")
         for k in ("climate", "leaderboard", "ngrc_tune")]
        + ["results/cross/lorenz_delta_sensitivity.csv"]
        + [f"results/significance/{s}_{k}.csv"
           for s in ("henon", "lorenz", "mackeyglass")
           for k in ("DM_matrix", "DM_pairs", "MCS")]
        # The per-seed forecast files are the primitive evidence: raw y_true /
        # y_pred per step, before any aggregation. They give a distribution of
        # cross-environment drift rather than a worst case on a derived table.
        # Discovered by the regeneration guard, which flagged them as rewritten
        # but unexamined.
        + [f"results/forecasts/{s}/{m}_seed{i}.csv"
           for s in ("henon", "lorenz", "mackeyglass")
           for m in ("NG-RCbest", "NG-RCd1", "NG-RCd2", "NG-RCd3",
                     "ELMppoly2", "ESNppoly2", "Linear-Ridge", "RandomForest")
           for i in (0, 1, 2)]
        + [f"results/forecasts/henon/{m}_seed{i}.csv"
           for m in ("elm", "esn") for i in (0, 1, 2)]
        + [f"results/forecasts/{s}/{m}_seed{i}.csv"
           for s in ("lorenz", "mackeyglass")
           for m in ("ELM", "ESN") for i in (0, 1, 2)],
    ),
    "noise": (
        "finite-shot / measurement-noise arm (~20-40 min)",
        ["python", "experiments/concentration_run.py"],
        # A different question from the deterministic arms. Shot sampling draws
        # from np.random.default_rng (PCG64), whose stream numpy guarantees is
        # bit-identical across platforms for a fixed seed -- so the randomness
        # itself is not the variable. What can differ is the PROBABILITY VECTOR
        # it draws against, which comes from the statevector and therefore from
        # BLAS. A draw sitting near a bin boundary can then land differently, and
        # a discrete count has no small perturbation: it moves by a whole shot.
        # So this arm can disagree more coarsely than the exact-expectation one,
        # and that is worth measuring rather than assuming either way.
        [f"results/concentration/{k}.csv"
         for k in ("scaling", "finite_shots", "finite_shot_budget",
                   "n8_npoints_spotcheck")],
    ),
    "entanglement": (
        "entanglement / ZZ-benefit arm (~45-90 min)",
        # experiments/run_entanglement.py is RESUMABLE: it reads the existing scores.csv and
        # entropy.csv into `done`/`ent_done` sets and skips every cell it finds,
        # appending only what is missing. Against the staged committed tree that
        # means it would skip everything, write nothing, and the regeneration
        # guard would (correctly) report NOT_REGENERATED for all three artifacts.
        # Clearing them first is what makes this arm actually recompute. Safe:
        # the deletion happens in the throwaway staging tree, never in the repo.
        ["sh", "-c",
         "rm -f results/entanglement/*.csv && python experiments/run_entanglement.py"],
        # The last GateQRC arm outside a scope, and the only unverified one on
        # the pure-state path that the vdot bug hit -- which is why it is here.
        # Its FeatureStore is DVC-tracked rather than git-tracked, so the staged
        # tree carries no cache and the quantum features are genuinely recomputed
        # rather than read back from disk.
        [f"results/entanglement/{k}.csv"
         for k in ("scores", "entropy", "separable_dm")],
    ),
}


def sh(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def docker_up():
    r = sh(["docker", "info", "--format", "{{.ServerVersion}}"])
    return r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[:2]


def stage_tree(dest: Path):
    """Extract the committed tree (HEAD) into dest -- no working-tree dirt."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(["git", "archive", "HEAD"], cwd=ROOT,
                               stdout=subprocess.PIPE)
    untar = subprocess.Popen(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
    archive.stdout.close()
    untar.communicate()
    if untar.returncode != 0:
        raise SystemExit("failed to stage the committed tree")


def compare(committed: Path, rebuilt: Path):
    """Per-column agreement between the committed artifact and the container's.

    Relative difference is only meaningful where the committed value is not
    essentially zero. Columns like `nrmse_std` for a deterministic model hold
    values around 1e-23, where a relative difference of 1e280 means "both are
    numerically zero" and nothing else. Those columns are reported by absolute
    difference instead, and the headline `max_rel_diff` is computed only over
    columns with a committed magnitude above ZERO_FLOOR.
    """
    import numpy as np
    import pandas as pd

    ZERO_FLOOR = 1e-12

    a = pd.read_csv(committed)
    b = pd.read_csv(rebuilt)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {"status": "SHAPE_DIFFERS",
                "detail": f"committed {a.shape} vs container {b.shape}"}
    if a.equals(b):
        return {"status": "IDENTICAL"}

    worst_rel, worst_rel_col = 0.0, None
    worst_abs, worst_abs_col = 0.0, None
    negligible = []          # differs, but both sides are numerically zero
    nonnum = []
    for c in a.columns:
        if not np.issubdtype(a[c].dtype, np.number):
            if not a[c].equals(b[c]):
                nonnum.append(c)
            continue
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        keep = ~(np.isnan(x) & np.isnan(y))
        if not keep.any():
            continue
        xa, ya = x[keep], y[keep]
        adiff = float(np.nanmax(np.abs(xa - ya)))
        if adiff > worst_abs:
            worst_abs, worst_abs_col = adiff, c
        big = np.abs(xa) > ZERO_FLOOR
        if big.any():
            rel = float(np.nanmax(np.abs(xa[big] - ya[big]) / np.abs(xa[big])))
            if rel > worst_rel:
                worst_rel, worst_rel_col = rel, c
        elif adiff > 0:
            negligible.append(c)

    return {
        "status": "DIFFERS" if (worst_rel or worst_abs or nonnum) else "EQUAL",
        "max_rel_diff": worst_rel,
        "worst_rel_column": worst_rel_col,
        "max_abs_diff": worst_abs,
        "worst_abs_column": worst_abs_col,
        "near_zero_columns": negligible,
        "non_numeric_mismatches": nonnum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="qde-repro:local")
    ap.add_argument("--scope", default="smoke", choices=sorted(SCOPES))
    ap.add_argument("--keep", action="store_true", help="keep the scratch tree")
    args = ap.parse_args()

    ok, info = docker_up()
    if not ok:
        print("docker is not reachable:", *info)
        print("Open Docker Desktop and complete its first-run setup, then re-run.")
        return 2

    label, cmd, artifacts = SCOPES[args.scope]
    print(f"scope: {args.scope} — {label}")

    scratch = Path(tempfile.mkdtemp(prefix="qde-container-"))
    tree = scratch / "tree"
    print(f"staging committed tree (HEAD) into {tree}")
    stage_tree(tree)

    host_env = sh([sys.executable, "-m", "qdepipe.envid"], cwd=ROOT).stdout
    cont_env = sh(["docker", "run", "--rm",
                   "-e", "QDE_LOCKFILE=/opt/requirements.lock.txt",
                   args.image, "python", "-m", "qdepipe.envid"]).stdout
    print("\n--- host ---\n" + host_env.strip())
    print("\n--- container ---\n" + cont_env.strip())

    # Marker for the regeneration check below. Any artifact whose mtime does not
    # advance past this was not rewritten by the run, and comparing it against
    # itself would be a free pass rather than evidence.
    marker = scratch / ".before"
    marker.touch()
    t0 = marker.stat().st_mtime

    docker_cmd = [
        "docker", "run", "--rm", "-t",
        "-v", f"{tree}:/work",
        "-e", "QDE_LOCKFILE=/opt/requirements.lock.txt",
        "-w", "/work",
        args.image, *cmd,
    ]
    print("\n--- running in container ---\n" + " ".join(docker_cmd) + "\n")
    rc = subprocess.run(docker_cmd, text=True).returncode
    print(f"\ncontainer exit code: {rc}")

    def as_json(s):
        try:
            return json.loads(s)
        except Exception:
            return None

    report = {
        "scope": args.scope,
        "image": args.image,
        "container_exit_code": rc,
        "host_environment": as_json(host_env),
        "container_environment": as_json(cont_env),
        "files": {},
    }

    # Which files did the run actually rewrite? Anything else in `artifacts` was
    # staged from git archive and left alone, and must never be scored as a match.
    rewritten = {
        str(p.relative_to(tree))
        for p in tree.rglob("*.csv")
        if p.stat().st_mtime > t0
    }

    # A nonzero exit means the run did not COMPLETE. Whatever it managed to write
    # before dying is partial or degenerate, and diffing it manufactures a verdict
    # (DIFFERS, with real-looking magnitudes) out of a crash. That is exactly how a
    # dead simulator in the container once got recorded as an NRMSE reproducibility
    # finding. A failed run is not evidence; refuse to score it.
    if rc != 0:
        report["verdict"] = "RUN_FAILED"
        report["detail"] = (
            f"container exited {rc}; no artifact comparison was performed "
            f"({len(rewritten)} file(s) had been written before the failure). "
            "Fix the run, then re-verify.")
    else:
        report["verdict"] = "COMPARED"
        for key in artifacts:
            rebuilt, committed = tree / key, ROOT / key
            if key not in rewritten:
                # Declared in the scope but the runner never touched it: a stale scope
                # list, or the runner changed. Either way it is not evidence.
                report["files"][key] = {"status": "NOT_REGENERATED",
                                        "detail": "declared in scope but not rewritten by the run"}
            elif not committed.exists():
                report["files"][key] = {"status": "NO_COMMITTED_COUNTERPART"}
            else:
                report["files"][key] = compare(committed, rebuilt)

    # The converse: files the run wrote that the scope does not list. These are
    # unexamined evidence, and silently ignoring them is how a scope list rots.
    unlisted = sorted(rewritten - set(artifacts))
    report["rewritten_but_not_in_scope"] = unlisted
    report["regenerated_count"] = len(rewritten)

    dest = ROOT / "docker" / "container_verification.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")

    if report["verdict"] == "RUN_FAILED":
        print(f"\n  !! RUN FAILED (exit {rc}). No comparison was performed.")
        print("     Partial artifacts from a failed run are not evidence.")

    if report["files"]:
        print("\n--- agreement with committed artifacts ---")
        counts = {}
        for k, v in sorted(report["files"].items()):
            counts[v["status"]] = counts.get(v["status"], 0) + 1
            extra = (f"  max rel diff {v['max_rel_diff']:.2e} in {v['worst_rel_column']}"
                     if v.get("max_rel_diff") else "")
            print(f"  {v['status']:10s} {k}{extra}")
        print("\n  " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
        stale = [k for k, v in report["files"].items()
                 if v["status"] == "NOT_REGENERATED"]
        if stale:
            print(f"\n  !! {len(stale)} file(s) declared in the scope were NOT rewritten by the run.")
            print("     They are excluded from the result. Fix the scope list:")
            for k in stale:
                print(f"       {k}")
        if unlisted:
            print(f"\n  !! {len(unlisted)} file(s) were rewritten but are not in the scope,")
            print("     so they went unexamined. Add them:")
            for k in unlisted:
                print(f"       {k}")

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"\nscratch kept at {scratch}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
