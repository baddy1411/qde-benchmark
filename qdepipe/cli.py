"""qde — a config-driven experiment-matrix CLI.

A *caller, not a re-implementer*. Every run reads a declarative YAML file, builds
validated :class:`ExperimentConfig` objects from it, and invokes the SAME
functions the 22 scripts already use:

* ``task: run`` iterates the matrix (systems x seeds x models) and calls
  ``run_experiment(build_forecaster(model, seed, cfg, store), cfg)`` per cell —
  the atomic unit every script is built on. Iterating it is a loop, not new
  orchestration.
* ``task: finite_shot | concentration | matched_budget`` dispatch to the wired
  library passes (``finite_shot_sweep``, ``scaling_sweep``,
  ``matched_budget_shots``) with seeds/systems/store threaded from the YAML.

Because it produces zero new computation, the CLI inherits the six gate-(b)
cache-transparency proofs and cannot move a result. The config is validated once,
against the step-1 ``ExperimentConfigModel`` — the single source of truth for
"valid experiment", shared by scripts and CLI.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import pandas as pd
import yaml

from .contracts import ExperimentConfigModel
from .experiment import ExperimentConfig, run_experiment
from .registry import MODELS, build_forecaster
from .feature_store import FeatureStore


# ---------------------------------------------------------------------------
# config -> ExperimentConfig (the single source of truth, validated once)
# ---------------------------------------------------------------------------
def _resolve_cfg(experiment: dict, system: str, seed: int) -> ExperimentConfig:
    """Validate an `experiment:` block + (system, seed) from the sweep into an
    ExperimentConfig. system/seed come from the sweep; if also present in the
    experiment block they are overridden here (sweep wins). Raises
    pydantic.ValidationError on any bad/unknown field (extra='forbid')."""
    fields = dict(experiment or {})
    fields["system"] = system
    fields["seed"] = seed
    model = ExperimentConfigModel(**fields)
    return ExperimentConfig(**model.model_dump())


def _store(spec: dict):
    return FeatureStore() if spec.get("cache") else None


# ---------------------------------------------------------------------------
# tasks — each returns a DataFrame; each calls proven functions, never reimplements
# ---------------------------------------------------------------------------
def task_run(spec: dict) -> pd.DataFrame:
    """The experiment matrix: one run_experiment per (system, seed, model) cell."""
    sw = spec.get("sweep") or {}
    systems = sw.get("systems") or ["henon"]
    seeds = sw.get("seeds") or [0]
    models = sw.get("models") or ["ngrc"]
    exp = spec.get("experiment") or {}
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; valid keys: {sorted(MODELS)}")
    store = _store(spec)
    track = spec.get("track")
    fdir = spec.get("forecasts_dir")
    if fdir:
        os.makedirs(fdir, exist_ok=True)
    rows = []
    for system, seed, model in itertools.product(systems, seeds, models):
        cfg = _resolve_cfg(exp, system, seed)
        r = run_experiment(build_forecaster(model, seed, cfg, store=store),
                           cfg, keep_forecasts=bool(fdir))
        if fdir and r.y_true is not None:
            # persist aligned (t, y_true, y_pred) so the DAG significance stage can
            # consume it. Additive — only when forecasts_dir is set.
            pd.DataFrame({"t": range(len(r.y_true)), "y_true": r.y_true,
                          "y_pred": r.y_pred}).to_csv(
                os.path.join(fdir, f"{system}__{model}__seed{seed}.csv"), index=False)
        if track:
            # observation only — logs WHAT ran; never alters the result above
            from .tracking import log_run
            log_run(cfg, r.metrics, model=model,
                    extra_tags={"system": system, "model_key": model},
                    run_name=f"{system}-{model}-s{seed}",
                    experiment=spec.get("mlflow_experiment", "qde"),
                    tracking_uri=spec.get("mlflow_uri"))
        rows.append({"system": system, "model_key": model, **r.row()})
    return pd.DataFrame(rows)


def task_finite_shot(spec: dict) -> pd.DataFrame:
    from .concentration import finite_shot_sweep
    sw = spec.get("sweep") or {}
    seeds = tuple(sw.get("seeds", (0, 1, 2)))
    n_list = tuple(sw.get("n_list", (4, 6, 8)))
    store = _store(spec)
    rows = []
    for system in (sw.get("systems") or ["henon"]):
        rows += finite_shot_sweep(system, seeds=seeds, n_list=n_list, store=store)
    return pd.DataFrame(rows)


def task_concentration(spec: dict) -> pd.DataFrame:
    from .concentration import scaling_sweep
    sw = spec.get("sweep") or {}
    seeds = tuple(sw.get("seeds", (0, 1, 2)))
    n_list = sw.get("n_list")
    store = _store(spec)
    rows = []
    for system in (sw.get("systems") or ["henon"]):
        rows += scaling_sweep(system, seeds=seeds,
                              n_list=tuple(n_list) if n_list else None, store=store)
    return pd.DataFrame(rows)


def task_matched_budget(spec: dict) -> pd.DataFrame:
    from .concentration import matched_budget_shots
    sw = spec.get("sweep") or {}
    seeds = tuple(sw.get("seeds", (0, 1, 2, 3, 4)))
    return pd.DataFrame(matched_budget_shots(seeds=seeds, store=_store(spec)))


TASKS = {
    "run": task_run,
    "finite_shot": task_finite_shot,
    "concentration": task_concentration,
    "matched_budget": task_matched_budget,
}

# what each task honors from the YAML — surfaced by `qde list-tasks` so a fixed
# internal is never mistaken for an override.
TASK_INFO = {
    "run": "matrix over sweep.{systems,seeds,models}; full `experiment:` ExperimentConfig applies per cell.",
    "finite_shot": "calls finite_shot_sweep over sweep.systems; honors sweep.{seeds,n_list}. Internal: n_points, encoding fixed.",
    "concentration": "calls scaling_sweep over sweep.systems; honors sweep.{seeds,n_list}. Internal: n_points, encoding fixed.",
    "matched_budget": "calls matched_budget_shots (Hénon, n_points=1500 FIXED); honors sweep.seeds only. The headline finite-shot pass.",
}


# ---------------------------------------------------------------------------
# validation / introspection
# ---------------------------------------------------------------------------
def _validate_spec(spec: dict) -> str:
    task = spec.get("task", "run")
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; valid: {sorted(TASKS)}")
    sw = spec.get("sweep") or {}
    systems = sw.get("systems") or ["henon"]
    seeds = sw.get("seeds") or [0]
    # validate the experiment block against the single-source-of-truth schema
    _resolve_cfg(spec.get("experiment") or {}, systems[0], seeds[0])
    if task == "run":
        bad = [m for m in (sw.get("models") or []) if m not in MODELS]
        if bad:
            raise SystemExit(f"unknown model(s) {bad}; valid keys: {sorted(MODELS)}")
    return task


def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _print_tasks() -> None:
    print("tasks (set `task:` in the YAML):\n")
    for name, info in TASK_INFO.items():
        print(f"  {name:15s} {info}")
    print(f"\nvalid `models:` keys (task: run): {sorted(MODELS)}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qde", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run an experiment from a YAML config")
    pr.add_argument("config")
    pv = sub.add_parser("validate", help="validate a YAML config without running")
    pv.add_argument("config")
    sub.add_parser("list-tasks", help="list tasks and the fields each honors")
    args = ap.parse_args(argv)

    if args.cmd == "list-tasks":
        _print_tasks()
        return 0

    spec = _load(args.config)

    if args.cmd == "validate":
        task = _validate_spec(spec)
        print(f"OK: config valid (task={task}).")
        return 0

    if args.cmd == "run":
        task = _validate_spec(spec)              # validate before running
        df = TASKS[task](spec)
        out = spec.get("output")
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            df.to_csv(out, index=False)
            print(f"wrote {out} ({len(df)} rows, task={task})")
        else:
            print(df.to_string(index=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
