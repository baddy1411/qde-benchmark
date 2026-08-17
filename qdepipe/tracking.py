"""Run provenance + MLflow tracking — closes the per-result provenance gap.

Before this, provenance was toolchain-only (``versions.txt``): no commit hash, no
run id, no config hash per result. This module records WHAT ran — git SHA, config
hash, run id, params, metrics — without touching HOW anything is computed.

ADDITIVE and no-op by default. ``mlflow`` (skinny) is an optional import: if it is
absent, or tracking is not requested, every function here is a safe no-op. The
frozen compute core, the wired cache, and the leakage invariants are untouched —
this is observation, not computation.

Until the first git commit, ``git_sha()`` reports ``"uncommitted"`` and
``git_dirty()`` is True; once the author commits deliberately, real SHAs flow.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
from typing import Optional

from . import contracts

try:                                   # tracking-only client; optional
    import mlflow as _mlflow
except Exception:                      # pragma: no cover
    _mlflow = None


# ---------------------------------------------------------------------------
# provenance primitives
# ---------------------------------------------------------------------------
def git_sha(short: bool = True) -> str:
    """HEAD commit SHA, or ``"uncommitted"`` if there is no repo / no commit yet."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return sha[:12] if short else sha
    except Exception:
        return "uncommitted"


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes (or there is no commit)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        # no HEAD yet -> rev-parse fails -> treat as dirty/unversioned
        subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return bool(out)
    except Exception:
        return True


def config_hash(cfg, length: int = 16) -> str:
    """Stable hash of an experiment config, reusing the feature-store key
    machinery (``contracts._canonical`` normaliser + blake2b) so config identity
    is computed the same way everywhere."""
    blob = json.dumps(contracts._canonical(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=max(1, length // 2)).hexdigest()


def provenance(cfg=None) -> dict:
    """The provenance stamp for a result: git SHA + dirty flag (+ config hash)."""
    d = {"git_sha": git_sha(), "git_dirty": git_dirty()}
    if cfg is not None:
        d["config_hash"] = config_hash(cfg)
    return d


# ---------------------------------------------------------------------------
# MLflow logging (no-op if mlflow absent)
# ---------------------------------------------------------------------------
def mlflow_available() -> bool:
    return _mlflow is not None


def _as_params(cfg) -> dict:
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    return {k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in d.items()}


def log_run(cfg, metrics: dict, *, model: Optional[str] = None, extra_tags: Optional[dict] = None,
            run_name: Optional[str] = None, experiment: str = "qde",
            tracking_uri: Optional[str] = None) -> Optional[str]:
    """Log one result to MLflow and return its run id (None if mlflow is absent).

    params = the config's fields; metrics = the finite numeric metrics; tags =
    git SHA, git-dirty flag, config hash (+ any extra). Pure observation: it reads
    the result, never alters it.
    """
    if _mlflow is None:
        return None
    if tracking_uri:
        _mlflow.set_tracking_uri(tracking_uri)
    _mlflow.set_experiment(experiment)
    tags = {"git_sha": git_sha(), "git_dirty": str(git_dirty()), "config_hash": config_hash(cfg)}
    if model is not None:
        tags["model"] = str(model)
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})
    with _mlflow.start_run(run_name=run_name) as run:
        _mlflow.log_params(_as_params(cfg))
        _mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                             if isinstance(v, (int, float)) and math.isfinite(v)})
        _mlflow.set_tags(tags)
        return run.info.run_id
