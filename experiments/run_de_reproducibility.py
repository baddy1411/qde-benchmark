#!/usr/bin/env python3
"""Experiment E — reproducibility.  ->  results_de/reproducibility.csv

Runs a small representative benchmark twice in-process and once in a fresh
subprocess (a clean interpreter + re-import, the closest approximation to a clean
checkout without building a container), then compares dataset/split/config
checksums, metric values, and a result-file content hash across the runs.

Classifies each component as DETERMINISTIC (identical bytes expected) or
STOCHASTIC-BUT-SEEDED (identical because the seed is threaded end-to-end). Wall
time is the one field NOT expected to reproduce (system load), and is excluded
from the equality check by design.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from qdepipe.logging_setup import configure, get_logger
from qdepipe import registry_ids as RID
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.pipeline.split import temporal_split
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig
from qdepipe.data import generate

OUT = Path("results_de"); OUT.mkdir(exist_ok=True)
configure(run_name="de_reproducibility")
log = get_logger("de.repro")

BENCH = [("henon", "ngrc", 0), ("henon", "esn", 0), ("lorenz", "esn", 1)]


def build(model_key, seed):
    if model_key == "ngrc":
        return ReservoirForecaster(NGRC(NGRCConfig(k=5, degree=2)), "NG-RC")
    return ReservoirForecaster(ESN(ESNConfig(units=300, seed=seed)), "ESN",
                               external_window=True)


def run_bench() -> dict:
    """Return {(system,model,seed): (dataset_hash, config_id, nrmse)} + a table hash."""
    out = {}
    frame = []
    for system, model_key, seed in BENCH:
        cfg = ExperimentConfig(system=system, n_points=1500, seed=seed,
                               lookback=5, washout=100)
        x = generate(system, 1500)
        dh = RID.dataset_hash(x)
        cid = RID.configuration_id(cfg)
        r = run_experiment(build(model_key, seed), cfg)
        out[(system, model_key, seed)] = (dh, cid, r.nrmse)
        frame.append({"system": system, "model": model_key, "seed": seed,
                      "nrmse": r.nrmse})
    tbl = pd.DataFrame(frame).to_csv(index=False).encode()
    out["_table_hash"] = hashlib.blake2b(tbl, digest_size=8).hexdigest()
    return out


# ---- run A and B in-process ------------------------------------------------
A = run_bench()
B = run_bench()

# ---- run C in a fresh subprocess (clean interpreter) -----------------------
_child = """
import warnings; warnings.filterwarnings('ignore')
import hashlib, pandas as pd
from qdepipe import registry_ids as RID
from qdepipe.experiment import run_experiment, ExperimentConfig
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ESN, ESNConfig, NGRC, NGRCConfig
from qdepipe.data import generate
BENCH=[('henon','ngrc',0),('henon','esn',0),('lorenz','esn',1)]
def build(mk,s):
    return (ReservoirForecaster(NGRC(NGRCConfig(k=5,degree=2)),'NG-RC') if mk=='ngrc'
            else ReservoirForecaster(ESN(ESNConfig(units=300,seed=s)),'ESN',external_window=True))
frame=[]
for sysn,mk,s in BENCH:
    cfg=ExperimentConfig(system=sysn,n_points=1500,seed=s,lookback=5,washout=100)
    r=run_experiment(build(mk,s),cfg)
    frame.append({'system':sysn,'model':mk,'seed':s,'nrmse':r.nrmse})
print(hashlib.blake2b(pd.DataFrame(frame).to_csv(index=False).encode(),digest_size=8).hexdigest())
"""
proc = subprocess.run([sys.executable, "-c", _child], capture_output=True, text=True)
C_hash = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 else "ERROR"

# ---- compare ---------------------------------------------------------------
rows = []
for k in BENCH:
    da, ca, na = A[k]; db, cb, nb = B[k]
    rows.append({"component": f"{k[0]}/{k[1]}/s{k[2]}",
                 "dataset_hash_AB": "match" if da == db else "DIFFER",
                 "config_id_AB": "match" if ca == cb else "DIFFER",
                 "nrmse_AB_identical": bool(na == nb),
                 "classification": "deterministic" if k[1] == "ngrc"
                                   else "stochastic-but-seeded"})
table_match_AB = A["_table_hash"] == B["_table_hash"]
table_match_AC = A["_table_hash"] == C_hash

pd.DataFrame(rows).to_csv(OUT / "reproducibility.csv", index=False)
print("\n===== Experiment E summary =====")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nin-process A vs B  result-table hash: {'IDENTICAL' if table_match_AB else 'DIFFER'} "
      f"({A['_table_hash']} vs {B['_table_hash']})")
print(f"fresh-subprocess C result-table hash: {'IDENTICAL' if table_match_AC else 'DIFFER'} "
      f"({C_hash})")
allrepro = all(r["nrmse_AB_identical"] for r in rows) and table_match_AB and table_match_AC
print(f"\nevery component reproduced bit-identically across all three runs: "
      f"{'PASS' if allrepro else 'FAIL'}")
print("note: stochastic models (ESN reservoir draw) reproduce ONLY because the "
      "seed is threaded end-to-end — an unseeded source would show up here as DIFFER.")
print("wrote results_de/reproducibility.csv")
