#!/usr/bin/env python3
"""Catalogue extension: four additional common classical baselines
(Theta, ElasticNet, ExtraTrees, KernelRidge) run in the EXACT frame of the
committed baseline leaderboard, written to a NEW artifact so the committed
leaderboard remains byte-comparable in the rebuild program.

Untuned context tier (research-design chapter): single fixed configuration,
five seeds, never argued from in headline conclusions.

Output: results/catalogue_ext.csv  (+ a merged rank preview printed).
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from synthesize import baseline_leaderboard
from qdepipe.registry import MODELS

NEW_KEYS = ["theta", "elastic_net", "extra_trees", "krr"]
SEEDS = tuple(range(5))

df = baseline_leaderboard(NEW_KEYS, seeds=SEEDS)
os.makedirs("results", exist_ok=True)
df.to_csv("results/catalogue_ext.csv", index=False)
print(df.to_string(index=False))

# merged preview against the committed leaderboard (context only)
lead = pd.read_csv("results/baseline_leaderboard.csv")
cols = [c for c in lead.columns if c in df.columns]
merged = pd.concat([lead[cols], df[cols]], ignore_index=True)
merged = merged.sort_values("nrmse_mean").reset_index(drop=True)
merged["rank"] = merged.index + 1
print("\n=== merged rank preview (Henon leaderboard + extension) ===")
print(merged[["rank", "model", "nrmse_mean"]].to_string(index=False))
