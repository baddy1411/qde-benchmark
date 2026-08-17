"""Zero-shot pretrained (foundation-model) context baseline: Chronos-Bolt.

Evaluates amazon/chronos-bolt-small — a pretrained time-series foundation
model — zero-shot (no training, no tuning on our data) on the three univariate
systems, through the SAME pipeline conventions as the cross-system battery:
identical trajectory generation, temporal split, train-fitted min-max scaling
(base_cfg from cross_system), the same one-step NRMSE convention, and the
identical run_closed_loop() harness for the autonomous battery (VPT in
Lyapunov times, log-spectral MSE, Wasserstein-1, divergence).

The model is deterministic (median-quantile read-out), like NG-RC, so one run
per system. Context window: the last 512 scaled points of the history.
Output: results/pretrained/zeroshot.csv. No committed artifact is modified.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline

from cross_system import base_cfg, RESULTS
from qdepipe import metrics
from qdepipe.closedloop import run_closed_loop
from qdepipe.data import generate as generate_system
from qdepipe.pipeline import make_scaler, temporal_split

MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT = int(os.environ.get("CHRONOS_CONTEXT", "512"))
N_STEPS = 300          # same rollout length as the committed climate battery

pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu",
                                           dtype=torch.float32)


class ChronosForecaster:
    """Duck-typed adapter for run_closed_loop: zero-shot, no fit."""
    name = "chronos_bolt_small"

    def onestep_predictor(self, u, split, cfg):
        def pred_fn(history: np.ndarray) -> float:
            ctx = torch.tensor(history[-CONTEXT:], dtype=torch.float32)
            q, _ = pipe.predict_quantiles(context=ctx, prediction_length=1,
                                          quantile_levels=[0.5])
            return float(q[0, 0, 0])
        return pred_fn


rows = []
for system in ("henon", "lorenz", "mackeyglass"):
    cfg = base_cfg(system)

    # --- one-step, teacher-forced over the test segment (batched) ------------
    x = generate_system(cfg.system, cfg.n_points)
    split = temporal_split(len(x), cfg.split_fracs, cfg.washout)
    scaler = make_scaler(cfg.scaler)
    scaler.fit(x[split.train_fit])
    u = np.asarray(scaler.transform(x), dtype=float).ravel()
    anchor, end = split.test.start, split.test.stop
    idxs = list(range(anchor, end))
    preds = np.empty(len(idxs))
    t0 = time.time()
    B = 64
    for b in range(0, len(idxs), B):
        batch = idxs[b:b + B]
        ctxs = torch.stack([torch.tensor(u[i - CONTEXT:i], dtype=torch.float32)
                            for i in batch])
        q, _ = pipe.predict_quantiles(context=ctxs, prediction_length=1,
                                      quantile_levels=[0.5])
        preds[b:b + len(batch)] = q[:, 0, 0].numpy()
    onestep_seconds = time.time() - t0
    true = u[anchor:end]
    nrmse = metrics.nrmse(true, preds)

    # --- autonomous rollout via the identical committed harness --------------
    t0 = time.time()
    cl = run_closed_loop(ChronosForecaster(), cfg, n_steps=N_STEPS)
    cl_seconds = time.time() - t0

    rows.append({"system": system, "model": "chronos_bolt_small",
                 "context": CONTEXT, "onestep_nrmse": nrmse,
                 "vpt_lyap": cl.vpt_lyap, "spectral_mse": cl.spectral_mse,
                 "wasserstein": cl.wasserstein, "diverged": cl.diverged,
                 "onestep_sec_per_step": onestep_seconds / len(idxs),
                 "closedloop_sec_per_step": cl_seconds / cl.n_steps})
    print(f"{system:12} 1-step NRMSE={nrmse:.4g}  VPT={cl.vpt_lyap:.3f} "
          f"spec={cl.spectral_mse:.3f} w1={cl.wasserstein:.4f} "
          f"div={cl.diverged}  {onestep_seconds/len(idxs)*1e3:.0f} ms/step",
          flush=True)

os.makedirs(os.path.join(RESULTS, "pretrained"), exist_ok=True)
out = os.path.join(RESULTS, "pretrained", f"zeroshot_ctx{CONTEXT}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
