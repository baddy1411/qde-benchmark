"""End-to-end smoke test: data -> pipeline -> ESN -> readout -> NRMSE.

Proves the harness runs and the leakage-safe scaling scope actually matters.
Run:  ../venv/bin/python smoke_test.py
"""
from __future__ import annotations

from qdepipe import device
from qdepipe.data import generate_henon, lyapunov_gate
from qdepipe.forecasters import ReservoirForecaster
from qdepipe.models import ESN, ESNConfig
from qdepipe.experiment import ExperimentConfig, run_experiment
from qdepipe.pipeline.postprocess import LeakyMemory


def main():
    device.info()
    print()

    # data sanity: the Lyapunov gate must pass or the signal is wrong
    x, _ = generate_henon(4000)
    le, lt, ok = lyapunov_gate(x)
    print(f"Hénon Lyapunov: LE={le:.4f}  T_lyap={lt:.2f} steps  gate={'PASS' if ok else 'FAIL'}")
    print()

    esn = ReservoirForecaster(
        ESN(ESNConfig(units=300, spectral_radius=0.9, leak_rate=0.3, seed=42)),
        "ESN",
    )

    print("ESN 1-step forecast under different data-engineering choices:")
    for scaler in ("minmax", "standard", "robust"):
        r = run_experiment(esn, ExperimentConfig(scaler=scaler, scaler_scope="train"))
        print("  ", r)

    print("\nLeakage ablation (same model, scaler scope only):")
    for scope in ("train", "global"):
        r = run_experiment(esn, ExperimentConfig(scaler="minmax", scaler_scope=scope))
        print(f"   scope={scope:6s} -> NRMSE={r.nrmse:.4f}")

    print("\nPost-processing axis (leaky memory on ESN features):")
    for eps in (None, 0.1, 0.3):
        post = [] if eps is None else [LeakyMemory(eps)]
        r = run_experiment(esn, ExperimentConfig(postprocess=post))
        tag = "off" if eps is None else f"eps={eps}"
        print(f"   leaky {tag:8s} -> NRMSE={r.nrmse:.4f}")


if __name__ == "__main__":
    main()
