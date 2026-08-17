"""Model catalog — the single source of truth for the classical benchmark suite.

Each entry knows how to BUILD its forecaster from (seed, cfg) and which
data-engineering axes apply to it. `build(seed, cfg)` takes the config so a model
whose lookback is an internal parameter (NG-RC's tap count, ELM's window) still
honours the shared `lookback` axis — keeping that axis comparable across every
model. The notebook builder reads this catalog to emit one notebook per model
with exactly the applicable experiments (and explicit "N/A" notes for the rest).

Axis keys (data-engineering independent variables; see THESIS_PLAN §2):
  scaler scope split washout lookback windowing horizon postprocess alpha
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .experiment import ExperimentConfig, Forecaster
from .models import ESN, ESNConfig, NGRC, NGRCConfig, ELM, ELMConfig
from .models import QNGRC, QNGRCConfig
from .models.gate_qrc import GateQRC, GateQRCConfig
from .forecasters import (
    ReservoirForecaster, WindowedForecaster, NaiveForecaster,
    StatsForecaster, ThetaForecaster, SequenceForecaster,
)

ALL_AXES = ["scaler", "scope", "split", "washout", "lookback",
            "windowing", "horizon", "postprocess", "alpha"]


@dataclass
class ModelEntry:
    key: str
    name: str
    group: str          # 'reservoir' | 'statistical' | 'ml' | 'dl'
    kind: str           # paradigm, for the notebook prose
    build: Callable[[int, ExperimentConfig], Forecaster]
    axes: list          # applicable data-engineering axes
    deterministic: bool = False
    closed_loop: bool = True
    base_overrides: dict = field(default_factory=dict)   # baseline cfg for this model
    blurb: str = ""

    def na_axes(self) -> list:
        return [a for a in ALL_AXES if a not in self.axes]


# ---- estimator factories (fresh, seeded per call) --------------------------
def _rf(seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def _histgb(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=seed)


def _xgb(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                        subsample=0.9, random_state=seed, verbosity=0, n_jobs=-1)


def _lgbm(seed):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.1,
                         random_state=seed, verbose=-1, n_jobs=-1)


def _svr(_seed):
    from sklearn.svm import SVR
    return SVR(C=10.0, gamma="scale")


def _knn(_seed):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=5)


def _ridge(alpha):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=max(alpha, 1e-12))


def _elastic(_seed):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000)


def _extratrees(seed):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def _krr(_seed):
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=1e-3, kernel="rbf")


# ---- the catalog -----------------------------------------------------------
_DE_ALL = ["scaler", "scope", "split", "washout", "horizon"]   # shared by all

MODELS: dict[str, ModelEntry] = {}


def _add(e: ModelEntry):
    MODELS[e.key] = e


# Reservoir / RC benchmarks ---------------------------------------------------
_add(ModelEntry(
    "esn", "ESN", "reservoir", "random recurrent reservoir + ridge readout",
    build=lambda s, c: ReservoirForecaster(ESN(ESNConfig(units=300, seed=s)), "ESN",
                                           external_window=True),
    axes=_DE_ALL + ["lookback", "windowing", "postprocess", "alpha"],
    deterministic=False, base_overrides={"lookback": 3, "windowing": "internal"},
    blurb="An Echo State Network: a fixed random recurrent reservoir whose states "
          "are read out by a trained ridge layer. The canonical classical reservoir "
          "and the primary baseline a QRC must beat."))

_add(ModelEntry(
    "esn_rpy", "ESN (reservoirpy)", "reservoir", "library ESN cross-check",
    build=lambda s, c: ReservoirForecaster(__import__("qdepipe.models.reservoirpy_esn",
                       fromlist=["RPyESN"]).RPyESN(units=300, seed=s), "ESN-rpy",
                       external_window=True),
    axes=_DE_ALL + ["lookback", "windowing", "postprocess", "alpha"],
    deterministic=False, base_overrides={"lookback": 3, "windowing": "internal"},
    blurb="The same ESN method via the standard `reservoirpy` library — a "
          "reproducibility cross-check that findings are not artifacts of our code."))

_add(ModelEntry(
    "ngrc", "NG-RC", "reservoir", "explicit delay + polynomial features + ridge",
    build=lambda s, c: ReservoirForecaster(NGRC(NGRCConfig(k=max(2, c.lookback), degree=2)), "NG-RC"),
    axes=_DE_ALL + ["lookback", "postprocess", "alpha"],
    deterministic=True, base_overrides={"lookback": 3},
    blurb="Next-Generation RC: a deterministic feature map of time-delays plus "
          "their polynomial products. On the Hénon map (itself a degree-2 "
          "polynomial of 2 lags) this is the natural model — and the quantum analog "
          "of a QRC's ZZ correlators."))

_add(ModelEntry(
    "qngrc", "qNG-RC", "quantum", "amplitude-encoded NVAR feature map + ridge",
    build=lambda s, c: ReservoirForecaster(QNGRC(QNGRCConfig(k=max(2, c.lookback), degree=2)), "qNG-RC"),
    axes=_DE_ALL + ["lookback", "postprocess", "alpha"],
    deterministic=True, base_overrides={"lookback": 3},
    blurb="Quantum NG-RC (Wang et al., Phys. Rev. A 111, 022609 (2025), arXiv:2502.16938): the NVAR feature map built "
          "by amplitude-encoding the delay window and taking tensor products, so the "
          "features are amplitude-NORMALISED monomials rather than NG-RC's raw ones. "
          "Exact-expectation idealisation (signs recovered, infinite shots), the same "
          "stance taken for the gate-QRC — it can only overstate the quantum side."))

_add(ModelEntry(
    "elm", "ELM", "reservoir", "random nonlinear projection + ridge",
    build=lambda s, c: ReservoirForecaster(ELM(ELMConfig(units=300, lookback=max(1, c.lookback), seed=s)), "ELM"),
    axes=_DE_ALL + ["lookback", "postprocess", "alpha"],
    deterministic=False, base_overrides={"lookback": 3},
    blurb="Extreme Learning Machine: a single random, untrained nonlinear layer "
          "with a trained ridge readout — the 'random features' control with no "
          "recurrence and no quantum structure."))

_add(ModelEntry(
    "linear", "Linear-Ridge", "reservoir", "linear ridge on lag windows",
    build=lambda s, c: WindowedForecaster(lambda: _ridge(c.alpha), "Linear-Ridge"),
    axes=_DE_ALL + ["lookback", "alpha"],
    deterministic=True, base_overrides={"lookback": 4},
    blurb="A plain linear ridge regression on the lag window — the linear-readout "
          "floor. Cannot represent the map's quadratic term, so it bounds what "
          "linearity alone achieves."))

# Statistical baselines -------------------------------------------------------
_add(ModelEntry(
    "naive", "Naive", "statistical", "persistence",
    build=lambda s, c: NaiveForecaster(),
    axes=["scaler", "scope", "split", "horizon"],
    deterministic=True,
    blurb="Persistence: predict the next value equals the current value. The "
          "zero-parameter reference floor (and scale-invariant — a useful sanity "
          "check that scaling choices don't change a model that ignores them)."))

_add(ModelEntry(
    "ar", "AR", "statistical", "linear autoregression",
    build=lambda s, c: StatsForecaster(kind="ar", order=(2, 0, 0)),
    axes=["scaler", "scope", "split", "washout", "horizon"],
    deterministic=True,
    blurb="A linear AR(2) model. Linear and stochastic — structurally unable to "
          "fit a deterministic quadratic map; sets the linear-stochastic floor."))

_add(ModelEntry(
    "arima", "ARIMA", "statistical", "ARIMA",
    build=lambda s, c: StatsForecaster(kind="arima", order=(2, 0, 1)),
    axes=["scaler", "scope", "split", "washout", "horizon"],
    deterministic=True,
    blurb="ARIMA(2,0,1). The classic Box–Jenkins benchmark; included to confirm "
          "that added MA terms do not rescue a linear model on chaos."))

_add(ModelEntry(
    "theta", "Theta", "statistical", "theta method",
    build=lambda s, c: ThetaForecaster(),
    axes=["scaler", "scope", "split", "horizon"],
    deterministic=True,
    blurb="The Theta method — the classic M3-competition winner; a "
          "drift-adjusted exponential-smoothing baseline."))

_add(ModelEntry(
    "ets", "ETS", "statistical", "exponential smoothing",
    build=lambda s, c: StatsForecaster(kind="ets", trend="add"),
    axes=["scaler", "scope", "split", "horizon"],
    deterministic=True,
    blurb="Exponential smoothing (error+trend). A smoothing baseline; expected to "
          "behave like a lagged persistence on a chaotic signal."))

# Classical ML ----------------------------------------------------------------
def _ml(key, name, fac, det, blurb, standardize=False):
    _add(ModelEntry(
        key, name, "ml", "tabular regressor on lag windows",
        build=lambda s, c, fac=fac, standardize=standardize:
            WindowedForecaster(lambda: fac(s), name, standardize=standardize),
        axes=_DE_ALL + ["lookback"], deterministic=det,
        base_overrides={"lookback": 4}, blurb=blurb))


_ml("random_forest", "RandomForest", _rf, False,
    "A random forest on lag-window features. Trees partition the lag space and "
    "capture the quadratic map well; a strong, easy classical benchmark.")
_ml("hist_gb", "HistGradientBoosting", _histgb, False,
    "Histogram gradient boosting (sklearn). Boosted trees — a top tabular method.")
_ml("xgboost", "XGBoost", _xgb, False,
    "XGBoost gradient-boosted trees — among the most common forecasting benchmarks.")
_ml("lightgbm", "LightGBM", _lgbm, False,
    "LightGBM gradient boosting — fast, widely used boosting benchmark.")
_ml("svr", "SVR", _svr, True,
    "Support-vector regression with an RBF kernel on standardized lag windows.",
    standardize=True)
_ml("knn", "kNN", _knn, True,
    "k-nearest-neighbours regression on standardized lag windows — a local, "
    "non-parametric baseline.", standardize=True)
_ml("elastic_net", "ElasticNet", _elastic, True,
    "Elastic-net linear regression on standardized lag windows — the common "
    "regularised linear baseline (L1+L2).", standardize=True)
_ml("extra_trees", "ExtraTrees", _extratrees, False,
    "Extremely randomised trees on lag windows — the standard companion "
    "ensemble to the random forest.")
_ml("krr", "KernelRidge", _krr, True,
    "Kernel ridge regression (RBF) on standardized lag windows — the closed-"
    "form kernel counterpart to SVR.", standardize=True)

# Deep learning ---------------------------------------------------------------
def _dl(key, name, arch, blurb, hidden=64):
    _add(ModelEntry(
        key, name, "dl", "torch sequence network on lag windows",
        build=lambda s, c, arch=arch, hidden=hidden:
            SequenceForecaster(arch=arch, hidden=hidden, epochs=200, patience=20,
                               seed=s, name=name),
        axes=_DE_ALL + ["lookback"], deterministic=False,
        base_overrides={"lookback": 8}, blurb=blurb))


_dl("mlp", "MLP", "mlp",
    "A feed-forward network on the flattened lag window. The simplest neural "
    "benchmark; learns its own features end-to-end.")
_dl("lstm", "LSTM", "lstm",
    "An LSTM recurrent network — the standard deep sequence-model benchmark.")
_dl("gru", "GRU", "gru",
    "A GRU recurrent network — a lighter gated RNN benchmark.")
_dl("cnn", "1D-CNN/TCN", "cnn",
    "A 1-D convolutional (temporal-convolution) network over the lag window.")
_dl("transformer", "Transformer", "transformer",
    "A small Transformer encoder over the lag window with self-attention.")

# Quantum reservoirs ----------------------------------------------------------
_QRC_AXES = ["scaler", "scope", "split", "washout", "horizon",
             "lookback", "postprocess", "alpha", "encoding"]


def _qrc(key, name, qcfg_fn, blurb):
    _add(ModelEntry(
        key, name, "quantum", "gate-based quantum reservoir + ridge readout",
        build=lambda s, c, fn=qcfg_fn: ReservoirForecaster(GateQRC(fn(s, c)), name),
        axes=_QRC_AXES, deterministic=False,
        base_overrides={"n_points": 1500, "lookback": 5}, blurb=blurb))


_qrc("qrc_v4", "QRC-v4",
     lambda s, c: GateQRCConfig(n_qubits=4, encoding="depth", r=1, coupling="cnot_rz",
                                channel="dephasing", gamma=0.1, V=4,
                                window=max(2, c.lookback), readout=("Z",),
                                encode_scale=c.encode_scale, seed=s),
     "Gate-based QRC: 4 qubits, CNOT+RZ ring with phase-damping, depth encoding, "
     "Z readout (F=16). The validated workhorse (reproduces the prior project's "
     "v4 result); the density-matrix variant with a dissipative channel.")

_qrc("qrc_v6", "QRC-v6",
     lambda s, c: GateQRCConfig(n_qubits=6, encoding="depth", r=1, coupling="isingxx",
                                J_strength=1.0, channel="none", V=4,
                                window=max(2, c.lookback), readout=("Z",),
                                encode_scale=c.encode_scale, seed=s),
     "Gate-based QRC: 6 qubits, IsingXX disorder ring (pure state), depth encoding, "
     "Z readout. Genuine quantum disorder rather than an inert layer.")

_qrc("qrc_rich", "QRC-rich",
     lambda s, c: GateQRCConfig(n_qubits=6, encoding="depth", r=1, coupling="isingxx",
                                J_strength=1.0, channel="none", V=4,
                                window=max(2, c.lookback), readout=("Z", "X", "Y", "ZZ"),
                                encode_scale=c.encode_scale, seed=s),
     "QRC-v6 with a richer readout (Z+X+Y+ZZ ring correlators, F=V·4n). The ZZ "
     "correlators are the quantum analog of NG-RC's quadratic terms — the key "
     "comparison for whether quantum features add anything over classical ones.")


# ---- public helpers --------------------------------------------------------
def build_forecaster(key: str, seed: int = 0, cfg: ExperimentConfig | None = None,
                     store=None) -> Forecaster:
    """Build a model's forecaster honouring the config (so model-internal axes
    like lookback stay comparable). `cfg` defaults to the model's baseline.

    `store`: optional, DEFAULT-OFF FeatureStore. Attached post-hoc to a reservoir
    forecaster (the only family with an expensive featurize) without touching any
    build lambda; store=None leaves the forecaster byte-for-byte unchanged."""
    e = MODELS[key]
    cfg = cfg or base_config(key)
    fc = e.build(seed, cfg)
    if store is not None and isinstance(fc, ReservoirForecaster):
        fc.store = store
    return fc


def base_config(key: str, **overrides) -> ExperimentConfig:
    """Baseline ExperimentConfig for a model (its sensible defaults + overrides)."""
    e = MODELS[key]
    return ExperimentConfig(**{**e.base_overrides, **overrides})


def by_group(group: str) -> list[str]:
    return [k for k, e in MODELS.items() if e.group == group]


GROUPS = ["reservoir", "statistical", "ml", "dl", "quantum"]
