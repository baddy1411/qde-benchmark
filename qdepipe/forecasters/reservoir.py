"""Reservoir forecaster — nonlinear feature map + trained linear (ridge) readout.

Wraps any `ReservoirModel` (ESN, NG-RC, ELM, QRC). The only trained parameters
are the ridge weights, so a comparison between two reservoir forecasters is a
comparison of their *feature maps* under an identical readout — which is exactly
the classical-vs-quantum question, posed fairly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..experiment import Forecaster, RunOutput, ExperimentConfig
from ..models.base import ReservoirModel
from ..pipeline import Identity, compose
from ..pipeline.base import Transformer
from ..pipeline.embedding import delay_embedding, supervised_pairs
from ..readout import ridge_fit, ridge_predict


class ReservoirForecaster(Forecaster):
    def __init__(self, reservoir: ReservoirModel, name: Optional[str] = None,
                 external_window: bool = False, store=None):
        """`external_window=True` makes the reservoir respond to the shared
        windowing axis (the runner feeds it delay vectors instead of the scalar
        series). NG-RC / ELM build their own delays, so they keep it False.

        `store`: optional, DEFAULT-OFF FeatureStore. It caches the ONE-STEP
        featurize in `run()` only; closed-loop rollout (`onestep_predictor`) is
        deliberately never cached — see the note there."""
        self.reservoir = reservoir
        self.name = name or getattr(reservoir, "name", "reservoir")
        self.external_window = external_window
        self.store = store

    def run(self, u: np.ndarray, split, cfg: ExperimentConfig) -> RunOutput:
        from ..feature_store import maybe_cached_featurize
        # `ext` decides both the input AND the cache key (external_window arg), so
        # they can never disagree. `inp` is determined by cfg, and cfg here is the
        # SAME object run_experiment used to scale u (verified), so the key from
        # (reservoir, cfg) identifies exactly this matrix. store=None -> byte-for-
        # byte self.reservoir.featurize(inp).
        ext = self.external_window and cfg.windowing == "shared"
        inp = delay_embedding(u, cfg.lookback, cfg.stride) if ext else u

        feats = maybe_cached_featurize(self.store, self.reservoir, inp, cfg, external_window=ext)

        post: Transformer = compose(*cfg.postprocess) if cfg.postprocess else Identity()
        post.fit(feats[split.train_fit])
        feats = post.transform(feats)

        X, y = supervised_pairs(feats, u, cfg.horizon)
        a, b, w = split.train.stop, split.test.start, split.washout
        tr = slice(w, a - cfg.horizon)
        te = slice(b, len(X))

        W = ridge_fit(X[tr], y[tr], alpha=cfg.alpha, bias=True)
        y_pred = ridge_predict(X[te], W, bias=True).ravel()

        return RunOutput(
            y_true=y[te], y_pred=y_pred,
            n_features=feats.shape[1], n_params=int(W.size),
            n_train=X[tr].shape[0], features=X[tr],
        )

    def _featurize(self, series, cfg):
        if self.external_window and cfg.windowing == "shared":
            return self.reservoir.featurize(delay_embedding(series, cfg.lookback, cfg.stride))
        return self.reservoir.featurize(series)

    def onestep_predictor(self, u, split, cfg):
        # NOT cached, by design. The rollout closure below featurizes a different
        # `history` suffix each step under a constant (reservoir, cfg) — the cache
        # key would be constant while the input varies, so a store here would alias
        # every step onto the first. The store's precondition ("inp is determined by
        # cfg") holds in run() but NOT in rollout, so onestep_predictor stays raw.
        from ..experiment import ExperimentConfig
        c = ExperimentConfig(**{**cfg.__dict__, "horizon": 1})
        feats = self._featurize(u, c)
        post: Transformer = compose(*c.postprocess) if c.postprocess else Identity()
        post.fit(feats[split.train_fit])
        feats = post.transform(feats)
        X, y = supervised_pairs(feats, u, 1)
        tr = slice(split.washout, split.train.stop - 1)
        W = ridge_fit(X[tr], y[tr], alpha=c.alpha, bias=True)

        # finite-memory reservoirs (QRC, NG-RC, ELM) only need a short suffix to
        # reproduce the last feature row → cheap rollout; ESN (unbounded) uses all.
        mw = getattr(self.reservoir, "memory_window", None)

        def predict(history):
            h = np.asarray(history, dtype=float)
            if mw is not None:
                h = h[-max(mw, c.lookback):]
            f = post.transform(self._featurize(h, c))
            return float(ridge_predict(f[-1][None], W, bias=True).ravel()[0])

        return predict
