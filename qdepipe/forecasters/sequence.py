"""Sequence-model forecasters (torch, GPU) — the modern deep-learning benchmarks.

One class covers MLP, LSTM, GRU, 1D-CNN/TCN and a small Transformer encoder, all
trained on the same causal lag windows the other models use, with early stopping
on a held-out tail of the training split (never on test). These are the
end-to-end-trained counterparts to the fixed-feature reservoirs: they get to
learn their representation, so they set the strong-classical bar a QRC is
measured against.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..experiment import Forecaster, RunOutput, ExperimentConfig, windowed_xy, last_window
from ..device import get_device


# ---- architectures ---------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, L, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):           # x: (N, L)
        return self.net(x).squeeze(-1)


class _RNN(nn.Module):
    def __init__(self, hidden, layers, kind):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):           # x: (N, L)
        out, _ = self.rnn(x.unsqueeze(-1))      # (N, L, H)
        return self.head(out[:, -1]).squeeze(-1)


class _CNN(nn.Module):
    def __init__(self, L, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=min(3, L), padding="same"), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=min(3, L), padding="same"), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):           # x: (N, L)
        h = self.net(x.unsqueeze(1)).squeeze(-1)   # (N, hidden)
        return self.head(h).squeeze(-1)


class _Transformer(nn.Module):
    def __init__(self, L, d_model=32, nhead=4, layers=1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):           # x: (N, L)
        h = self.proj(x.unsqueeze(-1)) + self.pos
        h = self.enc(h)
        return self.head(h[:, -1]).squeeze(-1)


def _build(arch, L, hidden, layers):
    if arch == "mlp":
        return _MLP(L, hidden)
    if arch in ("lstm", "gru"):
        return _RNN(hidden, layers, arch)
    if arch in ("cnn", "tcn"):
        return _CNN(L, hidden)
    if arch == "transformer":
        return _Transformer(L)
    raise ValueError(f"unknown arch {arch!r}")


class SequenceForecaster(Forecaster):
    def __init__(self, arch="lstm", hidden=64, layers=1, epochs=200, lr=1e-3,
                 patience=20, val_frac=0.2, batch=256, seed=42, name=None,
                 require_cuda=False):
        self.arch = arch
        self.hidden, self.layers = hidden, layers
        self.epochs, self.lr, self.patience = epochs, lr, patience
        self.val_frac, self.batch, self.seed = val_frac, batch, seed
        self.require_cuda = require_cuda
        self.name = name or arch.upper()

    def _fit(self, Xtr, ytr, L, dev):
        """Train the net with early stopping on a tail of the training rows."""
        torch.manual_seed(self.seed)
        n_val = max(1, int(self.val_frac * len(Xtr)))
        Xv, yv = Xtr[-n_val:], ytr[-n_val:]
        Xt, yt = Xtr[:-n_val], ytr[:-n_val]

        def to(a):
            return torch.as_tensor(np.asarray(a, np.float32), device=dev)

        Xt_, yt_, Xv_, yv_ = to(Xt), to(yt), to(Xv), to(yv)
        model = _build(self.arch, L, self.hidden, self.layers).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best, best_state, bad = np.inf, None, 0
        n = Xt_.shape[0]
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss_fn(model(Xt_[idx]), yt_[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(Xv_), yv_).item()
            if vloss < best - 1e-7:
                best = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, best, len(yt)

    def run(self, u: np.ndarray, split, cfg: ExperimentConfig) -> RunOutput:
        dev = get_device(require_cuda=self.require_cuda)
        Xtr, ytr, Xte, yte = windowed_xy(u, split, cfg)
        model, best, n_train = self._fit(Xtr, ytr, cfg.lookback, dev)
        with torch.no_grad():
            y_pred = model(torch.as_tensor(np.asarray(Xte, np.float32), device=dev)).cpu().numpy().ravel()
        return RunOutput(
            y_true=yte, y_pred=y_pred,
            n_features=cfg.lookback, n_params=int(sum(p.numel() for p in model.parameters())),
            n_train=n_train, features=None,
            extra={"val_loss": best, "arch": self.arch},
        )

    def onestep_predictor(self, u, split, cfg):
        c = ExperimentConfig(**{**cfg.__dict__, "horizon": 1})
        dev = get_device(require_cuda=self.require_cuda)
        Xtr, ytr, _, _ = windowed_xy(u, split, c)
        model, _, _ = self._fit(Xtr, ytr, cfg.lookback, dev)
        L, st = cfg.lookback, cfg.stride

        def predict(history):
            x = last_window(history, L, st).astype(np.float32)
            with torch.no_grad():
                return float(model(torch.as_tensor(x[None], device=dev)).cpu().numpy().ravel()[0])

        return predict
