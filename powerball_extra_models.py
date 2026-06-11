#!/usr/bin/env python3
"""
Extra Powerball model families that plug into the per-position bin-prediction
framework defined in ``autoresearch_powerball.py``.

Every model here operates on a single ball position at a time: it consumes a
1-D integer *bin series* (values in ``1..n_bins``) and produces predictions for
the next value. Two prediction surfaces are exposed by every model:

- ``predict_proba(context)`` -> length ``n_bins`` probability vector over bins
  ``1..n_bins`` (used by the evaluation harness for calibration: Brier/log-loss).
- ``predict_point(context)``  -> a single ``float`` bin estimate (used by the
  autoresearch loop, which then rounds it and maps the bin back to a number).

Model families implemented:

- ``MarginalFrequencyModel``    -- honest empirical-frequency baseline.
- ``DirichletModel``            -- Bayesian Dirichlet-multinomial with a prior
                                   and optional recency weighting.
- ``GradientBoostingModel``     -- XGBoost classifier on lagged-bin windows,
                                   GPU-accelerated when ``device='cuda'``.
- ``NeuralSequenceModel``       -- LSTM / GRU / Transformer over the bin
                                   sequence, GPU-accelerated.
- ``EnsembleModel``             -- probability-averaging stack of other models.

torch / xgboost are imported lazily so this module can be imported even in
environments where one of them is missing (the harness simply skips the
corresponding family).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

# XGBoost emits a one-time device-mismatch UserWarning on tiny inputs; harmless.
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

# Keep large model weights (e.g. TimesFM) in a writable, project-local HF cache
# (the default ~/.cache/huggingface may be read-only in this environment).
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _empirical_proba(series_bins: Sequence[int], n_bins: int, smoothing: float = 1e-3) -> np.ndarray:
    """Smoothed empirical bin frequency over ``1..n_bins``."""
    counts = np.bincount(np.asarray(series_bins, dtype=int), minlength=n_bins + 1)[1 : n_bins + 1]
    counts = counts.astype(float) + smoothing
    return counts / counts.sum()


def _make_windows(series_bins: np.ndarray, seq_len: int):
    """Supervised lagged windows: X[i] = bins[i-seq_len:i], y[i] = bins[i]."""
    X: List[np.ndarray] = []
    y: List[int] = []
    for i in range(seq_len, len(series_bins)):
        X.append(series_bins[i - seq_len : i])
        y.append(int(series_bins[i]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _expected_bin(proba: np.ndarray) -> float:
    """Posterior mean bin index (1-based), a smooth point estimate."""
    bins = np.arange(1, len(proba) + 1, dtype=float)
    return float(np.dot(bins, proba))


# --------------------------------------------------------------------------- #
# Baseline / Bayesian models
# --------------------------------------------------------------------------- #
class MarginalFrequencyModel:
    """Predict the next bin from its smoothed historical frequency.

    This is the honest baseline: it captures the order-statistics / bin-imbalance
    structure of the data without any temporal modelling. A "real" model has to
    beat *this*, not uniform-random.
    """

    def __init__(self, n_bins: int = 6, smoothing: float = 1e-3, **_ignored):
        self.n_bins = int(n_bins)
        self.smoothing = float(smoothing)
        self._proba = np.ones(self.n_bins) / self.n_bins

    def fit(self, series_bins: Sequence[int]) -> "MarginalFrequencyModel":
        if len(series_bins) > 0:
            self._proba = _empirical_proba(series_bins, self.n_bins, self.smoothing)
        return self

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        return self._proba

    def predict_point(self, context: Optional[Sequence[int]] = None) -> float:
        return _expected_bin(self._proba)


class DirichletModel:
    """Bayesian Dirichlet-multinomial estimator of the next-bin distribution.

    Posterior over bin probabilities is ``Dirichlet(alpha_prior + weighted_counts)``;
    the posterior predictive for a single draw is the posterior mean. Optional
    exponential recency weighting (``recency_halflife`` draws) lets recent draws
    count for more -- the only place "time" enters this otherwise memoryless model.
    """

    def __init__(self, n_bins: int = 6, alpha_prior: float = 1.0, recency_halflife: float = 0.0, **_ignored):
        self.n_bins = int(n_bins)
        self.alpha_prior = float(alpha_prior)
        self.recency_halflife = float(recency_halflife)
        self._proba = np.ones(self.n_bins) / self.n_bins

    def fit(self, series_bins: Sequence[int]) -> "DirichletModel":
        s = np.asarray(series_bins, dtype=int)
        if len(s) == 0:
            return self
        if self.recency_halflife and self.recency_halflife > 0:
            ages = np.arange(len(s))[::-1]  # most recent draw -> age 0
            weights = 0.5 ** (ages / self.recency_halflife)
        else:
            weights = np.ones(len(s))
        counts = np.zeros(self.n_bins)
        for b, w in zip(s, weights):
            if 1 <= b <= self.n_bins:
                counts[b - 1] += w
        alpha = counts + self.alpha_prior
        self._proba = alpha / alpha.sum()
        return self

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        return self._proba

    def predict_point(self, context: Optional[Sequence[int]] = None) -> float:
        return _expected_bin(self._proba)


# --------------------------------------------------------------------------- #
# Gradient boosting (XGBoost)
# --------------------------------------------------------------------------- #
class GradientBoostingModel:
    """XGBoost multiclass classifier on lagged-bin windows.

    A stronger drop-in for the Random-Forest-on-bins predictor. Uses the GPU
    (``device='cuda'``) when available; XGBoost's ``hist`` tree method falls back
    cleanly to CPU otherwise.
    """

    def __init__(
        self,
        n_bins: int = 6,
        seq_len: int = 15,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        device: str = "cuda",
        random_state: int = 42,
        **_ignored,
    ):
        self.n_bins = int(n_bins)
        self.seq_len = int(seq_len)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.device = device
        self.random_state = int(random_state)
        self._clf = None
        self._fallback = np.ones(self.n_bins) / self.n_bins
        self._fitted_tail: Optional[np.ndarray] = None

    def fit(self, series_bins: Sequence[int]) -> "GradientBoostingModel":
        import xgboost as xgb

        s = np.asarray(series_bins, dtype=int)
        self._fitted_tail = s[-self.seq_len :] if len(s) >= self.seq_len else s
        self._fallback = _empirical_proba(s, self.n_bins) if len(s) else self._fallback
        if len(s) <= self.seq_len + 4:
            self._clf = None
            return self
        X, y = _make_windows(s, self.seq_len)
        self._clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            tree_method="hist",
            device=self.device,
            random_state=self.random_state,
            verbosity=0,
        )
        # XGBClassifier label-encodes y internally; map bins 1..n_bins -> 0-based.
        self._clf.fit(X, y - 1)
        return self

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        if self._clf is None:
            return self._fallback
        if context is None:
            ctx = self._fitted_tail
        else:
            ctx = np.asarray(context, dtype=float)[-self.seq_len :]
        ctx = np.asarray(ctx, dtype=float).reshape(1, -1)
        p = self._clf.predict_proba(ctx)[0]
        full = np.zeros(self.n_bins)
        for cls, pi in zip(self._clf.classes_, p):  # classes_ are 0-based bin indices
            full[int(cls)] = pi
        total = full.sum()
        return full / total if total > 0 else self._fallback

    def predict_point(self, context: Optional[Sequence[int]] = None) -> float:
        return _expected_bin(self.predict_proba(context))


# --------------------------------------------------------------------------- #
# Neural sequence models (LSTM / GRU / Transformer)
# --------------------------------------------------------------------------- #
def _build_seqnet(n_bins, arch, emb, hidden, layers, dropout, max_len):
    import torch.nn as nn

    class _SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = "rnn" if arch in ("lstm", "gru") else "transformer"
            self.embedding = nn.Embedding(n_bins + 1, emb)  # index 0 unused (pad)
            if self.kind == "rnn":
                rnn_cls = nn.LSTM if arch == "lstm" else nn.GRU
                self.encoder = rnn_cls(
                    emb, hidden, num_layers=layers, batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden, n_bins)
            else:
                self.pos = nn.Parameter(0.02 * __import__("torch").randn(1, max_len, emb))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=emb, nhead=4, dim_feedforward=hidden,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
                self.head = nn.Linear(emb, n_bins)

        def forward(self, x):  # x: (B, T) long, bins in 1..n_bins
            h = self.embedding(x)
            if self.kind == "rnn":
                out, _ = self.encoder(h)
            else:
                T = x.size(1)
                h = h + self.pos[:, :T, :]
                out = self.encoder(h)
            return self.head(out[:, -1, :])  # logits over bins (0-based)

    return _SeqNet()


class NeuralSequenceModel:
    """LSTM / GRU / Transformer that predicts the next bin from the bin sequence.

    Trained once on the supplied history; ``predict_proba`` then runs a fixed-weight
    forward pass on the trailing context. As expected for an i.i.d. target, these
    converge toward the marginal distribution -- which is exactly why the harness
    compares them against the marginal-frequency baseline.
    """

    def __init__(
        self,
        n_bins: int = 6,
        seq_len: int = 20,
        arch: str = "lstm",
        emb: int = 16,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 60,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        device: str = "cuda",
        random_state: int = 42,
        **_ignored,
    ):
        self.n_bins = int(n_bins)
        self.seq_len = int(seq_len)
        self.arch = str(arch)
        self.emb = int(emb)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self._net = None
        self._fitted_tail: Optional[np.ndarray] = None
        self._fallback = np.ones(self.n_bins) / self.n_bins
        import torch

        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

    def fit(self, series_bins: Sequence[int]) -> "NeuralSequenceModel":
        import torch
        import torch.nn as nn

        s = np.asarray(series_bins, dtype=int)
        self._fitted_tail = s[-self.seq_len :] if len(s) >= self.seq_len else s
        self._fallback = _empirical_proba(s, self.n_bins) if len(s) else self._fallback
        if len(s) <= self.seq_len + 8:
            self._net = None
            return self

        torch.manual_seed(self.random_state)
        X, y = _make_windows(s, self.seq_len)
        Xt = torch.tensor(X, dtype=torch.long, device=self.device)
        yt = torch.tensor(y - 1, dtype=torch.long, device=self.device)  # 0-based bins

        net = _build_seqnet(
            self.n_bins, self.arch, self.emb, self.hidden, self.layers,
            self.dropout, max_len=max(self.seq_len, 8),
        ).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        net.train()
        n = Xt.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                logits = net(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        self._net = net
        return self

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        if self._net is None:
            return self._fallback
        import torch

        ctx = self._fitted_tail if context is None else np.asarray(context, dtype=int)[-self.seq_len :]
        ctx = np.asarray(ctx, dtype=int)
        if len(ctx) < self.seq_len:  # left-pad with the earliest value
            ctx = np.concatenate([np.full(self.seq_len - len(ctx), ctx[0] if len(ctx) else 1), ctx])
        with torch.no_grad():
            x = torch.tensor(ctx.reshape(1, -1), dtype=torch.long, device=self.device)
            p = torch.softmax(self._net(x), dim=-1).cpu().numpy()[0]
        return p

    def predict_point(self, context: Optional[Sequence[int]] = None) -> float:
        return _expected_bin(self.predict_proba(context))


# --------------------------------------------------------------------------- #
# Stacking ensemble
# --------------------------------------------------------------------------- #
class EnsembleModel:
    """Probability-averaging stack over already-fitted member models.

    Members must each expose ``predict_proba(context)``. Weights default to equal.
    The ensemble's point estimate is the posterior mean of the averaged
    distribution -- a simple, robust stack of the base families.
    """

    def __init__(self, members: List[object], weights: Optional[Sequence[float]] = None, n_bins: int = 6, **_ignored):
        self.members = list(members)
        self.n_bins = int(n_bins)
        if weights is None:
            weights = np.ones(len(self.members))
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def fit(self, series_bins: Sequence[int]) -> "EnsembleModel":
        for m in self.members:
            m.fit(series_bins)
        return self

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        acc = np.zeros(self.n_bins)
        for w, m in zip(self.weights, self.members):
            p = np.asarray(m.predict_proba(context), dtype=float)
            if len(p) == self.n_bins:
                acc += w * p
        total = acc.sum()
        return acc / total if total > 0 else np.ones(self.n_bins) / self.n_bins

    def predict_point(self, context: Optional[Sequence[int]] = None) -> float:
        return _expected_bin(self.predict_proba(context))


# --------------------------------------------------------------------------- #
# TimesFM 2.5 foundation forecaster (google/timesfm-2.5-200m-pytorch)
# --------------------------------------------------------------------------- #
_TIMESFM_CACHE: Dict[int, object] = {}


def get_timesfm(max_context: int = 2048, max_horizon: int = 16):
    """Load and compile the TimesFM 2.5 200M model once, then reuse it.

    Zero-shot foundation forecaster -- no fitting required. Cached per
    ``max_context`` so repeated calls are cheap. Raises ImportError if the
    ``timesfm`` package is unavailable.
    """
    key = int(max_context)
    if key not in _TIMESFM_CACHE:
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=int(max_context),
                max_horizon=int(max_horizon),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        _TIMESFM_CACHE[key] = model
    return _TIMESFM_CACHE[key]


def timesfm_forecast(series_list: List[np.ndarray], horizon: int = 1, max_context: int = 2048):
    """Batched next-step forecast for several univariate series.

    Returns ``(point, quantiles)`` where ``point`` is (n_series, horizon) and
    ``quantiles`` is (n_series, horizon, 10): index 0 is the mean, 1..9 are the
    0.1..0.9 deciles (in the series' own value space).
    """
    model = get_timesfm(max_context=max_context, max_horizon=max(horizon, 16))
    inputs = [np.asarray(s, dtype=float) for s in series_list]
    return model.forecast(horizon=horizon, inputs=inputs)


class TimesFMModel:
    """Per-position adapter for TimesFM on a single ball's *bin* series.

    Forecasts the next bin value zero-shot; ``predict_proba`` turns the model's
    decile forecast into a histogram over bins ``1..n_bins`` (wide deciles ->
    diffuse distribution, which is the honest outcome on i.i.d. lottery data).
    """

    def __init__(self, n_bins: int = 6, context_len: int = 512, max_context: int = 2048,
                 smoothing: float = 0.25, **_ignored):
        self.n_bins = int(n_bins)
        self.context_len = int(context_len)
        self.max_context = int(max_context)
        self.smoothing = float(smoothing)
        self._series = np.array([float((self.n_bins + 1) / 2)])
        self._cache_key = None
        self._cache_val = None

    def fit(self, series_bins: Sequence[int]) -> "TimesFMModel":
        self._series = np.asarray(series_bins, dtype=float)
        self._cache_key = None
        return self

    def _run(self, context: Optional[Sequence[int]]):
        s = self._series if context is None else np.asarray(context, dtype=float)
        s = s[-self.context_len :]
        key = (len(s), float(s[-1]) if len(s) else 0.0)
        if key == self._cache_key:
            return self._cache_val
        pf, qf = timesfm_forecast([s], horizon=1, max_context=self.max_context)
        point = float(pf[0, 0])
        deciles = np.asarray(qf[0, 0, 1:], dtype=float)  # 9 deciles in bin space
        samples = np.clip(np.round(deciles), 1, self.n_bins).astype(int)
        counts = np.bincount(samples, minlength=self.n_bins + 1)[1 : self.n_bins + 1].astype(float)
        counts += self.smoothing
        proba = counts / counts.sum()
        self._cache_key, self._cache_val = key, (point, proba)
        return self._cache_val

    def predict_proba(self, context: Optional[Sequence[int]] = None) -> np.ndarray:
        return self._run(context)[1]

    def predict_point(self, context: Optional[Sequence[int]] = None) -> float:
        return float(np.clip(self._run(context)[0], 1, self.n_bins))


# Registry of constructors keyed by the model-family name used across the project.
MODEL_REGISTRY = {
    "marginal": MarginalFrequencyModel,
    "dirichlet": DirichletModel,
    "gradient_boosting": GradientBoostingModel,
    "neural": NeuralSequenceModel,
    "timesfm": TimesFMModel,
}


def make_model(family: str, **params):
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: {family}")
    return MODEL_REGISTRY[family](**params)
