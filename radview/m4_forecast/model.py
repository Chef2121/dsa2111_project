"""M4 - PyTorch sequence model for rating-decline forecasting.

``DeclineGRU`` reads the last ``SEQ_LEN`` months of a restaurant's feature
vectors (rating trend, volume, overall + per-aspect complaint signals from M2/M3)
and outputs the probability that the rating will fall by >= DECLINE_THRESHOLD
stars over the next HORIZON_MONTHS months.

Why a sequence model: a gradient-boosted tree only sees hand-made window
features at month t; the GRU sees *how* each aspect's complaint rate evolved
month by month, which is exactly the "emerging problem" pattern we want to
catch early. LightGBM on the same features is kept as a baseline.

``SequenceData`` stores every business's months in one padded array and builds
(batch, SEQ_LEN, F) windows on the fly by index arithmetic, so memory stays
O(rows x features) rather than O(rows x SEQ_LEN x features).
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from radview.config import DEVICE as _DEVICE_SETTING
from radview.config import RANDOM_SEED

SEQ_LEN = 12
# CPU by default; RADVIEW_DEVICE=cuda opts in to a GPU when one is available.
DEVICE = torch.device("cuda" if _DEVICE_SETTING == "cuda" and torch.cuda.is_available() else "cpu")


class SequenceData:
    """Standardised, left-padded per-business feature sequences."""

    def __init__(self, frame: pd.DataFrame, features: list[str], mean: pd.Series, std: pd.Series):
        """Standardise ``features`` with the training mean/std (NaN -> 0, clipped to +-5), add a padding-flag
        channel, and store each restaurant's months left-padded so any month's 12-month window is one slice.
        """
        self.features = features
        self.mean, self.std = mean, std  # training statistics, reused for counterfactual frames
        z = ((frame[features] - mean) / std).fillna(0.0).clip(-5, 5).to_numpy(np.float32)
        z = np.concatenate([z, np.zeros((len(z), 1), np.float32)], axis=1)  # last channel = padding flag
        pad = np.zeros((SEQ_LEN - 1, z.shape[1]), np.float32)
        pad[:, -1] = 1.0
        blocks, pos = [], np.empty(len(frame), dtype=np.int64)
        offset = 0
        biz = frame["business_id"].to_numpy()
        # frame must be sorted by business, month
        starts = np.flatnonzero(np.r_[True, biz[1:] != biz[:-1]])
        ends = np.r_[starts[1:], len(frame)]
        for s, e in zip(starts, ends):
            blocks += [pad, z[s:e]]
            pos[s:e] = offset + SEQ_LEN - 1 + np.arange(e - s)
            offset += SEQ_LEN - 1 + (e - s)
        # The whole (rows x features) matrix is small, so it lives on the training device and
        # windows are gathered there without host<->GPU copies per batch.
        self.X = torch.from_numpy(np.concatenate(blocks)).to(DEVICE)
        self.pos = torch.from_numpy(pos).to(DEVICE)
        self._steps = torch.arange(-SEQ_LEN + 1, 1, device=DEVICE)

    @property
    def n_inputs(self) -> int:
        """Number of input channels (features + padding flag)."""
        return self.X.shape[1]

    def windows(self, rows: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return (len(rows), SEQ_LEN, F) input windows ending at the given row indices."""
        idx = self.pos[torch.as_tensor(rows, device=DEVICE)][:, None] + self._steps[None, :]
        return self.X[idx]


class DeclineGRU(nn.Module):
    """Linear projection (GELU, dropout) -> one-layer GRU -> small MLP head on the last hidden state.
    Outputs one decline logit per window.
    """
    def __init__(self, n_inputs: int, hidden: int = 64, dropout: float = 0.2):
        """``n_inputs`` channels in, ``hidden`` units in the projection and the GRU."""
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_inputs, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F) -> logits (B, 1)
        """x of shape (batch, months, channels) -> decline logit (batch, 1) from the final month's hidden state."""
        h, _ = self.gru(self.proj(x))
        return self.head(h[:, -1])


def predict(model: DeclineGRU, data: SequenceData, rows: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Uncalibrated decline probabilities (sigmoid of the logits) for the windows ending at ``rows``."""
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(rows), batch):
            out.append(torch.sigmoid(model(data.windows(rows[s:s + batch]))).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_gru(data: SequenceData, y: np.ndarray, train_rows: np.ndarray, valid_rows: np.ndarray,
              epochs: int = 30, batch: int = 256, lr: float = 2e-3, patience: int = 5, seed: int = RANDOM_SEED,
              verbose: bool = True) -> tuple[DeclineGRU, dict]:
    """Train with class-weighted BCE; early-stop on validation PR-AUC."""
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng = np.random.default_rng(seed)
    model = DeclineGRU(data.n_inputs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos_rate = float(y[train_rows].mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos_rate) / max(pos_rate, 1e-3), device=DEVICE))
    y_t = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_rows)
        total = 0.0
        for s in range(0, len(order), batch):
            rows = order[s:s + batch]
            logits = model(data.windows(rows)).squeeze(1)
            loss = loss_fn(logits, y_t[torch.as_tensor(rows, device=DEVICE)])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(rows)
        ap = average_precision_score(y[valid_rows], predict(model, data, valid_rows))
        history.append({"epoch": ep + 1, "train_loss": round(total / len(order), 4), "valid_pr_auc": round(float(ap), 4)})
        if verbose:
            print(f"  [M4-GRU] epoch {ep + 1:2d} loss {total / len(order):.4f} valid PR-AUC {ap:.4f}")
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_valid_pr_auc": round(float(best), 4), "history": history}


def explain(model: DeclineGRU, data: SequenceData, rows: np.ndarray, background_rows: np.ndarray,
            n_background: int = 200, seed: int = RANDOM_SEED) -> np.ndarray:
    """SHAP values (expected gradients) per input feature, summed over the time axis.

    Returns an array (len(rows), n_features) in *logit* space. Uses
    ``shap.GradientExplainer`` (the SHAP approximation for differentiable
    models); falls back to integrated gradients if shap is unavailable.
    """
    rng = np.random.default_rng(seed)
    bg_rows = rng.choice(background_rows, min(n_background, len(background_rows)), replace=False)
    # Explaining a few hundred rows is cheap, so do it on a CPU copy (shap + autograd are simplest there).
    model = copy.deepcopy(model).cpu().eval()
    x = data.windows(rows).cpu()
    bg = data.windows(bg_rows).cpu()
    try:
        import shap

        explainer = shap.GradientExplainer(model, bg)
        vals = explainer.shap_values(x, nsamples=100, rseed=seed)
        vals = np.asarray(vals[0] if isinstance(vals, list) else vals)
        vals = vals.reshape(x.shape)  # drop trailing output dim if present
    except Exception:  # pragma: no cover - fallback path
        base = bg.mean(0, keepdim=True)
        alphas = torch.linspace(0, 1, 32)
        grads = torch.zeros_like(x)
        for a in alphas:
            xi = (base + a * (x - base)).requires_grad_(True)
            model(xi).sum().backward()
            grads += xi.grad / len(alphas)
        vals = ((x - base) * grads).detach().numpy()
    return vals.sum(axis=1)[:, :-1]  # sum over time, drop padding channel


class GRUEnsemble(nn.Module):
    """Average of several GRUs trained with different seeds (lower variance)."""

    def __init__(self, models: list[DeclineGRU]):
        """Wrap the seed models so SHAP and prediction see a single network."""
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average of the members' logits."""
        return torch.stack([m(x) for m in self.models]).mean(0)


def fit_platt(p_valid: np.ndarray, y_valid: np.ndarray) -> tuple[float, float]:
    """Platt scaling on the validation set.

    Class weighting during training inflates raw probabilities; a 1-D logistic
    regression on the logit maps them back to observed decline frequencies
    (monotone, so AUROC / PR-AUC are unchanged).
    """
    from sklearn.linear_model import LogisticRegression

    logit = np.log(np.clip(p_valid, 1e-6, 1 - 1e-6) / np.clip(1 - p_valid, 1e-6, 1)).reshape(-1, 1)
    lr = LogisticRegression(C=1e4).fit(logit, y_valid)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p: np.ndarray, calib: tuple[float, float]) -> np.ndarray:
    """Platt scaling: sigmoid(a * logit(p) + b), with (a, b) fitted on the validation months."""
    a, b = calib
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))
