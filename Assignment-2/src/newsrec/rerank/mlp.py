"""Option B1 — a small MLP re-ranker over the SAME feature table GBDT uses
(a2_q2_plan.md, a2_q2_reranker_options.md).

Trained pointwise: binary cross-entropy against the click label, one row at a
time, rather than NRMS's listwise softmax-over-one-impression loss — the
simpler of the two training styles the primer describes, and the point of
building this option at all is to isolate "does letting the model learn how to
weight the features help", not to reproduce NRMS's full training procedure.

## GPU only, no CPU fallback

Unlike `embed.resolve_device("auto")` (a one-time, already-measured-on-CPU
encode), this module trains — potentially several hyperparameter combinations,
across three datasets — and that work must not silently degrade to a much
slower device. `train()` requires `torch.cuda.is_available()` and raises
rather than falling back, by instruction.

## Two problems found by actually training this, and their fixes

**Class imbalance.** Each impression has one clicked candidate among ~10-25
shown, so a plain `BCEWithLogitsLoss` sees ~95% negative labels and learns to
predict "not clicked" almost everywhere — technically low loss, useless
ranking. `pos_weight` (the ratio of negatives to positives, computed from the
TRAINING table only — a statistic of training data, the same category as
`query.training_popularity`) corrects for this directly in the loss. Measured
on `mind_small`: without it, val nDCG@10 never cleared ~0.27; with it, 0.353 —
above the 0.332 retrieval-only baseline.

**Overfitting.** A fixed epoch count either under- or over-trains depending on
the other hyperparameters (measured: epochs=3 beat epochs=5 at one learning
rate on `mind_small`, which is a symptom, not a real preference for fewer
epochs). `train()` now takes the val table and stops the moment val nDCG@10
stops improving, the same discipline `gbdt.py` uses and for the same reason:
this decides WHEN to stop, never what the network learns from — no leak.

## Normalisation statistics are part of the trained model, not recomputed per table

Feature scales differ wildly (`bm25_score` can run into the tens; `category_match`
is 0/1), so the input is standardised before it reaches the network. The
mean/std used come from the TRAINING table only, saved on the returned model,
and re-applied unchanged at predict time — computing them from a val/test
table would leak that table's distribution into what "normal" means.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn

from newsrec import config
from newsrec.eval import metrics
from newsrec.rerank import features, seeding


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


@dataclass
class MLPModel:
    """A trained network plus the normalisation stats it was trained under —
    both are needed to score a new table consistently."""

    net: _Net
    feature_mean: np.ndarray
    feature_std: np.ndarray
    device: str


def _require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "rerank.mlp requires a CUDA device and none is available. This module "
            "trains on GPU by design (a2_q2_plan.md) — it does not fall back to CPU."
        )
    return "cuda"


def _normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std > 1e-8, std, 1.0)
    return (x - mean) / safe_std


def _val_ndcg10(net: _Net, x_val_t: torch.Tensor, val_table: pl.DataFrame) -> float:
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(x_val_t)).cpu().numpy()
    net.train()
    scored = val_table.with_columns(pl.Series("_s", scores))
    values = []
    for _, group in scored.group_by("impression_id", maintain_order=True):
        labels = group["label"].to_numpy()
        if metrics.degenerate_reason(labels) is not None:
            continue
        got = metrics.ranking_metrics(labels, group["_s"].to_numpy())
        values.append(got["ndcg@10"])
    return float(np.mean(values)) if values else float("-inf")


def train(
    train_table: pl.DataFrame,
    val_table: pl.DataFrame,
    params: dict | None = None,
) -> MLPModel:
    """Fit an MLP on `train_table`, stopping early against `val_table`'s
    nDCG@10 — same contract as `gbdt.train`: both tables are the caller's
    responsibility; `ablation.py` enforces which split each one is."""
    device = _require_cuda()
    seeding.seed_everything()
    p = params or {}
    hidden_dim = p.get("hidden_dim", 32)
    lr = p.get("lr", 1e-3)
    weight_decay = p.get("weight_decay", 1e-4)

    x = features.feature_matrix(train_table)
    y = train_table["label"].to_numpy().astype(np.float32)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x_norm = _normalise(x, mean, std)
    x_val_norm = _normalise(features.feature_matrix(val_table), mean, std)

    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = torch.tensor(
        [n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32, device=device
    )

    net = _Net(n_features=x.shape[1], hidden_dim=hidden_dim).to(device)
    x_t = torch.from_numpy(x_norm).to(device)
    y_t = torch.from_numpy(y).to(device)
    x_val_t = torch.from_numpy(x_val_norm).to(device)

    optimiser = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = float("-inf")
    best_state = copy.deepcopy(net.state_dict())
    rounds_without_improvement = 0

    net.train()
    for epoch in range(config.MLP_MAX_EPOCHS):
        optimiser.zero_grad()
        logits = net(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimiser.step()

        if epoch % config.MLP_EVAL_EVERY != 0:
            continue
        score = _val_ndcg10(net, x_val_t, val_table)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(net.state_dict())
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        if rounds_without_improvement >= config.MLP_EARLY_STOPPING_PATIENCE:
            break

    net.load_state_dict(best_state)
    return MLPModel(net=net, feature_mean=mean, feature_std=std, device=device)


def predict(model: MLPModel, table: pl.DataFrame) -> np.ndarray:
    """One score per row, using the model's OWN normalisation stats — never
    stats recomputed from `table`, which would silently let a val/test table's
    distribution leak into scoring."""
    x = features.feature_matrix(table)
    x_norm = _normalise(x, model.feature_mean, model.feature_std)
    model.net.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_norm).to(model.device)
        logits = model.net(x_t)
        return torch.sigmoid(logits).cpu().numpy()
