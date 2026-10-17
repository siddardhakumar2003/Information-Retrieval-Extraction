"""Option A — a GBDT re-ranker over the hand-crafted feature table
(a2_q2_plan.md, a2_q2_reranker_options.md).

Trained with LightGBM's `lambdarank` objective: each impression is one
"group," and the loss is defined entirely in terms of the ordering WITHIN a
group, never across two different impressions' candidates — comparing a
candidate from one impression to a candidate from another is meaningless (a
different user, a different candidate pool).

Trains on the CPU with a fixed seed and `deterministic=True`, so a retrain gives
identical trees. It started on LightGBM's OpenCL GPU backend, but that backend
gave different trees on identical runs (measured on both EB-NeRD datasets), which
made every GBDT verdict unrepeatable. See `config.GBDT_FIXED_PARAMS`.

## Early stopping on the val set is why this module takes one

The first version trained a fixed number of trees and overfit badly: measured
on `mind_small`, train nDCG@10 reached 0.66 while val scored 0.30 — WORSE than
the 0.33 the retrieval-only baseline gets without any training at all. Handing
`train()` the val table so LightGBM can stop the moment val nDCG@10 stops
improving fixed this (val nDCG@10 rose to 0.34, above the baseline) without
touching a single feature. This is standard practice, not a leak: early
stopping only decides WHEN to stop adding trees, it never lets a val row's
label or feature values influence which split a tree makes.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from newsrec import config
from newsrec.rerank import features, seeding


def train(
    train_table: pl.DataFrame, val_table: pl.DataFrame, params: dict | None = None
) -> lgb.Booster:
    """Fit a `lambdarank` model on `train_table`, stopping early against
    `val_table`'s nDCG@10. Both tables are the CALLER's responsibility to have
    built from the right split — this function trusts its input completely,
    which is why `ablation.py`, not this module, is where the split discipline
    is enforced and tested.
    """
    seeding.seed_everything()
    merged = dict(config.GBDT_FIXED_PARAMS)
    merged.update(params or {})

    x_train = features.feature_matrix(train_table)
    y_train = train_table["label"].to_numpy()
    groups_train = features.group_sizes(train_table)
    train_ds = lgb.Dataset(
        x_train, label=y_train, group=groups_train, feature_name=list(features.FEATURE_COLUMNS)
    )

    x_val = features.feature_matrix(val_table)
    y_val = val_table["label"].to_numpy()
    groups_val = features.group_sizes(val_table)
    val_ds = lgb.Dataset(x_val, label=y_val, group=groups_val, reference=train_ds)

    return lgb.train(
        merged,
        train_ds,
        num_boost_round=config.GBDT_MAX_ROUNDS,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(config.GBDT_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )


def predict(model: lgb.Booster, table: pl.DataFrame) -> np.ndarray:
    """One score per row, from `FEATURE_COLUMNS` only — `feature_matrix` is the
    single place that carve-out happens, so this function cannot accidentally
    read the `label` column even if the caller's table still has it."""
    return model.predict(features.feature_matrix(table), num_iteration=model.best_iteration)
