"""A2 Q2 — the re-ranker's feature table, models, and the split discipline.

Same shape as `test_leakage.py`: fast canaries on small/synthetic input prove
the split/leakage rules can fail; the real-data test proves they hold on an
actual built store. A well-formed score built from the wrong split's rows
looks exactly like a real number.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.rerank import features, gbdt, mlp


def _table(rows: list[dict]) -> pl.DataFrame:
    schema = (
        {name: pl.Utf8 for name in features.ID_COLUMNS}
        | {"label": pl.Int64}
        | {name: pl.Float64 for name in features.FEATURE_COLUMNS}
    )
    return pl.DataFrame(rows, schema=schema, orient="row")


def _row(impression_id: str, article_id: str, label: int, **overrides) -> dict:
    base = {
        "impression_id": impression_id,
        "article_id": article_id,
        "user_id": "u1",
        "dataset": "d",
        "split": "train",
        "label": label,
        "bm25_score": 1.0,
        "semantic_score": 0.5,
        "n_clicks_in_history": 5.0,
        "recency_weighted_click_count": 3.0,
        "popularity_train": 2.0,
        "has_published_time": 1.0,
        "session_click_count": 0.0,
        "position_in_session": 1.0,
        "category_match": 0.0,
    }
    base.update(overrides)
    return base


# =============================================================================
# feature_matrix / group_sizes — the boundary between "row" and "model input"
# =============================================================================


def test_feature_matrix_excludes_ids_and_label():
    table = _table(
        [
            _row("i1", "a1", 1, bm25_score=9.0),
            _row("i1", "a2", 0, bm25_score=1.0),
        ]
    )
    x = features.feature_matrix(table)
    assert x.shape == (2, len(features.FEATURE_COLUMNS))
    # bm25_score is the first declared feature column; a label leak would
    # silently add or shift a column, not just contain a wrong value.
    assert x[0, features.FEATURE_COLUMNS.index("bm25_score")] == 9.0


def test_group_sizes_matches_candidates_per_impression():
    table = _table(
        [
            _row("i1", "a1", 1),
            _row("i1", "a2", 0),
            _row("i1", "a3", 0),
            _row("i2", "a4", 1),
        ]
    )
    assert features.group_sizes(table) == [3, 1]


# =============================================================================
# CANARY — predicting must never depend on the label column being present
# =============================================================================


def _skip_without_gpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("gpu marker: no CUDA device on this machine")


@pytest.mark.gpu
def test_gbdt_predict_ignores_a_label_column_left_in_the_table(monkeypatch):
    """A model that accidentally read `label` as a feature would need it
    present to score correctly; dropping the column and re-scoring must give
    identical predictions if `feature_matrix` really is the only input."""
    _skip_without_gpu()
    monkeypatch.setattr(config, "GBDT_MAX_ROUNDS", 5)
    monkeypatch.setattr(config, "GBDT_EARLY_STOPPING_ROUNDS", 5)
    rng = np.random.default_rng(0)
    n = 60
    rows = []
    for i in range(n // 3):
        for j in range(3):
            label = 1 if j == 0 else 0
            rows.append(
                _row(
                    f"i{i}",
                    f"a{i}_{j}",
                    label,
                    bm25_score=float(rng.normal()),
                    semantic_score=float(rng.normal()),
                )
            )
    table = _table(rows)
    model = gbdt.train(table, table, params={"num_leaves": 7})

    with_label = gbdt.predict(model, table)
    without_label = gbdt.predict(model, table.drop("label"))
    assert np.allclose(with_label, without_label)


@pytest.mark.gpu
def test_mlp_predict_ignores_a_label_column_left_in_the_table(monkeypatch):
    _skip_without_gpu()
    monkeypatch.setattr(config, "MLP_MAX_EPOCHS", 3)
    rng = np.random.default_rng(0)
    rows = [_row(f"i{i}", f"a{i}", i % 2, bm25_score=float(rng.normal())) for i in range(20)]
    table = _table(rows)
    model = mlp.train(table, table, params={"hidden_dim": 8, "lr": 1e-2})

    with_label = mlp.predict(model, table)
    without_label = mlp.predict(model, table.drop("label"))
    assert np.allclose(with_label, without_label)


def test_mlp_requires_cuda_and_does_not_fall_back_to_cpu(monkeypatch):
    """No GPU needed to run this one — it checks the REFUSAL, not training."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    table = _table([_row("i1", "a1", 1)])
    with pytest.raises(RuntimeError, match="CUDA"):
        mlp.train(table, table)


# =============================================================================
# Real-data: the split discipline (A2 Q2's own leakage risk)
# =============================================================================


@pytest.mark.realdata
@pytest.mark.slow
def test_train_table_contains_only_train_split_impressions(dataset):
    """The risk this stage adds on top of A1/A2 Q1's per-row leakage boundary:
    which ROWS end up in the training table, not what any single feature
    value contains."""
    if not config.PROCESSED_DIR.is_dir():
        pytest.skip("--realdata given but data/processed/ does not exist; run `make data`")
    if not (config.ARTIFACTS_DIR / "embeddings" / dataset).is_dir():
        pytest.skip("no embeddings artefact for this dataset; run `make embeddings`")

    import polars as pl

    from newsrec.data import feature_store

    impressions = feature_store.read_table(config.PROCESSED_DIR / dataset / "impressions.parquet")
    train_ids = set(impressions.filter(pl.col("split") == "train")["impression_id"].to_list())

    table = features.build_candidate_table(dataset, "train", size=100)
    assert len(table) > 0
    assert set(table["impression_id"].to_list()) <= train_ids
    assert set(table["split"].to_list()) == {"train"}


@pytest.mark.realdata
@pytest.mark.slow
def test_train_val_test_tables_are_disjoint(dataset):
    if not config.PROCESSED_DIR.is_dir():
        pytest.skip("--realdata given but data/processed/ does not exist; run `make data`")
    if not (config.ARTIFACTS_DIR / "embeddings" / dataset).is_dir():
        pytest.skip("no embeddings artefact for this dataset; run `make embeddings`")

    train = features.build_candidate_table(dataset, "train", size=50)
    val = features.build_candidate_table(dataset, "val", size=50)
    test = features.build_candidate_table(dataset, "test", size=50)

    train_ids = set(train["impression_id"].to_list())
    val_ids = set(val["impression_id"].to_list())
    test_ids = set(test["impression_id"].to_list())
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
