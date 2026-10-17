"""Q1.9 — the feature store round-trips faithfully."""

from __future__ import annotations

import pytest


def test_round_trip_preserves_dtypes_and_lists(tmp_path, dataset):
    """List columns are the failure mode worth testing.

    They are why D5 chose polars, and they are what a naive CSV round-trip
    silently flattens into strings.
    """
    import polars as pl

    from newsrec.data import feature_store

    df = pl.DataFrame({"article_id": ["A1", "A2"], "ids": [["X", "Y"], []], "score": [1.5, 2.0]})
    feature_store.write(df, tmp_path / "f.parquet")
    back = feature_store.read(tmp_path / "f.parquet")
    assert back.schema == df.schema
    assert back.to_dicts() == df.to_dicts()


def test_row_order_is_deterministic(processed_store, dataset):
    from newsrec.data import feature_store

    a = feature_store.read(processed_store / "articles.parquet")
    b = feature_store.read(processed_store / "articles.parquet")
    assert a["article_id"].to_list() == b["article_id"].to_list()
    assert a["article_id"].to_list() == sorted(a["article_id"].to_list())


def test_missing_partition_errors_clearly(tmp_path):
    """A missing partition must not read as an empty frame.

    An empty frame propagates as a zero metric, which looks like a result.
    """
    from newsrec.data import feature_store

    with pytest.raises(FileNotFoundError):
        feature_store.read(tmp_path / "does_not_exist.parquet")
