"""Q1.2-Q1.5 — the canonical schema contract.

These fail on schema drift rather than letting it surface as a wrong number
three stages later, which is the whole reason they are written before the
parsers exist.
"""

from __future__ import annotations

#: The core contract. Identical for every dataset — that is what Q1.2's
#: "unified schema" means, and this tuple is the executable form of it.
CORE_ARTICLES = (
    "article_id",
    "title",
    "abstract",
    "body",
    "category",
    "subcategory",
    "published_time",
    "dataset",
)
CORE_IMPRESSIONS = (
    "impression_id",
    "user_id",
    "impression_time",
    "candidate_ids",
    "clicked_ids",
    "split",
    "dataset",
)
CORE_HISTORY = ("user_id", "article_id", "click_time", "history_rank", "dataset")


def test_core_columns_identical_across_datasets(processed_store, dataset):
    """Q1.2 — no extra columns, no missing ones, same order, both datasets."""
    import polars as pl

    for name, expected in (
        ("articles", CORE_ARTICLES),
        ("impressions", CORE_IMPRESSIONS),
        ("history", CORE_HISTORY),
    ):
        got = tuple(pl.read_parquet(processed_store / f"{name}.parquet").columns)
        assert got == expected, f"{dataset}/{name}: {got} != {expected}"


def test_extras_join_one_to_one(processed_store, dataset):
    """Q1.3 — _extra tables join 1:1 on the core key and add no rows."""
    import polars as pl

    for core, extra, key in (
        ("articles", "articles_extra", "article_id"),
        ("impressions", "impressions_extra", "impression_id"),
    ):
        extra_path = processed_store / f"{extra}.parquet"
        if not extra_path.exists():
            continue
        c = pl.read_parquet(processed_store / f"{core}.parquet")
        e = pl.read_parquet(extra_path)
        assert e[key].n_unique() == len(e), f"{extra}: key is not unique"
        assert set(e[key]).issubset(set(c[key])), f"{extra}: orphan keys"


def test_referential_integrity(processed_store, dataset):
    """Q1.4 — article_id unique and non-null; every referenced id exists."""
    import polars as pl

    articles = pl.read_parquet(processed_store / "articles.parquet")
    assert articles["article_id"].null_count() == 0
    assert articles["article_id"].n_unique() == len(articles)

    known = set(articles["article_id"])
    history = pl.read_parquet(processed_store / "history.parquet")
    assert set(history["article_id"]) <= known, "history references unknown articles"

    impressions = pl.read_parquet(processed_store / "impressions.parquet")
    for col in ("candidate_ids", "clicked_ids"):
        seen = set(impressions[col].explode(empty_as_null=False).drop_nulls())
        assert seen <= known, f"impressions.{col} references unknown articles"


def test_timestamps_are_tz_aware_utc(processed_store, dataset):
    """Q1.5 — no naive datetime reaches the store.

    Naive timestamps compare fine and sort fine, so a leak here is invisible
    until two datasets with different implicit zones are compared.
    """
    import polars as pl

    for name in ("articles", "impressions", "history"):
        df = pl.read_parquet(processed_store / f"{name}.parquet")
        for col, dtype in df.schema.items():
            if isinstance(dtype, pl.Datetime):
                assert dtype.time_zone == "UTC", f"{name}.{col} is {dtype}, not UTC"


def test_labels_are_derivable_and_both_classes_present(processed_store, dataset):
    """clicked_ids is a subset of candidate_ids, and neither class is empty."""
    import polars as pl

    df = pl.read_parquet(processed_store / "impressions.parquet")
    n_clicked = df["clicked_ids"].list.len().sum()
    n_candidates = df["candidate_ids"].list.len().sum()
    assert n_clicked > 0, "no positive labels anywhere"
    assert n_candidates > n_clicked, "no negative labels anywhere"
