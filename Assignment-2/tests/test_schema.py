"""Unit tests for the canonical schema contract (Q1.2-Q1.5, Q9.3).

These exercise `schema.py` directly with hand-built frames, so they run today —
the store-level assertions in `test_schema_contracts.py` stay skipped until a
processed store exists.

Both directions throughout. A validator that rejects everything passes every
"does it catch X?" test while making the pipeline unbuildable, so each rule is
checked with a conforming frame AND a violating one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from newsrec import config
from newsrec.data import schema


def _articles(**overrides) -> pl.DataFrame:
    base = {
        "article_id": ["A1", "A2"],
        "title": ["First", "Second"],
        "abstract": ["a", "b"],
        "body": [None, "text"],
        "category": ["news", "sport"],
        "subcategory": ["world", "football"],
        "published_time": [datetime(2023, 5, 1, tzinfo=UTC)] * 2,
        "dataset": ["t", "t"],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema=dict(schema.ARTICLES.columns))


def _impressions(**overrides) -> pl.DataFrame:
    base = {
        "impression_id": ["I1", "I2"],
        "user_id": ["U1", "U2"],
        "impression_time": [datetime(2023, 5, 2, tzinfo=UTC)] * 2,
        "candidate_ids": [["A1", "A2"], ["A2"]],
        "clicked_ids": [["A1"], []],
        "split": ["train", "train"],
        "dataset": ["t", "t"],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema=dict(schema.IMPRESSIONS.columns))


def _history(**overrides) -> pl.DataFrame:
    base = {
        "user_id": ["U1", "U1"],
        "article_id": ["A1", "A2"],
        "click_time": [datetime(2023, 5, 1, tzinfo=UTC), None],
        # Both nullable: EB-NeRD populates click_time and leaves history_rank
        # null, MIND does the reverse. One row of each shape.
        "history_rank": [None, 0],
        "dataset": ["t", "t"],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema=dict(schema.HISTORY.columns))


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def test_core_tables_share_no_dataset_specific_columns():
    """'Unified schema' means the column list does not depend on the dataset."""
    for spec in schema.CORE_TABLES.values():
        assert "dataset" in spec.names, f"{spec.name} must carry its dataset key"
        for name in spec.names:
            assert not name.startswith(("mind_", "ebnerd_")), (
                f"{spec.name}.{name} is dataset-specific"
            )


def test_empty_frame_matches_the_spec():
    """An empty result must have the same dtypes as a populated one.

    A real hazard: a dataset can legitimately produce zero rows for a table, and
    an empty frame with Null-typed columns then breaks the concat downstream.
    """
    for spec in schema.CORE_TABLES.values():
        empty = spec.empty()
        assert tuple(empty.columns) == spec.names
        assert empty.schema == spec.dtypes
        assert schema.validate_table(empty, spec, raise_on_error=False) == []


def test_datetime_dtype_is_utc_microseconds():
    assert schema.UTC_DATETIME.time_zone == "UTC"
    assert schema.UTC_DATETIME.time_unit == "us"


# ---------------------------------------------------------------------------
# Single-table validators — conforming and violating
# ---------------------------------------------------------------------------


def test_conforming_frames_validate():
    schema.validate_table(_articles(), schema.ARTICLES)
    schema.validate_table(_impressions(), schema.IMPRESSIONS)
    schema.validate_table(_history(), schema.HISTORY)


def test_missing_column_is_rejected():
    bad = _articles().drop("body")
    with pytest.raises(schema.SchemaError, match="missing columns"):
        schema.validate_table(bad, schema.ARTICLES)


def test_extra_column_is_rejected():
    bad = _articles().with_columns(pl.lit(1).alias("surprise"))
    with pytest.raises(schema.SchemaError, match="unexpected columns"):
        schema.validate_table(bad, schema.ARTICLES)


def test_column_order_is_part_of_the_contract():
    reordered = _articles().select(["title", "article_id", *schema.ARTICLES.names[2:]])
    problems = schema.validate_table(reordered, schema.ARTICLES, raise_on_error=False)
    assert any("ORDER" in p for p in problems), problems


def test_wrong_dtype_is_rejected():
    cols = dict(schema.ARTICLES.columns) | {"article_id": pl.Int64}
    bad = _articles().with_columns(pl.Series("article_id", [1, 2])).cast(cols)
    with pytest.raises(schema.SchemaError, match="dtype is"):
        schema.validate_table(bad, schema.ARTICLES)


def test_naive_datetime_is_rejected():
    """Q1.5 — the failure this catches is invisible without it.

    Naive timestamps sort and compare perfectly well, so a naive column stays
    undetected until two datasets with different implicit zones are compared,
    by which point every split boundary is quietly wrong.
    """
    bad = _articles().with_columns(pl.col("published_time").dt.replace_time_zone(None))
    problems = schema.validate_table(bad, schema.ARTICLES, raise_on_error=False)
    assert any("naive" in p for p in problems), problems


def test_non_utc_timezone_is_rejected():
    bad = _articles().with_columns(
        pl.col("published_time").dt.convert_time_zone("Europe/Copenhagen")
    )
    problems = schema.validate_table(bad, schema.ARTICLES, raise_on_error=False)
    assert any("expected UTC" in p for p in problems), problems


def test_duplicate_key_is_rejected():
    bad = _articles(article_id=["A1", "A1"])
    with pytest.raises(schema.SchemaError, match="not unique"):
        schema.validate_table(bad, schema.ARTICLES)


def test_null_key_is_rejected():
    bad = _articles(article_id=["A1", None])
    with pytest.raises(schema.SchemaError, match="null"):
        schema.validate_table(bad, schema.ARTICLES)


def test_null_in_non_nullable_column_is_rejected():
    bad = _articles(title=["First", None])
    with pytest.raises(schema.SchemaError, match=r"title: 1 null"):
        schema.validate_table(bad, schema.ARTICLES)


def test_declared_nullable_columns_accept_nulls():
    """The other direction. MIND has no body, no published_time and no history
    timestamps; rejecting those would make half the corpus unrepresentable."""
    schema.validate_table(
        _articles(body=[None, None], published_time=[None, None]), schema.ARTICLES
    )
    schema.validate_table(_history(click_time=[None, None]), schema.HISTORY)
    schema.validate_table(_impressions(split=[None, None]), schema.IMPRESSIONS)


def test_invalid_split_label_is_rejected():
    bad = _impressions(split=["train", "holdout"])
    with pytest.raises(schema.SchemaError, match="invalid label"):
        schema.validate_table(bad, schema.IMPRESSIONS)


# ---------------------------------------------------------------------------
# Q9.3 — serving-unavailable columns must never reach a core table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", sorted(config.SERVING_UNAVAILABLE_FEATURES))
def test_serving_unavailable_column_in_core_table_is_rejected(column):
    """Structural enforcement, not review.

    Each of the seven is checked individually so that adding one to the config
    set without teaching the validator about it cannot pass silently.
    """
    bad = _impressions().with_columns(pl.lit(1.0).alias(column))
    problems = schema.validate_table(bad, schema.IMPRESSIONS, raise_on_error=False)
    assert any(column in p and "serving-unavailable" in p for p in problems), problems


# ---------------------------------------------------------------------------
# Cross-table validators
# ---------------------------------------------------------------------------


def test_valid_store_validates():
    schema.validate_store(
        {"articles": _articles(), "impressions": _impressions(), "history": _history()}
    )


def test_orphan_history_article_is_rejected():
    bad = _history(user_id=["U1", "U1"], article_id=["A1", "A_MISSING"])
    with pytest.raises(schema.SchemaError, match="absent from articles"):
        schema.validate_store(
            {"articles": _articles(), "impressions": _impressions(), "history": bad}
        )


def test_orphan_candidate_is_rejected():
    bad = _impressions(candidate_ids=[["A1", "A_MISSING"], ["A2"]])
    with pytest.raises(schema.SchemaError, match="absent from"):
        schema.validate_store({"articles": _articles(), "impressions": bad, "history": _history()})


def test_clicked_not_in_candidates_is_rejected():
    """A clicked id that was never shown is a positive no ranking can retrieve,
    so it depresses recall in a way that looks like model failure."""
    bad = _impressions(candidate_ids=[["A1"], ["A2"]], clicked_ids=[["A2"], []])
    with pytest.raises(schema.SchemaError, match="not present in"):
        schema.validate_store({"articles": _articles(), "impressions": bad, "history": _history()})


def test_missing_core_table_is_reported():
    problems = schema.validate_store({"articles": _articles()}, raise_on_error=False)
    assert any("missing core table 'impressions'" in p for p in problems), problems
    assert any("missing core table 'history'" in p for p in problems), problems


def test_extra_table_must_join_one_to_one():
    extra = pl.DataFrame({"article_id": ["A1", "A1"], "url": ["u", "u"]})
    problems = schema.check_extra_joins_one_to_one(
        _articles(), extra, "article_id", "articles_extra"
    )
    assert any("not unique" in p for p in problems), problems

    orphan = pl.DataFrame({"article_id": ["A_MISSING"], "url": ["u"]})
    problems = schema.check_extra_joins_one_to_one(
        _articles(), orphan, "article_id", "articles_extra"
    )
    assert any("no matching core row" in p for p in problems), problems

    good = pl.DataFrame({"article_id": ["A1", "A2"], "url": ["u", "v"]})
    assert (
        schema.check_extra_joins_one_to_one(_articles(), good, "article_id", "articles_extra") == []
    )


# ---------------------------------------------------------------------------
# conform()
# ---------------------------------------------------------------------------


def test_conform_reorders_and_casts():
    messy = _articles().select(["dataset", "article_id", *schema.ARTICLES.names[1:-1]])
    out = schema.conform(messy, schema.ARTICLES)
    assert tuple(out.columns) == schema.ARTICLES.names
    assert out.schema == schema.ARTICLES.dtypes


def test_conform_refuses_to_invent_or_discard_columns():
    """Narrow on purpose: both are contract breaches that must be seen.

    Silently adding a null column or dropping an unexpected one would let a
    parser bug reach the store looking like valid data.
    """
    with pytest.raises(schema.SchemaError, match="missing columns"):
        schema.conform(_articles().drop("category"), schema.ARTICLES)
    with pytest.raises(schema.SchemaError, match="unexpected columns"):
        schema.conform(_articles().with_columns(pl.lit(1).alias("x")), schema.ARTICLES)


def test_all_problems_reported_at_once():
    """Three bad columns should report three problems, not one at a time."""
    bad = _articles().drop("body").drop("category").with_columns(pl.lit(1).alias("junk"))
    problems = schema.validate_table(bad, schema.ARTICLES, raise_on_error=False)
    assert len(problems) >= 2, problems
