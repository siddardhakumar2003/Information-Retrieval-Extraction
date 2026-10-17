"""A2 Q1 — click-history, session, and article behavioural features.

Same discipline as `test_leakage.py`: canaries on manufactured fixtures prove a
check can fail, and the real-data tests prove the boundary holds on an actual
built store. A recency-weighted count or a session position that is well-formed
but built from the wrong clicks looks exactly like a real number.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from newsrec import config
from newsrec.data import behaviour_features, feature_store, leakage

UTC_DT = pl.Datetime("us", "UTC")


def _t(day: int, hour: int = 12) -> datetime:
    return datetime(2019, 11, day, hour, tzinfo=UTC)


def _impressions(rows):
    return pl.DataFrame(
        rows,
        schema={
            "impression_id": pl.Utf8,
            "user_id": pl.Utf8,
            "impression_time": UTC_DT,
            "candidate_ids": pl.List(pl.Utf8),
            "clicked_ids": pl.List(pl.Utf8),
            "split": pl.Utf8,
            "dataset": pl.Utf8,
        },
        orient="row",
    )


def _history(rows):
    """Full-contract history — `click_time` AND `history_rank`, both nullable."""
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "click_time": UTC_DT,
            "history_rank": pl.Int32,
            "dataset": pl.Utf8,
        },
        orient="row",
    )


def _articles(rows):
    return pl.DataFrame(
        rows,
        schema={"article_id": pl.Utf8, "title": pl.Utf8, "category": pl.Utf8},
        orient="row",
    )


# =============================================================================
# CANARY 1 — click-history features respect the strict boundary
# =============================================================================


def test_canary_future_click_does_not_count_or_appear():
    """A click AFTER the impression must not reach n_clicks or the recent list."""
    impressions = _impressions([("i1", "u1", _t(5), ["a1", "a2"], [], "train", "d")])
    history = _history(
        [
            ("u1", "past", _t(4), None, "d"),
            ("u1", "future", _t(6), None, "d"),  # after the impression
        ]
    )
    articles = _articles([("past", "Past title", "news"), ("future", "Future title", "news")])

    out = behaviour_features.build_click_history_features(history, impressions, articles, "d")
    row = out.row(0, named=True)
    assert row["n_clicks_in_history"] == 1
    assert row["recent_article_ids"] == ["past"]
    assert "future" not in row["recent_article_ids"]


def test_click_at_exactly_the_impression_time_is_excluded():
    """`strictly` before — a click at the same instant is excluded, same rule
    `select_history_for` already enforces for BM25 and the user embedding."""
    impressions = _impressions([("i1", "u1", _t(5), ["a1"], [], "train", "d")])
    history = _history([("u1", "simultaneous", _t(5), None, "d")])
    articles = _articles([("simultaneous", "T", "c")])

    out = behaviour_features.build_click_history_features(history, impressions, articles, "d")
    row = out.row(0, named=True)
    assert row["n_clicks_in_history"] == 0
    assert row["recent_article_ids"] == []


# =============================================================================
# Click-history arithmetic
# =============================================================================


def test_recency_weighted_click_count_matches_hand_computed_decay():
    """decay ** i for the i-th most recent click, summed. Hand-computed for a
    manufactured 3-click history so the exact semantics are pinned."""
    impressions = _impressions([("i1", "u1", _t(10), [], [], "train", "d")])
    history = _history(
        [
            ("u1", "newest", _t(9), None, "d"),
            ("u1", "middle", _t(8), None, "d"),
            ("u1", "oldest", _t(7), None, "d"),
        ]
    )
    articles = _articles([(a, a, "c") for a in ("newest", "middle", "oldest")])

    decay = 0.9
    out = behaviour_features.build_click_history_features(
        history, impressions, articles, "d", decay=decay
    )
    row = out.row(0, named=True)
    expected = decay**0 + decay**1 + decay**2
    assert row["n_clicks_in_history"] == 3
    assert row["recency_weighted_click_count"] == pytest.approx(expected, abs=1e-5)
    assert row["recent_article_ids"] == ["newest", "middle", "oldest"]


def test_recent_lists_are_capped_but_click_count_is_not():
    """A2 Q1.1: click count is over the FULL history; the stored lists are
    capped at `limit`. Both matter and must not be conflated."""
    impressions = _impressions([("i1", "u1", _t(10), [], [], "train", "d")])
    history = _history([("u1", f"a{i}", _t(1 + i), None, "d") for i in range(5)])
    articles = _articles([(f"a{i}", f"a{i}", "c") for i in range(5)])

    out = behaviour_features.build_click_history_features(
        history, impressions, articles, "d", limit=2
    )
    row = out.row(0, named=True)
    assert row["n_clicks_in_history"] == 5
    assert len(row["recent_article_ids"]) == 2
    # Most-recent-first: a4 clicked on day 5 is the newest.
    assert row["recent_article_ids"] == ["a4", "a3"]


def test_cold_start_user_gets_zero_not_null():
    impressions = _impressions([("i1", "u_cold", _t(10), [], [], "train", "d")])
    history = _history([]).cast({"click_time": UTC_DT})
    articles = _articles([])

    out = behaviour_features.build_click_history_features(history, impressions, articles, "d")
    row = out.row(0, named=True)
    assert row["n_clicks_in_history"] == 0
    assert row["recency_weighted_click_count"] == 0.0
    assert row["recent_article_ids"] == []


def test_mind_shaped_history_uses_rank_not_timestamp():
    """MIND ships no click timestamps — recency must fall back to history_rank,
    exactly as `leakage.select_history_for`'s structural path does."""
    impressions = _impressions([("i1", "u1", _t(10), [], [], "train", "d")])
    history = pl.DataFrame(
        [("u1", "older", None, 0, "d"), ("u1", "newer", None, 1, "d")],
        schema={
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "click_time": UTC_DT,
            "history_rank": pl.Int32,
            "dataset": pl.Utf8,
        },
        orient="row",
    )
    articles = _articles([("older", "o", "c"), ("newer", "n", "c")])

    out = behaviour_features.build_click_history_features(history, impressions, articles, "d")
    row = out.row(0, named=True)
    assert row["history_is_timestamped"] is False
    assert row["recent_article_ids"] == ["newer", "older"]


# =============================================================================
# Session features (EB-NeRD-shaped only)
# =============================================================================


def test_session_click_count_excludes_the_current_impressions_own_clicks():
    """Only clicks from EARLIER impressions in the same session count."""
    impressions = _impressions(
        [
            ("i1", "u1", _t(5, 8), ["a1"], ["a1"], "train", "d"),
            ("i2", "u1", _t(5, 9), ["a2"], ["a2"], "train", "d"),
            ("i3", "u1", _t(5, 10), ["a3"], [], "train", "d"),
        ]
    )
    extra = pl.DataFrame(
        {"impression_id": ["i1", "i2", "i3"], "session_id": ["s1", "s1", "s1"]},
        schema={"impression_id": pl.Utf8, "session_id": pl.Utf8},
    )

    out = behaviour_features.build_session_features(impressions, extra, "d")
    by_id = {r["impression_id"]: r for r in out.to_dicts()}
    assert by_id["i1"]["session_click_count"] == 0
    assert by_id["i1"]["position_in_session"] == 1
    assert by_id["i2"]["session_click_count"] == 1  # i1's click counts
    assert by_id["i2"]["position_in_session"] == 2
    assert by_id["i3"]["session_click_count"] == 2  # i1 + i2's clicks
    assert by_id["i3"]["position_in_session"] == 3


def test_session_features_absent_when_no_session_id():
    impressions = _impressions([("i1", "u1", _t(5), [], [], "train", "d")])
    out = behaviour_features.build_session_features(impressions, pl.DataFrame(), "d")
    assert out is None


def test_mind_has_no_session_features(dataset, processed_store):
    """Pins the disclosed MIND limitation rather than only asserting it in
    prose (a2_q1_plan.md)."""
    if dataset != "mind_small":
        pytest.skip("EB-NeRD-only assertion's counterpart; see the ebnerd variant")
    tables = {
        "impressions": feature_store.read_table(processed_store / "impressions.parquet"),
    }
    extra_path = processed_store / "impressions_extra.parquet"
    extra = feature_store.read_table(extra_path) if extra_path.exists() else None
    out = behaviour_features.build_session_features(tables["impressions"], extra, dataset)
    assert out is None


# =============================================================================
# Article popularity (A2 Q1.3) — leak canary
# =============================================================================


def test_popularity_train_ignores_val_and_test_clicks():
    impressions = _impressions(
        [
            ("i_train", "u1", _t(1), ["a1"], ["a1"], "train", "d"),
            ("i_val", "u1", _t(2), ["a2"], ["a2"], "val", "d"),
            ("i_test", "u1", _t(3), ["a2"], ["a2"], "test", "d"),
        ]
    )
    article_features = pl.DataFrame(
        {"article_id": ["a1", "a2"], "published_time": [None, None]},
        schema={"article_id": pl.Utf8, "published_time": UTC_DT},
    )
    out = behaviour_features.add_article_behaviour_columns(article_features, impressions)
    by_id = dict(out.select("article_id", "popularity_train").iter_rows())
    assert by_id["a1"] == 1
    assert by_id["a2"] == 0, (
        "a2 is clicked only outside training; counting it would be a leak "
        "(the same canary shape as test_query.py's training_popularity test)"
    )


def test_has_published_time_flag():
    impressions = _impressions([])
    article_features = pl.DataFrame(
        {"article_id": ["a1", "a2"], "published_time": [_t(1), None]},
        schema={"article_id": pl.Utf8, "published_time": UTC_DT},
    )
    out = behaviour_features.add_article_behaviour_columns(article_features, impressions)
    by_id = dict(out.select("article_id", "has_published_time").iter_rows())
    assert by_id["a1"] is True
    assert by_id["a2"] is False


# =============================================================================
# No post-click columns ever leak into these tables (A2 Q1.4 / Q9)
# =============================================================================


def test_no_post_click_columns_in_behaviour_tables():
    impressions = _impressions([("i1", "u1", _t(5), [], [], "train", "d")])
    history = _history([]).cast({"click_time": UTC_DT})
    articles = _articles([])
    click_history = behaviour_features.build_click_history_features(
        history, impressions, articles, "d"
    )
    session = behaviour_features.build_session_features(
        impressions, pl.DataFrame({"impression_id": ["i1"], "session_id": ["s1"]}), "d"
    )
    tables = {"user_click_history_features": click_history, "session_features": session}
    assert leakage.find_unavailable_columns(tables) == {}


# =============================================================================
# The build-time check function can fail (canary) and holds on real data
# =============================================================================


def test_check_raises_when_a_feature_disagrees_with_recomputation():
    impressions = _impressions([("i1", "u1", _t(10), [], [], "train", "d")])
    history = _history([("u1", "a1", _t(9), None, "d")])
    articles = _articles([("a1", "T", "c")])

    correct = behaviour_features.build_click_history_features(history, impressions, articles, "d")
    corrupted = correct.with_columns(pl.lit(999).cast(pl.Int32).alias("n_clicks_in_history"))

    with pytest.raises(behaviour_features.BehaviourLeakageError):
        behaviour_features.check(corrupted, None, history, impressions)


def test_check_passes_on_correctly_built_features():
    impressions = _impressions([("i1", "u1", _t(10), [], [], "train", "d")])
    history = _history([("u1", "a1", _t(9), None, "d")])
    articles = _articles([("a1", "T", "c")])

    correct = behaviour_features.build_click_history_features(history, impressions, articles, "d")
    report = behaviour_features.check(correct, None, history, impressions)
    assert report["problems"] == []


# =============================================================================
# Real-data tests — run against the actual built store
# =============================================================================


def test_click_history_features_respect_boundary_on_real_data(processed_store, dataset):
    """A2 Q1.4, real-data counterpart to the manufactured canaries above."""
    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    history = feature_store.read_table(processed_store / "history.parquet")
    articles = feature_store.read_table(processed_store / "articles.parquet")

    sample = impressions.sample(n=min(200, len(impressions)), seed=config.GLOBAL_SEED)
    out = behaviour_features.build_click_history_features(history, sample, articles, dataset)

    report = behaviour_features.check(out, None, history, sample)
    assert report["problems"] == []


def test_feature_store_builds_and_reads_behavioural_tables(processed_store, dataset, tmp_path):
    tables = {
        "articles": feature_store.read_table(processed_store / "articles.parquet"),
        "impressions": feature_store.read_table(processed_store / "impressions.parquet"),
        "history": feature_store.read_table(processed_store / "history.parquet"),
    }
    extra_path = processed_store / "impressions_extra.parquet"
    if extra_path.exists():
        tables["impressions_extra"] = feature_store.read_table(extra_path)

    feature_store.write_feature_store(tables, dataset, root=tmp_path)
    out = feature_store.read_feature_store(dataset, root=tmp_path)

    assert "user_click_history_features" in out
    assert len(out["user_click_history_features"]) == len(tables["impressions"])
    assert "popularity_train" in out["article_features"].columns
    assert "has_published_time" in out["article_features"].columns

    if "impressions_extra" in tables and "session_id" in tables["impressions_extra"].columns:
        assert "session_features" in out
    else:
        assert "session_features" not in out


def test_feature_store_manifest_reflects_behavioural_version(monkeypatch):
    """Bumping BEHAVIOURAL_FEATURE_VERSION must change the fingerprint, the same
    staleness discipline STORE_VERSION already gets."""
    before = feature_store.config_fingerprint("mind_small")
    monkeypatch.setattr(
        config, "BEHAVIOURAL_FEATURE_VERSION", config.BEHAVIOURAL_FEATURE_VERSION + 1
    )
    after = feature_store.config_fingerprint("mind_small")
    assert before != after
