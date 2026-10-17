"""Q2.5, Q2.6 — query construction and the cold-start fallback.

**The two canaries come first in this file, deliberately.** q2_plan's test list
names them, and Q1's mutation-testing lesson is that a canary written after the
code it guards tends to encode what the code does rather than what it should do.
Both were written before `query.py` existed and were confirmed to fail against
the empty module.

Canary 1 — the popularity table must be built from the TRAINING split only.
Canary 2 — a click at exactly the impression timestamp must not reach the query.

Popularity computed over the whole log still returns plausible articles, so that
failure is invisible without a manufactured fixture. The boundary canary was
justified the same way -- "clicks never share a microsecond with an impression"
-- and that turned out to be FALSE (6,518 such pairs in ebnerd_demo, 0.0341%;
62,030 in ebnerd_small, 0.0336%). The canary is still the right tool, because it
isolates the boundary where nothing else varies, but the claim that real data
cannot reach it was never measured until 2026-08-22. Same lesson as
`test_canary_selector_excludes_equal_timestamps` in Q1 (design_rationale 2.13).
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from newsrec import config
from newsrec.data import leakage
from newsrec.retrieval import query

UTC_ = UTC


def _dt(day: int, hour: int = 12) -> datetime:
    return datetime(2019, 11, day, hour, tzinfo=UTC_)


# `train` ends before day 14, `test` is day 15 — matching the MIND split shape.
IMPRESSIONS = pl.DataFrame(
    {
        "impression_id": ["i_train", "i_test"],
        "user_id": ["u1", "u1"],
        "impression_time": [_dt(10), _dt(15)],
        "candidate_ids": [["a1", "a2"], ["a1", "a2"]],
        "clicked_ids": [["a1"], ["a2"]],
        "split": ["train", "test"],
    }
)


def _history(article_ids, click_times, ranks=None, user="u1") -> pl.DataFrame:
    """A history frame matching the real `history` contract.

    The full schema is used deliberately, including `history_rank`. An earlier
    version of these fixtures omitted it and passed — until `query.py` switched
    to the dispatching selector, which reads it. A fixture that does not match
    the contract tests a table the pipeline never produces.
    """
    n = len(article_ids)
    return pl.DataFrame(
        {
            "user_id": [user] * n,
            "article_id": list(article_ids),
            "click_time": list(click_times),
            "history_rank": list(ranks) if ranks is not None else [None] * n,
        },
        schema={
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "click_time": pl.Datetime("us", "UTC"),
            "history_rank": pl.Int32,
        },
    )


# Timestamped, i.e. EB-NeRD-shaped. a3 is clicked AFTER the day-15 impression.
HISTORY = _history(["a1", "a2", "a3"], [_dt(9), _dt(14), _dt(16)])

ARTICLES = pl.DataFrame(
    {
        "article_id": ["a1", "a2", "a3"],
        "title": ["alpha story", "beta story", "gamma story"],
        "abstract": ["ABSTRACTWORD one", "ABSTRACTWORD two", "ABSTRACTWORD three"],
    }
)

SPEC = config.TokeniserSpec(name="toy", language="da")


# =============================================================================
# CANARY 1 (Q2.6, Q9) — popularity must come from training rows only
# =============================================================================


def test_canary_popularity_built_from_all_splits_is_detected():
    """The `total_inviews` leak in local form.

    An article clicked only in the TEST period must not appear in a popularity
    table used to serve TRAIN impressions — that is knowledge of the future.
    Here `a2` is clicked only in test and `a1` only in train, so a table built
    from everything ranks `a2` where a correct one does not rank it at all.
    """
    honest = query.training_popularity(IMPRESSIONS)
    assert "a1" in honest
    assert "a2" not in honest, (
        "a2 is clicked only in the test split; a popularity table containing it "
        "has been built from the future"
    )

    leaky = query.training_popularity(IMPRESSIONS, splits=("train", "val", "test"))
    assert "a2" in leaky, "the canary itself is broken — the leak did not occur"
    assert honest != leaky, "an honest and a leaked table must be distinguishable"


def test_canary_popularity_leak_is_visible_in_the_ranking_not_just_the_keys():
    """A leak that changed only counts, not membership, would slip past a
    key-set comparison. Assert the ordering moves too."""
    impressions = IMPRESSIONS.vstack(
        pl.DataFrame(
            {
                "impression_id": ["i_test2"],
                "user_id": ["u2"],
                "impression_time": [_dt(15, 13)],
                "candidate_ids": [["a1"]],
                "clicked_ids": [["a1"]],
                "split": ["test"],
            }
        )
    )
    honest = query.training_popularity(impressions)
    leaky = query.training_popularity(impressions, splits=("train", "val", "test"))
    assert honest["a1"] == 1
    assert leaky["a1"] == 2


def test_popularity_fallback_uses_only_the_training_table():
    fallback = query.cold_start_candidates(IMPRESSIONS, k=10)
    assert "a2" not in fallback


# =============================================================================
# CANARY 2 (Q2.5) — the strict boundary, manufactured
# =============================================================================


def test_canary_click_at_exactly_the_impression_time_is_excluded():
    """A click at the same instant as the impression is indistinguishable from a
    click *caused by* it — the label fed back in as a feature.

    Real data produces this rarely rather than never -- 0.0341% of ebnerd_demo
    (user, click, impression) triples, 0.0336% of ebnerd_small. Building it by
    hand is still right: a fixture isolates the boundary so a failure names its
    cause, where 0.03% of rows moving inside an aggregate does not.
    """
    # 'gamma' is absent from every other query, so its presence is unambiguous.
    boundary_history = _history(["a3"], [_dt(15)])  # EXACTLY the impression time
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test")

    terms = query.build_query(impression.row(0, named=True), boundary_history, ARTICLES, SPEC)
    assert "gamma" not in terms, "a click at the boundary reached the query"
    assert terms == [], "no legitimate history remains, so the query must be empty"


def test_canary_click_one_microsecond_before_is_included():
    """The control. A boundary check that excludes everything would pass the
    test above while making every query empty."""
    just_before = _history(["a3"], [datetime(2019, 11, 15, 11, 59, 59, 999_999, tzinfo=UTC_)])
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test")
    terms = query.build_query(impression.row(0, named=True), just_before, ARTICLES, SPEC)
    assert "gamma" in terms


def test_query_construction_goes_through_the_shared_selector(monkeypatch):
    """Q2.5, asserted structurally rather than only behaviourally.

    Q1's mutation testing showed a permissive selector is invisible on real
    data. The defence is that there is exactly ONE place the `<` boundary is
    written, so this asserts the query builder calls it rather than re-deriving
    an equivalent filter that could drift to `<=`.
    """
    called = {}
    original = leakage.select_history_for

    def spy(history, impressions):
        called["yes"] = True
        return original(history, impressions)

    monkeypatch.setattr(query.leakage, "select_history_for", spy)
    query.build_query(
        IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True),
        HISTORY,
        ARTICLES,
        SPEC,
    )
    assert called.get("yes"), "build_query did not use leakage.select_history_for"


def test_mind_shaped_history_produces_a_non_empty_query():
    """The regression this commit exists to fix, at the query layer.

    MIND ships no click timestamps. Calling the timestamp selector directly
    returned NOTHING for every MIND impression — null compared with anything is
    null — so 100% of MIND queries were empty and BM25 silently degraded to the
    popularity fallback for the whole English half of the assignment.

    Every other fixture in this file is EB-NeRD-shaped (timestamped), which is
    exactly why the bug survived the first version of this test file.
    """
    mind_history = _history(["a1", "a2"], [None, None], ranks=[0, 1])
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    terms = query.build_query(impression, mind_history, ARTICLES, SPEC)
    assert terms, "MIND-shaped history produced an empty query"
    assert "alpha" in terms and "beta" in terms


def test_mind_shaped_history_is_capped_by_rank_most_recent_first():
    """The D6 cap still means "most recent" when recency is a rank, not a time."""
    mind_history = _history([f"x{i}" for i in range(5)], [None] * 5, ranks=list(range(5)))
    titles = pl.DataFrame(
        {
            "article_id": [f"x{i}" for i in range(5)],
            "title": [f"term{i}" for i in range(5)],
            "abstract": [""] * 5,
        }
    )
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    terms = query.build_query(impression, mind_history, titles, SPEC, limit=2)
    assert set(terms) == {"term3", "term4"}, f"expected the highest ranks, got {terms}"


# =============================================================================
# D6 — what the query is made of
# =============================================================================


def test_future_clicks_never_reach_the_query():
    """`a3` is clicked on day 16, after the day-15 impression."""
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    terms = query.build_query(impression, HISTORY, ARTICLES, SPEC)
    assert "gamma" not in terms
    assert "alpha" in terms and "beta" in terms


def test_query_uses_titles_not_abstracts():
    """D6's deliberate asymmetry: documents are title + abstract, queries are
    titles only. A query built from 20 full abstracts would be enormous and
    diffuse."""
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    terms = query.build_query(impression, HISTORY, ARTICLES, SPEC)
    assert "abstractword" not in terms


def test_query_is_capped_at_the_configured_limit():
    many = _history(
        [f"x{i}" for i in range(50)],
        [datetime(2019, 11, 14, 0, i, tzinfo=UTC_) for i in range(50)],
    )
    titles = pl.DataFrame(
        {
            "article_id": [f"x{i}" for i in range(50)],
            "title": [f"term{i}" for i in range(50)],
            "abstract": [""] * 50,
        }
    )
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    terms = query.build_query(impression, many, titles, SPEC, limit=5)
    assert len(terms) == 5
    # The five MOST RECENT clicks, not the first five.
    assert set(terms) == {f"term{i}" for i in range(45, 50)}


def test_cap_takes_the_most_recent_not_an_arbitrary_five():
    """Recency is the whole justification for D6 option A. Taking an arbitrary
    subset would keep the cost saving and throw away the reasoning."""
    many = _history(["p", "q", "r", "s"], [_dt(11), _dt(12), _dt(13), _dt(14)])
    titles = pl.DataFrame(
        {
            "article_id": ["p", "q", "r", "s"],
            "title": ["pp", "qq", "rr", "ss"],
            "abstract": [""] * 4,
        }
    )
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    assert set(query.build_query(impression, many, titles, SPEC, limit=2)) == {"rr", "ss"}


def test_repeated_terms_are_preserved():
    """A term in several of a user's clicked titles genuinely matters more
    (D6 option D, rejected)."""
    repeat = _history(["a1", "a2"], [_dt(9), _dt(10)])
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    terms = query.build_query(impression, repeat, ARTICLES, SPEC)
    assert terms.count("story") == 2


def test_empty_history_yields_an_empty_query_rather_than_raising():
    """Q2.6 — cold start must be a defined outcome, not an exception."""
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    empty = HISTORY.clear()
    assert query.build_query(impression, empty, ARTICLES, SPEC) == []


def test_click_on_an_article_missing_from_the_corpus_is_skipped():
    """History can reference an article the article table does not carry. That
    must contribute nothing, not raise and not inject a null."""
    orphan = _history(["NOT_IN_CORPUS"], [_dt(9)])
    impression = IMPRESSIONS.filter(pl.col("impression_id") == "i_test").row(0, named=True)
    assert query.build_query(impression, orphan, ARTICLES, SPEC) == []


def test_default_limit_comes_from_config_not_a_literal():
    assert query.DEFAULT_LIMIT is config.QUERY_HISTORY_LIMIT


# =============================================================================
# Cold start (Q2.6)
# =============================================================================


def test_cold_start_returns_candidates_rather_than_nothing():
    got = query.cold_start_candidates(IMPRESSIONS, k=5)
    assert len(got) > 0


def _train_impressions(clicked: list[list[str]]) -> pl.DataFrame:
    n = len(clicked)
    return pl.DataFrame(
        {
            "impression_id": [str(i) for i in range(n)],
            "user_id": [f"u{i}" for i in range(n)],
            "impression_time": [_dt(10)] * n,
            "candidate_ids": [["z"]] * n,
            "clicked_ids": clicked,
            "split": ["train"] * n,
        }
    )


def test_cold_start_is_ordered_by_popularity():
    """`z` is clicked three times, `m` once — popularity is the primary key."""
    impressions = _train_impressions([["z", "m"], ["z"], ["z"]])
    assert query.cold_start_candidates(impressions, k=2) == ["z", "m"]


def test_cold_start_breaks_ties_by_article_id_ascending():
    """The secondary key, isolated. `z` and `m` are clicked twice each, so the
    order is decided alphabetically — and must be the same on every run."""
    impressions = _train_impressions([["z", "m"], ["z"], ["m"]])
    assert query.cold_start_candidates(impressions, k=2) == ["m", "z"]


def test_cold_start_respects_k():
    assert len(query.cold_start_candidates(IMPRESSIONS, k=1)) == 1


def test_cold_start_on_an_empty_training_split_returns_empty_not_error():
    only_test = IMPRESSIONS.filter(pl.col("split") == "test")
    assert query.cold_start_candidates(only_test, k=5) == []


def test_popularity_is_deterministic():
    first = query.cold_start_candidates(IMPRESSIONS, k=10)
    for _ in range(3):
        assert query.cold_start_candidates(IMPRESSIONS, k=10) == first


@pytest.mark.parametrize("k", [0, -1])
def test_cold_start_with_non_positive_k_is_empty(k):
    assert query.cold_start_candidates(IMPRESSIONS, k=k) == []
