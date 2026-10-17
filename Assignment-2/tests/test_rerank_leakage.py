"""A2 Q9.2 — no future click reaches a re-ranker's or NRMS's training rows.

A2 Q9 says: "Enforce the behaviour-window boundary — no future-click leakage.
Include a test asserting this." A1 already tests the boundary where it is
*enforced* (`tests/test_leakage.py`, `tests/test_behaviour_features.py`). This
file tests it where it is *consumed*: the tables a model is actually fitted on.

## Why a separate file rather than more cases in test_leakage.py

Two different claims, and only one of them is about features.

**Per-feature.** Every individual value a model sees must be derived only from
clicks strictly before its impression. A2 Q1 enforces that once, in
`leakage.select_history_for`, and gates it at build time with
`behaviour_features.check`.

**Per-row.** A model can leak by being fitted on the wrong *rows* — val or test
impressions in the training table — while every individual feature value in
those rows is itself perfectly leakage-safe. No feature-level check can see
that, because nothing about the value is wrong.

A2 Q3's model adds a third surface: the history **tensor**. It is gathered from
A2 Q1's already-gated `recent_article_ids`, so it should inherit the guarantee —
but "should inherit" is the kind of claim that is worth an assertion, because
the whole point of A2 Q1's OOM lesson was that reusing a tested function outside
its stated bound is a *different* claim (`design_rationale.md` §5.1).

## The canary

`test_leakage.py` establishes the rule that a boundary test which cannot be
shown to fail proves nothing (A1's Q9.6). Two canaries here follow it:

- `test_canary_a_click_at_or_after_the_impression_is_excluded` hands the
  **production selector** a history containing a click one hour before, one at
  *exactly* the impression time, and one an hour after. The middle one is the
  case that separates `<` from `<=` and the one an off-by-one gets wrong.
- `test_canary_a_widened_history_would_be_caught` simulates the bug the subset
  assertion guards against, so that assertion is known to be capable of failing.

Both are hand-built rather than drawn from the shared
`tests/fixtures/leakage_canary.json`. That fixture is shaped for
`leakage.check_store`'s per-user cases; what needs exercising here is a single
impression's boundary and the *gatherer's* behaviour, which the fixture does not
express. The trade is one more small corrupt input against reusing one that does
not fit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store, leakage
from newsrec.rerank import features

# =============================================================================
# The canary first — prove the boundary is observable at all
# =============================================================================


@pytest.mark.canary
def test_canary_a_click_at_or_after_the_impression_is_excluded():
    """The strict-`<` boundary, on hand-built rows.

    A click at *exactly* the impression time is the case that separates `<`
    from `<=`, and it is the one an off-by-one gets wrong. If this ever passes
    with the future click included, every assertion below is worthless.
    """
    impression_time = datetime(2019, 11, 14, 12, 0, tzinfo=UTC)
    history = pl.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "article_id": ["past", "exactly_now", "future"],
            "click_time": [
                impression_time - timedelta(hours=1),
                impression_time,
                impression_time + timedelta(hours=1),
            ],
        }
    )
    impressions = pl.DataFrame(
        {
            "impression_id": ["imp1"],
            "user_id": ["u1"],
            "impression_time": [impression_time],
            "split": ["train"],
        }
    )

    selected = leakage.select_history_before(history, impressions)
    returned = set(selected["article_id"].to_list()) if selected.height else set()

    assert "past" in returned, "the legitimate prior click was dropped"
    assert "future" not in returned, "a click after the impression was selected"
    assert "exactly_now" not in returned, (
        "a click at exactly the impression time was selected — the boundary is "
        "<= rather than the strict < the protocol requires"
    )


# =============================================================================
# Per-row: split membership in the tables a model is fitted on
# =============================================================================


@pytest.mark.realdata
@pytest.mark.slow
def test_no_training_row_belongs_to_a_val_or_test_impression(dataset, store_root):
    """The leak no feature-level check can see.

    Every value in a val impression's feature row is leakage-safe; the row
    simply must not be in the training table. `recall.sample_impressions`
    filters on split before anything else runs, and this asserts the result
    rather than trusting it.
    """
    table = features.build_candidate_table(dataset, "train", size=200, store_root=store_root)
    if table.is_empty():
        pytest.skip(f"{dataset} has no sampled train impressions in this store")

    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    split_of = dict(
        zip(
            impressions["impression_id"].to_list(),
            impressions["split"].to_list(),
            strict=True,
        )
    )
    wrong = {
        impression_id
        for impression_id in table["impression_id"].unique().to_list()
        if split_of.get(impression_id) != "train"
    }
    assert not wrong, (
        f"{len(wrong)} impressions in {dataset}'s TRAIN candidate table do not belong "
        f"to the train split, e.g. {sorted(wrong)[:5]}"
    )


@pytest.mark.realdata
@pytest.mark.slow
def test_the_three_tables_share_no_impression(dataset, store_root):
    """Hyperparameters chosen on val and results reported on test only mean
    something if the three sets are disjoint."""
    tables = {
        split: features.build_candidate_table(dataset, split, size=200, store_root=store_root)
        for split in ("train", "val", "test")
    }
    ids = {
        split: set(table["impression_id"].to_list())
        for split, table in tables.items()
        if not table.is_empty()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in ids and b in ids:
            assert not (ids[a] & ids[b]), f"{dataset}: {a} and {b} share impressions"


# =============================================================================
# Per-row: the NRMS history tensor
# =============================================================================


@pytest.mark.realdata
@pytest.mark.slow
def test_the_nrms_history_is_a_subset_of_the_strictly_prior_clicks(dataset, store_root):
    """A2 Q3's history tensor inherits A2 Q1's boundary — asserted, not assumed.

    The tensor is gathered from `user_click_history_features.recent_article_ids`,
    which `behaviour_features.check` already gates at build time. What this adds
    is that the *gathering* did not widen the set: each impression's rows must be
    a subset of the clicks that table says are legitimate, and no longer than
    `config.QUERY_HISTORY_LIMIT`.
    """
    store = store_root / "feature_store" / dataset
    click_history = feature_store.read_table(store / "user_click_history_features.parquet")
    if click_history.is_empty():
        pytest.skip(f"{dataset} has no click-history table in this store")

    sample = click_history.head(200)
    # A row index per article id, as the real caller uses.
    all_articles = sorted(
        {a for ids in sample["recent_article_ids"].to_list() for a in (ids or [])}
    )
    row_of_article = {a: i for i, a in enumerate(all_articles)}

    got = features.history_rows_by_impression(sample, row_of_article)
    article_of_row = {i: a for a, i in row_of_article.items()}

    for row in sample.iter_rows(named=True):
        impression_id = str(row["impression_id"])
        legitimate = list(row["recent_article_ids"] or [])
        rows = got[impression_id]
        assert len(rows) <= config.QUERY_HISTORY_LIMIT, (
            f"{impression_id}: history of {len(rows)} exceeds "
            f"QUERY_HISTORY_LIMIT={config.QUERY_HISTORY_LIMIT}"
        )
        gathered = [article_of_row[r] for r in rows]
        assert set(gathered) <= set(legitimate), (
            f"{impression_id}: the gathered history contains articles the "
            f"click-history table does not list as legitimately available: "
            f"{sorted(set(gathered) - set(legitimate))[:5]}"
        )
        # Order must survive too: A2 Q1 stores most-recent-first.
        assert gathered == [a for a in legitimate if a in row_of_article][: len(gathered)], (
            f"{impression_id}: the gathered history is not the table's order"
        )


def test_the_history_gatherer_cannot_invent_a_click():
    """The unit version of the assertion above, runnable with no dataset.

    An id absent from the embedding index is dropped. The alternative — mapping
    it to row 0 — would put a real, arbitrary article into a user's history and
    nothing would raise.
    """
    table = pl.DataFrame(
        {
            "impression_id": ["i1"],
            "recent_article_ids": [["known", "never_embedded"]],
        }
    )
    got = features.history_rows_by_impression(table, {"known": 4})
    assert got["i1"].tolist() == [4], "an unembedded article became a row number"


def test_a_cold_start_impression_gets_an_empty_history_not_someone_elses():
    table = pl.DataFrame({"impression_id": ["cold"], "recent_article_ids": [[]]})
    got = features.history_rows_by_impression(table, {"a": 0, "b": 1})
    assert got["cold"].size == 0


@pytest.mark.canary
def test_canary_a_widened_history_would_be_caught():
    """Proves the subset assertion above is capable of failing.

    Simulates the bug it guards against — a gatherer that returns an article
    the click-history table never listed — and checks that the same subset
    comparison rejects it.
    """
    legitimate = ["a", "b"]
    widened = ["a", "b", "c"]  # 'c' was never legitimately available
    assert not set(widened) <= set(legitimate)
    assert set(legitimate) <= set(legitimate)


# =============================================================================
# The serving-unavailable features must not reach a headline feature list
# =============================================================================


def test_no_serving_unavailable_feature_is_in_the_headline_feature_list():
    """A2 Q9.1's other half, as a structural condition.

    `read_time`, `scroll_percentage` and the dataset-wide `total_*` aggregates
    are usable only inside a labelled ablation. A model's headline feature list
    must contain none of them, and this fails if one is ever appended.
    """
    leaked = set(features.FEATURE_COLUMNS) & config.SERVING_UNAVAILABLE_FEATURES
    assert not leaked, (
        f"{sorted(leaked)} are in FEATURE_COLUMNS but are unavailable at serving "
        f"time — they belong only in an explicitly labelled ablation"
    )


def test_the_feature_list_has_no_post_click_field_under_another_name():
    """A weaker but useful check: nothing in the headline list is named after a
    post-click measurement. Catches `dwell_time` or `read_seconds` being added
    as a "new" feature rather than the forbidden one under a new spelling."""
    suspicious = ("read_time", "dwell", "scroll", "pageview", "inview", "next_")
    for column in features.FEATURE_COLUMNS:
        for token in suspicious:
            assert token not in column.lower(), (
                f"{column!r} looks like a post-click or dataset-wide measurement"
            )
