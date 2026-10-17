"""Q1.2, Q1.10, Q1.11 — the MIND parser.

Runs on the committed fixture by default (no dataset needed), with the real-file
canaries marked `realdata`.

Each test names the silent failure it exists to prevent. A parser test that only
checks "it produced some rows" passes just as happily on a parser that mis-fields
every row containing a quote.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from newsrec import config
from newsrec.data import mind, schema

FIXTURE = Path(__file__).parent / "fixtures" / "mind_small"


@pytest.fixture
def mind_root(tmp_path: Path) -> Path:
    """The committed fixture, already laid out the way MIND ships.

    The two periods hold DIFFERENT files, as the real ones do: dev restarts
    impression ids at 1 (exercising the collision path) and shares three of four
    articles with train (exercising dedupe), but contains no train-period rows.
    An earlier version copied one behaviors.tsv into both periods, which put a
    dev-date row inside the train file -- something real MIND never does, and
    which surfaced as a phantom split-crosscheck disagreement.
    """
    shutil.copytree(FIXTURE, tmp_path / "mind")
    return tmp_path / "mind"


@pytest.fixture
def parsed(mind_root: Path):
    return mind.parse(mind_root)


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


def test_output_satisfies_the_canonical_schema(parsed):
    """The whole point of building schema.py first."""
    schema.validate_store(
        {
            "articles": parsed.articles,
            "impressions": parsed.impressions,
            "history": parsed.history,
            "articles_extra": parsed.articles_extra,
            "impressions_extra": parsed.impressions_extra,
        }
    )


def test_timestamps_are_utc(parsed):
    """Q1.5 — localised once, at the parse boundary."""
    dtype = parsed.impressions.schema["impression_time"]
    assert dtype.time_zone == "UTC"
    first = parsed.impressions.sort("impression_time")["impression_time"][0]
    assert (first.year, first.month, first.day, first.hour) == (2019, 11, 9, 8)


def test_split_is_null_until_the_splitter_runs(parsed):
    """The parser must not invent a split; split.py is the only writer."""
    assert parsed.impressions["split"].null_count() == len(parsed.impressions)


# ---------------------------------------------------------------------------
# The four verified hazards
# ---------------------------------------------------------------------------


def test_impression_ids_are_namespaced_by_period(parsed):
    """Hazard 1. Both files number from 1, so every one of dev's 73,152 ids also
    occurs in train. Without namespacing, two different impressions merge under
    one primary key and the duplicate is invisible."""
    ids = parsed.impressions["impression_id"].to_list()
    assert len(ids) == len(set(ids)), "impression_id is not unique"
    assert "train:1" in ids and "dev:1" in ids


def test_original_impression_id_survives_for_q5(parsed):
    """Namespacing must not destroy the id the leaderboard addresses rows by."""
    extra = parsed.impressions_extra
    row = extra.filter(pl.col("impression_id") == "dev:1").row(0, named=True)
    assert row["source_impression_id"] == "1"
    assert row["source_period"] == "dev"


def test_titles_containing_quotes_are_not_mis_fielded(tmp_path):
    """Hazard 2. 422 real titles contain a double quote. With the CSV default,
    the quote swallows the following tab and every later field in that row
    shifts by one -- silent corruption that reads as ordinary bad text."""
    tricky = 'John Dorsey admits talks, but it "takes two to tango"'
    news = [
        ("N001", "sports", "nfl", tricky, "abstract here", "http://x/1", "[]", "[]"),
        ("N002", "news", "world", "Plain title", "b", "http://x/2", "[]", "[]"),
    ]
    beh = [("1", "U001", "11/09/2019 8:00:00 AM", "N002", "N001-1 N002-0")]
    for _period, sub in mind.SOURCE_PERIODS.items():
        d = tmp_path / sub
        d.mkdir(parents=True)
        (d / "news.tsv").write_text("".join("\t".join(r) + "\n" for r in news), encoding="utf-8")
        (d / "behaviors.tsv").write_text(
            "".join("\t".join(r) + "\n" for r in beh), encoding="utf-8"
        )

    result = mind.parse(tmp_path)
    row = result.articles.filter(pl.col("article_id") == "N001").row(0, named=True)
    assert row["title"] == tricky, "title was mangled by CSV quote handling"
    assert row["category"] == "sports", "field alignment shifted"
    assert row["abstract"] == "abstract here"


def test_article_dedupe_is_counted_and_lossless(parsed):
    """Hazard 3. 28,460 articles appear in both real files, byte-identical.

    The fixture mirrors that: 6 train articles + 4 dev articles, 3 of them
    shared, so 10 raw rows collapse to 7 unique with 3 duplicates.
    """
    counts = parsed.article_counts
    assert counts["raw"] == counts["unique"] + counts["duplicates"]
    assert counts == {"raw": 10, "duplicates": 3, "unique": 7}
    assert parsed.articles["article_id"].n_unique() == len(parsed.articles)


def test_conflicting_article_text_raises_rather_than_picking_one(tmp_path):
    """Hazard 3, the other direction. If the two files disagreed about an
    article's text, 'keep first' would silently resolve it by file order."""
    base = ("N001", "sports", "nfl", "Title A", "abs", "http://x/1", "[]", "[]")
    conflicting = ("N001", "sports", "nfl", "Title B", "abs", "http://x/1", "[]", "[]")
    beh = [("1", "U001", "11/09/2019 8:00:00 AM", "", "N001-1")]
    for period, sub in mind.SOURCE_PERIODS.items():
        d = tmp_path / sub
        d.mkdir(parents=True)
        row = base if period == "train" else conflicting
        (d / "news.tsv").write_text("\t".join(row) + "\n", encoding="utf-8")
        (d / "behaviors.tsv").write_text(
            "".join("\t".join(r) + "\n" for r in beh), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="different title/abstract"):
        mind.parse(tmp_path)


def test_history_is_per_user_and_untimestamped(parsed):
    """Hazard 4. History is constant per user in the real files, so a per-user
    table is exact. click_time is null because MIND ships no history timestamps."""
    assert parsed.history_is_per_user_constant
    assert parsed.history["click_time"].null_count() == len(parsed.history)
    u1 = set(parsed.history.filter(pl.col("user_id") == "U001")["article_id"])
    assert u1 == {"N001", "N002"}


def test_varying_history_is_counted_not_hidden(tmp_path):
    """The parser must not depend on a property it does not check.

    Constructed so one user's history GROWS between impressions -- which real
    MIND never does. The parser takes the earliest-by-timestamp history (the
    only choice that cannot import a click recorded after an impression we later
    score) and reports the anomaly rather than silently picking one.
    """
    news = [(f"N00{i}", "c", "s", f"T{i}", "a", "u", "[]", "[]") for i in range(1, 5)]
    beh = [
        ("1", "U001", "11/09/2019 8:00:00 AM", "N001", "N003-1 N004-0"),
        ("2", "U001", "11/14/2019 8:00:00 AM", "N001 N002", "N003-0 N004-1"),
    ]
    for _period, sub in mind.SOURCE_PERIODS.items():
        d = tmp_path / sub
        d.mkdir(parents=True)
        (d / "news.tsv").write_text("".join("\t".join(r) + "\n" for r in news), encoding="utf-8")
        (d / "behaviors.tsv").write_text(
            "".join("\t".join(r) + "\n" for r in beh), encoding="utf-8"
        )

    result = mind.parse(tmp_path)
    assert not result.history_is_per_user_constant
    assert result.counts["users_with_varying_history"] == 1
    # Earliest impression's history wins: N002 was only known later.
    assert set(result.history["article_id"]) == {"N001"}


# ---------------------------------------------------------------------------
# Accounting (Q1.10)
# ---------------------------------------------------------------------------


def test_rows_reconcile(parsed):
    assert parsed.reconciles
    assert parsed.counts["raw"] == 6  # 4 train + 2 dev
    assert parsed.counts["rejected"] == 0
    assert parsed.rejections == {}


def test_malformed_rows_are_rejected_by_reason_not_dropped(tmp_path):
    """A parser that discards quietly makes raw == parsed + rejected untrue, and
    then a 3-row loss is indistinguishable from a 30,000-row loss."""
    news = [(f"N00{i}", "c", "s", f"T{i}", "a", "u", "[]", "[]") for i in range(1, 4)]
    beh = [
        ("1", "U001", "11/09/2019 8:00:00 AM", "", "N001-1 N002-0"),  # good
        ("2", "U002", "not a timestamp", "", "N001-1"),  # unparseable_time
        ("3", "U003", "11/09/2019 9:00:00 AM", "", ""),  # empty_candidate_list
        ("4", "U004", "11/09/2019 9:00:00 AM", "", "N001-1 N002"),  # malformed token
    ]
    for _period, sub in mind.SOURCE_PERIODS.items():
        d = tmp_path / sub
        d.mkdir(parents=True)
        (d / "news.tsv").write_text("".join("\t".join(r) + "\n" for r in news), encoding="utf-8")
        (d / "behaviors.tsv").write_text(
            "".join("\t".join(r) + "\n" for r in beh), encoding="utf-8"
        )

    result = mind.parse(tmp_path)
    assert result.reconciles
    assert result.counts["raw"] == 8
    assert result.counts["parsed"] == 2
    assert result.counts["rejected"] == 6
    assert result.rejections["unparseable_time"] == 2
    assert result.rejections["empty_candidate_list"] == 2
    assert result.rejections["malformed_candidate_token"] == 2


def test_labels_align_with_candidates(parsed):
    """clicked_ids must be the candidates suffixed -1, positionally.

    An off-by-one here produces a plausible frame with the wrong positives, and
    every downstream metric is quietly measuring the wrong thing.
    """
    row = parsed.impressions.filter(pl.col("impression_id") == "train:1").row(0, named=True)
    assert row["candidate_ids"] == ["N003", "N004", "N005"]
    assert row["clicked_ids"] == ["N003"]

    row = parsed.impressions.filter(pl.col("impression_id") == "train:3").row(0, named=True)
    assert row["clicked_ids"] == ["N004"]


def test_cold_start_user_has_no_history_rather_than_an_error(parsed):
    """MIND has 3,238 train impressions with empty history -- a natural
    cold-start slice for Q4.5, not a parse failure."""
    assert "U003" in set(parsed.impressions["user_id"])
    assert parsed.history.filter(pl.col("user_id") == "U003").is_empty()


def test_history_rank_preserves_source_order(parsed):
    """MIND's history order IS its recency signal, and must survive parsing.

    The official documentation states "the clicked news articles are ordered by
    time", and MIND ships no click_time, so position is the only recency
    information available. An earlier version of the parser ended with
    `.sort("user_id", "article_id")` and replaced that order with an
    alphabetical one — silently, since the rows and counts were all still
    correct. It was found only when Q2 built a query and got nothing
    (design_rationale.md 1.16).

    The fixture's first user has history "N001 N002", so N001 must rank 0.
    """
    hist = parsed.history.filter(pl.col("user_id") == "U001").sort("history_rank")
    assert hist["article_id"].to_list() == ["N001", "N002"]
    assert hist["history_rank"].to_list() == [0, 1]


def test_history_rank_is_dense_and_zero_based_per_user(parsed):
    for user in parsed.history["user_id"].unique().to_list():
        ranks = parsed.history.filter(pl.col("user_id") == user)["history_rank"].to_list()
        assert sorted(ranks) == list(range(len(ranks))), f"{user}: ranks {ranks}"


def test_history_is_stored_in_rank_order_not_alphabetical(parsed):
    """The regression itself, pinned.

    A reader tidying this later might restore the alphabetical sort for
    "determinism". Both orders are deterministic; only one carries information.
    """
    for user in parsed.history["user_id"].unique().to_list():
        ranks = parsed.history.filter(pl.col("user_id") == user)["history_rank"].to_list()
        assert ranks == sorted(ranks), f"{user}: history not stored in rank order"


def test_missing_source_file_errors_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="make download"):
        mind.parse(tmp_path)


# ---------------------------------------------------------------------------
# Real-data canaries (Q1.11)
# ---------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.slow
def test_real_mind_known_counts():
    """Counted from behaviors.tsv on 2026-08-19. If a parser change moves these,
    it is a bug until proven otherwise."""
    root = config.RAW_DIR / "MIND"
    if not (root / "MINDsmall_train" / "behaviors.tsv").exists():
        pytest.skip("MIND not downloaded")

    result = mind.parse(root)
    assert result.counts["raw"] == 156965 + 73152
    assert result.counts["rejected"] == 0
    assert result.reconciles
    assert result.article_counts["raw"] == 51282 + 42416
    assert result.article_counts["unique"] == 65238
    assert result.article_counts["duplicates"] == 28460
    assert result.history_is_per_user_constant
    assert result.impressions["impression_id"].n_unique() == 156965 + 73152
    assert result.impressions["user_id"].n_unique() == 50000 + 50000 - 5943
