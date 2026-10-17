"""Q9.2-Q9.6 — the load-bearing tests.

Q9 asks for a test asserting the behaviour-window boundary. A leakage test that
never exercises its assertion passes forever and proves nothing, so this file
also contains the canary that proves the checker can fail.

The canary feeds the **same functions the pipeline runs** — `newsrec.data.leakage`
— deliberately corrupt input. A canary exercising a test-local copy of the logic
would prove the copy works, which is not the claim being made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store, leakage
from newsrec.retrieval import tokenise

UTC_DT = pl.Datetime("us", "UTC")


def test_serving_unavailable_set_matches_config():
    """Q9.2 — the set is defined once, and this is what it must contain.

    Runs today: it needs no pipeline, only config. If someone widens the set to
    make a metric look better, this fails.
    """
    assert config.POST_CLICK_FEATURES == {
        "read_time",
        "scroll_percentage",
        "next_read_time",
        "next_scroll_percentage",
    }
    assert config.FUTURE_AGGREGATE_FEATURES == {
        "total_inviews",
        "total_pageviews",
        "total_read_time",
    }
    assert config.SERVING_UNAVAILABLE_FEATURES == (
        config.POST_CLICK_FEATURES | config.FUTURE_AGGREGATE_FEATURES
    )


# ---------------------------------------------------------------------------
# Q9.4 — the per-impression window, on a real store
# ---------------------------------------------------------------------------


def test_history_strictly_precedes_impression(processed_store, dataset):
    """Q9.4 — strict <, never <=.

    On MIND click_time is null (no timestamps are distributed), so the boundary
    is structural there and this check is vacuous BY DESIGN, not by accident.
    Asserting it anyway on a null column would be a test that cannot fail —
    which is exactly what the canary below exists to rule out.

    Note what is checked and what is not. The raw history table is NOT expected
    to precede every impression: EB-NeRD's overlapping rolling windows put
    811,565 clicks after the earliest impression in `ebnerd_small`, by design.
    What must hold is that the history **as selected for an impression** is
    strictly earlier. So this runs the production selector and then validates
    its output — the same validator the canary breaks.
    """
    hist = feature_store.read_table(processed_store / "history.parquet")
    if hist["click_time"].null_count() == len(hist):
        pytest.skip(f"{dataset} ships no history timestamps; boundary is structural")

    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    pairs = leakage.select_history_before(hist, impressions)
    assert len(pairs) > 0, "selector returned nothing; the check would be vacuous"
    assert len(leakage.find_window_violations(pairs)) == 0


def test_raw_history_may_legitimately_extend_past_impressions(processed_store, dataset):
    """The corollary, asserted so nobody 'fixes' it later.

    Someone reading Q9.4 could reasonably tighten the pipeline until no history
    row follows any impression. That would silently discard real past clicks for
    later impressions and make the leakage test vacuous at the same time. This
    records that the overlap is expected, and that safety comes from the
    selector rather than from pruning the table.
    """
    hist = feature_store.read_table(processed_store / "history.parquet")
    if hist["click_time"].null_count() == len(hist):
        pytest.skip(f"{dataset} ships no history timestamps")

    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    earliest = impressions["impression_time"].min()
    later = hist.filter(pl.col("click_time") > earliest)
    assert len(later) > 0, (
        "history no longer overlaps impression time; the Q9.4 selector has "
        "nothing to exclude and this suite's boundary checks have gone vacuous"
    )


# ---------------------------------------------------------------------------
# Q9.5 — split-level boundaries
# ---------------------------------------------------------------------------


def test_features_derive_only_from_prior_rows(processed_store, feature_store_dir, dataset):
    """Q9.5 — features for split S use only rows before S's start boundary.

    This is the check that catches the boundary bug from design_rationale §2.6,
    where train was bounded by the split's END and could see four days of its
    own future while every count still looked plausible.
    """
    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    user_features = feature_store.read_table(feature_store_dir / "user_features.parquet")
    if not user_features["history_is_timestamped"].any():
        pytest.skip(f"{dataset} ships no history timestamps; boundary is structural")

    starts = feature_store.split_start_times(impressions)
    assert len(leakage.find_user_feature_violations(user_features, starts)) == 0


def test_no_id_spans_two_splits(processed_store, dataset):
    """Q9.5 — no impression_id and no (user_id, impression_time) in two splits.

    The second form is the one worth having: duplicate rows under different ids
    would put the same real event on both sides of the boundary, which a
    unique-id check cannot see.
    """
    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    assert len(leakage.find_ids_spanning_splits(impressions)) == 0


def test_no_unavailable_features_in_headline_frames(processed_store, dataset):
    """Q9.3 — the seven columns appear only in frames flagged as ablations."""
    for name in ("articles", "impressions", "history"):
        cols = set(feature_store.read_table(processed_store / f"{name}.parquet").columns)
        leaked = cols & config.SERVING_UNAVAILABLE_FEATURES
        assert not leaked, f"{name} exposes serving-unavailable columns: {sorted(leaked)}"


def test_unavailable_features_are_retained_in_the_extras(processed_store, dataset):
    """Q9.1's precondition, and the reason Q9.3 is a move rather than a delete.

    You cannot report a metric with and without a feature you discarded at parse
    time. EB-NeRD ships all seven, so they must be present somewhere.
    """
    if not dataset.startswith("ebnerd"):
        pytest.skip("MIND ships none of the serving-unavailable fields")
    extras: set[str] = set()
    for name in ("articles_extra", "impressions_extra"):
        path = processed_store / f"{name}.parquet"
        if path.exists():
            extras |= set(feature_store.read_table(path).columns)
    assert config.SERVING_UNAVAILABLE_FEATURES <= extras


def test_build_records_a_leakage_report(processed_store, dataset):
    """Q9 runs as a build gate, and its outcome is visible without running tests."""
    manifest = json.loads((processed_store / "_manifest.json").read_text())
    report = manifest["leakage"]
    assert report["problems"] == []
    assert report["window_violations"] == 0
    assert report["ids_spanning_splits"] == 0
    # "checked and clean" must be distinguishable from "could not be checked".
    expected = "structure" if dataset == "mind_small" else "timestamp"
    assert report["boundary_enforced_by"] == expected


# ---------------------------------------------------------------------------
# Still guarded: these need retrieval stages that do not exist yet
# ---------------------------------------------------------------------------


def test_bm25_query_respects_boundary(processed_store, dataset):
    """Q2.5 — the BM25 query uses only clicks strictly before the impression.

    Unguarded now that `newsrec.retrieval.query` exists. This runs against the
    REAL built store, which the canaries in tests/test_query.py cannot: they use
    manufactured fixtures precisely because real data never exercises the
    boundary. Both are needed — the canaries prove the check can fail, this
    proves it holds on the actual corpus.
    """
    from newsrec.retrieval import query

    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    history = feature_store.read_table(processed_store / "history.parquet")
    articles = feature_store.read_table(processed_store / "articles.parquet")
    spec = config.tokeniser_for(dataset)

    if history["click_time"].null_count() == len(history):
        pytest.skip("dataset ships no history timestamps; boundary is structural")

    # A seeded sample: the join is users x impressions x clicks and the point is
    # a spot audit, not a proof (the same reasoning as leakage.check_store).
    sample = impressions.sample(n=min(200, len(impressions)), seed=config.GLOBAL_SEED)

    titles = dict(articles.select("article_id", "title").drop_nulls("title").iter_rows())
    checked = 0
    for row in sample.iter_rows(named=True):
        terms = query.build_query(row, history, articles, spec)
        if not terms:
            continue
        checked += 1

        # Every term must be traceable to a title clicked strictly earlier.
        legitimate = history.filter(
            (pl.col("user_id") == row["user_id"])
            & pl.col("click_time").is_not_null()
            & (pl.col("click_time") < row["impression_time"])
        )["article_id"].to_list()
        allowed = set()
        for article_id in legitimate:
            if (title := titles.get(article_id)) is not None:
                allowed.update(tokenise.tokenise(title, spec))
        assert set(terms) <= allowed, (
            f"{row['impression_id']}: query contains terms with no legitimate source"
        )

    assert checked > 0, "sample produced no non-empty queries; the test asserted nothing"


def test_user_embedding_respects_boundary(processed_store, dataset):
    """Q3.4 — the user vector pools only clicks strictly before the impression.

    Unguarded now that `newsrec.retrieval.user_vector` exists. This runs against
    the REAL built store; the canaries in tests/test_user_vector.py use
    manufactured fixtures. Both are needed — the canaries prove the check can
    fail on a case constructed to be unambiguous, this proves it holds across the
    actual corpus.

    **Measured correction to a claim this project has repeated.** Elsewhere the
    codebase says a click and an impression "essentially never share a
    microsecond", so `<=` in place of `<` would change no number. That is **not
    true of EB-NeRD**: `ebnerd_demo` has **6,518** (user, click, impression)
    triples where `click_time` is *exactly* `impression_time` (0.0341% of
    19.1M pairs, across 784 users), and `ebnerd_small` has **62,030** (0.0336%,
    7,298 users). Mutating the selector to `<=` makes this test fail on real
    data, not merely the manufactured canary. The boundary is observably
    load-bearing rather than only theoretically so.

    The assertion is exact rather than statistical. A user vector is a weighted
    mean of specific article vectors, so it can be *recomputed* from the clicks
    that were legitimately available and compared. If a single future click had
    contributed, the recomputation would not match — and unlike a recall number,
    a vector mismatch has no plausible-looking failure mode.
    """
    import numpy as np

    from newsrec.retrieval import embed, user_vector

    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    history = feature_store.read_table(processed_store / "history.parquet")

    if history["click_time"].null_count() == len(history):
        pytest.skip("dataset ships no history timestamps; boundary is structural")

    # Stand-in vectors, one per article, deterministic per id. The boundary is a
    # property of WHICH vectors are pooled, not of what the encoder produced, so
    # this needs no model and runs in a clean clone with no network (OQ-9).
    article_ids = [
        str(a) for a in feature_store.read_table(processed_store / "articles.parquet")["article_id"]
    ]
    rng = np.random.default_rng(config.GLOBAL_SEED)
    vectors = rng.normal(size=(len(article_ids), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    embeddings = embed.ArticleEmbeddings(
        vectors=vectors,
        article_ids=article_ids,
        spec=config.EncoderSpec(
            model_id="test/stand-in",
            revision="0" * 40,
            dim=8,
            max_seq_length=128,
            passage_prefix="passage: ",
            query_prefix="query: ",
        ),
    )
    row_of = {a: i for i, a in enumerate(article_ids)}

    sample = impressions.sample(n=min(200, len(impressions)), seed=config.GLOBAL_SEED)
    users = sample["user_id"].unique().to_list()
    matrix, has_vector = user_vector.user_vectors(
        sample.select("impression_id", "user_id", "impression_time"),
        history.filter(pl.col("user_id").is_in(users)),
        embeddings,
    )

    checked = 0
    for i, row in enumerate(sample.iter_rows(named=True)):
        if not has_vector[i]:
            continue

        # Recompute from clicks that were legitimately available: strictly
        # before, most recent first, capped, corpus articles only.
        legitimate = (
            history.filter(
                (pl.col("user_id") == row["user_id"])
                & pl.col("click_time").is_not_null()
                & (pl.col("click_time") < row["impression_time"])
            )
            .sort(["click_time", "article_id"], descending=[True, False])
            .head(config.QUERY_HISTORY_LIMIT)["article_id"]
            .to_list()
        )
        rows = [row_of[a] for a in legitimate if a in row_of]
        if not rows:
            continue
        checked += 1
        assert np.allclose(matrix[i], user_vector.pool(embeddings.vectors[rows]), atol=1e-5), (
            f"{row['impression_id']}: user vector does not match a pooling of its "
            f"legitimately-available clicks — a future click contributed"
        )

    assert checked > 0, "sample produced no warm user vectors; the test asserted nothing"


# ---------------------------------------------------------------------------
# select_history_for — the two boundary rules, and the limit of the second
# ---------------------------------------------------------------------------

_IMP = pl.DataFrame(
    {
        "impression_id": ["i1"],
        "user_id": ["u1"],
        "impression_time": [datetime(2019, 11, 15, 12, tzinfo=UTC)],
    },
    schema={"impression_id": pl.Utf8, "user_id": pl.Utf8, "impression_time": UTC_DT},
)


def _mind_history(article_ids, ranks) -> pl.DataFrame:
    """MIND-shaped history: no timestamps, ordered by history_rank."""
    return pl.DataFrame(
        {
            "user_id": ["u1"] * len(article_ids),
            "article_id": list(article_ids),
            "click_time": [None] * len(article_ids),
            "history_rank": list(ranks),
        },
        schema={
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "click_time": UTC_DT,
            "history_rank": pl.Int32,
        },
    )


def _ebnerd_history(article_ids, times) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": ["u1"] * len(article_ids),
            "article_id": list(article_ids),
            "click_time": list(times),
            "history_rank": [None] * len(article_ids),
        },
        schema={
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "click_time": UTC_DT,
            "history_rank": pl.Int32,
        },
    )


def test_timestamp_dispatch_is_decided_by_the_data_not_a_dataset_name():
    assert leakage.history_is_timestamped(
        _ebnerd_history(["a"], [datetime(2019, 11, 1, tzinfo=UTC)])
    )
    assert not leakage.history_is_timestamped(_mind_history(["a"], [0]))


def test_untimestamped_history_returns_everything_ordered_by_rank():
    """The fix for the 100%-empty-query failure.

    Under the timestamp rule this returns nothing, because `null < anything` is
    null and the filter drops every row. Ordered most-recent-first, so the D6
    cap takes the newest clicks.
    """
    hist = _mind_history(["a", "b", "c"], [0, 1, 2])
    got = leakage.select_history_for(hist, _IMP)
    assert got["article_id"].to_list() == ["c", "b", "a"]
    assert leakage.select_history_before(hist, _IMP).is_empty()


def test_timestamped_history_still_applies_the_strict_boundary():
    """The EB-NeRD path is unchanged: the future click must not appear."""
    hist = _ebnerd_history(
        ["past", "boundary", "future"],
        [
            datetime(2019, 11, 14, tzinfo=UTC),
            datetime(2019, 11, 15, 12, tzinfo=UTC),  # exactly the impression
            datetime(2019, 11, 16, tzinfo=UTC),
        ],
    )
    got = leakage.select_history_for(hist, _IMP)
    assert got["article_id"].to_list() == ["past"]


def test_both_paths_return_the_same_columns():
    """Callers must not need to know which rule fired."""
    a = leakage.select_history_for(_mind_history(["a"], [0]), _IMP)
    b = leakage.select_history_for(
        _ebnerd_history(["a"], [datetime(2019, 11, 1, tzinfo=UTC)]), _IMP
    )
    assert tuple(a.columns) == tuple(b.columns) == leakage.SELECTION_COLUMNS


def test_history_with_neither_timestamp_nor_rank_is_refused():
    """Silently returning nothing is what caused the original bug. Raise."""
    broken = pl.DataFrame(
        {"user_id": ["u1"], "article_id": ["a"], "click_time": [None]},
        schema={"user_id": pl.Utf8, "article_id": pl.Utf8, "click_time": UTC_DT},
    )
    with pytest.raises(leakage.LeakageError, match="history_rank"):
        leakage.select_history_for(broken, _IMP)


def test_mind_future_row_is_NOT_caught_by_timestamp_check_structural_boundary_only():
    """THE CANARY. It asserts a limit, not a guarantee — read carefully.

    A MIND history row carrying a click *after* the impression is injected. The
    timestamp validator reports **zero violations**, and that is correct
    behaviour rather than a bug: MIND stores no click times, so there is nothing
    for a timestamp comparison to compare. `find_window_violations` explicitly
    treats a null `click_time` as unverifiable rather than as clean.

    What this pins is that MIND's safety rests entirely on `mind.py` keeping each
    user's EARLIEST impression's history (design_rationale §1.7), and NOT on any
    check in this module. If someone changed the parser to keep the latest
    history instead, future clicks would flow into every query and every leakage
    test here would still pass.

    The control at the end is what makes the test meaningful: the same injected
    row, given a real timestamp, IS caught. So the checker works; it simply
    cannot see through a column of nulls.
    """
    future = datetime(2019, 11, 20, tzinfo=UTC)
    hist = _mind_history(["past", "sneaky_future"], [0, 1])

    selected = leakage.select_history_for(hist, _IMP)
    assert "sneaky_future" in selected["article_id"].to_list(), (
        "the canary needs the injected row to be selected, or it proves nothing"
    )

    # The point: the timestamp check does not, and cannot, catch it.
    assert len(leakage.find_window_violations(selected)) == 0

    # And the report says so honestly rather than claiming a clean bill.
    assert not leakage.history_is_timestamped(hist)

    # CONTROL. Give that identical row a real timestamp and it is caught,
    # proving the validator bites and that the null column is the whole reason.
    timestamped = _ebnerd_history(
        ["past", "sneaky_future"],
        [
            datetime(2019, 11, 14, tzinfo=UTC),
            future,
        ],
    )
    pairs = _IMP.join(
        timestamped.select("user_id", "article_id", "click_time"), on="user_id", how="inner"
    )
    caught = leakage.find_window_violations(pairs)
    assert caught["article_id"].to_list() == ["sneaky_future"]


# ---------------------------------------------------------------------------
# Q9.6 — the canary
# ---------------------------------------------------------------------------


def _pairs_from_case(case: dict) -> pl.DataFrame:
    """Turn a canary case into the frame the production validator consumes.

    The fixture asserts that these history entries WOULD be used for this
    impression. The validator's job is to notice that some of them must not be.
    """
    impression_time = datetime.fromisoformat(case["impression_time"])
    return pl.DataFrame(
        {
            "impression_id": [f"{case['user_id']}-imp"] * len(case["history"]),
            "user_id": [case["user_id"]] * len(case["history"]),
            "impression_time": [impression_time] * len(case["history"]),
            "article_id": [h["article_id"] for h in case["history"]],
            "click_time": [datetime.fromisoformat(h["click_time"]) for h in case["history"]],
            "split": [case.get("split", "train")] * len(case["history"]),
        },
        schema={
            "impression_id": pl.Utf8,
            "user_id": pl.Utf8,
            "impression_time": pl.Datetime("us"),
            "article_id": pl.Utf8,
            "click_time": pl.Datetime("us"),
            "split": pl.Utf8,
        },
    ).with_columns(
        pl.col("impression_time").dt.replace_time_zone("UTC"),
        pl.col("click_time").dt.replace_time_zone("UTC"),
    )


def test_canary_fixture_is_well_formed(leakage_canary):
    """Runs today. The canary is worthless if it is not itself correct, so its
    shape is checked even before a checker exists to consume it."""
    cases = leakage_canary["cases"]
    assert set(cases) == {"future_click", "boundary_equal", "cross_split"}
    for name, case in cases.items():
        illegal = [h for h in case["history"] if not h["legal"]]
        assert illegal, f"canary case {name!r} contains no actual violation"
    assert all(h["legal"] for h in leakage_canary["clean_control"]["history"])


@pytest.mark.canary
@pytest.mark.parametrize("case", ["future_click", "boundary_equal"])
def test_canary_future_click_is_detected(leakage_canary, case):
    """Q9.6 — the checker MUST flag each deliberately corrupt case.

    boundary_equal is the one that matters most: a click at exactly the
    impression timestamp is what an off-by-one (<= instead of <) lets through,
    and it is indistinguishable from a click caused by the impression.

    The count is asserted, not just "something was found" — a checker that
    flagged every row would also pass a non-empty assertion.
    """
    spec = leakage_canary["cases"][case]
    pairs = _pairs_from_case(spec)
    violations = leakage.find_window_violations(pairs)

    expected = [h["article_id"] for h in spec["history"] if not h["legal"]]
    assert sorted(violations["article_id"].to_list()) == sorted(expected), (
        f"canary case {case!r}: expected exactly {expected} to be flagged"
    )


@pytest.mark.canary
def test_canary_cross_split_is_detected(leakage_canary):
    """Q9.5's canary: a click legitimately in an impression's past, but inside
    the split the impression belongs to.

    Passes the per-impression check and must fail the split check — which is
    why the two are separate functions.
    """
    spec = leakage_canary["cases"]["cross_split"]
    pairs = _pairs_from_case(spec)

    assert len(leakage.find_window_violations(pairs)) == 0, (
        "cross_split must be legal per-impression; otherwise it is not testing "
        "the split rule at all"
    )

    starts = {spec["split"]: datetime.fromisoformat(spec["split_start"]).replace(tzinfo=UTC)}
    violations = leakage.find_split_violations(pairs, starts)
    expected = [h["article_id"] for h in spec["history"] if not h["legal"]]
    assert sorted(violations["article_id"].to_list()) == sorted(expected)


@pytest.mark.canary
def test_canary_selector_excludes_equal_timestamps(leakage_canary):
    """Q9.6 for the SELECTOR, not just the validator.

    Found by mutation testing: changing `select_history_before` from `<` to
    `<=` broke no test. The validator's canary could not catch it, because the
    validator only ever sees what the selector hands it.

    The second reason originally given -- that real data never puts a click and
    an impression in the same microsecond -- is FALSE, measured 2026-08-22:
    ebnerd_demo has 6,518 such pairs (0.0341%) and ebnerd_small 62,030 (0.0336%).
    The off-by-one was undetectable because nothing LOOKED at those rows, not
    because they were absent. 0.03% of rows shifting inside an aggregate moves no
    number anyone was checking.

    This feeds the selector a hand-built case where a click lands EXACTLY on the
    impression time and asserts it is excluded. Still the right tool: a fixture
    isolates the boundary so a failure names its cause.
    """
    spec = leakage_canary["cases"]["boundary_equal"]
    impression_time = datetime.fromisoformat(spec["impression_time"]).replace(tzinfo=UTC)

    history = pl.DataFrame(
        {
            "user_id": [spec["user_id"]] * len(spec["history"]),
            "article_id": [h["article_id"] for h in spec["history"]],
            "click_time": [
                datetime.fromisoformat(h["click_time"]).replace(tzinfo=UTC) for h in spec["history"]
            ],
        },
        schema={"user_id": pl.Utf8, "article_id": pl.Utf8, "click_time": UTC_DT},
    )
    impressions = pl.DataFrame(
        {
            "impression_id": ["I1"],
            "user_id": [spec["user_id"]],
            "impression_time": [impression_time],
        },
        schema={"impression_id": pl.Utf8, "user_id": pl.Utf8, "impression_time": UTC_DT},
    )

    selected = leakage.select_history_before(history, impressions)
    kept = set(selected["article_id"].to_list())
    legal = {h["article_id"] for h in spec["history"] if h["legal"]}
    illegal = {h["article_id"] for h in spec["history"] if not h["legal"]}

    assert kept == legal, "selector kept the wrong set of clicks"
    assert not (kept & illegal), (
        "selector admitted a click timestamped exactly at the impression -- "
        "indistinguishable from a click CAUSED by that impression"
    )


@pytest.mark.canary
def test_canary_clean_control_is_accepted(leakage_canary):
    """The counterpart: the checker must ACCEPT the clean control.

    Without this, a checker that rejects everything would pass every canary
    test above and still be useless.
    """
    spec = leakage_canary["clean_control"]
    pairs = _pairs_from_case(spec)
    assert len(leakage.find_window_violations(pairs)) == 0

    starts = {spec["split"]: datetime.fromisoformat(spec["split_start"]).replace(tzinfo=UTC)}
    assert len(leakage.find_split_violations(pairs, starts)) == 0


@pytest.mark.canary
def test_corrupting_raw_history_does_NOT_trip_the_gate(fixture_raw_root, tmp_path, dataset):
    """Deliberately documents a limit of the store-level check.

    Pushing a history click past every impression does not make `check_store`
    fail -- and it should not. The selector filters that click out, which is
    exactly the protection working. The raw history table is allowed to contain
    clicks after some impressions (EB-NeRD ships 811,565 such rows), so a gate
    that fired here would fail on correct data.

    Written down as a test because the obvious next move on reading Q9.4 is to
    add such a check, and this records why that would be wrong. The thing that
    proves the boundary logic bites is the canary on find_window_violations,
    which feeds the validator a frame the selector never produced.
    """
    raw = fixture_raw_root / ("MIND" if dataset == "mind_small" else dataset)
    feature_store.build_dataset(dataset, raw_root=raw, root=tmp_path)
    tables = feature_store.read_processed(dataset, tmp_path / "processed")
    if tables["history"]["click_time"].null_count() == len(tables["history"]):
        pytest.skip(f"{dataset} has no history timestamps to corrupt")

    latest = tables["impressions"]["impression_time"].max()
    tables["history"] = tables["history"].with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(latest).dt.offset_by("1d"))
        .otherwise(pl.col("click_time"))
        .alias("click_time")
    )
    report = leakage.check_store(tables, sample_size=10_000)
    assert report["problems"] == [], "the selector should have excluded the future click"


@pytest.mark.canary
def test_check_store_raises_on_a_leaky_user_feature(fixture_raw_root, tmp_path, dataset):
    """The build gate must fail on the bug that actually happened.

    Reproduces design_rationale §2.6 directly: a user-feature row for the train
    split whose most recent click falls after the train split opened. That is
    the shape of the four-days-of-future bug, and the gate must stop it.
    """
    raw = fixture_raw_root / ("MIND" if dataset == "mind_small" else dataset)
    feature_store.build_dataset(dataset, raw_root=raw, root=tmp_path)
    tables = feature_store.read_processed(dataset, tmp_path / "processed")
    features = feature_store.read_feature_store(dataset, tmp_path / "feature_store")
    user_features = features["user_features"]
    if not user_features["history_is_timestamped"].any():
        pytest.skip(f"{dataset} has no history timestamps")

    starts = feature_store.split_start_times(tables["impressions"])
    poisoned = user_features.with_columns(
        pl.when(pl.col("split") == "train")
        .then(pl.lit(starts["train"]).dt.offset_by("1h"))
        .otherwise(pl.col("last_click_time"))
        .alias("last_click_time")
    )
    with pytest.raises(leakage.LeakageError, match=r"Q9\.5"):
        leakage.check_store(tables, poisoned)


@pytest.mark.canary
def test_check_store_raises_on_an_id_spanning_splits(fixture_raw_root, tmp_path, dataset):
    """Q9.5: the same event on both sides of the boundary is contamination."""
    raw = fixture_raw_root / ("MIND" if dataset == "mind_small" else dataset)
    feature_store.build_dataset(dataset, raw_root=raw, root=tmp_path)
    tables = feature_store.read_processed(dataset, tmp_path / "processed")

    first = (
        tables["impressions"]
        .head(1)
        .with_columns(
            pl.when(pl.col("split") == "train")
            .then(pl.lit("test"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
    )
    tables["impressions"] = pl.concat([tables["impressions"], first], how="vertical")
    with pytest.raises(leakage.LeakageError, match=r"Q9\.5"):
        leakage.check_store(tables)


@pytest.mark.canary
def test_check_store_raises_on_a_serving_unavailable_column(fixture_raw_root, tmp_path, dataset):
    """Q9.3 as a gate: a forbidden column in a core table must stop the build."""
    raw = fixture_raw_root / ("MIND" if dataset == "mind_small" else dataset)
    feature_store.build_dataset(dataset, raw_root=raw, root=tmp_path)
    tables = feature_store.read_processed(dataset, tmp_path / "processed")

    tables["impressions"] = tables["impressions"].with_columns(
        pl.lit(1.0).alias("scroll_percentage")
    )
    with pytest.raises(leakage.LeakageError, match=r"Q9\.3"):
        leakage.check_store(tables)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.parametrize("ds", ["mind_small", "ebnerd_demo", "ebnerd_small"])
def test_real_store_has_no_leakage(ds):
    store = config.PROCESSED_DIR / ds
    if not store.is_dir():
        pytest.skip(f"{ds} store not built; run `make data`")

    tables = feature_store.read_processed(ds)
    features = feature_store.read_feature_store(ds)
    report = leakage.check_store(tables, features["user_features"])
    assert report["problems"] == []
    assert report["window_violations"] == 0
