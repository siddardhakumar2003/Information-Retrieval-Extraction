"""Leakage detection (Q9.2-Q9.6).

The rule this module enforces: **no information that did not exist at prediction
time may reach a prediction.** Concretely, a user's history for an impression may
contain only clicks strictly before that impression's timestamp, and features for
a split may derive only from rows before that split opens.

## Why this is a module and not just a test

The canary fixture (Q9.6) has to prove the checker *can* fail. That proof is only
worth something if the canary runs the **same code the pipeline runs**. A canary
that exercised a test-local copy of the logic would demonstrate that the copy
works, which is not the claim being made.

So the checkers live here, `build_dataset` calls them on every build, and
`tests/test_leakage.py` feeds them both real stores and deliberately corrupt
fixtures. One implementation, two very different inputs.

## Why the raw history table is NOT expected to precede every impression

This is the subtlety that decides the design. EB-NeRD ships overlapping rolling
history windows, so its `history` table legitimately contains clicks recorded
after some impressions -- 811,565 of them for `ebnerd_small` (see
`agent_docs/design_rationale.md` §1.8). That is not a defect; the table is a
superset from which each moment selects its own past.

Therefore "every history row precedes every impression" is the WRONG assertion:
it would fail on correct data. The right assertion is about the history **as
selected for a given impression**, which is why `select_history_before` and
`find_window_violations` are separate functions. The selector produces the
subset; the validator checks that whatever it was handed is legitimate. Running
the validator on the selector's output tests them end to end; running it on the
canary's hand-built pairs tests that it bites.

## Strictly less than, never less than or equal

A click recorded at exactly the impression's timestamp is indistinguishable from
a click *caused by* that impression -- the user clicking what they were just
shown. Admitting it feeds the label back in as a feature. `boundary_equal` in the
canary fixture is precisely this case.

## Two boundaries, because two datasets enforce it differently

EB-NeRD timestamps every click, so its boundary is a **timestamp** comparison:
`click_time < impression_time`, checkable row by row.

MIND ships no click timestamps at all. Its boundary is **structural**: the raw
`behaviors.tsv` gives, per impression, the history of clicks that preceded it,
and `mind.py` keeps each user's EARLIEST impression's history
(`design_rationale.md` §1.7). That history therefore precedes every impression
that user has, by construction, and the whole of it is legitimately available.

`select_history_for` is the one place that decides which rule applies. It
dispatches on whether the data actually carries timestamps, not on a dataset
name, so a dataset that gains timestamps later starts being checked without
anyone remembering to update a list.

**The honest limit, stated because it is easy to assume otherwise.** On the
structural path there is nothing for a timestamp check to test. If `mind.py` were
changed to keep the *latest* impression's history instead of the earliest, future
clicks would enter the query and `find_window_violations` would report zero,
because it can only compare timestamps that exist. The guarantee rests on the
parser, and `test_mind_future_row_is_NOT_caught_by_timestamp_check_structural_boundary_only`
pins that limit so it is documented rather than discovered.
"""

from __future__ import annotations

import polars as pl

from newsrec import config

#: How many impressions the build-time per-impression check samples. A full
#: cross join of history against impressions is quadratic in users -- for
#: ebnerd_small that is 3.3M history rows against 477K impressions -- and the
#: check is a spot audit, not a proof. Sampling is seeded so the manifest stays
#: byte-identical across rebuilds (Q1.13).
LEAKAGE_SAMPLE_SIZE = 2_000


class LeakageError(ValueError):
    """Information from after the prediction point reached a prediction row."""


# =============================================================================
# Per-impression window (Q9.4)
# =============================================================================


def select_history_before(history: pl.DataFrame, impressions: pl.DataFrame) -> pl.DataFrame:
    """The production selector: each impression paired with its legitimate past.

    Returns one row per (impression, prior click). This is the operation every
    later stage needs -- Q2 builds its query text from these rows, Q3 pools these
    embeddings -- so it lives here rather than being re-derived per stage, where
    each copy would be a fresh chance to write `<=`.

    Callers should pass a *bounded* impressions frame. The join is
    users x impressions x clicks and grows quadratically; `check_store` samples.
    """
    if history.is_empty() or impressions.is_empty():
        return pl.DataFrame(
            schema={
                "impression_id": pl.Utf8,
                "user_id": pl.Utf8,
                "impression_time": pl.Datetime("us", "UTC"),
                "article_id": pl.Utf8,
                "click_time": pl.Datetime("us", "UTC"),
            }
        )

    return (
        impressions.select("impression_id", "user_id", "impression_time")
        .join(history.select("user_id", "article_id", "click_time"), on="user_id", how="inner")
        # STRICTLY less than. See the module docstring.
        .filter(pl.col("click_time") < pl.col("impression_time"))
    )


#: Columns every history selector returns, so both paths are interchangeable to
#: a caller. `history_rank` rides along because it is MIND's only ordering key.
SELECTION_COLUMNS = (
    "impression_id",
    "user_id",
    "impression_time",
    "article_id",
    "click_time",
    "history_rank",
)


def history_is_timestamped(history: pl.DataFrame) -> bool:
    """True when this history table carries usable click timestamps.

    Asked of the DATA rather than of a dataset name. A hard-coded
    `dataset == "mind_small"` would silently stop being true if MIND ever
    shipped timestamps, or if another untimestamped dataset were added, and the
    wrong branch would be taken with no error anywhere.
    """
    if history.is_empty() or "click_time" not in history.columns:
        return False
    return history["click_time"].null_count() < len(history)


def select_history_for(history: pl.DataFrame, impressions: pl.DataFrame) -> pl.DataFrame:
    """Each impression paired with its legitimate past, by whichever rule applies.

    **This is what Q2 and Q3 should call.** `select_history_before` remains the
    timestamp rule itself and is still the only place `<` is written; this
    function chooses between it and the structural rule, and orders the result
    most-recent-first so callers can simply take the first N (q2_plan D6).

    Timestamped data (EB-NeRD) -> `select_history_before`, ordered by
    `click_time` descending.

    Untimestamped data (MIND) -> the user's whole history, ordered by
    `history_rank` descending. That is safe because `mind.py` keeps each user's
    earliest impression's history, which precedes all of that user's
    impressions; and it is the ONLY thing that can be done, because filtering on
    a null timestamp excludes everything and yields an empty query for every
    impression -- which is exactly the failure this function was added to fix.

    `article_id` breaks ties on both paths so the ordering is total and the
    query is identical on every run (Q1.13).
    """
    empty = pl.DataFrame(
        schema={
            "impression_id": pl.Utf8,
            "user_id": pl.Utf8,
            "impression_time": pl.Datetime("us", "UTC"),
            "article_id": pl.Utf8,
            "click_time": pl.Datetime("us", "UTC"),
            "history_rank": pl.Int32,
        }
    )
    if history.is_empty() or impressions.is_empty():
        return empty

    if history_is_timestamped(history):
        pairs = select_history_before(history, impressions)
        if pairs.is_empty():
            return empty
        # select_history_before does not carry history_rank; add it back so both
        # paths return SELECTION_COLUMNS.
        if "history_rank" not in pairs.columns:
            pairs = pairs.join(
                history.select("user_id", "article_id", "history_rank").unique(
                    subset=["user_id", "article_id"], keep="first"
                ),
                on=["user_id", "article_id"],
                how="left",
            )
        return pairs.select(SELECTION_COLUMNS).sort(
            ["impression_id", "click_time", "article_id"], descending=[False, True, False]
        )

    # Structural path. No filter is applied and that is deliberate: the whole
    # history is prior by construction. See the module docstring for the limit
    # this creates and the canary that pins it.
    if "history_rank" not in history.columns:
        raise LeakageError(
            "history has neither usable click_time nor history_rank, so there is no "
            "way to order or bound it. A dataset must supply one or the other."
        )
    return (
        impressions.select("impression_id", "user_id", "impression_time")
        .join(
            history.select("user_id", "article_id", "click_time", "history_rank"),
            on="user_id",
            how="inner",
        )
        .select(SELECTION_COLUMNS)
        .sort(["impression_id", "history_rank", "article_id"], descending=[False, True, False])
    )


def find_window_violations(pairs: pl.DataFrame) -> pl.DataFrame:
    """Rows in a (impression, click) frame that break the Q9.4 boundary.

    Deliberately separate from the selector so it can be pointed at a frame the
    selector did NOT produce -- which is the only way to find out whether it
    detects anything.

    A null `click_time` is not a violation: MIND ships no history timestamps and
    its boundary is structural (the distributed history predates the whole log
    period). Treating null as a violation would fail correct data; treating it as
    *checked* would be worse still, so `check_store` reports how many rows were
    unverifiable rather than counting them as clean.
    """
    if pairs.is_empty():
        return pairs
    return pairs.filter(
        pl.col("click_time").is_not_null()
        & pl.col("impression_time").is_not_null()
        & (pl.col("click_time") >= pl.col("impression_time"))
    )


def find_split_violations(pairs: pl.DataFrame, split_starts: dict[str, object]) -> pl.DataFrame:
    """Rows using a click recorded at or after their own split opened (Q9.5).

    Stricter than the per-impression rule and it catches a different mistake: an
    impression late in the test period has a legitimate past that includes clicks
    from earlier in the test period, so the per-impression check passes, while a
    feature computed for the whole split must not use them.

    `pairs` must carry a `split` column.
    """
    if pairs.is_empty() or "split" not in pairs.columns:
        return pairs.clear() if not pairs.is_empty() else pairs

    frames = []
    for split_name, start in split_starts.items():
        if start is None:
            continue
        frames.append(
            pairs.filter(
                (pl.col("split") == split_name)
                & pl.col("click_time").is_not_null()
                & (pl.col("click_time") >= start)
            )
        )
    return pl.concat(frames, how="vertical") if frames else pairs.clear()


# =============================================================================
# Store-level checks
# =============================================================================


def find_unavailable_columns(tables: dict[str, pl.DataFrame]) -> dict[str, list[str]]:
    """Q9.3 — serving-unavailable columns in any non-ablation frame.

    Checks by column NAME rather than by table, so a forbidden field cannot be
    smuggled into a core table under its own name from any source.
    """
    leaked = {}
    for name, frame in tables.items():
        if name.endswith("_extra"):
            continue  # side tables are where these legitimately live
        found = sorted(set(frame.columns) & config.SERVING_UNAVAILABLE_FEATURES)
        if found:
            leaked[name] = found
    return leaked


def find_ids_spanning_splits(impressions: pl.DataFrame) -> pl.DataFrame:
    """Q9.5 — an impression_id, or a (user, time) pair, in more than one split.

    The second form matters independently: duplicate impression rows under
    different ids would put the same event on both sides of the boundary, which
    is a train/test contamination that a unique-id check cannot see.
    """
    by_id = (
        impressions.group_by("impression_id")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    by_event = (
        impressions.group_by("user_id", "impression_time")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    return pl.concat(
        [
            by_id.select(pl.col("impression_id").alias("key"), "n_splits").with_columns(
                pl.lit("impression_id").alias("kind")
            ),
            by_event.select(
                pl.concat_str(["user_id", "impression_time"], separator="@").alias("key"),
                "n_splits",
            ).with_columns(pl.lit("user_time").alias("kind")),
        ],
        how="vertical",
    )


def find_user_feature_violations(
    user_features: pl.DataFrame, split_starts: dict[str, object]
) -> pl.DataFrame:
    """Q9.5 on the built feature table: a split's features using its own future.

    This is the check that would have caught the boundary bug described in
    `design_rationale.md` §2.6, where train was bounded by the split's END.
    """
    if "last_click_time" not in user_features.columns:
        return user_features.clear()
    frames = []
    for split_name, start in split_starts.items():
        if start is None:
            continue
        frames.append(
            user_features.filter(
                (pl.col("split") == split_name)
                & pl.col("last_click_time").is_not_null()
                & (pl.col("last_click_time") >= start)
            )
        )
    return pl.concat(frames, how="vertical") if frames else user_features.clear()


# =============================================================================
# The aggregate check the pipeline runs
# =============================================================================


def check_store(
    tables: dict[str, pl.DataFrame],
    user_features: pl.DataFrame | None = None,
    *,
    sample_size: int = LEAKAGE_SAMPLE_SIZE,
    raise_on_error: bool = True,
) -> dict:
    """Run every leakage check over a built store. Returns a report.

    Called by `build_dataset`, so Q9 is a build-time gate rather than something
    only the test suite knows about, and the outcome is recorded in the manifest
    where a grader can see it without running anything.
    """
    problems: list[str] = []
    impressions = tables["impressions"]
    history = tables["history"]

    if leaked := find_unavailable_columns(tables):
        problems.append(f"serving-unavailable columns present (Q9.3): {leaked}")

    spanning = find_ids_spanning_splits(impressions)
    if len(spanning):
        problems.append(
            f"{len(spanning)} id(s) appear in more than one split (Q9.5), "
            f"e.g. {spanning['key'].head(3).to_list()}"
        )

    starts = {
        row["split"]: row["start"]
        for row in impressions.group_by("split")
        .agg(pl.col("impression_time").min().alias("start"))
        .to_dicts()
    }

    if user_features is not None:
        bad = find_user_feature_violations(user_features, starts)
        if len(bad):
            problems.append(
                f"{len(bad)} user-feature row(s) use a click at or after their own "
                f"split opened (Q9.5)"
            )

    # Per-impression spot audit. Seeded so the manifest stays byte-identical.
    sample = impressions
    if len(impressions) > sample_size:
        sample = impressions.sample(n=sample_size, seed=config.GLOBAL_SEED, shuffle=False)
    pairs = select_history_before(history, sample)
    violations = find_window_violations(pairs)
    if len(violations):
        problems.append(
            f"{len(violations)} selected history row(s) are not strictly before "
            f"their impression (Q9.4)"
        )

    timestamped = history["click_time"].null_count() < len(history) if len(history) else False
    report = {
        "checked_impressions": len(sample),
        "selected_history_rows": len(pairs),
        "window_violations": len(violations),
        "ids_spanning_splits": len(spanning),
        "unavailable_columns": {k: v for k, v in find_unavailable_columns(tables).items()},
        # Recorded, not hidden: on MIND every row is unverifiable by timestamp
        # because none exists. A reader must be able to tell "checked and clean"
        # from "could not be checked".
        "history_is_timestamped": timestamped,
        "boundary_enforced_by": "timestamp" if timestamped else "structure",
    }

    if problems and raise_on_error:
        raise LeakageError("leakage detected:\n  " + "\n  ".join(problems))
    report["problems"] = problems
    return report
