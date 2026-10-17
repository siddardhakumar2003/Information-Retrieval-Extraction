"""A2 Q1 — click-history, session, and article behavioural features.

Requirements covered: A2 Q1.1 (click-history), Q1.2 (session), Q1.3 (article
popularity/freshness ingredients), Q1.4 (the boundary, enforced by reuse).

## The one rule this module exists to keep, again

Same rule as A1's `retrieval/query.py` and `retrieval/user_vector.py`: a feature
for an impression may be built only from clicks strictly before that
impression's timestamp.

Q2 and Q3 enforce it by calling `leakage.select_history_for`, which joins every
impression to every one of that user's clicks and then filters — an approach
its own docstring says is only safe for a *bounded* impressions frame (a recall
sample, a leakage spot-audit). This module needs the answer for EVERY
impression, where that join is `users x impressions x clicks` and, on
`ebnerd_small`, materialises around 184 million rows before the filter even
runs — enough to OOM a machine outright. `_n_clicks_timestamped` /
`_n_clicks_structural` answer the same "how many clicks strictly precede this
impression" question with a `join_asof` (`allow_exact_matches=False` gives the
same strict `<` `select_history_before` writes) or a one-to-many broadcast join,
each O(impressions + history) rather than O(impressions x history). The
boundary rule itself is unchanged; only how cheaply it is evaluated is
different. See `_n_clicks_timestamped`'s docstring for the detail.

## Why this module, and not a bigger `feature_store.py`

`feature_store.build_article_features`/`build_user_features` are Assignment 1's
tested code. Nothing here edits them; this module adds two new tables plus two
new columns on the existing `article_features` table, called as an additional
step from `feature_store.write_feature_store`. A diff of A1's own functions
should show nothing changed.

See `agent_docs/a2_q1_plan.md` for the full design, including what was
deliberately left out (dwell time as a headline feature, a stored per-article
freshness column, the full user x article category-match join).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from newsrec import config
from newsrec.data import leakage
from newsrec.retrieval import query

#: A2 Q1.1's cap on "recent" clicks. Same constant Q2's BM25 query and Q3's user
#: vector already use, so "recent history" means the same window everywhere in
#: the codebase, and an ablation on it sweeps one value, not three.
DEFAULT_LIMIT = config.QUERY_HISTORY_LIMIT


class BehaviourLeakageError(leakage.LeakageError):
    """A behavioural feature disagreed with an independent recomputation.

    Subclasses `leakage.LeakageError` rather than introducing a new exception
    family — a grader (or a test) catching one already catches the other.
    """


# =============================================================================
# Click-history features (A2 Q1.1)
# =============================================================================


def _n_clicks_timestamped(history: pl.DataFrame, impressions: pl.DataFrame) -> pl.DataFrame:
    """`impression_id -> count of clicks strictly before it`, in O(impressions +
    history) rather than O(impressions x history).

    `leakage.select_history_for`/`select_history_before` answer this by joining
    every impression to every one of that user's clicks and then filtering —
    exactly the "users x impressions x clicks" cost its own docstring warns
    about, fine for the sampled callers Q2/Q3 have (a 2,000-impression recall
    sample, a leakage spot-audit) but not for a table meant to cover EVERY
    impression: on ebnerd_small that join alone is ~184M rows before the filter
    even runs, and building it eagerly is what OOM-killed the first version of
    this function.

    `join_asof` finds the same answer without ever materialising the product.
    Each click is given its own running count (`_n_inclusive`, 1-based: "this
    many clicks have happened up to and including this one"); then for every
    impression, `join_asof(..., strategy="backward", allow_exact_matches=False)`
    finds the LAST click strictly before it and reads off that running count —
    which, by construction, equals the number of clicks strictly before the
    impression. `allow_exact_matches=False` is what makes this "backward" search
    strict-less-than rather than less-than-or-equal, the same distinction
    `select_history_before` enforces with `<` (Q9.4 — a click recorded at
    exactly the impression's timestamp must not count).
    """
    ordered_history = history.sort(["user_id", "click_time", "article_id"]).with_columns(
        (pl.int_range(pl.len()).over("user_id") + 1).cast(pl.Int32).alias("_n_inclusive")
    )
    ordered_impressions = impressions.select("impression_id", "user_id", "impression_time").sort(
        ["user_id", "impression_time"]
    )
    # polars warns that it cannot verify sortedness when `by` groups are given —
    # both frames ARE sorted (globally by user_id, then by the join column
    # within each user, which is what join_asof actually needs); the warning is
    # about what it can cheaply check, not about a real problem here.
    joined = ordered_impressions.join_asof(
        ordered_history.select("user_id", "click_time", "_n_inclusive"),
        left_on="impression_time",
        right_on="click_time",
        by="user_id",
        strategy="backward",
        allow_exact_matches=False,
    )
    return joined.select(
        "impression_id", pl.col("_n_inclusive").fill_null(0).alias("n_clicks_in_history")
    )


def _n_clicks_structural(history: pl.DataFrame, impressions: pl.DataFrame) -> pl.DataFrame:
    """MIND's structural path: the whole of a user's history precedes every one
    of their impressions (Q1's `select_history_for` docstring), so the count is
    a per-user constant, broadcast onto impressions by a single one-to-many
    join rather than a per-impression selection."""
    counts = history.group_by("user_id").agg(pl.len().cast(pl.Int32).alias("_n_inclusive"))
    joined = impressions.select("impression_id", "user_id").join(counts, on="user_id", how="left")
    return joined.select(
        "impression_id", pl.col("_n_inclusive").fill_null(0).alias("n_clicks_in_history")
    )


def _ordered_click_lists(
    history: pl.DataFrame, articles: pl.DataFrame, order_by: list[str]
) -> pl.DataFrame:
    """One row per user: that user's clicked article_id/title/category, oldest
    first. `order_by` is `["click_time", "article_id"]` (EB-NeRD) or
    `["history_rank", "article_id"]` (MIND) — whichever key the caller already
    established as chronological, with `article_id` as the deterministic
    tie-break `select_history_for` also uses.

    A one-time O(history) join and group-by, independent of how many
    impressions a user has — this replaces the per-impression join the old
    implementation ran.
    """
    return (
        history.join(
            articles.select("article_id", "title", "category"), on="article_id", how="left"
        )
        .sort(["user_id", *order_by])
        .group_by("user_id", maintain_order=True)
        .agg(
            pl.col("article_id").alias("_hist_ids"),
            pl.col("title").alias("_hist_titles"),
            pl.col("category").alias("_hist_categories"),
        )
    )


def build_click_history_features(
    history: pl.DataFrame,
    impressions: pl.DataFrame,
    articles: pl.DataFrame,
    dataset: str,
    limit: int | None = None,
    decay: float | None = None,
) -> pl.DataFrame:
    """One row per impression: recent clicks, click count, recency-weighted count.

    `limit` bounds only the *stored lists* (`recent_article_ids`/titles/
    categories) — `n_clicks_in_history` and `recency_weighted_click_count` are
    computed over the FULL legitimate history, because both are meant to answer
    "how much history does this user have," which a cap would understate.

    Built in two cheap passes rather than one big join — see `_n_clicks_*` for
    why. `recency_weighted_click_count` is a closed form of `n_clicks_in_history`
    rather than a per-click sum: the weight of the i-th most recent click is
    `decay ** i` regardless of which article it is (`user_vector.pooling_weights`
    defines the same rule), so summed over the n clicks that exist it is exactly
    the geometric series `(1 - decay**n) / (1 - decay)` — no need to touch a
    single article_id to compute it.
    """
    limit = DEFAULT_LIMIT if limit is None else limit
    decay = config.USER_POOLING_DECAY if decay is None else decay
    timestamped = leakage.history_is_timestamped(history)
    empty_lists = pl.lit([], dtype=pl.List(pl.Utf8))

    base = impressions.select("impression_id", "user_id").with_columns(
        pl.lit(dataset).cast(pl.Utf8).alias("dataset"),
        pl.lit(timestamped).alias("history_is_timestamped"),
    )

    if history.is_empty() or impressions.is_empty():
        n_clicks = base.select("impression_id").with_columns(
            pl.lit(0).cast(pl.Int32).alias("n_clicks_in_history")
        )
        lists_by_user = pl.DataFrame(
            schema={
                "user_id": pl.Utf8,
                "_hist_ids": pl.List(pl.Utf8),
                "_hist_titles": pl.List(pl.Utf8),
                "_hist_categories": pl.List(pl.Utf8),
            }
        )
    elif timestamped:
        n_clicks = _n_clicks_timestamped(history, impressions)
        lists_by_user = _ordered_click_lists(history, articles, ["click_time", "article_id"])
    else:
        n_clicks = _n_clicks_structural(history, impressions)
        # MIND's structural path orders most-recent-first by history_rank
        # DESCENDING (Q1's select_history_for), so oldest-first — what the
        # gather below needs — is history_rank ASCENDING.
        lists_by_user = _ordered_click_lists(history, articles, ["history_rank", "article_id"])

    out = (
        base.join(n_clicks, on="impression_id", how="left")
        .with_columns(pl.col("n_clicks_in_history").fill_null(0))
        .join(lists_by_user, on="user_id", how="left")
        .with_columns(
            pl.col("_hist_ids").fill_null(empty_lists),
            pl.col("_hist_titles").fill_null(empty_lists),
            pl.col("_hist_categories").fill_null(empty_lists),
        )
        .with_columns(
            # The window is [n - limit, n) into the oldest-first per-user list —
            # the `limit` most recent legitimate clicks, still oldest-first.
            (pl.col("n_clicks_in_history") - limit).clip(lower_bound=0).alias("_start")
        )
        .with_columns(pl.int_ranges(pl.col("_start"), pl.col("n_clicks_in_history")).alias("_idx"))
        .with_columns(
            # Reversed so the stored lists are most-recent-first, matching
            # `select_history_for`'s ordering contract.
            pl.col("_hist_ids")
            .list.gather(pl.col("_idx"))
            .list.reverse()
            .alias("recent_article_ids"),
            pl.col("_hist_titles")
            .list.gather(pl.col("_idx"))
            .list.reverse()
            .alias("recent_titles"),
            pl.col("_hist_categories")
            .list.gather(pl.col("_idx"))
            .list.reverse()
            .alias("recent_categories"),
            pl.when(pl.col("n_clicks_in_history") == 0)
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(
                (1.0 - pl.lit(decay, dtype=pl.Float32) ** pl.col("n_clicks_in_history"))
                / (1.0 - decay)
            )
            .alias("recency_weighted_click_count"),
        )
    )
    return out.select(
        "impression_id",
        "user_id",
        "recent_article_ids",
        "recent_titles",
        "recent_categories",
        "n_clicks_in_history",
        "recency_weighted_click_count",
        "history_is_timestamped",
        "dataset",
    )


# =============================================================================
# Session features (A2 Q1.2) — EB-NeRD only
# =============================================================================


def build_session_features(
    impressions: pl.DataFrame, impressions_extra: pl.DataFrame | None, dataset: str
) -> pl.DataFrame | None:
    """One row per impression: within-session click count and position.

    Returns `None` when the dataset carries no `session_id` (MIND) — absent, not
    invented, following the same convention `articles_extra`/`impressions_extra`
    already use for columns that only exist on one dataset.
    """
    if impressions_extra is None or "session_id" not in impressions_extra.columns:
        return None
    if impressions_extra.is_empty() or impressions.is_empty():
        return None

    sessions = (
        impressions.select("impression_id", "user_id", "impression_time", "clicked_ids")
        .join(
            impressions_extra.select("impression_id", "session_id"), on="impression_id", how="inner"
        )
        .with_columns(pl.col("clicked_ids").list.len().cast(pl.Int32).alias("_n_clicked"))
        # Deterministic order within a session: by impression_time, tie-broken by
        # impression_id. `position_in_session` and `session_click_count` are both
        # defined by THIS order, so both are exclusive of the current row and
        # inclusive of nothing recorded at the same instant with a later id.
        .sort(["user_id", "session_id", "impression_time", "impression_id"])
    )

    sessions = sessions.with_columns(
        pl.int_range(pl.len()).over(["user_id", "session_id"]).cast(pl.Int32).alias("_rank0"),
        pl.col("_n_clicked").cum_sum().over(["user_id", "session_id"]).alias("_cum_inclusive"),
    ).with_columns(
        (pl.col("_rank0") + 1).alias("position_in_session"),
        # Exclusive cumulative sum: the running total INCLUDES this row's own
        # clicks, so subtracting them leaves only strictly-prior clicks in the
        # same session — the boundary A2 Q1.4 requires.
        (pl.col("_cum_inclusive") - pl.col("_n_clicked")).alias("session_click_count"),
        pl.lit(dataset).cast(pl.Utf8).alias("dataset"),
    )

    return sessions.select(
        "impression_id",
        "user_id",
        "session_id",
        "session_click_count",
        "position_in_session",
        "dataset",
    )


# =============================================================================
# Article features (A2 Q1.3) — additive columns on article_features
# =============================================================================


def add_article_behaviour_columns(
    article_features: pl.DataFrame, impressions: pl.DataFrame
) -> pl.DataFrame:
    """`popularity_train` and `has_published_time`, added to A1's article table.

    `popularity_train` reuses `query.training_popularity` — the same
    leakage-safe, training-split-only popularity table BM25's and the semantic
    retriever's cold-start fallback already use. Imported, not duplicated.
    """
    popularity = query.training_popularity(impressions)
    pop_table = pl.DataFrame(
        {
            "article_id": list(popularity.keys()),
            "popularity_train": list(popularity.values()),
        },
        schema={"article_id": pl.Utf8, "popularity_train": pl.Int64},
    )
    return article_features.join(pop_table, on="article_id", how="left").with_columns(
        pl.col("popularity_train").fill_null(0),
        pl.col("published_time").is_not_null().alias("has_published_time"),
    )


# =============================================================================
# Build-time leakage gate (A2 Q1.4)
# =============================================================================

#: Same spot-audit reasoning as `leakage.LEAKAGE_SAMPLE_SIZE`: a full recompute
#: over every impression is quadratic-shaped and unnecessary for a build gate.
CHECK_SAMPLE_SIZE = 2_000


def check(
    click_history_features: pl.DataFrame,
    session_features: pl.DataFrame | None,
    history: pl.DataFrame,
    impressions: pl.DataFrame,
    *,
    sample_size: int = CHECK_SAMPLE_SIZE,
    raise_on_error: bool = True,
) -> dict:
    """Recompute a sample of rows independently and compare (A2 Q1.4).

    Mirrors `leakage.check_store`: a spot audit run at build time, not only in
    the test suite, so a leaked behavioural feature is caught before anything
    downstream can read it.
    """
    problems: list[str] = []
    sample_ids = click_history_features["impression_id"]
    if len(sample_ids) > sample_size:
        sample_ids = click_history_features.select("impression_id").sample(
            n=sample_size, seed=config.GLOBAL_SEED, shuffle=False
        )["impression_id"]
    sample_ids_set = set(sample_ids.to_list())

    sample_impressions = impressions.filter(pl.col("impression_id").is_in(sample_ids_set))
    expected = build_click_history_features(
        history,
        sample_impressions,
        articles=pl.DataFrame(
            {"article_id": [], "title": [], "category": []},
            schema={"article_id": pl.Utf8, "title": pl.Utf8, "category": pl.Utf8},
        ),
        dataset="_check",
    ).select("impression_id", "n_clicks_in_history", "recency_weighted_click_count")

    actual = click_history_features.filter(pl.col("impression_id").is_in(sample_ids_set)).select(
        "impression_id", "n_clicks_in_history", "recency_weighted_click_count"
    )
    joined = expected.join(actual, on="impression_id", how="inner", suffix="_actual")
    mismatched = joined.filter(
        (pl.col("n_clicks_in_history") != pl.col("n_clicks_in_history_actual"))
        | (
            (
                pl.col("recency_weighted_click_count")
                - pl.col("recency_weighted_click_count_actual")
            ).abs()
            > 1e-4
        )
    )
    if len(mismatched):
        problems.append(
            f"{len(mismatched)} click-history row(s) disagree with an independent "
            f"recomputation from select_history_for (A2 Q1.4)"
        )

    if session_features is not None and not session_features.is_empty():
        # Recomputed over the FULL impressions table, not the sample: a session's
        # click count and position depend on every other impression in that same
        # session, most of which are outside the sample. Only the comparison
        # below is restricted to the sample; the recomputation must see whole
        # sessions or it reports mismatches that are sampling artefacts, not
        # real disagreements.
        expected_sessions = build_session_features(
            impressions,
            session_features.select("impression_id", "session_id"),
            dataset="_check",
        )
        if expected_sessions is not None and not expected_sessions.is_empty():
            actual_sessions = session_features.filter(pl.col("impression_id").is_in(sample_ids_set))
            joined_s = (
                expected_sessions.filter(pl.col("impression_id").is_in(sample_ids_set))
                .select("impression_id", "session_click_count", "position_in_session")
                .join(
                    actual_sessions.select(
                        "impression_id", "session_click_count", "position_in_session"
                    ),
                    on="impression_id",
                    how="inner",
                    suffix="_actual",
                )
            )
            mismatched_s = joined_s.filter(
                (pl.col("session_click_count") != pl.col("session_click_count_actual"))
                | (pl.col("position_in_session") != pl.col("position_in_session_actual"))
            )
            if len(mismatched_s):
                problems.append(
                    f"{len(mismatched_s)} session-feature row(s) disagree with an "
                    f"independent recomputation (A2 Q1.4)"
                )

    report = {
        "checked_impressions": len(sample_ids_set),
        "click_history_mismatches": len(mismatched),
        "problems": problems,
    }
    if problems and raise_on_error:
        raise BehaviourLeakageError(
            "behavioural feature leakage detected:\n  " + "\n  ".join(problems)
        )
    return report


# =============================================================================
# Reporting stage — `newsrec behaviour_features` (a2_q1_plan.md)
# =============================================================================
#
# The three tables above are built by `make data` (feature_store.write_feature_
# store), not by this stage — see the plan doc for why. This is only a report: it
# reads the already-built store and writes a summary a grader can read without
# recomputing anything, the same division `compare`/`ablations` already use
# between "build" and "report".

REPORT_NAME = "behavioural_features.json"


def _summarise_one(dataset: str) -> dict:
    from newsrec.data import feature_store

    store = feature_store.read_feature_store(dataset)
    click_history = store["user_click_history_features"]
    article_features = store["article_features"]

    summary: dict = {
        "dataset": dataset,
        "impressions": len(click_history),
        "click_history": {
            "n_clicks_mean": round(float(click_history["n_clicks_in_history"].mean() or 0.0), 3),
            "n_clicks_max": int(click_history["n_clicks_in_history"].max() or 0),
            "recency_weighted_mean": round(
                float(click_history["recency_weighted_click_count"].mean() or 0.0), 3
            ),
            "history_is_timestamped": bool(
                click_history["history_is_timestamped"].to_list()[0]
                if len(click_history)
                else False
            ),
        },
        "article_popularity": {
            "articles": len(article_features),
            "popularity_train_mean": round(
                float(article_features["popularity_train"].mean() or 0.0), 3
            ),
            "popularity_train_nonzero_share": round(
                float((article_features["popularity_train"] > 0).mean() or 0.0), 4
            ),
            "has_published_time_share": round(
                float(article_features["has_published_time"].mean() or 0.0), 4
            ),
        },
    }

    if "session_features" in store:
        sessions = store["session_features"]
        summary["session_features"] = {
            "impressions_with_session": len(sessions),
            "session_click_count_mean": round(
                float(sessions["session_click_count"].mean() or 0.0), 3
            ),
            "position_in_session_mean": round(
                float(sessions["position_in_session"].mean() or 0.0), 3
            ),
        }
    else:
        summary["session_features"] = {
            "note": "no session_id in this dataset — MIND ships no session concept (a2_q1_plan.md)"
        }

    return summary


def run(datasets: tuple[str, ...] | None = None, out_dir: Path | None = None) -> Path:
    """Write `behavioural_features.json` (A2 Q1)."""
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "behavioural_features",
        "requirement": "A2.Q1",
        "datasets": {d: _summarise_one(d) for d in datasets},
    }
    path = Path(out_dir) / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        ch = block["click_history"]
        lines.append(
            f"\n{dataset}  n={block['impressions']:,}  "
            f"clicks/impression mean={ch['n_clicks_mean']:.1f} max={ch['n_clicks_max']:,}  "
            f"recency-weighted mean={ch['recency_weighted_mean']:.2f}  "
            f"timestamped={ch['history_is_timestamped']}"
        )
        pop = block["article_popularity"]
        lines.append(
            f"  popularity_train mean={pop['popularity_train_mean']:.2f}  "
            f"nonzero={pop['popularity_train_nonzero_share']:.1%}  "
            f"has_published_time={pop['has_published_time_share']:.1%}"
        )
        sess = block["session_features"]
        if "note" in sess:
            lines.append(f"  sessions: {sess['note']}")
        else:
            lines.append(
                f"  sessions: n={sess['impressions_with_session']:,}  "
                f"click_count mean={sess['session_click_count_mean']:.2f}  "
                f"position mean={sess['position_in_session_mean']:.2f}"
            )
    return "\n".join(lines)
