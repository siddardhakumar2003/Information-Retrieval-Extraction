"""Parse EB-NeRD into the canonical tables (Q1.2, Q1.10).

Same contract as `mind.py`: parsing only, no policy, and every input row
accounted for. The datasets differ enough that the hazards are almost disjoint.

Verified in the distributed files before this was written:

* **No id collision.** `impression_id` is unique across `train/` and
  `validation/` (0 shared), unlike MIND where every dev id also occurs in train.
  Ids are therefore used as-is; the source period is still recorded in
  `impressions_extra` because Q5 needs to know which rows belong to which period.
* **The data is clean.** No nulls in any key column, no empty inview lists, no
  duplicate `article_id`, no orphan references, and `article_ids_clicked` is a
  subset of `article_ids_inview` in every row. The rejection paths below
  therefore fire zero times today — they exist so that a future file which is
  *not* clean is counted rather than silently reshaped.
* **Everything is numeric.** Ids are `Int32`/`UInt32` and are cast to `Utf8` so
  the join key has one type across both datasets (see `q1_plan.md`).
* **Timestamps are tz-naive**, so they are localised to UTC here, at the parse
  boundary, exactly as MIND's naive strings are.

## The history hazard, which is the opposite of MIND's

MIND ships one history per user, constant across the whole corpus, with **no
timestamps**. EB-NeRD ships a **rolling 21-day window per period**, timestamped:

    train/history.parquet        2023-04-27 07:00 .. 2023-05-18 07:00
    validation/history.parquet   2023-05-04 07:00 .. 2023-05-25 07:00

The validation window *overlaps the entire train period*. For 633 of the 1,217
users present in both files, the validation history is strictly longer.

So the two files cannot be treated as interchangeable, and taking either one
alone is wrong: `train/` alone discards real clicks that precede a validation
impression, while `validation/` alone would hand a train impression clicks
recorded up to a week in its future.

**What this module does:** unions both files into one timestamped click table and
deduplicates. That is safe *only because every EB-NeRD click carries a
timestamp*, so the Q9.4 rule `click_time < impression_time` can be enforced
downstream on rows. It is also strictly more correct than either file alone: a
train impression late in the period legitimately has clicks that only appear in
the validation window, and those are genuinely in its past.

**What this module relies on:** that the strict boundary IS applied wherever
features are built. Without it, the union leaks — a train impression would see
clicks from its future. This is exactly the failure `tests/test_leakage.py` and
its canary exist to catch, and it is why `click_time` is non-null here while it
is null for all of MIND.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from newsrec import config
from newsrec.data import schema

#: Source period -> subdirectory. `train` becomes our train+val after splitting;
#: `validation` becomes our test, the window the RecSys leaderboard evaluates.
SOURCE_PERIODS: tuple[str, ...] = ("train", "validation")

#: Kept in `articles_extra`. The last three are the future-leaking aggregates in
#: `config.FUTURE_AGGREGATE_FEATURES`: they are computed over the whole corpus
#: window, so every row carries popularity from its own future. Retained because
#: Q9.1 needs paired with/without numbers -- you cannot ablate a feature thrown
#: away at parse time -- but they live here so that reaching them costs a join.
ARTICLE_EXTRA_COLUMNS = (
    "url",
    "article_type",
    "premium",
    "ner_clusters",
    "entity_groups",
    "topics",
    "sentiment_label",
    "sentiment_score",
    "last_modified_time",
    "total_inviews",
    "total_pageviews",
    "total_read_time",
)

#: Kept in `impressions_extra`. The first four are `config.POST_CLICK_FEATURES`:
#: you cannot know how long someone read an article you have not yet shown them.
IMPRESSION_EXTRA_COLUMNS = (
    "read_time",
    "scroll_percentage",
    "next_read_time",
    "next_scroll_percentage",
    "device_type",
    "session_id",
    "is_sso_user",
    "is_subscriber",
    "gender",
    "age",
    "postcode",
)


def _as_id(expr: pl.Expr) -> pl.Expr:
    """Numeric id -> Utf8, so the join key has one type across both datasets."""
    return expr.cast(pl.Int64, strict=False).cast(schema.ID)


def _as_id_list(expr: pl.Expr) -> pl.Expr:
    return expr.list.eval(pl.element().cast(pl.Int64, strict=False).cast(schema.ID)).cast(
        schema.ID_LIST
    )


def _utc(expr: pl.Expr) -> pl.Expr:
    """Localise a tz-naive timestamp to UTC, at the parse boundary (Q1.5)."""
    return expr.dt.replace_time_zone("UTC")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"EB-NeRD source missing: {path}. Run `make download`.")
    return path


# =============================================================================
# Articles
# =============================================================================


def parse_articles(root: Path, dataset: str) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    raw = pl.read_parquet(_require(root / "articles.parquet"))
    raw_rows = len(raw)
    unique = raw.unique(subset=["article_id"], keep="first", maintain_order=True)

    articles = unique.select(
        _as_id(pl.col("article_id")).alias("article_id"),
        pl.col("title").cast(schema.TEXT).fill_null(""),
        # EB-NeRD's `subtitle` is the field that plays MIND's `abstract` role:
        # a short summary shown with the headline. Mapped here so Q2 indexes
        # comparable text on both datasets (title + abstract).
        pl.col("subtitle").cast(schema.TEXT).alias("abstract"),
        pl.col("body").cast(schema.TEXT),
        pl.col("category_str").cast(schema.TEXT).fill_null("").alias("category"),
        # subcategory is List(Int16) here and a plain string in MIND. Joined into
        # a stable sorted string so the column has ONE type across datasets, with
        # the raw list preserved in articles_extra. The codes are numeric and
        # carry no shared vocabulary with MIND's words -- they are comparable
        # within a dataset, never across, and the design note says so.
        pl.when(pl.col("subcategory").list.len() == 0)
        .then(None)
        .otherwise(
            pl.col("subcategory")
            .list.sort()
            .list.eval(pl.element().cast(schema.TEXT))
            .list.join("|")
        )
        .cast(schema.TEXT)
        .alias("subcategory"),
        _utc(pl.col("published_time")).alias("published_time"),
        pl.lit(dataset).alias("dataset"),
    ).sort("article_id")
    articles = schema.conform(articles, schema.ARTICLES)

    extra = unique.select(
        _as_id(pl.col("article_id")).alias("article_id"),
        pl.col("subcategory").alias("subcategory_codes"),
        *[pl.col(c) for c in ARTICLE_EXTRA_COLUMNS if c in unique.columns],
    ).sort("article_id")

    counts = {
        "raw": raw_rows,
        "duplicates": raw_rows - len(unique),
        "unique": len(unique),
    }
    return articles, extra, counts


# =============================================================================
# Behaviors
# =============================================================================


def parse_behaviors(
    root: Path, dataset: str
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int], dict[str, int]]:
    frames = []
    for period in SOURCE_PERIODS:
        df = pl.read_parquet(_require(root / period / "behaviors.parquet"))
        frames.append(df.with_columns(pl.lit(period).alias("source_period")))
    raw = pl.concat(frames, how="vertical_relaxed")
    raw_rows = len(raw)

    prepared = raw.with_columns(
        _as_id_list(pl.col("article_ids_inview")).alias("_candidates"),
        _as_id_list(pl.col("article_ids_clicked")).alias("_clicked"),
    )

    rejects = {
        "null_impression_time": pl.col("impression_time").is_null(),
        "empty_candidate_list": pl.col("_candidates").list.len() == 0,
        # Would make a positive that no ranking can ever retrieve, depressing
        # recall for a reason that looks like model failure.
        "clicked_not_in_candidates": pl.col("_clicked")
        .list.set_difference(pl.col("_candidates"))
        .list.len()
        > 0,
    }
    prepared = prepared.with_columns(
        [cond.alias(f"_reject_{name}") for name, cond in rejects.items()]
    ).with_columns(pl.any_horizontal([pl.col(f"_reject_{n}") for n in rejects]).alias("_rejected"))
    rejection_counts = {
        name: int(prepared.select(pl.col(f"_reject_{name}").sum()).item()) for name in rejects
    }
    kept = prepared.filter(~pl.col("_rejected"))

    impressions = schema.conform(
        kept.select(
            _as_id(pl.col("impression_id")).alias("impression_id"),
            _as_id(pl.col("user_id")).alias("user_id"),
            _utc(pl.col("impression_time")).alias("impression_time"),
            pl.col("_candidates").alias("candidate_ids"),
            pl.col("_clicked").alias("clicked_ids"),
            # Written by the splitter and by nothing else.
            pl.lit(None, dtype=schema.ID).alias("split"),
            pl.lit(dataset).alias("dataset"),
        ),
        schema.IMPRESSIONS,
    )

    impressions_extra = kept.select(
        _as_id(pl.col("impression_id")).alias("impression_id"),
        pl.col("impression_id").cast(schema.TEXT).alias("source_impression_id"),
        pl.col("source_period"),
        # The article the user was reading when the impression was served, when
        # the impression happened on an article page rather than the front page.
        _as_id(pl.col("article_id")).alias("context_article_id"),
        *[pl.col(c) for c in IMPRESSION_EXTRA_COLUMNS if c in kept.columns],
    )

    counts = {"raw": raw_rows, "parsed": len(kept), "rejected": raw_rows - len(kept)}
    return impressions, impressions_extra, counts, rejection_counts


# =============================================================================
# History
# =============================================================================


def parse_history(root: Path, dataset: str) -> tuple[pl.DataFrame, dict[str, int]]:
    """Both history files -> one deduplicated, timestamped click table.

    See the module docstring: the two files are overlapping rolling windows, not
    alternatives. The union is correct and leak-free ONLY under the strict
    `click_time < impression_time` filter that Q9.4 mandates downstream.
    """
    per_period: dict[str, int] = {}
    frames = []
    for period in SOURCE_PERIODS:
        raw = pl.read_parquet(_require(root / period / "history.parquet"))
        exploded = (
            raw.select(
                _as_id(pl.col("user_id")).alias("user_id"),
                pl.col("article_id_fixed"),
                pl.col("impression_time_fixed"),
            )
            # Zip the parallel list columns positionally. Exploding them
            # separately would pair click i with timestamp j the moment any two
            # lists differ in length -- a silent, catastrophic misalignment.
            .explode("article_id_fixed", "impression_time_fixed", empty_as_null=False)
            .select(
                pl.col("user_id"),
                _as_id(pl.col("article_id_fixed")).alias("article_id"),
                _utc(pl.col("impression_time_fixed")).alias("click_time"),
                pl.lit(dataset).alias("dataset"),
            )
            .drop_nulls(["article_id"])
        )
        per_period[period] = len(exploded)
        frames.append(exploded)

    combined = pl.concat(frames, how="vertical")
    history = (
        combined.unique(
            subset=["user_id", "article_id", "click_time"], keep="first", maintain_order=True
        )
        .sort("user_id", "click_time", "article_id")
        # `history_rank` is MIND's positional recency proxy and is deliberately
        # NULL here. EB-NeRD ships a real `click_time` for every click, so the
        # ordering is carried by the timestamp itself and a position column
        # would be a second, redundant source of truth for the same fact.
        # Filling it with a plausible integer would invite a caller to sort by
        # it and get an answer that only accidentally matches the timestamps.
        .with_columns(pl.lit(None, dtype=schema.RANK).alias("history_rank"))
    )
    history = schema.conform(history, schema.HISTORY)

    counts = {
        **{f"raw_{p}": n for p, n in per_period.items()},
        "raw": sum(per_period.values()),
        "unique": len(history),
        "duplicates_across_periods": sum(per_period.values()) - len(history),
    }
    return history, counts


# =============================================================================


def parse(raw_root: Path | None = None, dataset: str = "ebnerd_demo") -> schema.ParseResult:
    """Parse an EB-NeRD variant. `dataset` selects the registry entry."""
    if dataset not in config.DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(config.DATASETS)}")
    root = raw_root or (config.RAW_DIR / config.DATASETS[dataset].raw_subdir)

    articles, articles_extra, article_counts = parse_articles(root, dataset)
    impressions, impressions_extra, counts, rejections = parse_behaviors(root, dataset)
    history, history_counts = parse_history(root, dataset)

    source_files = [
        {"path": config.repo_relative(root / "articles.parquet"), "period": None},
        *[
            {"path": config.repo_relative(root / period / name), "period": period}
            for period in SOURCE_PERIODS
            for name in ("behaviors.parquet", "history.parquet")
        ],
    ]

    result = schema.ParseResult(
        articles=articles,
        impressions=impressions,
        history=history,
        articles_extra=articles_extra,
        impressions_extra=impressions_extra,
        counts=counts,
        rejections={k: v for k, v in rejections.items() if v},
        article_counts=article_counts,
        history_counts=history_counts,
        source_files=source_files,
    )
    if not result.reconciles:
        raise AssertionError(
            f"row accounting failed: raw={counts['raw']} != parsed={counts['parsed']} "
            f"+ rejected={counts['rejected']}"
        )
    return result
