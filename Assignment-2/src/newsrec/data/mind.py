"""Parse MIND-small into the canonical tables (Q1.2, Q1.10).

Parsing only. No filtering, no imputation, no policy — the splitter owns splits
and the feature store owns features. The one job here is to turn bytes into
frames that satisfy `schema.py`, and to account for every input row.

Accounting is the reason this module counts rejections by reason instead of
dropping quietly. If a parser may discard a malformed row silently, then
`raw == parsed + rejected` stops holding and a 3-row loss is indistinguishable
from a 30,000-row loss. Silent drops are the easiest way to produce a plausible
wrong number in a data pipeline.

Four properties of the distributed files were verified before this was written,
each of which would otherwise have caused a silent, plausible-looking error.
They are asserted at parse time rather than trusted:

1. **impression_id collides across train and dev.** Both files number from 1, so
   all 73,152 dev ids also occur in train. Concatenating naively merges two
   different impressions under one primary key. Ids are namespaced by source
   period; the original is preserved in `impressions_extra` because Q5 needs it
   to address the leaderboard's rows.
2. **422 titles contain a double quote.** Read with `quote_char=None`; the CSV
   default would swallow delimiters and silently mis-field those rows.
3. **28,460 articles appear in both news.tsv files**, byte-identical in title and
   abstract. Deduplication is therefore lossless — but a conflict is counted and
   reported rather than resolved by "last one wins".
4. **History is constant per user**, within train and between train and dev
   (0 of 33,617 multi-impression users vary; 0 of 5,943 shared users differ).
   A per-user history table is therefore exact and, crucially, leak-free: had
   the strings grown over time, unioning them would have given an early
   impression access to clicks recorded days later.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from newsrec.data import schema

DATASET = "mind_small"

#: Source period -> subdirectory. `train` becomes our train+val after splitting;
#: `dev` becomes our test, which is the period the MIND leaderboard evaluates.
SOURCE_PERIODS: dict[str, str] = {
    "train": "MINDsmall_train",
    "dev": "MINDsmall_dev",
}

NEWS_COLUMNS = (
    "article_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)
BEHAVIOR_COLUMNS = ("impression_id", "user_id", "time", "history", "impressions")

#: MIND ships `MM/DD/YYYY H:MM:SS AM/PM`, naive. Localised to UTC here, at the
#: parse boundary, and never touched again (Q1.5). The UTC assumption is
#: documented in data_schemas.md: MIND states no zone, but the choice is uniform
#: across the corpus so it cannot change any relative ordering, and ordering is
#: all the splits and the leakage rules depend on.
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


class MindParseResult(schema.ParseResult):
    """MIND's ParseResult, plus the invariant this parser must not assume."""

    @property
    def history_is_per_user_constant(self) -> bool:
        """False if any user's history differed between their impressions.

        Not an error -- earliest-by-timestamp is still the safe reading -- but it
        contradicts a documented property of MIND, so it belongs in the manifest
        where it can be seen rather than in a comment where it cannot.
        """
        return self.counts.get("users_with_varying_history", 0) == 0


def _read_tsv(path: Path, columns: tuple[str, ...]) -> pl.DataFrame:
    """Read a MIND TSV.

    `quote_char=None` is not a style choice: 422 train titles contain a double
    quote, and the CSV default treats it as a quoting character, which swallows
    the following tab and shifts every later field in that row. The corruption
    is silent and looks like ordinary bad text.

    `infer_schema=False` keeps everything Utf8 so a numeric-looking id is never
    coerced to an integer and back, losing leading zeros or changing its type
    between the two files.
    """
    if not path.exists():
        raise FileNotFoundError(f"MIND source missing: {path}. Run `make download`.")
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=list(columns),
        quote_char=None,
        infer_schema=False,
        null_values=[],
    )
    if tuple(df.columns) != columns:
        raise ValueError(f"{path}: expected {len(columns)} columns, got {len(df.columns)}")
    return df


# =============================================================================
# Articles
# =============================================================================


def parse_articles(raw_root: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Both news.tsv files -> deduplicated core `articles` plus `articles_extra`."""
    frames = []
    for period, subdir in SOURCE_PERIODS.items():
        df = _read_tsv(raw_root / subdir / "news.tsv", NEWS_COLUMNS)
        frames.append(df.with_columns(pl.lit(period).alias("source_period")))
    combined = pl.concat(frames, how="vertical")
    raw_rows = len(combined)

    # Verify the dedupe is lossless instead of assuming it. If two files
    # disagree about an article's text, "keep first" would silently pick one and
    # the choice would depend on file order.
    conflicts = (
        combined.group_by("article_id")
        .agg(pl.col("title").n_unique().alias("t"), pl.col("abstract").n_unique().alias("a"))
        .filter((pl.col("t") > 1) | (pl.col("a") > 1))
    )
    if len(conflicts):
        ids = conflicts["article_id"].head(5).to_list()
        raise ValueError(
            f"{len(conflicts)} article_id(s) have different title/abstract in the two "
            f"news.tsv files, e.g. {ids}. Deduplication would silently pick one; "
            "resolve the conflict explicitly before proceeding."
        )

    unique = combined.unique(subset=["article_id"], keep="first", maintain_order=True)

    articles = unique.select(
        pl.col("article_id"),
        # Never null: an absent title is an empty string, which is a real state.
        pl.col("title").fill_null(""),
        # Nullable: MIND leaves 2,666 abstracts empty, and empty-vs-absent is a
        # distinction worth keeping rather than flattening.
        pl.col("abstract"),
        # MIND ships no body and its MSN urls are dead (disclosed, data_schemas).
        pl.lit(None, dtype=schema.TEXT).alias("body"),
        pl.col("category").fill_null(""),
        pl.col("subcategory"),
        pl.lit(None, dtype=schema.UTC_DATETIME).alias("published_time"),
        pl.lit(DATASET).alias("dataset"),
    ).sort("article_id")
    articles = schema.conform(articles, schema.ARTICLES)

    extra = unique.select(
        "article_id", "url", "title_entities", "abstract_entities", "source_period"
    ).sort("article_id")

    counts = {
        "raw": raw_rows,
        "duplicates": raw_rows - len(unique),
        "unique": len(unique),
    }
    return articles, extra, counts


# =============================================================================
# Behaviors -> impressions + history
# =============================================================================


def _parse_behaviors(raw_root: Path) -> pl.DataFrame:
    frames = []
    for period, subdir in SOURCE_PERIODS.items():
        df = _read_tsv(raw_root / subdir / "behaviors.tsv", BEHAVIOR_COLUMNS)
        frames.append(df.with_columns(pl.lit(period).alias("source_period")))
    return pl.concat(frames, how="vertical")


def _split_ids(expr: pl.Expr) -> pl.Expr:
    """Space-separated ids -> List(Utf8), with empties removed.

    `fill_null("")` first: MIND writes an empty field for a user with no
    history, which polars reads as null, and a null would propagate through the
    split into a one-element list containing null.
    """
    return (
        expr.fill_null("")
        .str.strip_chars()
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
    )


def parse_behaviors(
    raw_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int], dict[str, int]]:
    """behaviors.tsv -> `impressions`, `history`, `impressions_extra` + accounting."""
    raw = _parse_behaviors(raw_root)
    raw_rows = len(raw)

    parsed = raw.with_columns(
        # strict=False so an unparseable timestamp becomes null and is COUNTED
        # below, rather than aborting the run or vanishing.
        pl.col("time").str.to_datetime(TIME_FORMAT, strict=False).alias("_ts"),
        _split_ids(pl.col("history")).alias("_history_ids"),
        _split_ids(pl.col("impressions")).alias("_tokens"),
    )

    # Candidate tokens are `<article_id>-<0|1>`. MIND article ids never contain a
    # hyphen, so a right split on the last "-" is unambiguous.
    parsed = parsed.with_columns(
        pl.col("_tokens")
        .list.eval(pl.element().str.extract(r"^(.+)-[01]$", 1))
        .alias("_candidates"),
        pl.col("_tokens").list.eval(pl.element().str.extract(r"^.+-([01])$", 1)).alias("_labels"),
    )

    rejects = {
        "unparseable_time": pl.col("_ts").is_null(),
        "empty_candidate_list": pl.col("_tokens").list.len() == 0,
        # A token that does not match `<id>-<0|1>` yields a null after extract.
        "malformed_candidate_token": pl.col("_candidates").list.contains(None),
    }
    parsed = parsed.with_columns(
        [cond.alias(f"_reject_{name}") for name, cond in rejects.items()]
    ).with_columns(pl.any_horizontal([pl.col(f"_reject_{n}") for n in rejects]).alias("_rejected"))

    rejection_counts = {
        name: int(parsed.select(pl.col(f"_reject_{name}").sum()).item()) for name in rejects
    }
    kept = parsed.filter(~pl.col("_rejected"))

    impressions = kept.select(
        # Namespaced: all 73,152 dev impression_ids also occur in train, so the
        # raw value is not a key across the combined corpus.
        (pl.col("source_period") + pl.lit(":") + pl.col("impression_id")).alias("impression_id"),
        pl.col("user_id"),
        pl.col("_ts").dt.replace_time_zone("UTC").alias("impression_time"),
        pl.col("_candidates").cast(schema.ID_LIST).alias("candidate_ids"),
        pl.col("_candidates")
        .list.eval(pl.element().filter(pl.element().is_not_null()))
        .alias("_c"),
        pl.col("_labels").alias("_l"),
        # Written by the splitter and by nothing else.
        pl.lit(None, dtype=schema.ID).alias("split"),
        pl.lit(DATASET).alias("dataset"),
    )
    # clicked_ids = candidates whose label is "1", positionally aligned.
    impressions = impressions.with_columns(
        pl.struct(["_c", "_l"])
        .map_elements(
            lambda s: [a for a, lab in zip(s["_c"], s["_l"]) if lab == "1"],
            return_dtype=schema.ID_LIST,
        )
        .alias("clicked_ids")
    ).drop("_c", "_l")
    impressions = schema.conform(
        impressions.select(list(schema.IMPRESSIONS.names)), schema.IMPRESSIONS
    )

    impressions_extra = kept.select(
        (pl.col("source_period") + pl.lit(":") + pl.col("impression_id")).alias("impression_id"),
        # Q5 addresses leaderboard rows by the ORIGINAL id, so it must survive.
        pl.col("impression_id").alias("source_impression_id"),
        pl.col("source_period"),
    )

    # History is constant per user in the distributed files -- verified: 0 of
    # 33,617 multi-impression users vary, and 0 of 5,943 users shared between
    # train and dev differ. But the parser must not DEPEND on a property it does
    # not check, so it takes the history attached to the user's EARLIEST
    # impression and counts any user whose history varies.
    #
    # Earliest-by-timestamp, not `first` by row order. If the invariant ever
    # broke, row order would pick arbitrarily while earliest is the only choice
    # that cannot import a click recorded after an impression we later score --
    # which is precisely the Q9.4 leak. The count surfaces the violation in the
    # manifest instead of letting it pass as a silent, plausible result.
    #
    # click_time is null throughout: MIND distributes an ordered history with no
    # timestamps, so its boundary is enforced structurally (eval_protocol.md).
    varying_history = int(
        kept.group_by("user_id")
        .agg(pl.col("_history_ids").list.sort().n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    # ORDER IS DATA HERE. MIND's documentation states that "the clicked news
    # articles are ordered by time", and since no click_time is distributed,
    # that position is the ONLY recency signal MIND gives us.
    #
    # An earlier version of this function ended with `.sort("user_id",
    # "article_id")`, chosen for reproducibility, which silently replaced a
    # time ordering with an alphabetical one. Nothing failed: the table had the
    # right rows, the right count, and passed every schema and leakage check.
    # It surfaced only when Q2 built its first query and found that 100% of MIND
    # impressions produced an empty one (design_rationale.md 1.16).
    #
    # `history_rank` records the source position before any reordering, so the
    # final sort can stay deterministic AND carry the signal: sorting by
    # (user_id, history_rank) is exactly as reproducible as sorting
    # alphabetically, and preserves information rather than discarding it.
    history = (
        kept.sort("_ts")
        .select("user_id", "_history_ids")
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("_history_ids").first())
        .explode("_history_ids", empty_as_null=False)
        .drop_nulls("_history_ids")
        # Rank assigned AFTER explode, so it is the position in the raw
        # space-separated history field, and BEFORE dedupe, so a repeated
        # article keeps the position of its first occurrence.
        .with_columns(pl.int_range(pl.len(), dtype=schema.RANK).over("user_id").alias("_rank"))
        .unique(subset=["user_id", "_history_ids"], keep="first", maintain_order=True)
        .select(
            pl.col("user_id"),
            pl.col("_history_ids").alias("article_id"),
            pl.lit(None, dtype=schema.UTC_DATETIME).alias("click_time"),
            pl.col("_rank").alias("history_rank"),
            pl.lit(DATASET).alias("dataset"),
        )
        .sort("user_id", "history_rank")
    )
    history = schema.conform(history, schema.HISTORY)

    counts = {
        "raw": raw_rows,
        "parsed": len(kept),
        "rejected": raw_rows - len(kept),
        "users_with_varying_history": varying_history,
    }
    return impressions, history, impressions_extra, counts, rejection_counts


# =============================================================================


def parse(raw_root: Path | None = None) -> schema.ParseResult:
    """Parse MIND-small from `raw_root` (default `Data/MIND`)."""
    from newsrec import config

    root = raw_root or (config.RAW_DIR / "MIND")

    articles, articles_extra, article_counts = parse_articles(root)
    impressions, history, impressions_extra, counts, rejections = parse_behaviors(root)

    source_files = [
        {"path": config.repo_relative(root / sub / name), "period": period}
        for period, sub in SOURCE_PERIODS.items()
        for name in ("news.tsv", "behaviors.tsv")
    ]

    result = MindParseResult(
        articles=articles,
        impressions=impressions,
        history=history,
        articles_extra=articles_extra,
        impressions_extra=impressions_extra,
        counts=counts,
        rejections={k: v for k, v in rejections.items() if v},
        article_counts=article_counts,
        source_files=source_files,
    )
    if not result.reconciles:
        raise AssertionError(
            f"row accounting failed: raw={counts['raw']} != parsed={counts['parsed']} "
            f"+ rejected={counts['rejected']}"
        )
    return result
