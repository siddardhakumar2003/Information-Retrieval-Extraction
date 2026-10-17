"""The canonical schema contract (Q1.2-Q1.5).

This module is the *definition* of "unified schema", and it is deliberately the
first thing built: the parsers in `mind.py` and `ebnerd.py` exist to satisfy it,
so it has to say precisely what satisfaction means before either is written.

Three core tables — `articles`, `impressions`, `history` — with **identical
column names and dtypes across every dataset**. Anything that exists in one
dataset and not the other lives in an `_extra` side table keyed by the core key.

Why the split rather than a union of all columns: a union forces every
downstream stage to know which dataset it is looking at, and "unified schema"
becomes a directory name rather than a guarantee. It also puts the Q9
serving-unavailable columns in the same frame as the safe ones, one typo away
from a leaked headline number. In a side table, using one requires a deliberate
join — see `SERVING_UNAVAILABLE_FEATURES` in `config`.

Everything here is declarative plus validators. No I/O, no parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from newsrec import config


@dataclass
class ParseResult:
    """What every dataset parser returns: canonical tables plus accounting.

    The counts are not decoration. Q1.10 requires `raw == parsed + rejected`
    with rejects attributed to reasons, because a parser allowed to drop a
    malformed row quietly makes a 3-row loss indistinguishable from a
    30,000-row loss.
    """

    articles: pl.DataFrame
    impressions: pl.DataFrame
    history: pl.DataFrame
    articles_extra: pl.DataFrame
    impressions_extra: pl.DataFrame
    counts: dict = field(default_factory=dict)
    rejections: dict = field(default_factory=dict)
    article_counts: dict = field(default_factory=dict)
    history_counts: dict = field(default_factory=dict)
    source_files: list = field(default_factory=list)

    @property
    def reconciles(self) -> bool:
        return self.counts["raw"] == self.counts["parsed"] + self.counts["rejected"]

    def as_store(self) -> dict:
        """The table mapping `validate_store` expects."""
        return {
            "articles": self.articles,
            "impressions": self.impressions,
            "history": self.history,
            "articles_extra": self.articles_extra,
            "impressions_extra": self.impressions_extra,
        }


class SchemaError(ValueError):
    """A frame violates the canonical contract.

    Its own type so callers can distinguish a contract breach from an ordinary
    bad value: the first means the pipeline is wrong, the second means the data
    is. Messages name the table, the column and the expectation, because a
    schema failure three stages upstream is otherwise a guessing game.
    """


# =============================================================================
# Dtypes
#
# Spelled out once. `Datetime("us", "UTC")` is the only datetime permitted
# anywhere in the store (Q1.5): naive timestamps compare and sort perfectly
# well, so a naive column is invisible until two datasets with different
# implicit zones are compared, at which point every split is quietly wrong.
# =============================================================================

UTC_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")
ID = pl.Utf8
TEXT = pl.Utf8
ID_LIST = pl.List(pl.Utf8)
#: 0-based position within an ordered per-user sequence. Signed and 32-bit: no
#: user has 2 billion clicks, and a signed type makes an accidental -1 visible
#: rather than wrapping to a huge positive rank.
RANK = pl.Int32


@dataclass(frozen=True)
class TableSpec:
    """One canonical table: its columns in order, its key, and its rules."""

    name: str
    #: Ordered. Column order is part of the contract so that a diff of two
    #: stores is readable and `test_core_columns_identical_across_datasets`
    #: compares tuples rather than sets.
    columns: tuple[tuple[str, pl.DataType], ...]
    #: Columns forming the primary key. Empty when the table has no unique key
    #: (history is intentionally many-rows-per-user).
    key: tuple[str, ...] = ()
    #: Columns that may contain nulls. Everything else must be fully populated.
    nullable: frozenset[str] = field(default_factory=frozenset)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    @property
    def dtypes(self) -> dict[str, pl.DataType]:
        return dict(self.columns)

    def empty(self) -> pl.DataFrame:
        """An empty frame with exactly this schema.

        Used by parsers as a starting point and by tests as a reference, so the
        dtype of an empty result is never accidentally different from the dtype
        of a populated one — a real hazard when a dataset legitimately yields
        zero rows for a table.
        """
        return pl.DataFrame(schema=dict(self.columns))


# =============================================================================
# Core tables
# =============================================================================

ARTICLES = TableSpec(
    name="articles",
    columns=(
        ("article_id", ID),
        ("title", TEXT),
        ("abstract", TEXT),
        ("body", TEXT),
        ("category", TEXT),
        ("subcategory", TEXT),
        ("published_time", UTC_DATETIME),
        ("dataset", ID),
    ),
    key=("article_id",),
    # body and published_time are null for ALL of MIND: news.tsv ships neither,
    # and the MSN urls it links are dead. Disclosed in data_schemas.md and in
    # the design note rather than papered over with an empty string, which would
    # make "has a body" indistinguishable from "body is empty".
    nullable=frozenset({"body", "published_time", "abstract", "subcategory"}),
)

IMPRESSIONS = TableSpec(
    name="impressions",
    columns=(
        ("impression_id", ID),
        ("user_id", ID),
        ("impression_time", UTC_DATETIME),
        ("candidate_ids", ID_LIST),
        ("clicked_ids", ID_LIST),
        ("split", ID),
        ("dataset", ID),
    ),
    key=("impression_id",),
    # `split` is null between parsing and splitting: the parser must not invent
    # it and the splitter is the only writer. Nullable so the intermediate state
    # is representable instead of forcing a placeholder value that looks real.
    nullable=frozenset({"split"}),
)

HISTORY = TableSpec(
    name="history",
    columns=(
        ("user_id", ID),
        ("article_id", ID),
        ("click_time", UTC_DATETIME),
        ("history_rank", RANK),
        ("dataset", ID),
    ),
    # No key: one row per click, exploded from the raw per-user form. Exploded
    # because every leakage check is `click_time < impression_time` (Q9.4),
    # which is a filter on rows; inside a list column the boundary test has to
    # reach into a nested structure, and that is where an off-by-one hides.
    key=(),
    # click_time is null for ALL of MIND, which distributes an ordered history
    # with no timestamps. The leakage checker must handle this rather than
    # assume a timestamp exists — otherwise it is silently vacuous on half the
    # corpus.
    #
    # history_rank is the OTHER half of that disclosure, and exists because the
    # first version of this table did not have it. MIND's documentation states
    # "the clicked news articles are ordered by time", so position in the raw
    # history field IS the recency signal — the only one MIND gives. The
    # original parser ended with `.sort("user_id", "article_id")` for
    # determinism and destroyed it, which was invisible until Q2 tried to build
    # a query and found 100% of MIND impressions producing an empty one. See
    # design_rationale.md 1.16.
    #
    # It is null for EB-NeRD, which has real timestamps and therefore needs no
    # positional proxy. That asymmetry is deliberate and disclosed rather than
    # filled with a plausible-looking integer.
    nullable=frozenset({"click_time", "history_rank"}),
)

CORE_TABLES: dict[str, TableSpec] = {t.name: t for t in (ARTICLES, IMPRESSIONS, HISTORY)}

#: Side tables. Their columns are dataset-specific by design, so only the join
#: key is fixed here; the contract is "joins 1:1 onto its core table".
EXTRA_TABLES: dict[str, str] = {
    "articles_extra": "article_id",
    "impressions_extra": "impression_id",
}

VALID_SPLITS = ("train", "val", "test")


# =============================================================================
# Validators
#
# Each returns a list of problems rather than raising on the first one. A parser
# that gets three columns wrong should report three columns, not make you fix
# and re-run three times.
# =============================================================================


def check_columns(df: pl.DataFrame, spec: TableSpec) -> list[str]:
    """Exact columns, exact order, exact dtypes (Q1.2)."""
    problems: list[str] = []
    got = tuple(df.columns)
    if got != spec.names:
        missing = [c for c in spec.names if c not in got]
        extra = [c for c in got if c not in spec.names]
        if missing:
            problems.append(f"{spec.name}: missing columns {missing}")
        if extra:
            problems.append(f"{spec.name}: unexpected columns {extra}")
        if not missing and not extra:
            problems.append(f"{spec.name}: column ORDER differs — got {got}, expected {spec.names}")
        return problems

    for name, expected in spec.columns:
        actual = df.schema[name]
        if actual != expected:
            problems.append(f"{spec.name}.{name}: dtype is {actual}, expected {expected}")
    return problems


def check_nulls(df: pl.DataFrame, spec: TableSpec) -> list[str]:
    """Only declared-nullable columns may contain nulls."""
    problems = []
    for name, _ in spec.columns:
        if name in spec.nullable or name not in df.columns:
            continue
        if (n := df[name].null_count()) > 0:
            problems.append(f"{spec.name}.{name}: {n} null(s) in a non-nullable column")
    return problems


def check_key(df: pl.DataFrame, spec: TableSpec) -> list[str]:
    """Primary key unique and non-null (Q1.4)."""
    if not spec.key:
        return []
    problems = []
    for name in spec.key:
        if name not in df.columns:
            continue
        if (n := df[name].null_count()) > 0:
            problems.append(f"{spec.name}.{name}: key column has {n} null(s)")
    if all(k in df.columns for k in spec.key):
        n_unique = df.select(spec.key).n_unique()
        if n_unique != len(df):
            problems.append(
                f"{spec.name}: key {spec.key} is not unique — "
                f"{len(df)} rows, {n_unique} distinct ({len(df) - n_unique} duplicate(s))"
            )
    return problems


def check_timestamps_utc(df: pl.DataFrame, spec: TableSpec) -> list[str]:
    """Every datetime column is tz-aware UTC (Q1.5)."""
    problems = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone != "UTC":
            tz = dtype.time_zone or "naive"
            problems.append(
                f"{spec.name}.{name}: datetime is {tz}, expected UTC. "
                "Localise at the parse boundary, never downstream."
            )
    return problems


def check_no_serving_unavailable(df: pl.DataFrame, spec: TableSpec) -> list[str]:
    """Q9.3 — the seven serving-unavailable columns never appear in a core table.

    Enforced structurally rather than by review. These are retained in the
    `_extra` tables because Q9.1 requires paired with/without numbers, and you
    cannot ablate a feature discarded at parse time — but reaching them must
    always cost a deliberate join.
    """
    leaked = sorted(set(df.columns) & config.SERVING_UNAVAILABLE_FEATURES)
    if leaked:
        return [
            f"{spec.name}: exposes serving-unavailable column(s) {leaked}. "
            "These belong in the _extra table (Q9.3); a core table must never "
            "carry a feature that is unknown at ranking time."
        ]
    return []


def check_splits(df: pl.DataFrame) -> list[str]:
    """`split` holds only the three valid labels, or null before splitting."""
    if "split" not in df.columns:
        return []
    seen = set(df["split"].drop_nulls().unique().to_list())
    if invalid := sorted(seen - set(VALID_SPLITS)):
        return [f"impressions.split: invalid label(s) {invalid}, expected {list(VALID_SPLITS)}"]
    return []


def validate_table(df: pl.DataFrame, spec: TableSpec, *, raise_on_error: bool = True) -> list[str]:
    """Run every single-table check. Returns problems; raises by default."""
    # These two run FIRST and unconditionally, because both failure modes also
    # trip check_columns and would otherwise never report their own message.
    # A Q9.3 leak surfacing as "unexpected columns", or a naive timestamp as
    # "dtype is Datetime(us, None)", tells you what changed but not why it
    # matters -- and these are the two that most matter.
    problems = check_no_serving_unavailable(df, spec)
    problems += check_timestamps_utc(df, spec)

    shape_problems = check_columns(df, spec)
    problems += shape_problems
    # The value checks are only meaningful once the shape is right; otherwise a
    # missing column produces a cascade of consequences that hide the cause.
    if not shape_problems:
        problems += check_nulls(df, spec)
        problems += check_key(df, spec)
        if spec.name == "impressions":
            problems += check_splits(df)
    if problems and raise_on_error:
        raise SchemaError(f"{spec.name} violates the canonical schema:\n  " + "\n  ".join(problems))
    return problems


# =============================================================================
# Cross-table validators
# =============================================================================


def check_referential_integrity(
    articles: pl.DataFrame, impressions: pl.DataFrame, history: pl.DataFrame
) -> list[str]:
    """Every referenced article_id exists in `articles` (Q1.4)."""
    known = set(articles["article_id"].to_list())
    problems = []

    if len(history):
        orphans = set(history["article_id"].to_list()) - known
        if orphans:
            problems.append(
                f"history: {len(orphans)} article_id(s) absent from articles, "
                f"e.g. {sorted(orphans)[:5]}"
            )

    for col in ("candidate_ids", "clicked_ids"):
        if col not in impressions.columns or not len(impressions):
            continue
        seen = set(impressions[col].explode(empty_as_null=False).drop_nulls().to_list())
        if orphans := seen - known:
            problems.append(
                f"impressions.{col}: {len(orphans)} article_id(s) absent from "
                f"articles, e.g. {sorted(orphans)[:5]}"
            )
    return problems


def check_clicked_subset_of_candidates(impressions: pl.DataFrame) -> list[str]:
    """Every clicked article was actually shown.

    Labels are derived as `candidate_id in clicked_ids`, so a clicked id that is
    not a candidate silently produces a positive that no ranking can ever
    retrieve — depressing recall for a reason that looks like model failure.
    """
    if not len(impressions):
        return []
    bad = impressions.filter(
        pl.col("clicked_ids").list.set_difference(pl.col("candidate_ids")).list.len() > 0
    )
    if len(bad):
        ids = bad["impression_id"].head(5).to_list()
        return [
            f"impressions: {len(bad)} row(s) have clicked_ids not present in "
            f"candidate_ids, e.g. {ids}"
        ]
    return []


def check_extra_joins_one_to_one(
    core: pl.DataFrame, extra: pl.DataFrame, key: str, extra_name: str
) -> list[str]:
    """An `_extra` table joins 1:1 onto its core table (Q1.3)."""
    problems = []
    if key not in extra.columns:
        return [f"{extra_name}: missing join key {key!r}"]
    if extra.select(key).n_unique() != len(extra):
        problems.append(f"{extra_name}: join key {key!r} is not unique")
    if orphans := set(extra[key].to_list()) - set(core[key].to_list()):
        problems.append(f"{extra_name}: {len(orphans)} key(s) with no matching core row")
    return problems


def validate_store(tables: dict[str, pl.DataFrame], *, raise_on_error: bool = True) -> list[str]:
    """Validate a whole processed store: every core table plus cross-table rules.

    `tables` maps table name to frame; `_extra` tables are optional and checked
    only when present, because MIND and EB-NeRD do not both have both.
    """
    problems: list[str] = []

    for name, spec in CORE_TABLES.items():
        if name not in tables:
            problems.append(f"missing core table {name!r}")
            continue
        problems += validate_table(tables[name], spec, raise_on_error=False)

    if any(t not in tables for t in CORE_TABLES):
        # Cross-table checks would fail confusingly with a table missing.
        if problems and raise_on_error:
            raise SchemaError("store violates the canonical schema:\n  " + "\n  ".join(problems))
        return problems

    articles, impressions, history = (
        tables["articles"],
        tables["impressions"],
        tables["history"],
    )
    problems += check_referential_integrity(articles, impressions, history)
    problems += check_clicked_subset_of_candidates(impressions)

    for extra_name, key in EXTRA_TABLES.items():
        if extra_name not in tables:
            continue
        core = articles if key == "article_id" else impressions
        problems += check_extra_joins_one_to_one(core, tables[extra_name], key, extra_name)

    if problems and raise_on_error:
        raise SchemaError("store violates the canonical schema:\n  " + "\n  ".join(problems))
    return problems


def conform(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """Reorder and cast a frame to the contract, then validate it.

    For parsers: build the columns by any convenient means, then hand the frame
    here. Casting is narrow on purpose — it will not invent a missing column or
    drop an unexpected one, because both are contract breaches that must be
    seen rather than repaired.
    """
    missing = [c for c in spec.names if c not in df.columns]
    if missing:
        raise SchemaError(f"{spec.name}: cannot conform, missing columns {missing}")
    extra = [c for c in df.columns if c not in spec.names]
    if extra:
        raise SchemaError(
            f"{spec.name}: cannot conform, unexpected columns {extra}. "
            "Dataset-specific fields belong in the _extra table."
        )
    out = df.select([pl.col(name).cast(dtype) for name, dtype in spec.columns])
    validate_table(out, spec)
    return out
