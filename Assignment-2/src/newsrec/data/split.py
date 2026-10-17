"""Temporal train/val/test split (Q1.6-Q1.8).

**Interaction data is never split randomly.** Not a preference — there is no
fraction parameter and no shuffle parameter anywhere in this module or in
`config.SplitSpec`, so a random split is unrepresentable rather than merely
discouraged. The only input is a timestamp and two boundaries read from config.

The rule, identical for both datasets:

    impression_time <  val_start                      -> train
    val_start       <= impression_time <  test_start  -> val
    test_start      <= impression_time                -> test

Boundaries come from `config.DATASETS[key].split` and are never written here.
A test asserts the produced split equals the configured boundaries, so changing
config changes behaviour and nothing downstream can quietly disagree.

## Why comparison is on datetimes, not dates

MIND's periods are calendar-aligned (2019-11-09 00:00:19 .. 2019-11-14 23:59:13)
but **EB-NeRD's break at 07:00 UTC**: its train period ends 2023-05-25 06:59:52
and validation begins 2023-05-25 07:00:15. A date-level comparison would cut
25 May in half and put the first seven hours of the *validation* period into
train — future data in the training set, and invisible to any assertion that
compares dates. See OQ-4.

## The cross-check this module performs, and does not depend on

Both datasets ship their periods in separate directories, so there are two ways
to decide which rows are test: the timestamp, and the folder the row came from.
The timestamp is the definition — folder provenance is what we *believe*, and a
splitter that silently shuffled would still put rows in the right folders.

But the two agreeing is strong evidence that the configured boundary is right,
and verified on the real data they agree exactly (MIND 73,152; ebnerd_demo
25,356; zero rows differ either way). `crosscheck_against_source_periods` reports
disagreement so a mis-set boundary is caught immediately rather than at the point
where a metric looks slightly off.
"""

from __future__ import annotations

import polars as pl

from newsrec import config
from newsrec.data import schema

SPLITS = schema.VALID_SPLITS  # ("train", "val", "test")

#: Deterministic ordering for the output (Q1.13). impression_time alone is not a
#: total order -- MIND has thousands of ties per second -- so impression_id
#: breaks them, giving byte-identical output across rebuilds.
SORT_KEY = ("impression_time", "impression_id")


class SplitError(ValueError):
    """The split could not be produced, or violates its own contract."""


def assign(impressions: pl.DataFrame, spec: config.SplitSpec) -> pl.DataFrame:
    """Write the `split` column from `impression_time` and `spec`.

    Refuses to guess. Every precondition that would make the result quietly
    wrong is checked and raises with the boundary and the observed range in the
    message, because "all rows landed in train" is otherwise a silent outcome
    that looks like a working pipeline.
    """
    if "impression_time" not in impressions.columns:
        raise SplitError("impressions has no impression_time column")

    dtype = impressions.schema["impression_time"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        raise SplitError(
            f"impression_time is {dtype}, expected tz-aware UTC. Comparing a naive "
            "timestamp against a tz-aware boundary is either an error or a silent "
            "offset; localise at the parse boundary (Q1.5)."
        )

    if impressions.is_empty():
        raise SplitError("cannot split an empty impressions table")

    lo = impressions["impression_time"].min()
    hi = impressions["impression_time"].max()
    # A boundary outside the observed range puts every row in one bucket. That
    # is the generalisation of the placeholder-date failure OQ-4 guarded against:
    # the pipeline runs, produces a split, and is entirely wrong.
    for name, boundary in (("val_start", spec.val_start), ("test_start", spec.test_start)):
        if not (lo < boundary <= hi):
            raise SplitError(
                f"{name}={boundary.isoformat()} lies outside the observed "
                f"impression range [{lo.isoformat()}, {hi.isoformat()}]. "
                "Every row would land in a single split. Check "
                "config.DATASETS[...].split against the data."
            )

    out = impressions.with_columns(
        pl.when(pl.col("impression_time") < spec.val_start)
        .then(pl.lit("train"))
        .when(pl.col("impression_time") < spec.test_start)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .cast(schema.ID)
        .alias("split")
    ).sort(list(SORT_KEY))

    problems = validate(out, spec)
    if problems:
        raise SplitError("split violates its contract:\n  " + "\n  ".join(problems))
    return out


def summarise(impressions: pl.DataFrame) -> dict[str, int]:
    """Rows per split, always with all three keys present."""
    counts = dict(
        impressions.group_by("split").len().iter_rows()  # type: ignore[arg-type]
    )
    return {name: int(counts.get(name, 0)) for name in SPLITS}


def validate(impressions: pl.DataFrame, spec: config.SplitSpec) -> list[str]:
    """Q1.6-Q1.8, asserted on timestamps rather than on provenance."""
    problems: list[str] = []
    sizes = summarise(impressions)

    for name, n in sizes.items():
        if n == 0:
            problems.append(f"{name} split is empty")
    if sum(sizes.values()) != len(impressions):
        problems.append("rows carry a split label outside train/val/test")
    if any(n == 0 for n in sizes.values()):
        return problems  # ordering checks below need every split populated

    bounds = {
        name: (
            impressions.filter(pl.col("split") == name)["impression_time"].min(),
            impressions.filter(pl.col("split") == name)["impression_time"].max(),
        )
        for name in SPLITS
    }

    # Q1.6 -- the ordering that makes the split temporal.
    if bounds["train"][1] >= bounds["val"][0]:
        problems.append(
            f"train overlaps val: max(train)={bounds['train'][1]} >= min(val)={bounds['val'][0]}"
        )
    if bounds["val"][1] >= bounds["test"][0]:
        problems.append(
            f"val overlaps test: max(val)={bounds['val'][1]} >= min(test)={bounds['test'][0]}"
        )

    # Q1.8 -- the produced boundary equals the configured one.
    if bounds["train"][1] >= spec.val_start:
        problems.append(f"train contains {bounds['train'][1]}, at or after val_start")
    if bounds["val"][0] < spec.val_start:
        problems.append(f"val starts at {bounds['val'][0]}, before val_start")
    if bounds["val"][1] >= spec.test_start:
        problems.append(f"val contains {bounds['val'][1]}, at or after test_start")
    if bounds["test"][0] < spec.test_start:
        problems.append(f"test starts at {bounds['test'][0]}, before test_start")

    # Q1.7 -- disjoint on impression_id.
    if "impression_id" in impressions.columns:
        ids = impressions.select("impression_id", "split").unique()
        if ids["impression_id"].n_unique() != len(ids):
            problems.append("an impression_id appears in more than one split")

    return problems


def crosscheck_against_source_periods(
    impressions: pl.DataFrame, impressions_extra: pl.DataFrame
) -> dict[str, int]:
    """Compare the timestamp-derived test split with the shipped period folders.

    Evidence, not definition. The timestamp is what decides the split; this
    reports how far folder provenance agrees, so a mis-set boundary shows up as
    a number here rather than as a slightly-off metric three stages later.

    Returns counts of the disagreement in both directions; both are 0 on the
    real MIND and EB-NeRD files.
    """
    if "source_period" not in impressions_extra.columns:
        return {"only_by_timestamp": 0, "only_by_folder": 0, "checked": 0}

    joined = impressions.join(
        impressions_extra.select("impression_id", "source_period"), on="impression_id", how="inner"
    )
    by_timestamp = set(joined.filter(pl.col("split") == "test")["impression_id"].to_list())
    by_folder = set(
        joined.filter(pl.col("source_period").is_in(["dev", "validation"]))[
            "impression_id"
        ].to_list()
    )
    return {
        "only_by_timestamp": len(by_timestamp - by_folder),
        "only_by_folder": len(by_folder - by_timestamp),
        "checked": len(joined),
    }


def split_dataset(result: schema.ParseResult, dataset: str) -> schema.ParseResult:
    """Apply the configured split to a parsed dataset, in place of its nulls."""
    if dataset not in config.DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(config.DATASETS)}")
    spec = config.DATASETS[dataset].split

    result.impressions = assign(result.impressions, spec)
    result.counts["splits"] = summarise(result.impressions)
    result.counts["split_crosscheck"] = crosscheck_against_source_periods(
        result.impressions, result.impressions_extra
    )
    return result
