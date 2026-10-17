"""Q1.6-Q1.8 — the temporal splitter.

Unit tests against hand-built frames, so they run today. The store-level
assertions in `test_temporal_split.py` stay skipped until a store exists.

The theme throughout: a splitter that is wrong still produces a split. Every
failure mode here yields a plausible three-way partition, so each is tested
explicitly rather than trusted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from newsrec import config
from newsrec.data import schema, split

VAL_START = datetime(2023, 5, 24, 7, 0, tzinfo=UTC)
TEST_START = datetime(2023, 5, 25, 7, 0, tzinfo=UTC)
SPEC = config.SplitSpec(val_start=VAL_START, test_start=TEST_START, rationale="test fixture")


def _impressions(times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": [f"I{i}" for i in range(len(times))],
            "user_id": ["U1"] * len(times),
            "impression_time": times,
            "candidate_ids": [["A1"]] * len(times),
            "clicked_ids": [["A1"]] * len(times),
            "split": [None] * len(times),
            "dataset": ["t"] * len(times),
        },
        schema=dict(schema.IMPRESSIONS.columns),
    )


SPANNING = [
    VAL_START - timedelta(days=2),  # train
    VAL_START - timedelta(seconds=1),  # train, right up to the boundary
    VAL_START,  # val, exactly on it
    TEST_START - timedelta(seconds=1),  # val
    TEST_START,  # test, exactly on it
    TEST_START + timedelta(days=3),  # test
]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_boundaries_are_half_open_lower_inclusive():
    """`val_start <= t < test_start` is val.

    The exact-boundary rows are the ones an off-by-one moves, and they are
    invisible in aggregate counts on a large dataset.
    """
    out = split.assign(_impressions(SPANNING), SPEC)
    assert out.sort("impression_time")["split"].to_list() == [
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
    ]


def test_split_sizes_are_reported_for_all_three():
    out = split.assign(_impressions(SPANNING), SPEC)
    assert split.summarise(out) == {"train": 2, "val": 2, "test": 2}


def test_output_is_deterministically_ordered():
    """Q1.13 — impression_time alone is not a total order (MIND has thousands of
    ties per second), so impression_id breaks them."""
    tied = [VAL_START - timedelta(days=1)] * 3 + SPANNING
    a = split.assign(_impressions(tied), SPEC)
    b = split.assign(_impressions(tied), SPEC)
    assert a["impression_id"].to_list() == b["impression_id"].to_list()
    assert a.equals(b)


def test_result_satisfies_the_schema():
    out = split.assign(_impressions(SPANNING), SPEC)
    schema.validate_table(out, schema.IMPRESSIONS)
    assert out["split"].null_count() == 0


# ---------------------------------------------------------------------------
# Refusals — each would otherwise produce a plausible, wrong split
# ---------------------------------------------------------------------------


def test_naive_timestamps_are_refused():
    """Comparing a naive timestamp to a tz-aware boundary is either an error or
    a silent offset. Neither is acceptable at a split boundary."""
    naive = _impressions(SPANNING).with_columns(
        pl.col("impression_time").dt.replace_time_zone(None)
    )
    with pytest.raises(split.SplitError, match="expected tz-aware UTC"):
        split.assign(naive, SPEC)


def test_boundary_outside_the_data_range_is_refused():
    """The generalisation of the OQ-4 placeholder failure.

    With an out-of-range boundary every row lands in one split: the pipeline
    runs, produces a partition, and is entirely wrong.
    """
    spec = config.SplitSpec(
        val_start=datetime(1971, 1, 1, tzinfo=UTC),
        test_start=datetime(1971, 1, 2, tzinfo=UTC),
        rationale="placeholder-like",
    )
    with pytest.raises(split.SplitError, match="outside the observed"):
        split.assign(_impressions(SPANNING), spec)


def test_boundary_that_empties_a_split_is_refused():
    """A boundary inside the range but past every row still empties val."""
    only_early = _impressions([VAL_START - timedelta(days=i) for i in (3, 2, 1)])
    with pytest.raises(split.SplitError):
        split.assign(only_early, SPEC)


def test_empty_input_is_refused():
    with pytest.raises(split.SplitError, match="empty impressions"):
        split.assign(schema.IMPRESSIONS.empty(), SPEC)


def test_random_split_is_unrepresentable():
    """Q1.6 asserted at the level of the API rather than the output.

    A test on counts cannot distinguish a temporal split from a lucky shuffle.
    This asserts the splitter has no way to express one: SplitSpec carries only
    boundaries, and assign() takes only a frame and a spec.
    """
    import inspect

    fields = set(config.SplitSpec.__dataclass_fields__)
    assert fields == {"val_start", "test_start", "rationale"}
    for banned in ("frac", "fraction", "ratio", "shuffle", "seed", "random_state"):
        assert banned not in fields
        assert banned not in inspect.signature(split.assign).parameters


# ---------------------------------------------------------------------------
# validate() catches a bad split it did not produce
# ---------------------------------------------------------------------------


def test_validate_detects_interleaved_splits():
    """The failure a shuffle produces. Labels are overwritten by hand so
    validate() is exercised on a split it did not create."""
    out = split.assign(_impressions(SPANNING), SPEC)
    shuffled = out.with_columns(
        pl.Series("split", ["train", "test", "train", "test", "train", "val"])
    )
    problems = split.validate(shuffled, SPEC)
    assert any("overlaps" in p for p in problems), problems


def test_validate_detects_an_empty_split():
    out = split.assign(_impressions(SPANNING), SPEC)
    no_val = out.with_columns(
        pl.when(pl.col("split") == "val")
        .then(pl.lit("train"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    assert any("val split is empty" in p for p in split.validate(no_val, SPEC))


def test_validate_detects_a_boundary_mismatch():
    """A split that is internally ordered but does not match the CONFIGURED
    boundary — the case Q1.8 exists for."""
    out = split.assign(_impressions(SPANNING), SPEC)
    later = config.SplitSpec(
        val_start=TEST_START,
        test_start=TEST_START + timedelta(days=1),
        rationale="shifted",
    )
    assert split.validate(out, later)


# ---------------------------------------------------------------------------
# Cross-check against folder provenance
# ---------------------------------------------------------------------------


def test_crosscheck_reports_agreement():
    out = split.assign(_impressions(SPANNING), SPEC)
    extra = pl.DataFrame(
        {
            "impression_id": out["impression_id"],
            "source_period": [
                "train" if s != "test" else "validation" for s in out["split"].to_list()
            ],
        }
    )
    result = split.crosscheck_against_source_periods(out, extra)
    assert result == {"only_by_timestamp": 0, "only_by_folder": 0, "checked": 6}


def test_crosscheck_reports_disagreement():
    """Evidence, not definition: a mismatch is reported, never corrected."""
    out = split.assign(_impressions(SPANNING), SPEC)
    extra = pl.DataFrame(
        {"impression_id": out["impression_id"], "source_period": ["train"] * len(out)}
    )
    result = split.crosscheck_against_source_periods(out, extra)
    assert result["only_by_timestamp"] == 2
    assert result["only_by_folder"] == 0


# ---------------------------------------------------------------------------
# Real data (Q1.8)
# ---------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("mind_small", {"train": 126_695, "val": 30_270, "test": 73_152}),
        ("ebnerd_demo", {"train": 21_318, "val": 3_406, "test": 25_356}),
        ("ebnerd_small", {"train": 200_328, "val": 32_559, "test": 244_647}),
    ],
)
def test_real_split_sizes_and_crosscheck(dataset, expected):
    """The configured boundaries produce exactly these splits, and the
    timestamp rule reproduces the shipped folder assignment with zero
    disagreement in either direction."""
    from newsrec.data import ebnerd, mind

    if dataset == "mind_small":
        root = config.RAW_DIR / "MIND"
        if not (root / "MINDsmall_train" / "behaviors.tsv").exists():
            pytest.skip("MIND not downloaded")
        parsed = mind.parse(root)
    else:
        root = config.RAW_DIR / config.DATASETS[dataset].raw_subdir
        if not (root / "articles.parquet").exists():
            pytest.skip(f"{dataset} not downloaded")
        parsed = ebnerd.parse(root, dataset)

    parsed = split.split_dataset(parsed, dataset)
    assert parsed.counts["splits"] == expected
    assert parsed.counts["split_crosscheck"]["only_by_timestamp"] == 0
    assert parsed.counts["split_crosscheck"]["only_by_folder"] == 0
    assert split.validate(parsed.impressions, config.DATASETS[dataset].split) == []
