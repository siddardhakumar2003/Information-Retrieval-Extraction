"""Q1.2, Q1.10 — the EB-NeRD parser.

Runs on the committed JSON fixture (materialised to Parquet at test time), with
real-file canaries marked `realdata`.

The hazards here are almost disjoint from MIND's, so the tests are too. The one
that matters most is history: EB-NeRD ships overlapping rolling windows, so the
union of both files spans *into* impression time and is only safe under the
strict Q9.4 boundary. That property is asserted here explicitly rather than left
as a comment, because it is the assumption the whole design rests on.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from newsrec import config
from newsrec.data import ebnerd, schema

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ebnerd_root(tmp_path_factory) -> Path:
    """Materialise the committed JSON fixture into the Parquet layout.

    Fixtures are committed as JSON because `*.parquet` is itself in
    config.FORBIDDEN_PATH_PATTERNS, and a binary fixture cannot be reviewed in a
    diff. The parser still gets its real input format.
    """
    dest = tmp_path_factory.mktemp("ebnerd_demo")
    result = subprocess.run(
        [sys.executable, str(FIXTURES / "build_fixtures.py"), "ebnerd_demo", str(dest)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return dest


@pytest.fixture(scope="module")
def parsed(ebnerd_root: Path):
    return ebnerd.parse(ebnerd_root, "ebnerd_demo")


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


def test_output_satisfies_the_canonical_schema(parsed):
    schema.validate_store(parsed.as_store())


def test_numeric_ids_become_strings(parsed):
    """Q1.2 — EB-NeRD ids are Int32/UInt32, MIND's are strings like N12345.

    Casting to Utf8 here is what makes the join key one type everywhere. Leaving
    them numeric would make 'unified schema' false at the first cross-dataset
    join and push a dataset conditional into every downstream module.
    """
    assert parsed.articles.schema["article_id"] == schema.ID
    assert parsed.impressions.schema["user_id"] == schema.ID
    assert parsed.impressions.schema["candidate_ids"] == schema.ID_LIST
    assert parsed.articles["article_id"][0] == "101"


def test_timestamps_localised_to_utc(parsed):
    """Q1.5 — EB-NeRD Parquet stores tz-NAIVE datetimes."""
    for frame, col in (
        (parsed.articles, "published_time"),
        (parsed.impressions, "impression_time"),
        (parsed.history, "click_time"),
    ):
        assert frame.schema[col].time_zone == "UTC", col
    first = parsed.impressions.sort("impression_time")["impression_time"][0]
    assert first == datetime(2023, 5, 19, 10, 0, tzinfo=UTC)


def test_split_is_null_until_the_splitter_runs(parsed):
    assert parsed.impressions["split"].null_count() == len(parsed.impressions)


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------


def test_subtitle_maps_to_abstract(parsed):
    """EB-NeRD's subtitle plays MIND's abstract role, so Q2 indexes comparable
    text (title + abstract) on both datasets."""
    row = parsed.articles.filter(pl.col("article_id") == "101").row(0, named=True)
    assert row["abstract"].startswith("Forhandlingerne")
    assert row["category"] == "nyheder"  # category_str, not the numeric code


def test_subcategory_list_becomes_a_stable_string(parsed):
    """subcategory is List(Int16) here and a plain string in MIND.

    Joined into a sorted, pipe-separated string so the column has ONE type
    across datasets. Sorted so the value is stable rather than dependent on
    source ordering.
    """
    assert parsed.articles.schema["subcategory"] == schema.TEXT
    row = parsed.articles.filter(pl.col("article_id") == "101").row(0, named=True)
    assert row["subcategory"] == "11"


def test_null_body_is_preserved_not_emptied(parsed):
    """Article 103 has a null body in the fixture. Null and empty are different
    states and flattening them loses that."""
    row = parsed.articles.filter(pl.col("article_id") == "103").row(0, named=True)
    assert row["body"] is None


def test_serving_unavailable_columns_are_in_extras_only(parsed):
    """Q9.3 — all seven retained, none in a core table.

    Retained because Q9.1 needs paired with/without numbers and you cannot
    ablate a feature discarded at parse time; in the extras so that reaching one
    costs a deliberate join.
    """
    for frame in (parsed.articles, parsed.impressions, parsed.history):
        assert not set(frame.columns) & config.SERVING_UNAVAILABLE_FEATURES

    extras = set(parsed.articles_extra.columns) | set(parsed.impressions_extra.columns)
    assert config.SERVING_UNAVAILABLE_FEATURES <= extras, (
        "an ablation cannot be run on a feature that was thrown away at parse time"
    )


def test_original_impression_id_and_period_survive_for_q5(parsed):
    extra = parsed.impressions_extra
    assert set(extra["source_period"]) == {"train", "validation"}
    row = extra.filter(pl.col("impression_id") == "1001").row(0, named=True)
    assert row["source_impression_id"] == "1001"


# ---------------------------------------------------------------------------
# History — the load-bearing hazard
# ---------------------------------------------------------------------------


def test_history_is_timestamped_unlike_mind(parsed):
    """The asymmetry the leakage checker must handle.

    MIND ships no history timestamps, so its boundary is structural. EB-NeRD
    ships them, so its boundary is checkable directly. A leakage test written to
    assume timestamps exist would be silently vacuous on MIND; one written to
    assume they do not would never check EB-NeRD at all.
    """
    assert parsed.history["click_time"].null_count() == 0


def test_history_unions_both_periods_and_deduplicates(parsed):
    """The two files are overlapping rolling windows, not alternatives.

    Taking `train/` alone discards real clicks preceding a validation
    impression; taking `validation/` alone hands a train impression clicks from
    its future. The union of timestamped clicks, deduplicated, is the only
    reading that is both complete and checkable.
    """
    counts = parsed.history_counts
    assert counts["raw"] == counts["raw_train"] + counts["raw_validation"]
    assert counts["unique"] + counts["duplicates_across_periods"] == counts["raw"]
    assert counts["duplicates_across_periods"] > 0, "fixture must exercise the overlap"

    # No duplicate (user, article, click_time) triples survive.
    triples = parsed.history.select("user_id", "article_id", "click_time")
    assert triples.n_unique() == len(triples)


def test_parallel_list_columns_stay_aligned(parsed):
    """article_id_fixed and impression_time_fixed are PARALLEL lists.

    Exploding them separately would pair click i with timestamp j the moment two
    lists differ in length -- a silent, catastrophic misalignment that produces
    a perfectly valid-looking table of wrong facts. The fixture gives user 501
    two clicks with known, distinct timestamps.
    """
    rows = parsed.history.filter(pl.col("user_id") == "501").sort("click_time").to_dicts()
    pairs = {(r["article_id"], r["click_time"]) for r in rows}
    assert ("101", datetime(2023, 5, 10, 8, 0, tzinfo=UTC)) in pairs
    assert ("102", datetime(2023, 5, 12, 20, 15, tzinfo=UTC)) in pairs


def test_history_overlaps_impression_time_so_the_boundary_is_mandatory(parsed):
    """The assumption the union rests on, asserted rather than assumed.

    Because the validation history window covers the train period, the unioned
    history contains clicks that fall AFTER some impressions. That is fine --
    and only fine -- because every click is timestamped and Q9.4 filters on
    `click_time < impression_time`. If this test ever fails, the union stopped
    overlapping and the strict filter would be silently untested.
    """
    earliest_impression = parsed.impressions["impression_time"].min()
    later = parsed.history.filter(pl.col("click_time") > earliest_impression)
    assert len(later) > 0, (
        "history no longer overlaps impression time; the Q9.4 filter would be "
        "vacuous on this data and the leakage canary is doing the real work"
    )


def test_user_with_empty_history_yields_no_rows(parsed):
    """User 503 has empty parallel lists -- a cold-start user, not a parse error."""
    assert parsed.history.filter(pl.col("user_id") == "503").is_empty()
    assert "503" in set(parsed.impressions["user_id"])


# ---------------------------------------------------------------------------
# Accounting (Q1.10)
# ---------------------------------------------------------------------------


def test_rows_reconcile(parsed):
    assert parsed.reconciles
    assert parsed.counts["raw"] == 6  # 4 train + 2 validation
    assert parsed.counts["rejected"] == 0
    assert parsed.rejections == {}


def _write_behaviors(root: Path, rows: list[dict]) -> None:
    """Minimal but schema-shaped behaviours for both periods.

    Ids are offset per period because real EB-NeRD ids are disjoint across
    train/ and validation/; reusing them would trip the primary-key check for a
    reason that has nothing to do with what the test is checking.
    """
    for offset, period in enumerate(ebnerd.SOURCE_PERIODS):
        d = root / period
        d.mkdir(parents=True, exist_ok=True)
        shifted = [{**r, "impression_id": r["impression_id"] + offset * 1000} for r in rows]
        pl.DataFrame(shifted).write_parquet(d / "behaviors.parquet")
        pl.DataFrame(
            {
                "user_id": [501],
                "article_id_fixed": [[101]],
                "impression_time_fixed": [[datetime(2023, 4, 30, 8, 0)]],
                "read_time_fixed": [[1.0]],
                "scroll_percentage_fixed": [[1.0]],
            }
        ).write_parquet(d / "history.parquet")


def test_malformed_rows_are_rejected_by_reason_not_dropped(tmp_path):
    """These paths fire zero times on today's clean files. They exist so a future
    file that is NOT clean gets counted rather than silently reshaped."""
    pl.DataFrame(
        {
            "article_id": [101],
            "title": ["t"],
            "subtitle": ["s"],
            "body": ["b"],
            "category": [1],
            "category_str": ["nyheder"],
            "subcategory": [[11]],
            "published_time": [datetime(2023, 5, 1)],
        }
    ).write_parquet(tmp_path / "articles.parquet")

    rows = [
        {  # good
            "impression_id": 1,
            "user_id": 501,
            "article_id": None,
            "article_ids_inview": [101],
            "article_ids_clicked": [101],
            "impression_time": datetime(2023, 5, 20, 8, 0),
        },
        {  # empty_candidate_list
            "impression_id": 2,
            "user_id": 501,
            "article_id": None,
            "article_ids_inview": [],
            "article_ids_clicked": [],
            "impression_time": datetime(2023, 5, 20, 9, 0),
        },
        {  # clicked_not_in_candidates
            "impression_id": 3,
            "user_id": 501,
            "article_id": None,
            "article_ids_inview": [101],
            "article_ids_clicked": [999],
            "impression_time": datetime(2023, 5, 20, 10, 0),
        },
        {  # null_impression_time
            "impression_id": 4,
            "user_id": 501,
            "article_id": None,
            "article_ids_inview": [101],
            "article_ids_clicked": [101],
            "impression_time": None,
        },
    ]
    _write_behaviors(tmp_path, rows)

    result = ebnerd.parse(tmp_path, "ebnerd_demo")
    assert result.reconciles
    assert result.counts["raw"] == 8  # 4 rows x 2 periods
    assert result.counts["parsed"] == 2
    assert result.rejections["empty_candidate_list"] == 2
    assert result.rejections["clicked_not_in_candidates"] == 2
    assert result.rejections["null_impression_time"] == 2


def test_missing_source_file_errors_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="make download"):
        ebnerd.parse(tmp_path, "ebnerd_demo")


def test_unknown_dataset_key_is_rejected(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset"):
        ebnerd.parse(tmp_path, "ebnerd_enormous")


# ---------------------------------------------------------------------------
# Real-data canaries
# ---------------------------------------------------------------------------


@pytest.mark.realdata
def test_fixture_dtypes_match_the_real_files(ebnerd_root):
    """The fixture must have the REAL schema, not merely the right values.

    The first build_fixtures.py inferred Utf8 for ISO timestamps, so ebnerd.py
    passed on the fixture and failed on the real Parquet. A fixture with the
    wrong dtypes tests the parser against input it will never see.
    """
    real_root = config.RAW_DIR / config.DATASETS["ebnerd_demo"].raw_subdir
    if not (real_root / "articles.parquet").exists():
        pytest.skip("ebnerd_demo not downloaded")

    for rel in ("articles.parquet", "train/behaviors.parquet", "train/history.parquet"):
        real = pl.read_parquet(real_root / rel).schema
        fake = pl.read_parquet(ebnerd_root / rel).schema
        for col, dtype in fake.items():
            assert col in real, f"{rel}: fixture has column {col!r} the real file lacks"
            assert dtype == real[col], f"{rel}.{col}: fixture is {dtype}, real file is {real[col]}"


@pytest.mark.realdata
@pytest.mark.parametrize("dataset", ["ebnerd_demo", "ebnerd_small"])
def test_real_ebnerd_counts(dataset):
    """Counted from the Parquet files on 2026-08-20."""
    root = config.RAW_DIR / config.DATASETS[dataset].raw_subdir
    if not (root / "articles.parquet").exists():
        pytest.skip(f"{dataset} not downloaded")

    result = ebnerd.parse(root, dataset)
    schema.validate_store(result.as_store())
    assert result.reconciles
    assert result.counts["rejected"] == 0, result.rejections
    assert result.history["click_time"].null_count() == 0
    # No id collision across periods, unlike MIND.
    assert result.impressions["impression_id"].n_unique() == len(result.impressions)

    expected = {"ebnerd_demo": 50_080, "ebnerd_small": 477_534}[dataset]
    assert result.counts["raw"] == expected

    if dataset == "ebnerd_demo":
        assert result.article_counts["unique"] == 11_777
        assert result.history_counts["duplicates_across_periods"] > 0
