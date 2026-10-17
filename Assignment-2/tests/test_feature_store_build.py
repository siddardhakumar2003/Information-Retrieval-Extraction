"""Q1.9, Q1.10, Q9.5 — the derived feature tables and the manifest.

`test_feature_store.py` covers the Parquet round-trip contract. This file covers
what is actually *in* the store, and above all that the per-split user features
respect the boundary Q9.5 requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store

# ---------------------------------------------------------------------------
# Article features
# ---------------------------------------------------------------------------


def test_article_features_expose_one_text_field(feature_store_dir: Path, dataset: str):
    """`text` is title + abstract, built in ONE place.

    Q2's index and Q3's encoder must not be able to disagree about what an
    article's text is; if each concatenated its own, a tokenisation change in
    one would silently make the lexical/semantic comparison invalid.
    """
    af = feature_store.read_table(feature_store_dir / "article_features.parquet")
    assert "text" in af.columns
    row = af.row(0, named=True)
    assert row["title"] in row["text"]
    if row["abstract"]:
        assert row["abstract"] in row["text"]
    assert af["text"].null_count() == 0


def test_article_features_record_body_availability(feature_store_dir: Path, dataset: str):
    """MIND ships no body at all; EB-NeRD does. `has_body` makes the disclosed
    deviation queryable instead of a footnote."""
    af = feature_store.read_table(feature_store_dir / "article_features.parquet")
    if dataset == "mind_small":
        assert not af["has_body"].any(), "MIND has no body text (disclosed deviation)"
    else:
        assert af["has_body"].any()


def test_article_features_cover_every_article(feature_store_dir: Path, processed_store: Path):
    af = feature_store.read_table(feature_store_dir / "article_features.parquet")
    articles = feature_store.read_table(processed_store / "articles.parquet")
    assert set(af["article_id"]) == set(articles["article_id"])


# ---------------------------------------------------------------------------
# User features — Q9.5
# ---------------------------------------------------------------------------


def test_user_features_are_keyed_by_user_and_split(feature_store_dir: Path):
    """Not by user alone. One row per user would be computed over the whole
    history and then used to score every split."""
    uf = feature_store.read_table(feature_store_dir / "user_features.parquet")
    assert {"user_id", "split"} <= set(uf.columns)
    assert uf.select("user_id", "split").n_unique() == len(uf)
    assert set(uf["split"]) <= {"train", "val", "test"}


def test_user_features_never_use_clicks_from_after_the_split_starts(
    feature_store_dir: Path, processed_store: Path, dataset: str
):
    """Q9.5, asserted directly on the produced table.

    The boundary for split S is the START of S. Getting this backwards -- using
    S's end -- lets a train impression on 20 May use clicks from 24 May while
    every count still looks plausible. That is exactly the bug this test was
    written to catch, and it did.
    """
    uf = feature_store.read_table(feature_store_dir / "user_features.parquet")
    if not uf["history_is_timestamped"].any():
        pytest.skip(f"{dataset} ships no history timestamps; boundary is structural")

    impressions = feature_store.read_table(processed_store / "impressions.parquet")
    starts = feature_store.split_start_times(impressions)

    for split_name, start in starts.items():
        rows = uf.filter(pl.col("split") == split_name)
        if rows.is_empty():
            continue
        latest = rows["last_click_time"].max()
        assert latest < start, (
            f"{split_name}: a user feature uses a click at {latest}, which is at or "
            f"after the split starts at {start} -- that is the user's own future"
        )


def test_user_feature_history_grows_monotonically_across_splits(
    feature_store_dir: Path, dataset: str
):
    """Later splits see more history, never less.

    A useful cross-check on the boundary direction: if the filter were
    inverted or dropped, this ordering would break or collapse to equality.
    """
    uf = feature_store.read_table(feature_store_dir / "user_features.parquet")
    if not uf["history_is_timestamped"].any():
        pytest.skip(f"{dataset} ships no history timestamps")

    totals = {
        row["split"]: row["clicks"]
        for row in uf.group_by("split").agg(pl.col("n_clicks").sum().alias("clicks")).to_dicts()
    }
    assert totals["train"] < totals["val"] <= totals["test"], totals


def test_untimestamped_history_is_handled_explicitly(feature_store_dir: Path, dataset: str):
    """MIND's history predates the whole log period, so it is legitimately
    available to every split.

    Handled by an explicit branch, not by letting `click_time < boundary` on an
    all-null column return nothing -- which would empty MIND's user features
    without raising anything.
    """
    uf = feature_store.read_table(feature_store_dir / "user_features.parquet")
    if dataset != "mind_small":
        pytest.skip("MIND-specific")
    assert not uf["history_is_timestamped"].any()
    assert uf["n_clicks"].sum() > 0, "MIND user features are empty; the null filter swallowed them"
    assert uf["recency_seconds"].null_count() == len(uf), "no timestamps means no recency"


# ---------------------------------------------------------------------------
# Manifest (Q1.10)
# ---------------------------------------------------------------------------


def test_manifest_is_complete(processed_store: Path, dataset: str):
    manifest = json.loads((processed_store / "_manifest.json").read_text())
    for key in (
        "dataset",
        "config_fingerprint",
        "counts",
        "splits",
        "rejections",
        "articles",
        "tables",
        "source_files",
    ):
        assert key in manifest, f"manifest is missing {key!r}"
    assert manifest["dataset"] == dataset
    assert manifest["config_fingerprint"] == feature_store.config_fingerprint(dataset)


def test_manifest_has_no_wall_clock_field(processed_store: Path):
    """Determinism (Q1.13) beats provenance-by-timestamp.

    A `built_at` would make every rebuild differ while nothing meaningful
    changed, and the determinism test could then only pass by excluding the very
    file that describes the build. What the build ran ON is captured by the
    source checksums and the config fingerprint.
    """
    manifest = json.loads((processed_store / "_manifest.json").read_text())
    banned = {"built_at", "timestamp", "generated_at", "created_at", "run_at"}
    assert not banned & set(manifest), (
        f"manifest carries a wall-clock field: {banned & set(manifest)}"
    )


def test_manifest_records_a_checksum_per_table(processed_store: Path):
    manifest = json.loads((processed_store / "_manifest.json").read_text())
    for name, info in manifest["tables"].items():
        assert len(info["sha256"]) == 64, name
        assert info["rows"] == len(feature_store.read_table(processed_store / f"{name}.parquet"))


def test_split_crosscheck_agrees(processed_store: Path):
    """The timestamp rule reproduces the shipped folder assignment exactly."""
    manifest = json.loads((processed_store / "_manifest.json").read_text())
    assert manifest["split_crosscheck"]["only_by_timestamp"] == 0
    assert manifest["split_crosscheck"]["only_by_folder"] == 0


# ---------------------------------------------------------------------------
# Store read contract
# ---------------------------------------------------------------------------


def test_read_processed_validates_against_the_schema(store_root: Path, dataset: str):
    """Validated on READ as well as on write: a store can be stale, hand-edited
    or produced by older code, and every downstream stage trusts its shape."""
    tables = feature_store.read_processed(dataset, store_root / "processed")
    assert set(tables) >= {"articles", "impressions", "history"}


def test_missing_partition_raises_rather_than_reading_empty(tmp_path: Path):
    """An empty frame propagates into a zero metric, and a zero looks like a
    result."""
    with pytest.raises(FileNotFoundError, match="make data"):
        feature_store.read_table(tmp_path / "nope.parquet")


def test_unknown_dataset_is_rejected(tmp_path: Path):
    with pytest.raises(KeyError, match="unknown dataset"):
        feature_store.build_dataset("ebnerd_enormous", root=tmp_path)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.parametrize("ds", ["mind_small", "ebnerd_demo", "ebnerd_small"])
def test_real_store_builds_and_reconciles(tmp_path, ds):
    root = config.RAW_DIR / ("MIND" if ds == "mind_small" else config.DATASETS[ds].raw_subdir)
    marker = "MINDsmall_train/behaviors.tsv" if ds == "mind_small" else "articles.parquet"
    if not (root / marker).exists():
        pytest.skip(f"{ds} not downloaded")

    manifest = feature_store.build_dataset(ds, raw_root=root, root=tmp_path)
    counts = manifest["counts"]
    assert counts["raw"] == counts["parsed"] + counts["rejected"]
    assert sum(manifest["splits"].values()) == counts["parsed"]
    articles = manifest["articles"]
    assert articles["raw"] - articles["duplicates"] == articles["unique"]
    assert manifest["split_crosscheck"]["only_by_timestamp"] == 0
    assert manifest["split_crosscheck"]["only_by_folder"] == 0
