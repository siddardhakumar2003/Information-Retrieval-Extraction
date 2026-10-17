"""Q1.6-Q1.8 — the split is temporal, asserted on timestamps.

Deliberately never asserts on which source FOLDER a row came from. Folder
provenance is what we believe about the data; timestamps are what is true. A
splitter that silently shuffled would still put rows in the right folders.
"""

from __future__ import annotations

from itertools import pairwise


def test_split_boundaries_ordered(processed_store, dataset):
    """Q1.6 — max(train) <= min(val) and max(val) <= min(test), on real times."""
    import polars as pl

    df = pl.read_parquet(processed_store / "impressions.parquet")
    parts = {
        s: df.filter(pl.col("split") == s)["impression_time"] for s in ("train", "val", "test")
    }
    assert parts["train"].max() <= parts["val"].min(), "train overlaps val"
    assert parts["val"].max() <= parts["test"].min(), "val overlaps test"


def test_splits_partition_input(processed_store, dataset):
    """Q1.7 — disjoint on impression_id, union is the whole input, none empty."""
    import polars as pl

    df = pl.read_parquet(processed_store / "impressions.parquet")
    ids = {
        s: set(df.filter(pl.col("split") == s)["impression_id"]) for s in ("train", "val", "test")
    }
    for s, group in ids.items():
        assert group, f"{s} split is empty"
    assert not (ids["train"] & ids["val"]), "train and val overlap"
    assert not (ids["val"] & ids["test"]), "val and test overlap"
    assert not (ids["train"] & ids["test"]), "train and test overlap"
    assert len(ids["train"] | ids["val"] | ids["test"]) == len(df), "rows lost or duplicated"


def test_boundaries_match_config(processed_store, dataset):
    """Q1.8 — the produced split equals config, not a literal somewhere.

    Compares full timestamps, never dates. EB-NeRD's periods break at 07:00 UTC
    (OQ-4), so a date-level comparison here would pass while the splitter had
    put the first seven hours of the validation period into train.
    """
    import polars as pl

    from newsrec import config

    spec = config.DATASETS[dataset].split
    df = pl.read_parquet(processed_store / "impressions.parquet")
    train_max = df.filter(pl.col("split") == "train")["impression_time"].max()
    val_min = df.filter(pl.col("split") == "val")["impression_time"].min()
    val_max = df.filter(pl.col("split") == "val")["impression_time"].max()
    test_min = df.filter(pl.col("split") == "test")["impression_time"].min()

    assert train_max < spec.val_start <= val_min
    assert val_max < spec.test_start <= test_min


def test_split_is_contiguous_not_shuffled(processed_store, dataset):
    """A regression guard: a random split would interleave timestamps.

    Each split's timestamp range must not overlap another's. This is the
    property a shuffle destroys and a date filter preserves.
    """
    import polars as pl

    df = pl.read_parquet(processed_store / "impressions.parquet")
    ranges = []
    for s in ("train", "val", "test"):
        col = df.filter(pl.col("split") == s)["impression_time"]
        ranges.append((col.min(), col.max()))
    for (_, end), (start, _) in pairwise(ranges):
        assert end <= start, "split timestamp ranges interleave — was this a shuffle?"


def test_split_boundaries_are_tz_aware_and_ordered():
    """Runs today — needs no pipeline, only config.

    Guards the OQ-4 finding directly. Naive boundaries are rejected by
    SplitSpec.__post_init__, and EB-NeRD's must sit at 07:00 UTC rather than
    midnight: a midnight boundary silently moves seven hours of the validation
    period into train.
    """
    from datetime import UTC

    from newsrec import config

    for key, spec in ((k, d.split) for k, d in config.DATASETS.items()):
        assert spec.val_start.tzinfo is not None, f"{key}: naive val_start"
        assert spec.test_start.tzinfo is not None, f"{key}: naive test_start"
        assert spec.val_start < spec.test_start, f"{key}: boundaries out of order"

    mind = config.DATASETS["mind_small"].split
    assert (mind.val_start.hour, mind.test_start.hour) == (0, 0), "MIND is calendar-aligned"

    for key in ("ebnerd_demo", "ebnerd_small"):
        spec = config.DATASETS[key].split
        assert (spec.val_start.hour, spec.test_start.hour) == (7, 7), (
            f"{key}: EB-NeRD collection periods run 07:00-to-07:00 UTC (OQ-4); "
            "a midnight boundary would put validation data into train"
        )
        assert spec.test_start.astimezone(UTC).isoformat() == "2023-05-25T07:00:00+00:00"
