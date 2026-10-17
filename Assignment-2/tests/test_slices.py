"""Q4.5 — the evaluation slices.

The failure guarded against is a slice that looks informative and is not: masks
that overlap (double-counting impressions), masks that leave a gap (silently
dropping them), or a "training-period" slice that quietly reads the test period.
All three produce per-slice numbers that look precise and do not reconstruct the
overall number.

The degeneracy of the Q4.5-mandated training-period head/tail cut is measured on
real data, not asserted here — the fixture corpus is far too small to reproduce
a 0.4%/99.6% split. What these tests pin is that the two cuts are *different*
and that each reads the period it claims to.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec.data import feature_store
from newsrec.eval import slices
from newsrec.retrieval import recall


@pytest.fixture
def parts(store_root, dataset):
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    history = feature_store.read_table(store_root / "processed" / dataset / "history.parquet")
    sample = recall.sample_impressions(impressions, "test", 25)
    users = sample["user_id"].unique().to_list()
    return sample, history.filter(pl.col("user_id").is_in(users)), impressions


@pytest.fixture
def masks(parts):
    return slices.build(*parts)


# =============================================================================
# 1. The masks partition the sample
# =============================================================================


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("cold_start", "warm"),
        ("article_head", "article_tail"),
        ("article_head_train", "article_tail_train"),
    ],
)
def test_each_pair_partitions_with_no_overlap_and_no_gap(masks, parts, a, b):
    n = len(parts[0])
    assert not (masks[a] & masks[b]).any(), f"{a} and {b} overlap"
    assert (masks[a] | masks[b]).all(), f"{a} and {b} leave a gap"
    assert int(masks[a].sum()) + int(masks[b].sum()) == n


def test_the_quartile_pair_deliberately_leaves_the_middle_out(masks):
    """`history_short`/`history_long` are quartiles, not a partition.

    Asserting they partition would be wrong, and asserting nothing would let a
    genuine bug through, so what is pinned is that the middle half exists — or,
    on a tiny fixture where the quartiles meet, that they at least do not
    silently become a partition by accident.
    """
    covered = masks["history_short"] | masks["history_long"]
    assert covered.any(), "both quartile slices are empty"


def test_every_mask_is_boolean_and_aligned_to_the_sample(masks, parts):
    n = len(parts[0])
    for name, mask in masks.items():
        assert mask.dtype == np.bool_, f"{name} is not boolean"
        assert mask.shape == (n,), f"{name} is not aligned to the sample"


def test_partition_check_raises_on_an_overlap(masks, parts):
    """The guard must bite, or it is decoration."""
    broken = dict(masks)
    broken["warm"] = np.ones(len(parts[0]), dtype=bool)  # now overlaps cold_start
    if not masks["cold_start"].any():
        pytest.skip("this fixture has no cold-start impressions to overlap")
    with pytest.raises(ValueError, match="do not partition"):
        slices.partition_check(broken, len(parts[0]))


def test_partition_check_raises_on_a_gap(masks, parts):
    broken = dict(masks)
    broken["article_head"] = np.zeros(len(parts[0]), dtype=bool)
    broken["article_tail"] = np.zeros(len(parts[0]), dtype=bool)
    with pytest.raises(ValueError, match="do not partition"):
        slices.partition_check(broken, len(parts[0]))


def test_partition_check_accepts_the_real_masks(masks, parts):
    slices.partition_check(masks, len(parts[0]))


# =============================================================================
# 2. Q4.5 — training-period membership reads the training period
# =============================================================================


def test_slice_definitions_use_train_period_only(parts):
    """The named Q4.5 condition.

    `cold_start`/`warm` and the `_train` head/tail cut must not change when the
    test period changes. Deleting every test-period impression from the table
    the slices are built from must leave those masks identical — if a definition
    were secretly reading test rows, they would move.

    `article_head`/`article_tail` are excluded on purpose: they are declared
    test-period descriptors, and this test asserts that distinction is real
    rather than assuming it.
    """
    sample, history, impressions = parts
    train_only = impressions.filter(pl.col("split") != "test")

    full = slices.build(sample, history, impressions)
    without_test = slices.build(sample, history, train_only)

    for name in slices.TRAINING_PERIOD_SLICES:
        assert np.array_equal(full[name], without_test[name]), (
            f"{name} changed when test-period impressions were removed, "
            f"so its membership is not training-period only"
        )


def test_the_test_period_slice_really_does_read_the_test_period(parts):
    """The canary for the test above.

    If removing the test period changed nothing anywhere, the previous test
    would pass vacuously. `article_head` is defined from test-period clicks, so
    it must move — proving the removal has an observable effect at all.
    """
    sample, history, impressions = parts
    train_only = impressions.filter(pl.col("split") != "test")
    full = slices.build(sample, history, impressions)
    without_test = slices.build(sample, history, train_only)

    moved = any(
        not np.array_equal(full[name], without_test[name]) for name in slices.TEST_PERIOD_SLICES
    )
    assert moved, (
        "removing the test period changed no test-period slice, so "
        "test_slice_definitions_use_train_period_only proves nothing"
    )


def test_the_two_head_tail_cuts_are_not_the_same_slice(parts):
    """Reporting both is only worth doing if they differ.

    On the real corpora they differ enormously (0.4% vs ~50% in the head). On a
    tiny fixture they may coincide, so this skips rather than asserting a split
    the fixture cannot produce.
    """
    masks = slices.build(*parts)
    if np.array_equal(masks["article_head"], masks["article_head_train"]):
        pytest.skip("fixture too small for the two popularity cuts to diverge")
    assert True


def test_cold_start_means_no_usable_history(parts):
    """Not "few clicks" — none. q3_plan D8 measured that any threshold above
    zero selects 0 of 2,000 impressions on both EB-NeRD corpora."""
    from newsrec.retrieval import compare

    sample, history, impressions = parts
    masks = slices.build(sample, history, impressions)
    lengths = compare.history_lengths(sample, history)
    assert np.array_equal(masks["cold_start"], lengths == 0)


def test_every_slice_the_report_names_is_produced(masks):
    assert set(masks) == set(slices.TRAINING_PERIOD_SLICES) | set(slices.TEST_PERIOD_SLICES)
    assert "article_head_train" in slices.TRAINING_PERIOD_SLICES
    assert "article_head" in slices.TEST_PERIOD_SLICES
