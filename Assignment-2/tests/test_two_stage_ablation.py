"""A2 Q9.1 — the two-stage with/without serving-unavailable ablation.

The claim this file protects is narrow and important: the columns the ablation
adds really are the forbidden ones, they only ever appear in the *ablation*
variant, and the four it leaves out are left out for a stated structural
reason rather than by oversight.

An ablation that quietly tested a safe column, or a headline number that had
absorbed a forbidden one, would both look completely normal.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.eval import metrics
from newsrec.rerank import features, unavailable

REPORT = config.METRICS_DIR / unavailable.REPORT_NAME


# =============================================================================
# The columns really are the forbidden ones
# =============================================================================


def test_every_added_column_is_a_serving_unavailable_feature():
    """The whole point. An ablation that added a permitted column would report
    a delta that means nothing."""
    for column in unavailable.ABLATION_FEATURE_COLUMNS:
        assert column in config.SERVING_UNAVAILABLE_FEATURES, column


def test_every_excluded_column_is_also_a_serving_unavailable_feature():
    for column in unavailable.CONSTANT_WITHIN_IMPRESSION:
        assert column in config.SERVING_UNAVAILABLE_FEATURES, column


def test_the_added_and_excluded_columns_together_cover_the_whole_forbidden_set():
    """No forbidden column may be silently unaccounted for. Each is either
    tested or explicitly explained."""
    covered = set(unavailable.ABLATION_FEATURE_COLUMNS) | set(
        unavailable.CONSTANT_WITHIN_IMPRESSION
    )
    missing = config.SERVING_UNAVAILABLE_FEATURES - covered
    assert not missing, (
        f"{sorted(missing)} are unavailable at serving time but this ablation "
        f"neither tests them nor explains why not"
    )


def test_no_added_column_leaks_into_the_headline_feature_list():
    """The headline model must never see one. `tests/test_rerank_leakage.py`
    asserts this too; repeated here because this module is the one that
    deliberately handles them."""
    leaked = set(features.FEATURE_COLUMNS) & set(unavailable.ABLATION_FEATURE_COLUMNS)
    assert not leaked, sorted(leaked)


def test_the_two_groups_do_not_overlap():
    assert not set(unavailable.ABLATION_FEATURE_COLUMNS) & set(
        unavailable.CONSTANT_WITHIN_IMPRESSION
    )


# =============================================================================
# The structural argument for excluding four of the seven
# =============================================================================


def test_a_feature_constant_within_an_impression_cannot_change_its_ranking():
    """The reason `read_time` and friends are excluded, demonstrated rather
    than asserted.

    A re-ranker orders candidates *within* one impression. Adding the same
    number to every candidate's score in an impression leaves the order
    untouched, and AUC, MRR and nDCG are all computed inside an impression. So
    a per-impression feature is structurally incapable of inflating them —
    which is a stronger statement than "we did not test it".
    """
    table = pl.DataFrame(
        {
            "impression_id": ["i0"] * 5 + ["i1"] * 5,
            "article_id": [f"a{j}" for j in range(10)],
            "label": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "popularity_train": [0] * 10,
        }
    )
    scores = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5])
    # A per-impression feature shifts every candidate of an impression equally.
    shifted = scores + np.array([7.0] * 5 + [-3.0] * 5)

    before = unavailable._ranked_metrics(table, scores)
    after = unavailable._ranked_metrics(table, shifted)
    for metric in metrics.RANKING_METRICS:
        np.testing.assert_allclose(before[metric], after[metric])


# =============================================================================
# The join
# =============================================================================


def _table() -> pl.DataFrame:
    data = {
        "impression_id": ["i0", "i0", "i1"],
        "article_id": ["a1", "a2", "a3"],
        "label": [1, 0, 1],
    }
    for name in features.FEATURE_COLUMNS:
        data[name] = [0.0, 0.0, 0.0]
    return pl.DataFrame(data)


def test_a_dataset_shipping_none_of_the_columns_adds_nothing(tmp_path):
    table = _table()
    joined, columns = unavailable.with_unavailable(table, "mind_small", store_root=tmp_path)
    assert columns == ()
    assert joined.columns == table.columns


@pytest.mark.realdata
def test_mind_ships_none_of_the_added_columns():
    """Not a defect — MIND simply has no such column, and the report must say
    so explicitly rather than omitting the dataset."""
    assert unavailable.available_columns("mind_small") == ()


@pytest.mark.realdata
def test_ebnerd_ships_all_three_added_columns():
    assert set(unavailable.available_columns("ebnerd_small")) == set(
        unavailable.ABLATION_FEATURE_COLUMNS
    )


# =============================================================================
# The written report
# =============================================================================


def _report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT.name} not built; run `make two-stage-ablation` (needs a GPU)")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_the_report_covers_both_datasets():
    report = _report()
    assert set(report["datasets"]) >= set(config.HEADLINE_DATASETS)


def test_a_dataset_without_the_columns_says_so_explicitly():
    """`ablation_possible: false` plus a note, not a missing key. An absent
    dataset reads as an oversight."""
    report = _report()
    for dataset, block in report["datasets"].items():
        assert "ablation_possible" in block, dataset
        if not block["ablation_possible"]:
            assert block.get("note"), f"{dataset} is not applicable but gives no reason"


def test_the_ablation_variant_carries_a_paired_delta_against_the_headline():
    report = _report()
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"]:
            continue
        variant = block["variants"][unavailable.ABLATION_VARIANT]
        for metric in metrics.RANKING_METRICS:
            delta = variant[metric].get("delta_vs_headline")
            assert delta is not None, f"{dataset}/{metric}"
            excludes_zero = delta["ci_low"] > 0 or delta["ci_high"] < 0
            assert delta["established"] == excludes_zero, f"{dataset}/{metric}"


def test_the_headline_variant_carries_no_delta():
    """It is the baseline of this comparison; a delta against itself would be
    zero and would invite reading it as a result."""
    report = _report()
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"]:
            continue
        variant = block["variants"][unavailable.HEADLINE_VARIANT]
        for metric in metrics.RANKING_METRICS:
            assert "delta_vs_headline" not in variant[metric], f"{dataset}/{metric}"


def test_the_report_explains_which_columns_it_left_out_and_why():
    report = _report()
    block = report["excluded_because_constant_within_an_impression"]
    assert set(block["columns"]) == set(unavailable.CONSTANT_WITHIN_IMPRESSION)
    assert "within an impression" in block["why"]


def test_the_report_names_every_serving_unavailable_feature():
    report = _report()
    assert set(report["serving_unavailable_features"]) == config.SERVING_UNAVAILABLE_FEATURES
