"""Q9.1 — what the serving-unavailable features would buy.

The failure guarded against is an ablation that quietly measures nothing: a leak
column that resolves to zeros for every article, a "paired" delta computed
against a different candidate set, or a dataset silently omitted because it has
no such columns. Each produces a report that looks complete and says nothing.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from newsrec import config
from newsrec.eval import ablations, metrics


def test_the_leak_column_is_one_of_the_forbidden_features():
    """It must come from config, not be a locally chosen 'interesting' column."""
    assert ablations.LEAK_COLUMN in config.SERVING_UNAVAILABLE_FEATURES


def test_the_blend_weight_is_not_tuned():
    """0.5 is an equal vote. A tuned weight would make this an attempt to win
    rather than a measurement of what the leak is worth."""
    assert ablations.LEAK_ALPHA == 0.5


def test_mind_reports_that_it_has_nothing_to_ablate(store_root):
    """MIND ships none of these columns.

    The report must say so rather than omit the dataset — an absent dataset
    reads as an oversight, an explicit 'not applicable' does not.
    """
    if not (store_root / "processed" / "mind_small").is_dir():
        pytest.skip("no mind_small store")
    got = ablations.evaluate_dataset("mind_small", store_root=store_root)
    assert got["ablation_possible"] is False
    assert "nothing to ablate" in got["note"]
    assert got["leak_feature"]["available"] is False


def test_a_missing_value_is_zero_and_the_article_is_kept(tmp_path):
    """Dropping an article would change the candidate count between variants,
    which is the one thing a paired comparison cannot survive."""
    frame = pl.DataFrame({"article_id": ["a", "b", "c"], ablations.LEAK_COLUMN: [10, None, 30]})
    d = tmp_path / "processed" / "x"
    d.mkdir(parents=True)
    frame.write_parquet(d / "articles_extra.parquet")

    values, coverage = ablations.leak_values("x", store_root=tmp_path)
    assert set(values) == {"a", "c"}, "the null must be absent from the mapping"
    assert coverage["articles_total"] == 3
    assert coverage["articles_with_value"] == 2
    assert coverage["null_share"] == pytest.approx(1 / 3, abs=1e-4)


def test_a_dataset_with_no_extra_table_is_handled_not_crashed(tmp_path):
    values, coverage = ablations.leak_values("absent", store_root=tmp_path)
    assert values == {}
    assert coverage["available"] is False


def test_a_dataset_whose_extra_table_lacks_the_column_is_handled(tmp_path):
    d = tmp_path / "processed" / "y"
    d.mkdir(parents=True)
    pl.DataFrame({"article_id": ["a"], "something_else": [1]}).write_parquet(
        d / "articles_extra.parquet"
    )
    values, coverage = ablations.leak_values("y", store_root=tmp_path)
    assert values == {}
    assert ablations.LEAK_COLUMN in coverage["reason"]


# =============================================================================
# The report
# =============================================================================


@pytest.fixture(scope="module")
def report() -> dict:
    path = config.METRICS_DIR / ablations.REPORT_NAME
    if not path.exists():
        pytest.skip("ablations.json not built; run `make ablations`")
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_headline_metric_has_a_paired_ablation(report):
    """The named Q9.1 condition.

    Not just AUC: every metric the design note might quote needs its pair, or
    the pairing is selective.
    """
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"]:
            continue
        for variant in ablations.VARIANTS:
            if variant == "headline_semantic":
                continue
            for metric in metrics.RANKING_METRICS:
                cell = block["variants"][variant][metric]
                assert "delta_vs_headline" in cell, f"{dataset}/{variant}/{metric}"
                for field in ("mean", "ci_low", "ci_high", "established"):
                    assert field in cell["delta_vs_headline"]


def test_the_deltas_are_consistent_with_the_variant_means(report):
    """A paired delta must equal the difference of the two means it pairs.

    Catches a delta computed against the wrong baseline — which would look
    entirely plausible in the report.
    """
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"]:
            continue
        base = block["variants"]["headline_semantic"]
        for variant in ablations.VARIANTS:
            if variant == "headline_semantic":
                continue
            for metric in metrics.RANKING_METRICS:
                cell = block["variants"][variant][metric]
                gap = cell["mean"] - base[metric]["mean"]
                # 2e-6, not 1e-6: the report rounds each mean and the delta to
                # six places independently, so the difference of two rounded
                # means can sit up to 2e-6 from the rounded delta. A delta
                # computed against the WRONG baseline — the failure this test
                # exists for — would be out by orders of magnitude more.
                assert cell["delta_vs_headline"]["mean"] == pytest.approx(gap, abs=2e-6), (
                    f"{dataset}/{variant}/{metric}"
                )


def test_every_variant_scored_the_same_impressions(report):
    """Different n between variants means they scored different lists, and the
    pairing is fictional."""
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"]:
            continue
        for metric in metrics.RANKING_METRICS:
            ns = {v: block["variants"][v][metric]["n"] for v in ablations.VARIANTS}
            assert len(set(ns.values())) == 1, f"{dataset}/{metric}: {ns}"


def test_the_leak_actually_resolved_to_values(report):
    """The quiet failure: a leak column that is null for everything measures
    nothing while producing a complete-looking report."""
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"]:
            continue
        cov = block["leak_feature"]
        assert cov["articles_with_value"] > 0, dataset
        assert cov["null_share"] < 1.0, dataset


def test_the_report_names_the_forbidden_set_from_config(report):
    assert set(report["serving_unavailable_features"]) == set(config.SERVING_UNAVAILABLE_FEATURES)
    assert report["leak_column"] == ablations.LEAK_COLUMN


def test_datasets_without_the_features_are_listed_not_dropped(report):
    listed = set(report["datasets_without_the_features"])
    actual = {d for d, b in report["datasets"].items() if not b["ablation_possible"]}
    assert listed == actual


def test_the_impression_set_matches_the_headline_reports(report):
    """The ablation must describe the same impressions as eval.json, or the
    'paired with the headline' claim is false."""
    other = config.METRICS_DIR / "eval.json"
    if not other.exists():
        pytest.skip("eval.json not built")
    committed = json.loads(other.read_text(encoding="utf-8"))["datasets"]
    for dataset, block in report["datasets"].items():
        if not block["ablation_possible"] or dataset not in committed:
            continue
        assert (
            block["impression_set_fingerprint"]
            == (committed[dataset]["impression_set_fingerprint"])
        ), dataset
