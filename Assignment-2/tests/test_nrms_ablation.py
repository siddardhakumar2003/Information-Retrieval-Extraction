"""A2 Q3.3/Q3.4/Q3.7 — the ablation's protocol, not the model's quality.

These tests do not care whether NRMS wins. They care that the comparison is
**able to mean something**: that the model was fitted on train rows only, that
the three tables are disjoint, that the delta which A2.Q3.4 grades is actually
present and paired, and that no gain is claimed without an interval excluding
zero.

A model can leak by training on the wrong *rows* even when every individual
feature value is itself leakage-safe, which is why the split-membership checks
live here rather than in `tests/test_nrms.py`.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.eval import metrics
from newsrec.rerank import nrms_ablation

REPORT = config.METRICS_DIR / nrms_ablation.REPORT_NAME


# =============================================================================
# The delta machinery, on hand-built numbers
# =============================================================================


def test_a_delta_that_straddles_zero_is_not_established():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(500)
    base = rng.standard_normal(500)
    delta = nrms_ablation._delta(values, base)
    assert delta["ci_low"] < 0 < delta["ci_high"]
    assert delta["established"] is False


def test_a_clear_improvement_is_established_and_positive():
    base = np.full(500, 0.30)
    values = base + 0.05
    delta = nrms_ablation._delta(values, base)
    assert delta["mean"] == pytest.approx(0.05, abs=1e-6)
    assert delta["established"] is True
    assert delta["ci_low"] > 0


def test_a_clear_regression_is_established_and_negative():
    """ "Established" must not mean "better". A2 Q2's result was an established
    LOSS on every dataset, and the report has to be able to say so."""
    base = np.full(500, 0.30)
    values = base - 0.05
    delta = nrms_ablation._delta(values, base)
    assert delta["established"] is True
    assert delta["ci_high"] < 0


def test_the_delta_is_paired_not_a_difference_of_two_independent_intervals():
    """Two methods that track each other closely have a tight delta interval
    even when each method's own interval is wide. An unpaired subtraction would
    lose that, and would report "not established" for a real effect.
    """
    rng = np.random.default_rng(1)
    base = rng.standard_normal(500) * 2.0  # wide spread across impressions
    values = base + 0.10  # ... but a constant, small improvement on each

    paired = nrms_ablation._delta(values, base)
    assert paired["established"] is True

    own_interval_width = (
        nrms_ablation._summary(base)["ci_high"] - nrms_ablation._summary(base)["ci_low"]
    )
    paired_width = paired["ci_high"] - paired["ci_low"]
    assert paired_width < own_interval_width / 10


# =============================================================================
# Score/row alignment through the metric layer
# =============================================================================


def _scored_table(n_impressions: int = 4, n_candidates: int = 5) -> pl.DataFrame:
    rows = {
        "impression_id": [],
        "article_id": [],
        "label": [],
        "popularity_train": [],
    }
    for i in range(n_impressions):
        for j in range(n_candidates):
            rows["impression_id"].append(f"i{i}")
            rows["article_id"].append(f"a{i}_{j}")
            rows["label"].append(1 if j == 0 else 0)
            rows["popularity_train"].append(0)
    return pl.DataFrame(rows)


def test_ranked_metrics_returns_one_value_per_impression():
    table = _scored_table()
    got = nrms_ablation._ranked_metrics(table, np.zeros(table.height))
    for name in metrics.RANKING_METRICS:
        assert len(got[name]) == 4


def test_scores_align_with_the_candidate_order_of_their_impression():
    """The bug with no error message.

    A perfect scorer — the label itself — must produce a perfect nDCG. If the
    score vector were offset or grouped differently from the table's rows, this
    would come out mid-range and look like a plausible model, with nothing
    raising anywhere. `submit/score_fast.py`'s docstring warns about the same
    class of failure and A1 shipped it once for real
    (`design_rationale.md` §1.21).
    """
    table = _scored_table()
    perfect = table["label"].to_numpy().astype(np.float64)
    got = nrms_ablation._ranked_metrics(table, perfect)
    assert all(v == pytest.approx(1.0) for v in got["ndcg@10"])
    assert all(v == pytest.approx(1.0) for v in got["mrr"])


def test_an_inverted_score_is_measurably_worse_than_a_perfect_one():
    """Guards against a metric that returns 1.0 for anything."""
    table = _scored_table()
    labels = table["label"].to_numpy().astype(np.float64)
    perfect = nrms_ablation._ranked_metrics(table, labels)
    inverted = nrms_ablation._ranked_metrics(table, -labels)
    assert np.mean(inverted["ndcg@10"]) < np.mean(perfect["ndcg@10"])


def test_a_degenerate_impression_is_skipped_not_scored_as_zero():
    """An impression where every candidate was clicked has no ranking to get
    right; scoring it as 0 would drag the mean down for a non-reason. The same
    rule `eval/metrics.py` applies everywhere else."""
    table = pl.DataFrame(
        {
            "impression_id": ["i0"] * 3,
            "article_id": ["a", "b", "c"],
            "label": [1, 1, 1],
            "popularity_train": [0, 0, 0],
        }
    )
    got = nrms_ablation._ranked_metrics(table, np.array([1.0, 2.0, 3.0]))
    assert got["ndcg@10"] == []


# =============================================================================
# The written report — conditions on the artefact a grader reads
# =============================================================================


def _report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT.name} not built; run `make nrms` (needs a GPU)")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_the_report_covers_both_datasets():
    report = _report()
    assert set(report["datasets"]) >= set(config.HEADLINE_DATASETS), (
        "every A2 question requires BOTH datasets; a MIND-only result is incomplete"
    )


def test_every_dataset_carries_all_three_variants():
    report = _report()
    for dataset, block in report["datasets"].items():
        expected = {nrms_ablation.BASELINE_VARIANT, *config.NRMS_VARIANTS}
        assert set(block["variants"]) == expected, dataset


def test_every_variant_reports_all_four_ranking_metrics_with_intervals():
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, metric_block in block["variants"].items():
            assert set(metric_block) == set(metrics.RANKING_METRICS), f"{dataset}/{name}"
            for metric, entry in metric_block.items():
                assert "ci_low" in entry and "ci_high" in entry, f"{dataset}/{name}/{metric}"
                assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]


def test_the_graded_comparison_is_present_on_every_dataset():
    """A2.Q3.3/Q3.4: the ablation has to isolate the ONE change, which means a
    delta against the reproduction, not only against retrieval-only."""
    report = _report()
    for dataset, block in report["datasets"].items():
        variant = block["variants"]["nrms_category_aware"]
        for metric in metrics.RANKING_METRICS:
            delta = variant[metric].get("delta_vs_reproduction")
            assert delta is not None, f"{dataset}/{metric} has no delta vs the reproduction"
            assert {"mean", "ci_low", "ci_high", "established"} <= set(delta)


def test_every_claimed_gain_has_an_interval_excluding_zero():
    """A2.Q3.4, stated as a condition on the artefact.

    Any delta reported as `established` must genuinely exclude zero, in the
    direction its mean points. This does not require the change to have helped
    — a negative established delta is a fine and reportable result. It requires
    that `established` never disagrees with the interval beside it.
    """
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, metric_block in block["variants"].items():
            for metric, entry in metric_block.items():
                for key in ("delta_vs_baseline", "delta_vs_reproduction"):
                    delta = entry.get(key)
                    if delta is None:
                        continue
                    excludes_zero = delta["ci_low"] > 0 or delta["ci_high"] < 0
                    assert delta["established"] == excludes_zero, (
                        f"{dataset}/{name}/{metric}/{key}: established="
                        f"{delta['established']} but interval is "
                        f"[{delta['ci_low']}, {delta['ci_high']}]"
                    )


def test_the_report_declares_its_deviations_from_published_nrms():
    """A2.Q3.1 is claimed as "reproduced with the deviations declared". If the
    deviations are not in the artefact, the claim is not supported by it."""
    report = _report()
    deviations = report.get("declared_deviations_from_published_nrms")
    assert deviations, "no deviations declared"
    joined = " ".join(deviations).lower()
    assert "news encoder" in joined, "the substituted news encoder is not declared"
    assert "oq-12" in joined, "the unread reference configuration is not disclosed"


def test_the_negative_sampling_shortfall_is_reported():
    """`a2_q2_insights.md` names short EB-NeRD candidate lists as an unverified
    hypothesis for GBDT's failure. This report is where it becomes a number."""
    report = _report()
    for dataset, block in report["datasets"].items():
        sampling = block["negative_sampling"]
        assert sampling["n_negatives"] == config.NRMS_NEG_SAMPLES
        assert 0.0 <= sampling["share"] <= 1.0, dataset


def test_the_baseline_is_the_same_column_the_q2_ablation_used():
    """Q2's "before re-ranking" and Q3's must be the same number, not two
    similar ones, or the two reports cannot be read side by side."""
    from newsrec.rerank import ablation as q2_ablation

    assert nrms_ablation.BASELINE_SCORE_COLUMN == q2_ablation.BASELINE_SCORE_COLUMN


def test_the_test_impression_count_matches_the_q2_ablation():
    """Both stages sample the test split with `config.RECALL_SAMPLE_SIZE` and
    `config.GLOBAL_SEED`, so they describe the same impressions."""
    report = _report()
    q2_path = config.METRICS_DIR / "reranker_ablation.json"
    if not q2_path.exists():
        pytest.skip("reranker_ablation.json not built")
    q2 = json.loads(q2_path.read_text(encoding="utf-8"))
    for dataset, block in report["datasets"].items():
        if dataset not in q2["datasets"]:
            continue
        assert block["test_impressions"] == q2["datasets"][dataset]["test_impressions"], dataset
