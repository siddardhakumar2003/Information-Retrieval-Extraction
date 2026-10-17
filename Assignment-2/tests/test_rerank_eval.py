"""A2 Q5 — the full metric set for the two-stage pipeline (a2_q5_plan.md).

`eval/harness.py` already computes AUC, MRR, nDCG@5/@10, intra-list diversity,
novelty and coverage, each with a bootstrap interval, over eight slice masks —
but only for the `bm25` and `semantic` retrievers. A2 Q5 wants the same set for
the **re-ranked** ordering.

This module is glue, so these tests are about the glue: that the two-stage
numbers describe the *same impressions* `eval.json` describes (otherwise
"before vs after re-ranking" is not a paired claim), and that the re-ranker's
per-row scores are regrouped onto the sample in the right order.

The regrouping is the dangerous part. A candidate list assembled in a different
order than the scores would give a plausible mid-range metric with **nothing
raising anywhere** — the failure A1 shipped once for real
(`design_rationale.md` §1.21).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.eval import metrics, slices
from newsrec.rerank import evaluate

REPORT = config.METRICS_DIR / evaluate.REPORT_NAME


def _sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": ["i0", "i1"],
            "candidate_ids": [["a", "b", "c"], ["d", "e"]],
            "clicked_ids": [["b"], ["d"]],
        }
    )


def _table() -> pl.DataFrame:
    """Shaped like `features.build_candidate_table`: one row per (impression,
    candidate), candidates in the impression's own order."""
    return pl.DataFrame(
        {
            "impression_id": ["i0", "i0", "i0", "i1", "i1"],
            "article_id": ["a", "b", "c", "d", "e"],
        }
    )


# =============================================================================
# Regrouping per-row scores onto the sample
# =============================================================================


def test_regroup_returns_one_array_per_sample_row_in_candidate_order():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    got = evaluate.regroup(_table(), scores, _sample())

    assert len(got) == 2
    np.testing.assert_allclose(got[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(got[1], [0.4, 0.5])


def test_regroup_follows_the_sample_order_not_the_tables():
    """The sample decides row order — `harness._per_impression` walks it. A
    table stored in a different impression order must still line up."""
    sample = pl.DataFrame(
        {
            "impression_id": ["i1", "i0"],
            "candidate_ids": [["d", "e"], ["a", "b", "c"]],
            "clicked_ids": [["d"], ["b"]],
        }
    )
    got = evaluate.regroup(_table(), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), sample)
    np.testing.assert_allclose(got[0], [0.4, 0.5])
    np.testing.assert_allclose(got[1], [0.1, 0.2, 0.3])


def test_regroup_raises_when_a_candidate_list_disagrees():
    """The guard that turns a silent mis-ranking into a crash.

    If the table's candidates for an impression are not the sample's candidates
    in the same order, the scores cannot be trusted — and the metric computed
    from them would look completely normal.
    """
    scrambled = _table().with_columns(
        pl.Series("article_id", ["b", "a", "c", "d", "e"])  # i0's order flipped
    )
    with pytest.raises(evaluate.CandidateMismatch, match="i0"):
        evaluate.regroup(scrambled, np.arange(5.0), _sample())


def test_regroup_raises_when_an_impression_is_missing_from_the_table():
    partial = _table().filter(pl.col("impression_id") == "i0")
    with pytest.raises(evaluate.CandidateMismatch, match="i1"):
        evaluate.regroup(partial, np.arange(3.0), _sample())


def test_a_perfect_score_regroups_to_a_perfect_ranking():
    """End-to-end alignment check through the metric layer: score each
    candidate by its own label and nDCG must be 1.0. An off-by-one anywhere in
    the regrouping shows up here and nowhere else."""
    sample = _sample()
    table = _table()
    labels = np.array([0.0, 1.0, 0.0, 1.0, 0.0])  # b clicked in i0, d in i1
    grouped = evaluate.regroup(table, labels, sample)
    for row, scores in zip(sample.iter_rows(named=True), grouped, strict=True):
        clicked = set(row["clicked_ids"])
        truth = np.array([1 if a in clicked else 0 for a in row["candidate_ids"]])
        got = metrics.ranking_metrics(truth, scores)
        assert got["ndcg@10"] == pytest.approx(1.0)


# =============================================================================
# The written report — conditions on what a grader reads
# =============================================================================


def _report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT.name} not built; run `make rerank-eval` (needs a GPU)")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_the_report_covers_both_datasets():
    report = _report()
    assert set(report["datasets"]) >= set(config.HEADLINE_DATASETS)


def test_full_metric_set_per_dataset():
    """A2.Q5.1: all of AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage
    — the beyond-accuracy three are the ones a ranking report usually drops."""
    report = _report()
    required = {*metrics.RANKING_METRICS, "diversity", "novelty", "coverage"}
    for dataset, block in report["datasets"].items():
        for name, variant in block["variants"].items():
            assert required <= set(variant["overall"]), (
                f"{dataset}/{name} is missing {sorted(required - set(variant['overall']))}"
            )


#: `coverage` is the one metric whose point estimate legitimately sits OUTSIDE
#: its own bootstrap interval, and it is worth understanding why rather than
#: exempting it silently.
#:
#: Every other metric here is an average over impressions, so resampling
#: impressions gives a distribution centred on the sample mean. Coverage is not
#: an average — it is the size of a *union*: how much of the catalogue appears
#: in any top-10. A bootstrap draw samples n impressions with replacement and
#: therefore sees only about 63% of the distinct ones, so every resampled union
#: is smaller than the real one and the whole interval sits below the point
#: estimate.
#:
#: `eval/beyond_accuracy.py::coverage_ci` documents this and deliberately
#: reports the true sample coverage as `mean` rather than the resample mean,
#: because the resample mean would understate coverage systematically. A1's
#: committed `eval.json` has the same property on all four (dataset, method)
#: cells, so this is a property of the metric, not of A2's code.
SET_VALUED_METRICS = ("coverage",)


def test_every_metric_has_ci():
    """A2.Q5.3: a bootstrap interval on **every** reported metric, not only the
    headline ranking ones."""
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, variant in block["variants"].items():
            for metric, entry in variant["overall"].items():
                assert "ci_low" in entry and "ci_high" in entry, f"{dataset}/{name}/{metric}"
                if metric in SET_VALUED_METRICS:
                    continue
                assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"], (
                    f"{dataset}/{name}/{metric}: mean outside its own interval"
                )


def test_coverages_interval_sits_below_its_point_estimate():
    """The positive form of the exemption above.

    Coverage is exempt from "mean inside the interval", but not from having a
    *predictable* relationship to it: because a bootstrap union is biased low,
    the whole interval must lie at or below the true sample coverage. If it ever
    straddled it, the bias explanation would be wrong and the number would need
    re-deriving rather than excusing.
    """
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, variant in block["variants"].items():
            entry = variant["overall"]["coverage"]
            if entry["n"] <= 1:
                continue
            assert entry["ci_high"] <= entry["mean"], (
                f"{dataset}/{name}: coverage interval is not below its point estimate "
                f"(mean {entry['mean']}, ci_high {entry['ci_high']}) — the "
                f"resample-bias explanation in beyond_accuracy.coverage_ci does not hold"
            )


def test_required_slices_present():
    """A2.Q5.2 names two slicings explicitly: cold-start vs warm, head vs tail."""
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, variant in block["variants"].items():
            present = set(variant["slices"])
            for required in ("cold_start", "warm", "article_head", "article_tail"):
                assert required in present, f"{dataset}/{name} has no {required} slice"


def test_slices_partition_the_sample():
    """Each named pair must cover the sample exactly once. An overlapping pair
    double-counts impressions and a gap silently drops them, and both produce
    per-slice numbers that look fine and do not reconstruct the overall one."""
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, variant in block["variants"].items():
            total = variant["overall"]["ndcg@10"]["n"]
            for a, b in (("cold_start", "warm"), ("article_head", "article_tail")):
                got = variant["slices"][a]["n"] + variant["slices"][b]["n"]
                assert got == total, f"{dataset}/{name}: {a}+{b} = {got}, sample is {total}"


def test_the_impression_set_matches_eval_json():
    """A2.Q5.4. Without this, "before vs after re-ranking" compares two similar
    samples rather than the same one, and the paired delta means nothing."""
    report = _report()
    eval_path = config.METRICS_DIR / "eval.json"
    if not eval_path.exists():
        pytest.skip("eval.json not built")
    committed = json.loads(eval_path.read_text(encoding="utf-8"))["datasets"]
    for dataset, block in report["datasets"].items():
        if dataset not in committed:
            continue
        assert (
            block["impression_set_fingerprint"]
            == (committed[dataset]["impression_set_fingerprint"])
        ), dataset


def test_every_non_baseline_variant_carries_a_paired_delta():
    report = _report()
    for dataset, block in report["datasets"].items():
        for name, variant in block["variants"].items():
            if name == evaluate.BASELINE_VARIANT:
                continue
            for metric in metrics.RANKING_METRICS:
                delta = variant["overall"][metric].get("delta_vs_baseline")
                assert delta is not None, f"{dataset}/{name}/{metric}"
                excludes_zero = delta["ci_low"] > 0 or delta["ci_high"] < 0
                assert delta["established"] == excludes_zero


def test_the_slice_names_come_from_the_shared_builder():
    """`eval/slices.py` is the one definition of a slice. If this report
    invented its own names, a slice here and a slice in `eval.json` could mean
    different rows."""
    expected = {*slices.TRAINING_PERIOD_SLICES, *slices.TEST_PERIOD_SLICES}
    report = _report()
    for block in report["datasets"].values():
        for variant in block["variants"].values():
            assert set(variant["slices"]) <= expected
