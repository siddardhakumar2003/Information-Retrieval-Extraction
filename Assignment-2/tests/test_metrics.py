"""Q4.1-Q4.3 — the four ranking metrics and the degenerate-impression rule.

Two oracles, deliberately. `sklearn` agreement proves we implemented *something
standard*; the hand-computed worked example proves we implemented *the thing we
meant*. A bug that is standard-but-wrong-for-us passes the first and fails the
second; a transcription slip fails the first.

The degenerate rule is tested on **manufactured** impressions, because the real
evaluated sample contains zero of them (0.0% on all three datasets). A rule that
never fires on real data is exactly the kind that rots unnoticed, so it gets
fixtures of its own rather than trust.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score, roc_auc_score

from newsrec import config
from newsrec.eval import metrics

# -----------------------------------------------------------------------------
# The hand-computed worked example (Q4.1)
#
# Five candidates. Scores descending give the order C B A E D, i.e. ranked
# labels [0, 1, 1, 0, 1] — a miss, then two hits, then a miss, then a hit.
# -----------------------------------------------------------------------------
WORKED_LABELS = [1, 1, 0, 1, 0]
WORKED_SCORES = [0.5, 0.8, 0.9, 0.2, 0.3]


def test_the_worked_example_auc_matches_the_hand_computation():
    # Ranked best-first: C(0) B(1) A(1) E(0) D(1). Positives sit at ranks 2, 3
    # and 5; negatives at 1 and 4. Of the 3x2 = 6 (positive, negative) pairs the
    # positive wins 2 — B and A each beat the negative at rank 4, and neither
    # beats the negative at rank 1, while the positive at rank 5 beats neither.
    assert metrics.auc(WORKED_LABELS, WORKED_SCORES) == pytest.approx(2 / 6)


def test_the_worked_example_mrr_is_the_first_hit_only():
    # First clicked candidate is at rank 2. Later hits do not contribute.
    assert metrics.mrr(WORKED_LABELS, WORKED_SCORES) == pytest.approx(1 / 2)


def test_the_worked_example_ndcg_at_5_matches_the_hand_computation():
    dcg = 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert metrics.ndcg_at_k(WORKED_LABELS, WORKED_SCORES, 5) == pytest.approx(dcg / idcg)


def test_ndcg_at_5_truncates_and_differs_from_ndcg_at_10_when_a_hit_falls_outside():
    # Six candidates, the only click ranked last. nDCG@5 cannot see it; nDCG@10
    # can. If the cutoff were ignored the two would be equal.
    labels = [0, 0, 0, 0, 0, 1]
    scores = [6, 5, 4, 3, 2, 1]
    assert metrics.ndcg_at_k(labels, scores, 5) == 0.0
    assert metrics.ndcg_at_k(labels, scores, 10) == pytest.approx(1 / math.log2(7))


def test_the_ideal_dcg_uses_this_impressions_click_count_so_a_perfect_ranking_scores_one():
    # Three clicks ranked first, second and third. A fixed single-click ideal
    # would make this 1.79, i.e. "better than perfect".
    labels = [1, 1, 1, 0, 0]
    scores = [9, 8, 7, 1, 0]
    assert metrics.ndcg_at_k(labels, scores, 10) == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# The sklearn oracle (Q4.2)
# -----------------------------------------------------------------------------


def _random_impression(rng, n_candidates=None):
    n = int(rng.integers(2, 40)) if n_candidates is None else n_candidates
    labels = rng.integers(0, 2, size=n)
    if labels.sum() in (0, n):  # keep it non-degenerate; that rule is tested below
        labels[0], labels[-1] = 1, 0
    return labels, rng.normal(size=n)


def test_auc_agrees_with_sklearn_on_random_impressions():
    rng = np.random.default_rng(config.GLOBAL_SEED)
    for _ in range(200):
        labels, scores = _random_impression(rng)
        assert metrics.auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_auc_agrees_with_sklearn_when_scores_tie():
    """Ties are where a rank-based AUC most easily goes wrong.

    Scores are drawn from a three-value alphabet so tied groups are the norm
    rather than a rarity. Average-rank handling is what makes this pass;
    ordinal ranks would drift by a few percent and still look plausible.
    """
    rng = np.random.default_rng(config.GLOBAL_SEED + 1)
    for _ in range(200):
        n = int(rng.integers(4, 30))
        labels = rng.integers(0, 2, size=n)
        if labels.sum() in (0, n):
            labels[0], labels[-1] = 1, 0
        scores = rng.integers(0, 3, size=n).astype(float)
        assert metrics.auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_an_all_tied_impression_scores_exactly_one_half():
    # No information in the ranking, so no credit. An implementation that broke
    # ties by position would score this well above 0.5 on a lucky label order.
    labels = [1, 0, 0, 1, 0, 0, 0, 0]
    assert metrics.auc(labels, [0.0] * len(labels)) == 0.5


def test_ndcg_agrees_with_sklearn_on_random_impressions():
    rng = np.random.default_rng(config.GLOBAL_SEED + 2)
    for _ in range(200):
        labels, scores = _random_impression(rng)
        for k in config.NDCG_K_VALUES:
            expected = ndcg_score([labels], [scores], k=k)
            assert metrics.ndcg_at_k(labels, scores, k) == pytest.approx(expected)


def test_mrr_matches_a_direct_search_of_the_ranked_labels():
    rng = np.random.default_rng(config.GLOBAL_SEED + 3)
    for _ in range(200):
        labels, scores = _random_impression(rng)
        ranked = np.asarray(labels)[np.argsort(-np.asarray(scores), kind="stable")]
        expected = 1.0 / (int(np.flatnonzero(ranked == 1)[0]) + 1)
        assert metrics.mrr(labels, scores) == pytest.approx(expected)


# -----------------------------------------------------------------------------
# Ties break deterministically (Q1.13)
# -----------------------------------------------------------------------------


def test_ties_break_by_candidate_position_not_by_reversed_order():
    """`argsort(-s)` and `argsort(s)[::-1]` differ exactly here.

    The click is the first of three tied candidates. Breaking ties by list
    position gives rank 1; reversing an ascending sort gives rank 3. Both are
    deterministic, so a determinism test alone would not catch the swap.
    """
    labels = [1, 0, 0]
    scores = [0.0, 0.0, 0.0]
    assert metrics.mrr(labels, scores) == 1.0


def test_the_metrics_are_identical_across_repeated_calls_on_tied_scores():
    labels = [0, 1, 0, 1, 0, 0, 1]
    scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    first = metrics.ranking_metrics(labels, scores)
    for _ in range(5):
        assert metrics.ranking_metrics(labels, scores) == first


# -----------------------------------------------------------------------------
# The degenerate rule (Q4.3)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "reason"),
    [
        ([1], "single_candidate"),
        ([0], "single_candidate"),
        ([], "single_candidate"),
        ([1, 1, 1], "all_positive"),
        ([0, 0, 0, 0], "all_negative"),
    ],
)
def test_degenerate_impressions_are_named(labels, reason):
    assert metrics.degenerate_reason(labels) == reason


@pytest.mark.parametrize("labels", [[1, 0], [0, 1, 1], [1, 0, 0, 1]])
def test_ordinary_impressions_are_not_degenerate(labels):
    assert metrics.degenerate_reason(labels) is None


@pytest.mark.parametrize("labels", [[1], [1, 1, 1], [0, 0, 0]])
def test_ranking_metrics_returns_none_for_a_degenerate_impression(labels):
    """`None`, never 0.5 and never a dict of NaNs.

    An undefined value replaced by a plausible one moves the mean while looking
    fine — the exact failure `eval_protocol.md` names.
    """
    assert metrics.ranking_metrics(labels, [1.0] * len(labels)) is None


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0]])
def test_auc_raises_rather_than_inventing_a_number_for_a_degenerate_impression(labels):
    with pytest.raises(ValueError, match="undefined"):
        metrics.auc(labels, [1.0, 2.0, 3.0])


def test_a_single_candidate_impression_would_have_scored_one_had_it_been_included():
    """Why the exclusion is stricter than the protocol's AUC-only letter.

    MRR and nDCG are 1.0 here whatever the retriever does. Averaging them in
    would drag every mean toward 1.0 in proportion to how many trivial
    impressions the sample happened to contain — so they are excluded too, and
    this test pins the reason rather than leaving it in a docstring.
    """
    assert metrics.mrr([1], [0.0]) == 1.0
    assert metrics.ndcg_at_k([1], [0.0], 10) == 1.0
    assert metrics.ranking_metrics([1], [0.0]) is None


def test_ranking_metrics_reports_every_metric_the_report_expects():
    got = metrics.ranking_metrics(WORKED_LABELS, WORKED_SCORES)
    assert set(got) == set(metrics.RANKING_METRICS)
    assert set(metrics.RANKING_METRICS) == {"auc", "mrr", "ndcg@5", "ndcg@10"}


def test_the_ndcg_cutoffs_come_from_config_not_from_a_local_literal():
    """Changing `config.NDCG_K_VALUES` must change what the harness reports.

    A hard-coded (5, 10) inside `ranking_metrics` would pass every other test in
    this file and silently ignore the configured value.
    """
    got = metrics.ranking_metrics(WORKED_LABELS, WORKED_SCORES, k_values=(3,))
    assert set(got) == {"auc", "mrr", "ndcg@3"}
