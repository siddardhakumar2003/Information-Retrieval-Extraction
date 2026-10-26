import numpy as np
import pytest

from annbench.recall import mean_recall_at_k, recall_at_k


def test_perfect_match():
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert recall_at_k(gt, result, k=5) == pytest.approx(1.0)


def test_zero_overlap():
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    assert recall_at_k(gt, result, k=5) == pytest.approx(0.0)


def test_partial_overlap():
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # 3 of the 5 returned distances are within the gt's k-th (5.0) threshold
    result = np.array([1.0, 2.0, 3.0, 999.0, 999.0])
    assert recall_at_k(gt, result, k=5) == pytest.approx(3 / 5)


def test_exact_tie_at_kth_position_counts_as_correct():
    # True 10th-nearest distance is 5.0; an alternate point at exactly the
    # same distance (a legitimate tie) must NOT be penalised.
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # The algorithm returns a different equidistant point tied at the boundary.
    result = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    assert recall_at_k(gt, result, k=10) == pytest.approx(1.0)


def test_near_tie_within_floating_point_epsilon():
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Distance recomputed with a tiny float error, just over the raw threshold.
    result = np.array([1.0, 2.0, 3.0, 4.0, 5.0 + 1e-6])
    assert recall_at_k(gt, result, k=5) == pytest.approx(1.0)


def test_extra_results_beyond_threshold_do_not_inflate_recall():
    gt = np.array([1.0, 2.0, 3.0])
    # Only k=3 slots are counted, even if more candidates were returned.
    result = np.array([1.0, 2.0, 3.0, 3.0, 3.0])
    assert recall_at_k(gt, result, k=3) == pytest.approx(1.0)


def test_rejects_too_few_ground_truth_distances():
    gt = np.array([1.0, 2.0])
    result = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        recall_at_k(gt, result, k=5)


def test_mean_recall_at_k_averages_over_queries():
    gt = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = np.array(
        [
            [1.0, 2.0, 3.0],  # recall 1.0
            [999.0, 999.0, 999.0],  # recall 0.0
        ]
    )
    assert mean_recall_at_k(gt, result, k=3) == pytest.approx(0.5)
