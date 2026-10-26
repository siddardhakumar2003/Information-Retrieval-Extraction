"""Our own recall@k implementation (the assignment asks us not to rely on a
library's), with explicit handling of ties at the k-th neighbour's distance.

Why ties matter: if several points are equidistant from the query at the k-th
position, there can be more than one valid set of "the k nearest neighbours".
Comparing retrieved ids against one fixed ground-truth id list would wrongly
penalise a correct answer that happens to return a different (but equally
close) point. Instead we compare *distances* against the k-th true distance:
any returned point at or within that distance counts as correct.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-4


def recall_at_k(
    gt_distances: np.ndarray,
    result_distances: np.ndarray,
    k: int = 10,
    eps: float = DEFAULT_EPS,
) -> float:
    """Recall@k for a single query.

    gt_distances: 1-D array of ground-truth distances, sorted ascending, length >= k.
    result_distances: 1-D array of the distances returned by the index being evaluated,
        length == the number of results actually returned (<= k is treated as a miss
        for the missing slots).
    eps: relative tolerance on the k-th-distance threshold, to absorb floating-point
        noise between an index's distance computation and the ground truth's.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if gt_distances.shape[0] < k:
        raise ValueError(f"need at least {k} ground-truth distances, got {gt_distances.shape[0]}")

    threshold = gt_distances[k - 1] * (1.0 + eps)
    hits = int(np.sum(result_distances[:k] <= threshold))
    return min(hits, k) / k


def mean_recall_at_k(
    gt_distances: np.ndarray,
    result_distances: np.ndarray,
    k: int = 10,
    eps: float = DEFAULT_EPS,
) -> float:
    """Recall@k averaged over a batch of queries.

    gt_distances, result_distances: 2-D arrays, shape (n_queries, >=k).
    """
    n = gt_distances.shape[0]
    if result_distances.shape[0] != n:
        raise ValueError("gt_distances and result_distances must have the same number of queries")

    recalls = np.empty(n, dtype=np.float64)
    for i in range(n):
        recalls[i] = recall_at_k(gt_distances[i], result_distances[i], k=k, eps=eps)
    return float(np.mean(recalls))
