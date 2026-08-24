"""Fusion methods for combining BM25 and semantic scores at the impression level."""

import numpy as np


def reciprocal_rank_fusion(candidate_ids, retriever_scores, k_rrf=60):
    """
    Reciprocal Rank Fusion: combine RRF scores from multiple retrievers.

    Args:
        candidate_ids: list of article IDs (order-preserving, original order).
        retriever_scores: dict[str, dict] e.g. {"bm25": {aid: score}, "semantic": {aid: score}}
        k_rrf: RRF constant, default 60.

    Returns:
        dict[article_id] -> rrf_score (higher is better).
    """
    rrf_scores = {}

    for aid in candidate_ids:
        rrf_score = 0.0
        for retriever_name, scores_dict in retriever_scores.items():
            # Rank within this retriever: sort candidates by score descending,
            # assign ranks 1..N. Missing articles get rank N+1 (worst).
            ranked = sorted(candidate_ids, key=lambda x: scores_dict.get(x, -np.inf), reverse=True)
            rank = ranked.index(aid) + 1 if aid in ranked else len(candidate_ids) + 1
            rrf_score += 1.0 / (k_rrf + rank)
        rrf_scores[aid] = rrf_score

    return rrf_scores


def weighted_fusion(candidate_ids, retriever_scores, weights):
    """
    Weighted fusion with min-max normalization per retriever over candidates.

    Args:
        candidate_ids: list of article IDs.
        retriever_scores: dict[str, dict] e.g. {"bm25": {aid: score}, "semantic": {aid: score}}.
        weights: dict[str, float] e.g. {"semantic": 0.6, "bm25": 0.4}.

    Returns:
        dict[article_id] -> fused_score.
    """
    fused = {}

    # Normalize each retriever's scores to [0, 1] over the candidate set.
    normalized_scores = {}
    for retriever_name, scores_dict in retriever_scores.items():
        scores_list = [scores_dict.get(aid, 0.0) for aid in candidate_ids]
        min_score = min(scores_list) if scores_list else 0.0
        max_score = max(scores_list) if scores_list else 0.0

        if max_score == min_score:
            # Degenerate case: all scores equal, map to 0.5.
            normalized_scores[retriever_name] = {aid: 0.5 for aid in candidate_ids}
        else:
            normalized_scores[retriever_name] = {
                aid: (scores_dict.get(aid, 0.0) - min_score) / (max_score - min_score)
                for aid in candidate_ids
            }

    # Weighted combination.
    for aid in candidate_ids:
        fused[aid] = sum(
            weights.get(retriever_name, 0.0) * normalized_scores[retriever_name][aid]
            for retriever_name in retriever_scores.keys()
        )

    return fused


def scores_to_rank_permutation(candidate_ids_in_original_order, scores):
    """
    Convert scores dict to a rank permutation [1, 2, ..., N] in original order.

    Ties broken by score descending, then article_id ascending (for determinism).

    Args:
        candidate_ids_in_original_order: list of article IDs in the order they appeared.
        scores: dict[article_id] -> score.

    Returns:
        list of ranks [r1, r2, ..., rN] where r_i is the rank of candidate_ids_in_original_order[i].
    """
    # Sort candidates by score (descending) with tie-break by article_id (ascending).
    sorted_aids = sorted(
        candidate_ids_in_original_order,
        key=lambda aid: (-scores.get(aid, 0.0), aid)
    )

    # Map article_id -> rank (1-indexed).
    aid_to_rank = {aid: rank + 1 for rank, aid in enumerate(sorted_aids)}

    # Return ranks in original order.
    return [aid_to_rank[aid] for aid in candidate_ids_in_original_order]
