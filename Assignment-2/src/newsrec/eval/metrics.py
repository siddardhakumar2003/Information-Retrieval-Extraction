"""AUC, MRR, nDCG@5, nDCG@10, with a stated rule for degenerate impressions.

Requirements covered: Q4.1, Q4.2, Q4.3
Design: agent_docs/q4_plan.md D1, D3

## What these measure, and what they do NOT measure

Every function here scores **one impression's candidate list**: the articles the
publisher actually showed that user, some clicked and some not. That is a
different question from Q2/Q3's recall@K, which asks whether a retriever can
find the clicked article anywhere in a 65,238-article corpus.

The distinction is load-bearing and belongs in the design note in these words:
**recall@K measures candidate generation over the corpus; AUC/MRR/nDCG measure
ranking within a supplied list.** Neither supersedes the other, and the two are
not two views of one number. Ranking the corpus shortlist instead would leave
AUC undefined for roughly 97% of impressions, because recall at K=200 is ~0.03
(q4_plan D1, option B).

## Averaged over impressions, never pooled

Each metric is computed per impression and then averaged. Pooling every
(candidate, label) pair into one big list would let a single MIND impression
with 279 candidates outweigh thirty EB-NeRD impressions with nine.

## Ties

Two candidates with identical scores have no true order, and a retriever that
scores every candidate 0 (an empty BM25 query) produces nothing *but* ties.

- **AUC** uses the rank-based Mann-Whitney form with **average ranks** for tied
  groups, which is what `sklearn.metrics.roc_auc_score` does. An all-tied
  impression therefore scores exactly 0.5 -- correctly, since the ranking
  carries no information.
- **MRR and nDCG** need an actual permutation, so ties break by **candidate
  position** in the impression's list (a stable descending sort). That is
  arbitrary but deterministic, so the number is identical on every run (Q1.13).
  It is also mildly optimistic on all-tied impressions, which is why the
  all-tied case is a test in its own right rather than an assumption.

## Degenerate impressions (Q4.3)

`eval_protocol.md` fixes the rule for AUC: all-positive, all-negative and
single-candidate impressions are **excluded from the average, counted and
reported** -- never imputed as 0.5, because an undefined value replaced by a
plausible one moves the mean while looking fine.

This module applies that same exclusion to **MRR and nDCG as well**, which is
stricter than the protocol's letter. The reason: on a single-candidate
impression MRR and nDCG are 1.0 by construction whatever the retriever does, and
on an all-positive impression they are 1.0 as well. Those are not measurements
of the system; averaging them in would drag every mean toward 1.0 in proportion
to how many trivial impressions the sample happens to contain. One exclusion
rule for all four metrics also means one count to report rather than four.

**Measured, there are zero degenerate impressions in the evaluated sample** --
0.0% on all three datasets. `recall.sample_impressions` already drops
impressions with no click, which rules out the all-negative case, and no sampled
impression turned out all-positive or single-candidate. The rule is therefore
exercised by manufactured fixtures rather than by real data, exactly as the
leakage boundary canaries are, and the real count of 0 is written to the report
so a reader can see it was checked rather than assumed.
"""

from __future__ import annotations

import numpy as np

from newsrec import config

#: The four per-impression ranking metrics, in report order. Named here so the
#: harness, the report writer and the tests all agree on the key set without
#: three separate literal lists drifting apart.
RANKING_METRICS: tuple[str, ...] = (
    "auc",
    "mrr",
    *(f"ndcg@{k}" for k in config.NDCG_K_VALUES),
)


def degenerate_reason(labels: np.ndarray | list[int]) -> str | None:
    """Why this impression has no defined ranking metric, or `None` if it does.

    Returns one of `"single_candidate"`, `"all_positive"`, `"all_negative"`, or
    `None`. An empty candidate list counts as `"single_candidate"` -- there is
    nothing to rank either way, and inventing a fourth category for a case that
    cannot occur would be noise.
    """
    arr = np.asarray(labels, dtype=np.int64)
    if arr.size <= 1:
        return "single_candidate"
    positives = int(arr.sum())
    if positives == arr.size:
        return "all_positive"
    if positives == 0:
        return "all_negative"
    return None


def _order(scores: np.ndarray) -> np.ndarray:
    """Candidate indices best-first, ties broken by position in the list.

    `np.argsort(-scores, kind="stable")` and not `argsort(scores)[::-1]`: the
    reversal of a stable ascending sort puts tied candidates in *reverse* list
    order, so the two differ on exactly the all-tied impressions this has to get
    right.
    """
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def auc(labels: np.ndarray | list[int], scores: np.ndarray | list[float]) -> float:
    """ROC AUC over one impression's candidates.

    The Mann-Whitney form: the probability that a randomly chosen clicked
    candidate outranks a randomly chosen unclicked one, with tied pairs counting
    a half. Callers must have checked `degenerate_reason` first; this raises
    rather than returning a plausible-looking number for an undefined case.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    reason = degenerate_reason(y)
    if reason is not None:
        raise ValueError(f"AUC is undefined for a {reason} impression")

    # Average ranks within tied score groups, ascending -- the tie handling that
    # makes this agree with sklearn to floating-point precision (Q4.2).
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    for end in range(1, s.size + 1):
        if end == s.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mrr(labels: np.ndarray | list[int], scores: np.ndarray | list[float]) -> float:
    """Reciprocal rank of the **first** clicked candidate.

    Not the mean over all clicked candidates: MRR asks how far a user has to
    read before finding something worth clicking, and the answer is decided by
    the first hit.
    """
    y = np.asarray(labels, dtype=np.int64)
    ranked = y[_order(scores)]
    hits = np.flatnonzero(ranked == 1)
    if hits.size == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def ndcg_at_k(labels: np.ndarray | list[int], scores: np.ndarray | list[float], k: int) -> float:
    """nDCG@k with binary gains and the standard log2 discount.

    The ideal DCG uses **the actual number of clicks in this impression**,
    capped at k. Using a fixed ideal of one click would score a two-click
    impression above 1.0; using the whole candidate list would make a
    nine-candidate impression incomparable to a 279-candidate one.

    An impression with no clicks has an ideal DCG of 0 and no defined nDCG; that
    is an all-negative impression, which `degenerate_reason` already excludes,
    so 0.0 here is unreachable via the harness and kept only so the function is
    total.
    """
    y = np.asarray(labels, dtype=np.int64)
    ranked = y[_order(scores)][:k]
    discounts = 1.0 / np.log2(np.arange(2, ranked.size + 2))
    dcg = float((ranked * discounts).sum())

    ideal_hits = min(int(y.sum()), k)
    if ideal_hits == 0:
        return 0.0
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg


def ranking_metrics(
    labels: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    k_values: tuple[int, ...] | None = None,
) -> dict[str, float] | None:
    """All four metrics for one impression, or `None` if it is degenerate.

    `None` rather than a dict of NaNs so a caller cannot average it by accident:
    `np.mean` of a NaN-bearing array is NaN, which is at least loud, but a
    caller filtering with `if row:` gets the exclusion right without knowing the
    rule. The degenerate count is the caller's to keep -- `degenerate_reason`
    gives it the label.
    """
    k_values = config.NDCG_K_VALUES if k_values is None else k_values
    if degenerate_reason(labels) is not None:
        return None
    out = {"auc": auc(labels, scores), "mrr": mrr(labels, scores)}
    for k in k_values:
        out[f"ndcg@{k}"] = ndcg_at_k(labels, scores, k)
    return out
