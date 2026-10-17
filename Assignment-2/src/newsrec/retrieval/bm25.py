"""BM25-Okapi scoring over our inverted index (q2_plan.md D5).

Requirements covered: Q2.3 (correct, proved against an oracle), Q2.4 (edge
cases), Q2.7 (top-K with a deterministic tie-break).

## The formula

For a query term `q` and document `d`, BM25 contributes:

    idf(q) * ( tf * (k1 + 1) ) / ( tf + k1 * (1 - b + b * |d| / avgdl) )

Three ideas, each doing one job:

* **tf** -- how often `q` occurs in `d`. The `tf / (tf + k1 * ...)` shape makes
  this *saturate*: the second occurrence of a word adds much more than the
  tenth, because a document mentioning "police" ten times is not ten times more
  about police. `k1` sets how fast that flattening happens.
* **idf** -- how rare `q` is across the corpus. Matching a rare word says much
  more than matching a common one.
* **|d| / avgdl** -- document length relative to the average, weighted by `b`.
  Without it a long document wins simply by containing more words. `b=0`
  disables it entirely, `b=1` applies it fully.

A document that does not contain `q` gets `tf = 0`, so the numerator is zero and
it contributes nothing. That is why scoring only ever walks the posting lists of
the query's own terms rather than the whole corpus.

## Why the idf here has a quirk we did not choose

Okapi's classic idf is `log(N - df + 0.5) - log(df + 0.5)`, which goes **negative**
once a term appears in more than half the documents. A negative contribution
means a document is *penalised* for containing a query term, which is
incoherent.

`rank_bm25.BM25Okapi` deals with this by flooring every negative idf at
`epsilon * average_idf`, where the average is taken over the raw (un-floored)
idfs of the whole vocabulary. We inherit that rule exactly, including its
oddities, because D7 makes `rank_bm25` our correctness oracle and that proof only
holds if the two implementations agree exactly rather than approximately. A
cleaner rule -- clamping at zero, or Lucene's `+1` smoothing -- would be defensible
on its own terms and would break the proof. See design_rationale.md 1.15.

## Where this is deliberately not the oracle

`rank_bm25` cannot build an index over a corpus with an empty vocabulary or zero
total length: it divides by `len(self.idf)` and by `avgdl` unguarded and raises
ZeroDivisionError. Those corpora are real for us (an all-empty-text corpus is a
legitimate degenerate input), so we return finite zeros instead. That is outside
the oracle's domain, so nothing is being contradicted -- there is simply nothing
to compare against.
"""

from __future__ import annotations

import math

import numpy as np

from newsrec import config
from newsrec.retrieval.index import InvertedIndex


def raw_idf(n_docs: int, doc_freq: int) -> float:
    """Okapi's classic idf, before any floor. Negative when df > N/2."""
    return math.log(n_docs - doc_freq + 0.5) - math.log(doc_freq + 0.5)


def idf_table(index: InvertedIndex, epsilon: float | None = None) -> dict[str, float]:
    """term -> idf, with rank_bm25's floor applied.

    Deliberately a faithful reimplementation of `BM25Okapi._calc_idf`:

    1. compute the raw idf for every term;
    2. average those raw values over the whole vocabulary -- **including the
       negative ones**, which is what rank_bm25 does;
    3. replace every negative idf with `epsilon * average_idf`.

    Step 2 is the part that looks like a mistake and is not ours to fix. The
    average includes values that are about to be discarded, so the floor depends
    on terms that will never use it. Reproducing it is the price of an exact
    oracle comparison.
    """
    epsilon = config.BM25_EPSILON if epsilon is None else epsilon
    n = index.n_docs
    if not index.vocab:
        return {}

    raw = {term: raw_idf(n, index.doc_freq(term)) for term in index.vocab}
    average = sum(raw.values()) / len(raw)
    floor = epsilon * average
    return {term: (floor if value < 0 else value) for term, value in raw.items()}


def idf_array(index: InvertedIndex, epsilon: float | None = None) -> np.ndarray:
    """`idf_table` as an array indexed by term_id, for the hot path."""
    table = idf_table(index, epsilon)
    out = np.zeros(len(index.vocab), dtype=np.float64)
    for term, tid in index.vocab.items():
        out[tid] = table[term]
    return out


def score(
    index: InvertedIndex,
    query_terms: list[str],
    *,
    k1: float | None = None,
    b: float | None = None,
    epsilon: float | None = None,
    idf: np.ndarray | None = None,
) -> np.ndarray:
    """BM25 score of every document against `query_terms`.

    Returns an array of length `index.n_docs`. Terms are **not** deduplicated:
    a query containing "city" twice contributes twice, which is what
    `BM25Okapi.get_scores` does and what preserves the signal that a term
    recurring across a user's clicked titles matters more (D6 option D).

    `idf` may be supplied precomputed. Building it walks the whole vocabulary,
    so recomputing it per query would dominate the cost of a 230,117-query run.
    """
    k1 = config.BM25_K1 if k1 is None else k1
    b = config.BM25_B if b is None else b

    scores = np.zeros(index.n_docs, dtype=np.float64)
    if not query_terms or index.n_docs == 0 or not index.vocab:
        return scores

    if idf is None:
        idf = idf_array(index, epsilon)

    avgdl = index.avg_doc_length
    # An all-empty corpus has avgdl == 0, and the length term divides by it.
    # Every document then has length 0 too, so the ratio is genuinely 0/0; the
    # meaningful value is "no length penalty", i.e. treat the ratio as 0.
    if avgdl > 0:
        length_norm = k1 * (1 - b + b * index.doc_lengths / avgdl)
    else:
        length_norm = np.full(index.n_docs, k1 * (1 - b), dtype=np.float64)

    for term in query_terms:
        tid = index.vocab.get(term)
        if tid is None:
            continue  # out of vocabulary: contributes nothing, exactly as tf=0 would
        docs, freqs = index.postings(term)
        if docs.size == 0:
            continue
        tf = freqs.astype(np.float64)
        denom = tf + length_norm[docs]
        scores[docs] += idf[tid] * (tf * (k1 + 1)) / denom

    return scores


def top_k(
    scores: np.ndarray,
    k: int,
    *,
    tie_break: np.ndarray | None = None,
) -> np.ndarray:
    """The `k` highest-scoring doc_ids, descending, with a total order.

    Q2.7 requires exactly `k` results (or all of them when the corpus is
    smaller), deduplicated, in descending score order, with ties broken
    deterministically.

    The tie-break matters more than it looks. Ties are common -- every document
    that matches no query term scores exactly 0 -- and on the MIND leaderboard
    roughly half of all candidate slots are tied at 0 (design_rationale 2.14).
    Without an explicit rule the order would be whatever `argpartition` happened
    to produce, so recall@K could move between runs and a real improvement would
    be indistinguishable from reordering noise.

    `tie_break` is an optional per-document array, higher meaning better, used
    only to order documents whose scores are already equal. D8 passes
    training-period popularity here. It can never outrank a genuine score
    difference, because score is the primary key. doc_id ascending is the final
    key, so the order is total and no tie is ever left to chance.
    """
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)

    doc_ids = np.arange(n, dtype=np.int64)
    if tie_break is None:
        tie_break = np.zeros(n, dtype=np.float64)

    # ONE code path, sliced afterwards. `np.lexsort` sorts by the LAST key
    # first, so this reads bottom-up: score descending, then tie_break
    # descending, then doc_id ascending as the final total order.
    #
    # A note on cost, because it departs from what q2_plan D8 assumed.
    # D8 planned `argpartition` for O(n) selection instead of an O(n log n)
    # sort. That works when the ranking key is a single array, but not here: a
    # partition on score alone would choose arbitrarily among documents tied at
    # the k-th boundary, and the tie-break would never get to see the ones it
    # should have preferred. Since roughly half of all candidate slots tie at
    # zero (design_rationale 2.14), that is the common case rather than a corner.
    # Correctness first; the measured cost is in reports/metrics/scale.json.
    #
    # Writing this as a single expression rather than branching on `k < n` is
    # also deliberate: an earlier version had two branches computing the same
    # order, and a mutation that broke one of them was invisible because the
    # tests happened to exercise only the other.
    return np.lexsort((doc_ids, -tie_break, -scores))[:k].astype(np.int64)
