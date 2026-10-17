"""BM25 over candidate lists at submission scale (Q5.1).

Design: agent_docs/q5_plan.md, agent_docs/q4_plan.md

## Why the Q4 path cannot be reused

`harness._bm25_scores` calls `bm25.score`, which scores the **whole corpus** and
then reads off the candidate rows. At 6-16 ms per impression that is roughly
4 hours for MIND's 2,370,727 test rows and 22 hours for EB-NeRD's 13,536,710 --
and almost all of it is spent scoring 120,000 articles nobody asked about.

We only ever need the ~39 candidates the publisher showed.

## The factorisation that makes this cheap

BM25 splits into a query part and a document part:

    score(q, d) = SUM over t of  qtf(t) * idf(t)  *  docweight(t, d)

    docweight(t, d) = tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len_d / avgdl))

`docweight` does not depend on the query, so it is computed **once** for the
whole corpus and stored as a sparse document-by-term matrix. Scoring one
impression is then a sparse dot product between that impression's query vector
and 39 rows of that matrix.

## Why the inner loop looks like that

The obvious vectorisation -- gather a sparse row per (impression, candidate)
pair and multiply -- needs about 3.5 GB per 50,000 impressions, because each
query row carries ~150 non-zeros and there are 39 pairs per impression. It does
not fit.

Instead each impression scatters its query into one **reused dense buffer** over
the vocabulary, and the candidates' non-zeros are gathered with a vectorised
ragged range and reduced with `np.add.reduceat`. The buffer is allocated once
and only the touched entries are reset, so the per-impression cost is
proportional to the query length plus the candidates' total term count -- about
1,200 numbers -- rather than to the vocabulary or the corpus.

Nothing here redefines BM25. `k1`, `b` and the idf come from `config` and
`bm25.idf_array`, the same values Q2 measured with, and `test_bm25_fast.py`
asserts agreement with `bm25.score` on real fixtures.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

from newsrec import config
from newsrec.retrieval import bm25, tokenise
from newsrec.retrieval import index as idx


def doc_term_matrix(
    index: idx.InvertedIndex,
    k1: float | None = None,
    b: float | None = None,
) -> sparse.csr_matrix:
    """The query-independent half of BM25, as docs x vocab.

    Built once per corpus. `index` stores postings term-major (CSR over terms),
    so this transposes into doc-major, which is the orientation scoring needs:
    we look up candidates, never terms.
    """
    k1 = config.BM25_K1 if k1 is None else k1
    b = config.BM25_B if b is None else b

    n_terms = len(index.vocab)
    term_ids = np.repeat(np.arange(n_terms, dtype=np.int64), np.diff(index.offsets))
    docs = index.doc_ids.astype(np.int64)
    tf = index.term_freqs.astype(np.float64)

    avgdl = index.avg_doc_length
    if avgdl > 0:
        norm = k1 * (1.0 - b + b * index.doc_lengths[docs] / avgdl)
    else:
        norm = np.full(docs.size, k1 * (1.0 - b), dtype=np.float64)
    weights = tf * (k1 + 1.0) / (tf + norm)

    return sparse.csr_matrix((weights, (docs, term_ids)), shape=(index.n_docs, max(n_terms, 1)))


def _ragged_indices(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate `range(s, s+l)` for each (s, l), without a Python loop.

    The standard trick: lay out one flat `arange` over the total length, then
    offset each element by its own row's start minus the running total of the
    rows before it. This is the only reason the inner loop is fast enough --
    building the same index list in Python costs more than the arithmetic it
    feeds.
    """
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(lengths)
    offsets = np.repeat(starts - np.concatenate(([0], ends[:-1])), lengths)
    return offsets + np.arange(total, dtype=np.int64)


class CandidateScorer:
    """Scores an impression's candidates against a query, reusing one buffer."""

    def __init__(self, matrix: sparse.csr_matrix, idf: np.ndarray):
        self.indptr = matrix.indptr
        self.indices = matrix.indices
        self.data = matrix.data
        self.idf = idf
        #: Allocated once. Per-impression cost must not scale with vocabulary
        #: size, so this is scattered into and then reset entry by entry.
        self._buffer = np.zeros(matrix.shape[1], dtype=np.float64)

    def score(
        self,
        term_ids: np.ndarray,
        term_weights: np.ndarray,
        doc_rows: np.ndarray,
    ) -> np.ndarray:
        """BM25 for `doc_rows` against one query. Zeros for an empty query."""
        out = np.zeros(doc_rows.size, dtype=np.float64)
        if term_ids.size == 0 or doc_rows.size == 0:
            return out

        buf = self._buffer
        buf[term_ids] = term_weights
        try:
            starts = self.indptr[doc_rows]
            lengths = self.indptr[doc_rows + 1] - starts
            flat = _ragged_indices(starts, lengths)
            if flat.size:
                contrib = self.data[flat] * buf[self.indices[flat]]
                # reduceat needs the start of each segment; empty segments would
                # read the wrong bucket, so they are masked out afterwards.
                segment_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
                summed = np.add.reduceat(contrib, segment_starts)
                out = np.where(lengths > 0, summed, 0.0)
        finally:
            # Reset only what was touched. Zeroing the whole buffer would make
            # every impression cost O(vocabulary).
            buf[term_ids] = 0.0
        return out


def query_terms(
    history_ids: list[str],
    titles: dict[str, list[str]],
    vocab: dict[str, int],
    idf: np.ndarray,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(term_ids, qtf * idf)` for one user's history.

    Repetition is preserved before weighting -- a word appearing in five of the
    user's clicked titles counts five times, exactly as `bm25.score` treats a
    repeated query term (q2_plan D6). Out-of-vocabulary terms are dropped, which
    is what a term-frequency of zero would do anyway.
    """
    limit = config.QUERY_HISTORY_LIMIT if limit is None else limit
    if not history_ids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    counts: dict[int, int] = {}
    # Most recent first, then capped: the raw history is oldest-first.
    for article_id in history_ids[-limit:]:
        for term in titles.get(article_id, ()):
            tid = vocab.get(term)
            if tid is not None:
                counts[tid] = counts.get(tid, 0) + 1

    if not counts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    term_ids = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    qtf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    return term_ids, qtf * idf[term_ids]


def tokenised_titles(
    articles: pl.DataFrame,
    spec: config.TokeniserSpec,
) -> dict[str, list[str]]:
    """article_id -> title terms, tokenised once per article.

    Once per ARTICLE, not once per (impression, article): a popular article
    appears in thousands of histories, and re-tokenising it each time was the
    dominant cost in Q2's first recall run.
    """
    return {
        article_id: tokenise.tokenise(title, spec)
        for article_id, title in articles.select("article_id", "title")
        .drop_nulls("title")
        .iter_rows()
    }


def build(articles: pl.DataFrame, spec: config.TokeniserSpec):
    """Everything scoring needs for one corpus: index, weights, idf, scorer."""
    index = idx.build(articles.select("article_id", "text"), spec)
    idf = bm25.idf_array(index)
    matrix = doc_term_matrix(index)
    return index, CandidateScorer(matrix, idf), idf
