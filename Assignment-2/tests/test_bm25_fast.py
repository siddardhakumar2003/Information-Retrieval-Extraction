"""BM25 over candidate lists at submission scale.

This module exists purely for speed — 0.334 ms per impression against the Q4
path's 16.1 ms, a 48x difference that turns four hours into thirteen minutes.
Speed is not a reason to compute something different, so the load-bearing test
is agreement with `bm25.score`, which Q2 already validated against `rank_bm25`.

Everything else here guards the two optimisations that make it fast and could
each silently corrupt a score: the reused scatter buffer, and the vectorised
ragged range.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.retrieval import bm25, recall
from newsrec.retrieval import query as query_mod
from newsrec.submit import bm25_fast


@pytest.fixture
def corpus(store_root, dataset):
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    spec = config.tokeniser_for(dataset)
    index, scorer, idf = bm25_fast.build(articles, spec)
    return index, scorer, idf, spec


def _query_vector(terms, index, idf):
    counts: dict[int, int] = {}
    for term in terms:
        tid = index.vocab.get(term)
        if tid is not None:
            counts[tid] = counts.get(tid, 0) + 1
    if not counts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    tid = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    qtf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    return tid, qtf * idf[tid]


# =============================================================================
# 1. The load-bearing test: same numbers as the slow path
# =============================================================================


def test_scores_match_bm25_score_on_real_queries(corpus, store_root, dataset):
    """Agreement with the implementation Q2 validated against rank_bm25.

    Real queries built from real history, not synthetic term lists: a scorer can
    agree on toy input and disagree on the length normalisation that only bites
    on documents of realistic length.
    """
    index, scorer, idf, spec = corpus
    processed = store_root / "processed" / dataset
    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")
    titles = feature_store.read_table(processed / "articles.parquet").select("article_id", "title")

    sample = recall.sample_impressions(impressions, "test", 25)
    users = sample["user_id"].unique().to_list()
    queries = query_mod.build_queries(
        sample, history.filter(pl.col("user_id").is_in(users)), titles, spec
    )
    doc_of = {a: i for i, a in enumerate(index.article_ids)}

    checked = 0
    for row in sample.iter_rows(named=True):
        terms = queries.get(row["impression_id"], [])
        rows = np.array([doc_of[a] for a in row["candidate_ids"] if a in doc_of], dtype=np.int64)
        if rows.size == 0:
            continue
        expected = bm25.score(index, terms, idf=idf)[rows]
        got = scorer.score(*_query_vector(terms, index, idf), rows)
        assert got == pytest.approx(expected, abs=1e-9)
        checked += 1
    assert checked > 0, "no impression was actually compared"


def test_the_doc_term_matrix_holds_the_query_independent_half(corpus):
    """`docweight` must not depend on the query — that is what makes it
    precomputable. Scoring a single-term query must return exactly that term's
    stored weight times its idf."""
    index, scorer, idf, _ = corpus
    matrix = bm25_fast.doc_term_matrix(index)
    term = next(iter(index.vocab))
    tid = index.vocab[term]
    docs, _ = index.postings(term)
    rows = docs.astype(np.int64)[:5]
    if rows.size == 0:
        pytest.skip("term has no postings")

    got = scorer.score(np.array([tid]), np.array([idf[tid]]), rows)
    expected = np.asarray(matrix[rows, tid].todense()).ravel() * idf[tid]
    assert got == pytest.approx(expected, abs=1e-12)


# =============================================================================
# 2. The reused buffer must not leak between impressions
# =============================================================================


def test_scoring_twice_gives_the_same_answer(corpus):
    """The buffer is scattered into and reset by hand. A missed reset would
    make an impression's score depend on whatever was scored before it —
    order-dependent, reproducible-looking, and wrong."""
    index, scorer, idf, _ = corpus
    terms = list(index.vocab)[:6]
    rows = np.arange(min(8, index.n_docs), dtype=np.int64)
    tid, w = _query_vector(terms, index, idf)

    first = scorer.score(tid, w, rows)
    second = scorer.score(tid, w, rows)
    assert first == pytest.approx(second)


def test_an_earlier_query_cannot_contaminate_a_later_one(corpus):
    """The specific leak: score a big query, then a disjoint small one.

    If the buffer were not reset, the second score would carry terms from the
    first. Compared against the small query scored on a FRESH scorer, which
    cannot have been contaminated.
    """
    index, scorer, idf, _ = corpus
    vocab = list(index.vocab)
    if len(vocab) < 12:
        pytest.skip("fixture vocabulary too small to build disjoint queries")

    big, small = vocab[:8], vocab[8:10]
    rows = np.arange(min(10, index.n_docs), dtype=np.int64)

    scorer.score(*_query_vector(big, index, idf), rows)
    after = scorer.score(*_query_vector(small, index, idf), rows)

    fresh = bm25_fast.CandidateScorer(bm25_fast.doc_term_matrix(index), idf)
    clean = fresh.score(*_query_vector(small, index, idf), rows)
    assert after == pytest.approx(clean, abs=1e-12)


def test_an_empty_query_scores_zero_not_stale_values(corpus):
    """Cold start. Zeros let the popularity tie-break decide, exactly as
    `bm25.score` does for an empty term list."""
    index, scorer, idf, _ = corpus
    rows = np.arange(min(5, index.n_docs), dtype=np.int64)
    scorer.score(*_query_vector(list(index.vocab)[:5], index, idf), rows)
    got = scorer.score(np.empty(0, dtype=np.int64), np.empty(0), rows)
    assert got.tolist() == [0.0] * rows.size


# =============================================================================
# 3. The ragged range
# =============================================================================


def test_ragged_indices_concatenates_the_ranges():
    got = bm25_fast._ragged_indices(np.array([10, 100, 50]), np.array([3, 2, 1]))
    assert got.tolist() == [10, 11, 12, 100, 101, 50]


def test_ragged_indices_handles_empty_segments():
    """A document with no terms in the vocabulary contributes no postings.

    `np.add.reduceat` reads the wrong bucket for a zero-length segment, so the
    caller masks those out — this pins that the index builder itself copes.
    """
    got = bm25_fast._ragged_indices(np.array([5, 9, 9]), np.array([2, 0, 3]))
    assert got.tolist() == [5, 6, 9, 10, 11]


def test_ragged_indices_of_nothing_is_empty():
    assert bm25_fast._ragged_indices(np.array([1, 2]), np.array([0, 0])).size == 0


def test_a_document_with_no_matching_terms_scores_zero(corpus):
    """The empty-segment case end to end, where reduceat would misread."""
    index, scorer, idf, _ = corpus
    if index.n_docs < 3:
        pytest.skip("fixture too small")
    # A query of one term, scored against every document: those without the
    # term must be exactly zero rather than borrowing a neighbour's sum.
    term = next(iter(index.vocab))
    tid = index.vocab[term]
    docs, _ = index.postings(term)
    has = set(docs.tolist())
    rows = np.arange(index.n_docs, dtype=np.int64)
    got = scorer.score(np.array([tid]), np.array([idf[tid]]), rows)
    for doc in range(index.n_docs):
        if doc not in has:
            assert got[doc] == 0.0, f"doc {doc} has no '{term}' but scored {got[doc]}"


# =============================================================================
# 4. Query construction
# =============================================================================


def test_repeated_terms_count_repeatedly(corpus):
    """A word in five of the user's titles must weigh more than one in a single
    title — q2_plan D6, and what `bm25.score` does with a repeated term."""
    index, _, idf, _ = corpus
    term = next(iter(index.vocab))
    titles = {"a": [term], "b": [term], "c": [term]}
    once, w_once = bm25_fast.query_terms(["a"], titles, index.vocab, idf)
    thrice, w_thrice = bm25_fast.query_terms(["a", "b", "c"], titles, index.vocab, idf)
    assert once.tolist() == thrice.tolist()
    assert w_thrice[0] == pytest.approx(3 * w_once[0])


def test_the_history_cap_takes_the_most_recent(corpus):
    """Raw history is oldest-first, so the cap must take the TAIL.

    Taking the head would build every query from the user's oldest clicks — a
    query that is well-formed, plausible, and about the wrong user.
    """
    index, _, idf, _ = corpus
    vocab = list(index.vocab)
    if len(vocab) < 2:
        pytest.skip("fixture vocabulary too small")
    old, new = vocab[0], vocab[1]
    titles = {"oldest": [old], "newest": [new]}
    tid, _ = bm25_fast.query_terms(["oldest", "newest"], titles, index.vocab, idf, limit=1)
    assert tid.tolist() == [index.vocab[new]], "the cap kept the oldest click"


def test_an_out_of_vocabulary_term_is_dropped(corpus):
    index, _, idf, _ = corpus
    titles = {"a": ["zzzz-not-a-real-term-zzzz"]}
    tid, w = bm25_fast.query_terms(["a"], titles, index.vocab, idf)
    assert tid.size == 0 and w.size == 0


def test_a_history_with_no_known_articles_is_an_empty_query(corpus):
    index, _, idf, _ = corpus
    tid, w = bm25_fast.query_terms(["absent"], {}, index.vocab, idf)
    assert tid.size == 0 and w.size == 0


def test_titles_are_tokenised_once_per_article(corpus, store_root, dataset):
    _, _, _, spec = corpus
    titles = feature_store.read_table(
        store_root / "processed" / dataset / "articles.parquet"
    ).select("article_id", "title")
    got = bm25_fast.tokenised_titles(titles, spec)
    assert len(got) <= len(titles)
    assert all(isinstance(v, list) for v in got.values())


# =============================================================================
# 5. Gaps found by mutation
#
# Both of these survived a first round: `query_terms` was never exercised
# against the reference implementation (the agreement test built its query
# vector with a local helper instead), and no fixture document has zero terms,
# so the empty-segment mask was never reached.
# =============================================================================


def test_query_terms_feeds_the_scorer_to_the_same_answer_as_bm25_score(corpus, store_root, dataset):
    """End to end through the REAL query builder, not a test-local stand-in.

    Dropping the idf from `query_terms` left every other test passing: the
    repeated-term test only checks a ratio, which idf cancels out of, and the
    agreement test was building its own query vector. This closes both.
    """
    index, scorer, idf, spec = corpus
    processed = store_root / "processed" / dataset
    titles_df = feature_store.read_table(processed / "articles.parquet").select(
        "article_id", "title"
    )
    titles = bm25_fast.tokenised_titles(titles_df, spec)

    history_ids = list(titles)[:5]
    if not history_ids:
        pytest.skip("fixture has no titled articles")

    # The same terms the slow path would build from this history.
    expected_terms: list[str] = []
    for article_id in history_ids[-config.QUERY_HISTORY_LIMIT :]:
        expected_terms.extend(titles[article_id])

    rows = np.arange(min(20, index.n_docs), dtype=np.int64)
    expected = bm25.score(index, expected_terms, idf=idf)[rows]
    got = scorer.score(*bm25_fast.query_terms(history_ids, titles, index.vocab, idf), rows)
    assert got == pytest.approx(expected, abs=1e-9)


def test_a_document_with_no_terms_at_all_scores_zero():
    """The empty-segment case, on a hand-built matrix.

    `np.add.reduceat` does not return 0 for a zero-length segment — it returns
    the element at that start index, i.e. the next document's first posting. So
    an article with no indexable text would silently inherit its neighbour's
    score. No fixture article is empty, so this has to be constructed.
    """
    from scipy import sparse

    # Three docs, three terms; doc 1 has no postings at all.
    matrix = sparse.csr_matrix(
        (
            np.array([5.0, 7.0, 9.0]),
            (np.array([0, 2, 2]), np.array([0, 0, 1])),
        ),
        shape=(3, 3),
    )
    assert matrix.indptr[1] == matrix.indptr[2], "doc 1 must be the empty one"

    idf = np.array([1.0, 1.0, 1.0])
    scorer = bm25_fast.CandidateScorer(matrix, idf)
    got = scorer.score(np.array([0]), np.array([1.0]), np.array([0, 1, 2], dtype=np.int64))

    assert got[1] == 0.0, f"empty document inherited a neighbour's score: {got.tolist()}"
    assert got[0] == pytest.approx(5.0)
    assert got[2] == pytest.approx(7.0)
