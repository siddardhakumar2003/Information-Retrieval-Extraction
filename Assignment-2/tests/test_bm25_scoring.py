"""Q2.3, Q2.4, Q2.7 — the BM25 scorer, proved against an oracle.

D7 settled that we write BM25 ourselves because Q2.1 asks us to build it. That
makes correctness our problem, so it is established the only way it can be:
`rank_bm25.BM25Okapi` is run over the same corpus with the same parameters and
the scores must agree.

The comparison is only meaningful if the two implementations agree *exactly*.
That is why we inherit rank_bm25's idf floor rather than a cleaner rule of our
own — see design_rationale.md 1.15. A test that allowed "close enough" would
pass for an implementation with a real bug in it.

`rank_bm25` appears nowhere outside this file.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from rank_bm25 import BM25Okapi

from newsrec import config
from newsrec.retrieval import bm25, tokenise
from newsrec.retrieval import index as idx

NAIVE = config.TokeniserSpec(name="toy", language="da")

# Deliberately varied: repeated terms, a term in every document, a rare term,
# documents of different lengths, and a very short document.
CORPUS = [
    "police arrest man in city centre",
    "police police report a stolen car in the city",
    "man bites dog in a quiet city street",
    "the city council votes on a new city budget for the city",
    "dog",
]
TOY = pl.DataFrame({"article_id": [f"d{i}" for i in range(len(CORPUS))], "text": CORPUS})


def _oracle(corpus_texts, k1, b, epsilon=None):
    tokenised = [tokenise.tokenise(t, NAIVE) for t in corpus_texts]
    kwargs = {"k1": k1, "b": b}
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    return BM25Okapi(tokenised, **kwargs)


def _ours(corpus_texts):
    frame = pl.DataFrame(
        {"article_id": [f"d{i}" for i in range(len(corpus_texts))], "text": corpus_texts}
    )
    return idx.build(frame, NAIVE)


# =============================================================================
# Q2.3 — the oracle comparison
# =============================================================================


@pytest.mark.parametrize(
    "query",
    [
        "police",
        "city",  # in 4 of 5 documents — the near-universal case
        "police city",
        "dog man",
        "city city city",  # repeated query terms must count repeatedly
        "aardvark",  # entirely out of vocabulary
        "police aardvark",  # half out of vocabulary
        "the",  # a term in most documents
    ],
)
def test_matches_rank_bm25_reference(query):
    """Q2.3 — our scores equal the oracle's to within 1e-6."""
    k1, b = config.BM25_K1, config.BM25_B
    terms = tokenise.tokenise(query, NAIVE)

    expected = _oracle(CORPUS, k1, b).get_scores(terms)
    got = bm25.score(_ours(CORPUS), terms, k1=k1, b=b)

    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("k1", [0.9, 1.2, 1.5, 2.0])
@pytest.mark.parametrize("b", [0.0, 0.5, 0.75, 1.0])
def test_matches_oracle_across_the_parameter_grid(k1, b):
    """The agreement must not be a coincidence at one (k1, b).

    b=0 disables length normalisation entirely and b=1 applies it fully; if our
    formula had a term in the wrong place, these corners would expose it even
    when the defaults happened to line up.

    k1=0 is excluded on purpose and covered separately below: the oracle is not
    defined there.
    """
    terms = tokenise.tokenise("police city dog", NAIVE)
    expected = _oracle(CORPUS, k1, b).get_scores(terms)
    got = bm25.score(_ours(CORPUS), terms, k1=k1, b=b)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)


def test_k1_zero_is_finite_here_and_not_in_the_oracle():
    """A deliberate, documented divergence — we are RIGHT and the oracle is not.

    At k1=0 the denominator `tf + k1*(...)` collapses to `tf`, so the oracle
    evaluates 0/0 for every document that does not contain the query term and
    emits NaN. It computes over the full corpus array, so it cannot avoid this.
    We only ever walk the posting lists of terms that actually occur, so a
    non-matching document contributes nothing and scores a clean 0.

    Asserted rather than skipped, so that if the oracle is ever fixed upstream
    the divergence is noticed rather than silently inherited.
    """
    terms = tokenise.tokenise("police", NAIVE)
    oracle_scores = _oracle(CORPUS, 0.0, 0.0).get_scores(terms)
    ours = bm25.score(_ours(CORPUS), terms, k1=0.0, b=0.0)

    assert np.isnan(oracle_scores).any(), "oracle no longer produces NaN at k1=0"
    assert np.all(np.isfinite(ours))
    # Where the oracle IS defined — the documents that match — we still agree.
    matched = ~np.isnan(oracle_scores)
    np.testing.assert_allclose(ours[matched], oracle_scores[matched], atol=1e-6, rtol=0)
    assert np.all(ours[~matched] == 0.0)


def test_idf_table_matches_the_oracles():
    """Compare the intermediate, not only the total.

    Two different idf tables can still produce equal scores on a lucky query.
    Checking idf directly means the floor rule itself is verified, not inferred.
    """
    index = _ours(CORPUS)
    oracle = _oracle(CORPUS, config.BM25_K1, config.BM25_B)
    ours = bm25.idf_table(index)
    assert set(ours) == set(oracle.idf)
    for term, value in oracle.idf.items():
        assert ours[term] == pytest.approx(value, abs=1e-12), term


def test_negative_idf_is_floored_exactly_as_the_oracle_does():
    """The inherited quirk, asserted rather than left implicit (rationale 1.15).

    'common' is in every document, so Okapi's raw idf is negative — which would
    mean a document is penalised for containing a query term. rank_bm25 replaces
    every negative idf with epsilon * average_idf, and so do we.
    """
    corpus = ["common alpha", "common beta", "common gamma", "common delta"]
    index = _ours(corpus)
    oracle = _oracle(corpus, config.BM25_K1, config.BM25_B)
    ours = bm25.idf_table(index)

    raw = bm25.raw_idf(index.n_docs, index.doc_freq("common"))
    assert raw < 0, "this corpus is supposed to produce a negative raw idf"
    assert ours["common"] == pytest.approx(oracle.idf["common"], abs=1e-12)
    assert ours["common"] != pytest.approx(raw)


def test_floor_uses_the_configured_epsilon():
    corpus = ["common alpha", "common beta", "common gamma", "common delta"]
    index = _ours(corpus)
    for eps in (0.0, 0.25, 0.5):
        oracle = _oracle(corpus, config.BM25_K1, config.BM25_B, epsilon=eps)
        ours = bm25.idf_table(index, epsilon=eps)
        assert ours["common"] == pytest.approx(oracle.idf["common"], abs=1e-12)


# =============================================================================
# Q2.4 — edge cases score without error and without NaN
# =============================================================================


def _assert_finite(scores, n):
    assert scores.shape == (n,)
    assert np.all(np.isfinite(scores)), f"non-finite score: {scores}"


def test_empty_query_scores_zero_everywhere():
    index = _ours(CORPUS)
    scores = bm25.score(index, [])
    _assert_finite(scores, index.n_docs)
    assert np.all(scores == 0.0)


def test_out_of_vocabulary_query_scores_zero_everywhere():
    index = _ours(CORPUS)
    scores = bm25.score(index, ["aardvark", "zzzz"])
    _assert_finite(scores, index.n_docs)
    assert np.all(scores == 0.0)


def test_single_document_corpus():
    """N=1 means df=1 for every term, so every raw idf is negative and the floor
    fires for the whole vocabulary."""
    index = _ours(["only one document here"])
    scores = bm25.score(index, ["document"])
    _assert_finite(scores, 1)


def test_zero_length_document_in_a_normal_corpus():
    """ebnerd_small really contains 19 of these. The empty document must score
    0 — not NaN, and not an exception from dividing by its length."""
    index = _ours(["alpha beta", "", "beta gamma"])
    scores = bm25.score(index, ["beta"])
    _assert_finite(scores, 3)
    assert scores[1] == 0.0


def test_corpus_of_only_empty_documents():
    """avgdl is 0 here, and BM25 divides by avgdl.

    Outside the oracle's domain — rank_bm25 raises ZeroDivisionError building
    this — so there is nothing to compare against; the requirement is only that
    we return finite zeros rather than NaN.
    """
    index = _ours(["", ""])
    scores = bm25.score(index, ["anything"])
    _assert_finite(scores, 2)
    assert np.all(scores == 0.0)


def test_the_idf_floor_can_itself_be_negative():
    """The inherited quirk in its sharpest form (rationale 1.15).

    The floor is `epsilon * average_idf`, and that average is taken over the RAW
    idfs — including the negative ones. In a corpus where the negatives dominate
    the average, the floor is itself negative, so flooring reduces the penalty
    without removing it.

    Concretely here: 'city' is in both documents, raw idf -1.609, average_idf
    -0.2012, floor -0.0503. The consequence is genuinely backwards — the
    document mentioning 'city' three times scores LOWER (-0.0839) than the one
    mentioning it once (-0.0452), because more of a negatively-weighted term is
    worse.

    This is not a bug in our code: we reproduce the oracle exactly, which is what
    D7 requires. It is asserted here so the behaviour is a documented, tested
    property rather than a surprise discovered later inside a recall number.
    """
    corpus = ["city city city short", "city and a much longer document here"]
    index = _ours(corpus)
    oracle = _oracle(corpus, config.BM25_K1, config.BM25_B)

    assert bm25.idf_table(index)["city"] < 0, "the floor is negative in this corpus"

    scores = bm25.score(index, ["city"])
    _assert_finite(scores, 2)
    np.testing.assert_allclose(scores, oracle.get_scores(["city"]), atol=1e-12, rtol=0)
    assert scores[0] < scores[1], "more of a negatively-weighted term scores lower"


def test_term_in_every_document_ranks_normally_when_the_floor_is_positive():
    """The ordinary case, for contrast with the test above.

    With enough distinct vocabulary the average raw idf is positive, so the
    floor is positive and a near-universal term behaves the way intuition
    expects: more occurrences in a shorter document ranks higher.

    Measured for this corpus: 'city' is in 4 of 6 documents, raw idf -0.5878,
    floored to +0.3067. So the SAME rule that produced a backwards ordering
    above produces a sensible one here — the sign of the floor depends on the
    rest of the vocabulary, not on the term being scored.
    """
    corpus = [
        "city city city short",
        "city and a much longer document here about things",
        "city council meets",
        "city budget rises",
        "alpha beta gamma delta epsilon zeta",
        "eta theta iota kappa lambda mu",
    ]
    index = _ours(corpus)
    assert bm25.raw_idf(index.n_docs, index.doc_freq("city")) < 0
    assert bm25.idf_table(index)["city"] > 0
    scores = bm25.score(index, ["city"])
    _assert_finite(scores, 6)
    np.testing.assert_allclose(
        scores,
        _oracle(corpus, config.BM25_K1, config.BM25_B).get_scores(["city"]),
        atol=1e-12,
        rtol=0,
    )
    assert scores[0] > scores[1]


def test_scores_are_reproducible():
    index = _ours(CORPUS)
    first = bm25.score(index, ["police", "city"])
    for _ in range(3):
        np.testing.assert_array_equal(bm25.score(index, ["police", "city"]), first)


# =============================================================================
# Q2.7 — top-K
# =============================================================================


def test_top_k_returns_exactly_k_in_descending_order():
    index = _ours(CORPUS)
    scores = bm25.score(index, ["city", "police"])
    top = bm25.top_k(scores, 3)
    assert len(top) == 3
    assert list(top) == sorted(top, key=lambda d: -scores[d])


def test_top_k_returns_everything_when_k_exceeds_the_corpus():
    index = _ours(CORPUS)
    scores = bm25.score(index, ["city"])
    assert len(bm25.top_k(scores, 999)) == index.n_docs


def test_top_k_has_no_duplicates():
    index = _ours(CORPUS)
    scores = bm25.score(index, ["city"])
    top = bm25.top_k(scores, index.n_docs)
    assert len(set(top)) == len(top)


def test_ties_break_deterministically_and_not_by_array_order():
    """Every document scores 0 for an unknown term, so the whole corpus is tied.

    Without an explicit tie-break the order would be whatever argpartition
    happened to produce, and recall@K could differ between runs — making a real
    improvement indistinguishable from reordering noise (D8).
    """
    index = _ours(CORPUS)
    scores = bm25.score(index, ["aardvark"])
    assert list(bm25.top_k(scores, 3)) == list(bm25.top_k(scores, 3))
    # Default secondary key is doc_id ascending.
    assert list(bm25.top_k(scores, 3)) == [0, 1, 2]


def test_tie_break_key_is_injectable():
    """D8 orders tied candidates by training-period popularity before falling
    back to article_id. The popularity table belongs to the retrieval stage, so
    the scorer takes it as an argument rather than knowing about it."""
    index = _ours(CORPUS)
    scores = bm25.score(index, ["aardvark"])  # all tied at 0
    popularity = np.array([0, 5, 0, 9, 1], dtype=np.int64)
    assert list(bm25.top_k(scores, 3, tie_break=popularity)) == [3, 1, 4]


def test_tie_break_does_not_override_the_score():
    """Popularity must only order *equal* scores. If it could outrank a genuine
    score difference, the retriever would quietly become a popularity ranker."""
    index = _ours(CORPUS)
    scores = bm25.score(index, ["police"])
    popularity = np.zeros(index.n_docs, dtype=np.int64)
    popularity[-1] = 10_000  # the lowest-scoring document, made very popular
    top = bm25.top_k(scores, index.n_docs, tie_break=popularity)
    assert list(top) == sorted(top, key=lambda d: (-scores[d], -popularity[d], d))
    assert scores[top[0]] >= scores[top[1]]
