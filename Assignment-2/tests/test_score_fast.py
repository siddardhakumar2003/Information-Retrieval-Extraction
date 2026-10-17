"""Q5 — the vectorised submission scorer.

The failure guarded against is a fast path that computes something slightly
different from the slow one. `eval/harness.py` is the definition of a semantic
score; this module only changes the shape of the arithmetic, so the load-bearing
test is that the two agree on the same input.

Two failures here are silent rather than loud, and both get their own test:

- **A key-type mismatch.** EB-NeRD candidate ids are `Int32` in the parquet. If
  the embedding table is keyed on strings, every lookup misses, every candidate
  scores `UNSCORABLE`, and the ranking collapses to the `article_id` tie-break —
  which is exactly the baseline. The submission would be well-formed, would be
  accepted, and would score like the baseline.
- **Writing the permutation instead of its inverse.** Well-formed, accepted,
  scores near random.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.retrieval import embed, user_vector
from newsrec.submit import score_fast


def _spec(dim: int = 8) -> config.EncoderSpec:
    return config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )


@pytest.fixture
def corpus() -> embed.ArticleEmbeddings:
    """Six articles on the unit sphere, ids as ints to mirror EB-NeRD."""
    rng = np.random.default_rng(7)
    v = rng.normal(size=(6, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return embed.ArticleEmbeddings(vectors=v, article_ids=[10, 11, 12, 13, 14, 15], spec=_spec())


@pytest.fixture
def history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "article_id_fixed": [[10, 11], [12], []],
        }
    )


# =============================================================================
# 1. The fast path computes the same thing as the slow one
# =============================================================================


def test_scores_match_a_naive_per_row_dot_product(corpus):
    """The whole point of the module: same numbers, different shape."""
    rng = np.random.default_rng(3)
    users = rng.normal(size=(4, 8)).astype(np.float32)
    users /= np.linalg.norm(users, axis=1, keepdims=True)

    article_rows = np.array([0, 3, 5, 1, 2], dtype=np.int64)
    user_rows = np.array([0, 1, 1, 3, 2], dtype=np.int64)

    got = score_fast.score_chunk(article_rows, user_rows, corpus.vectors, users)
    want = np.array(
        [float(corpus.vectors[a] @ users[u]) for a, u in zip(article_rows, user_rows, strict=True)],
        dtype=np.float32,
    )
    assert got == pytest.approx(want, abs=1e-6)


def test_a_missing_embedding_scores_below_every_real_cosine(corpus):
    """Not dropped: dropping a candidate changes that impression's length and
    breaks the positional contract with the truth file."""
    users = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    got = score_fast.score_chunk(
        np.array([-1, 0], dtype=np.int64), np.array([0, 0], dtype=np.int64), corpus.vectors, users
    )
    assert got[0] == score_fast.UNSCORABLE
    assert got[0] < -1.0, "must sort below any achievable cosine"
    assert got.size == 2, "the candidate is still there"


def test_a_user_without_a_vector_scores_unscorable(corpus):
    users = np.zeros((0, 8), dtype=np.float32)
    got = score_fast.score_chunk(
        np.array([0], dtype=np.int64), np.array([-1], dtype=np.int64), corpus.vectors, users
    )
    assert got[0] == score_fast.UNSCORABLE


def test_chunking_the_flat_array_changes_nothing(corpus, monkeypatch):
    """A chunk boundary must not be observable in the output."""
    rng = np.random.default_rng(11)
    users = rng.normal(size=(3, 8)).astype(np.float32)
    users /= np.linalg.norm(users, axis=1, keepdims=True)
    frame = pl.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "user_id": [1, 2, 1],
            "article_ids_inview": [[10, 12, 14], [11, 13], [15, 10, 11, 12]],
        }
    )
    row_of_user = {1: 0, 2: 1}
    row_of_article = score_fast.article_row_index(corpus)

    monkeypatch.setattr(score_fast, "SCORE_CHUNK", 1_000_000)
    whole = score_fast.score_frame(frame, row_of_user, users, corpus, row_of_article).to_list()
    monkeypatch.setattr(score_fast, "SCORE_CHUNK", 2)
    chunked = score_fast.score_frame(frame, row_of_user, users, corpus, row_of_article).to_list()
    assert whole == chunked


# =============================================================================
# 2. Ranks are the inverse permutation, with no ties
# =============================================================================


def test_ranks_are_the_inverse_permutation_best_first():
    """Highest score gets rank 1, and the rank sits at the candidate's position."""
    frame = pl.DataFrame(
        {
            "row_ix": [0, 0, 0],
            "article_id": [10, 11, 12],
            "score": [0.1, 0.9, 0.5],
        }
    )
    # Best is article 11 (position 1), then 12 (position 2), then 10 (position 0).
    assert score_fast.rank_within_impressions(frame).to_list() == [[3, 1, 2]]


def test_ranks_restart_within_each_impression():
    frame = pl.DataFrame(
        {
            "row_ix": [0, 0, 1, 1, 1],
            "article_id": [10, 11, 12, 13, 14],
            "score": [0.2, 0.8, 0.5, 0.1, 0.9],
        }
    )
    assert score_fast.rank_within_impressions(frame).to_list() == [[2, 1], [2, 3, 1]]


def test_tied_scores_break_by_article_id_and_never_repeat_a_rank():
    """`1/rank` needs distinct ranks; float32 cosines tie readily."""
    frame = pl.DataFrame(
        {
            "row_ix": [0, 0, 0, 0],
            "article_id": [14, 11, 13, 12],
            "score": [0.5, 0.5, 0.5, 0.5],
        }
    )
    ranks = score_fast.rank_within_impressions(frame).to_list()[0]
    assert sorted(ranks) == [1, 2, 3, 4], "a tie produced a repeated rank"
    # All tied, so article_id ascending decides: 11 < 12 < 13 < 14.
    assert ranks == [4, 1, 3, 2]


def test_every_impression_gets_a_permutation_of_one_to_n():
    rng = np.random.default_rng(5)
    row_ix, article_id, score = [], [], []
    for i, n in enumerate([1, 2, 5, 9, 250]):
        row_ix += [i] * n
        article_id += list(range(n))
        score += list(rng.normal(size=n))
    frame = pl.DataFrame({"row_ix": row_ix, "article_id": article_id, "score": score})
    for ranks in score_fast.rank_within_impressions(frame).to_list():
        assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_a_higher_cosine_always_outranks_a_lower_one(corpus):
    """Guards against a sign slip in the negation, which would invert the whole
    submission — a well-formed file scoring worse than random."""
    frame = pl.DataFrame({"row_ix": [0, 0], "article_id": [10, 11], "score": [-0.9, 0.9]})
    ranks = score_fast.rank_within_impressions(frame).to_list()[0]
    assert ranks[1] == 1, "the higher score must get rank 1"


# =============================================================================
# 3. User vectors
# =============================================================================


def test_user_vectors_pool_the_same_way_the_offline_path_does(corpus, history):
    row_of_user, matrix = score_fast.user_vectors_from_fixed_history(history, corpus)
    assert set(row_of_user) == {1, 2}, "user 3 has no history and must be absent"

    expected = user_vector.pool(corpus.vectors[[1, 0]])  # most recent first: 11, then 10
    assert matrix[row_of_user[1]] == pytest.approx(expected, abs=1e-6)


def test_user_vectors_are_unit_norm(corpus, history):
    _, matrix = score_fast.user_vectors_from_fixed_history(history, corpus)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)


def test_a_user_whose_history_has_no_embeddable_article_is_absent(corpus):
    history = pl.DataFrame({"user_id": [9], "article_id_fixed": [[999, 998]]})
    row_of_user, matrix = score_fast.user_vectors_from_fixed_history(history, corpus)
    assert row_of_user == {}
    assert matrix.shape[0] == 0


def test_the_history_cap_is_the_shared_config_value(corpus):
    """Read from config, not written here, so the submission and the offline
    path cannot disagree about how much history counts."""
    long_history = pl.DataFrame({"user_id": [1], "article_id_fixed": [[10, 11, 12, 13, 14, 15]]})
    _, capped = score_fast.user_vectors_from_fixed_history(long_history, corpus, limit=2)
    # Most recent two are 14, 15 -> rows 4 and 5, most-recent-first.
    assert capped[0] == pytest.approx(user_vector.pool(corpus.vectors[[5, 4]]), abs=1e-6)


# =============================================================================
# 4. The silent failure: a key-type mismatch
# =============================================================================


def test_coverage_reports_zero_when_the_id_types_do_not_match(corpus):
    """The bug that would ship the baseline while looking like a real run.

    EB-NeRD candidate ids are Int32. Keyed on strings, every lookup misses,
    every candidate scores UNSCORABLE, and the ranking falls through to the
    article_id tie-break — producing a valid, accepted, baseline-quality
    submission with nothing anywhere reporting a problem.
    """
    string_keyed = score_fast.article_row_index(corpus, cast=str)
    candidates = pl.Series([10, 11, 12], dtype=pl.Int32)
    assert score_fast.coverage(string_keyed, candidates) == 0.0

    int_keyed = score_fast.article_row_index(corpus)
    assert score_fast.coverage(int_keyed, candidates) == 1.0


def test_coverage_is_a_fraction_of_candidate_slots(corpus):
    row_of = score_fast.article_row_index(corpus)
    assert score_fast.coverage(row_of, pl.Series([10, 999, 11, 998])) == 0.5


def test_a_total_coverage_miss_makes_every_score_unscorable(corpus):
    """Downstream of the mismatch: what the submission would actually contain."""
    users = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    frame = pl.DataFrame(
        {"impression_id": [1], "user_id": [1], "article_ids_inview": [[10, 11, 12]]}
    )
    ranks = score_fast.score_frame(frame, {1: 0}, users, corpus, {}).to_list()[0]
    # Nothing scorable, so article_id ascending decides — i.e. the baseline.
    assert ranks == [1, 2, 3]


# =============================================================================
# 5. Lines
# =============================================================================


def test_lines_carry_the_impression_id_and_the_ranks_in_row_order():
    frame = pl.DataFrame({"impression_id": [7, 0, 0]})
    ranks = pl.Series([[2, 1], [1], [3, 1, 2]])
    assert list(score_fast.lines_for_frame(frame, ranks)) == ["7 [2,1]", "0 [1]", "0 [3,1,2]"]


def test_repeated_impression_ids_each_get_their_own_line():
    """EB-NeRD's 200,000 beyond-accuracy rows all carry id 0."""
    frame = pl.DataFrame({"impression_id": [0, 0, 0]})
    ranks = pl.Series([[1], [1], [1]])
    assert len(list(score_fast.lines_for_frame(frame, ranks))) == 3
