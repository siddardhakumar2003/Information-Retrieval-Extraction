"""The EB-NeRD semantic submission.

Every test here exists because of one bug. The first real run produced a
perfectly valid, correctly formatted, accepted-shape submission that was
byte-identical to the `article_id` baseline, while the coverage guard reported
1.0000.

The cause: the scorer resolves article ids in **two independent places**. The
candidate side was handed an integer-keyed index and worked. The user side built
its own `{str: row}` default from `embeddings.article_ids`, so all 807,677 users
failed lookup, every score became UNSCORABLE, and every ranking fell through to
the tie-break. The guard measured candidates because candidates were what its
author was thinking about.

So these tests check **both** halves, and check them against the specific shape
of the data rather than against a toy where `str` and `int` keys would both work.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.retrieval import embed
from newsrec.submit import score_fast
from newsrec.submit import semantic_predict as sp


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
    """Ids as STRINGS of digits, exactly like the real EB-NeRD table.

    This is the detail that matters: with `["a", "b"]` a str/int mix would raise
    somewhere, but with `["10", "11"]` it silently matches nothing — which is
    the real failure.
    """
    rng = np.random.default_rng(2)
    v = rng.normal(size=(4, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return embed.ArticleEmbeddings(vectors=v, article_ids=["10", "11", "12", "13"], spec=_spec())


@pytest.fixture
def history() -> pl.DataFrame:
    """`article_id_fixed` as Int32, exactly like the real history parquet."""
    return pl.DataFrame(
        {"user_id": [1, 2], "article_id_fixed": [[10, 11], [12]]},
        schema={"user_id": pl.UInt32, "article_id_fixed": pl.List(pl.Int32)},
    )


# =============================================================================
# 1. The bug: both halves resolve ids, and both can fail independently
# =============================================================================


def test_string_keyed_index_finds_no_users(corpus, history):
    """The exact defect, pinned so it cannot come back silently.

    Int32 history ids against a str-keyed table match nothing. No exception, no
    warning — just an empty mapping and a submission that is the baseline.
    """
    row_of_user, matrix = score_fast.user_vectors_from_fixed_history(
        history, corpus, row_of_article=score_fast.article_row_index(corpus, cast=str)
    )
    assert row_of_user == {}, "str keys must NOT match Int32 history ids"
    assert matrix.shape[0] == 0


def test_integer_keyed_index_finds_every_user(corpus, history):
    row_of_user, matrix = score_fast.user_vectors_from_fixed_history(
        history, corpus, row_of_article=score_fast.article_row_index(corpus, cast=int)
    )
    assert set(row_of_user) == {1, 2}
    assert matrix.shape == (2, corpus.dim)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)


def test_the_default_index_is_the_one_that_failed(corpus, history):
    """`user_vectors_from_fixed_history` builds a str-keyed default.

    Documented as a trap rather than fixed by guessing the caller's id type:
    guessing would work here and fail somewhere we have not looked.
    """
    row_of_user, _ = score_fast.user_vectors_from_fixed_history(history, corpus)
    assert row_of_user == {}, "the default is str-keyed; callers must pass their own"


def test_candidate_coverage_can_be_perfect_while_user_coverage_is_zero(corpus, history):
    """Why one guard was not enough.

    This is the state the first run was in: every candidate resolves, no user
    does, and a check on candidates alone reports everything is fine.
    """
    int_keyed = score_fast.article_row_index(corpus, cast=int)
    candidates = pl.Series([10, 11, 12], dtype=pl.Int32)
    assert score_fast.coverage(int_keyed, candidates) == 1.0

    row_of_user, _ = score_fast.user_vectors_from_fixed_history(history, corpus)
    assert len(row_of_user) == 0


# =============================================================================
# 2. The guards that would have caught it
# =============================================================================


def test_low_user_coverage_refuses_to_write(tmp_path, monkeypatch, corpus, history):
    """The check that was missing. It must raise, not warn."""
    monkeypatch.setattr(sp, "load_pieces", lambda *a, **k: (corpus, {}, {}, np.zeros((0, 8))))
    monkeypatch.setattr(sp, "measure_coverage", lambda *a, **k: 1.0)
    monkeypatch.setattr(sp.ebnerd_test, "check_shape", lambda *a, **k: {"rows": 2})
    monkeypatch.setattr(sp.pl, "scan_parquet", lambda *a, **k: pl.LazyFrame({"user_id": [1, 2]}))
    with pytest.raises(sp.CoverageTooLow, match="users"):
        sp.write_ebnerd_semantic(tmp_path)


def test_low_candidate_coverage_refuses_to_write(tmp_path, monkeypatch, corpus):
    monkeypatch.setattr(sp, "load_pieces", lambda *a, **k: (corpus, {}, {1: 0}, np.zeros((1, 8))))
    monkeypatch.setattr(sp, "measure_coverage", lambda *a, **k: 0.0)
    monkeypatch.setattr(sp.ebnerd_test, "check_shape", lambda *a, **k: {"rows": 2})
    with pytest.raises(sp.CoverageTooLow, match="candidate slots"):
        sp.write_ebnerd_semantic(tmp_path)


def test_the_thresholds_leave_no_room_for_a_partial_failure():
    """0.99, not 0.5: the real values are 1.0000 and the failure is 0.0000, so
    anything between is a state we have never seen and should stop on."""
    assert sp.MIN_COVERAGE >= 0.99
    assert sp.MIN_USER_COVERAGE >= 0.99


def test_the_error_names_the_cause_not_just_the_symptom(tmp_path, monkeypatch, corpus):
    """A future reader hitting this needs to know where to look. 'coverage is
    0.0' is a symptom; 'candidates are Int32 and the table keys on str' is the
    bug."""
    monkeypatch.setattr(sp, "load_pieces", lambda *a, **k: (corpus, {}, {1: 0}, np.zeros((1, 8))))
    monkeypatch.setattr(sp, "measure_coverage", lambda *a, **k: 0.0)
    monkeypatch.setattr(sp.ebnerd_test, "check_shape", lambda *a, **k: {"rows": 2})
    with pytest.raises(sp.CoverageTooLow, match="Int32"):
        sp.write_ebnerd_semantic(tmp_path)


# =============================================================================
# 3. Scoring actually changes the ranking
# =============================================================================


def test_semantic_ranking_differs_from_the_article_id_baseline(corpus, history):
    """The end-to-end symptom of the bug: identical output to the baseline.

    A submission that ranks by cosine must not agree with alphabetical ordering
    except by coincidence, so this uses a candidate list whose id order and
    similarity order are deliberately different.
    """
    row_of_article = score_fast.article_row_index(corpus, cast=int)
    row_of_user, matrix = score_fast.user_vectors_from_fixed_history(
        history, corpus, row_of_article=row_of_article
    )
    frame = pl.DataFrame(
        {"impression_id": [1], "user_id": [1], "article_ids_inview": [[10, 11, 12, 13]]},
        schema={
            "impression_id": pl.UInt32,
            "user_id": pl.UInt32,
            "article_ids_inview": pl.List(pl.Int32),
        },
    )
    ranks = score_fast.score_frame(frame, row_of_user, matrix, corpus, row_of_article)
    got = ranks.to_list()[0]

    assert sorted(got) == [1, 2, 3, 4]
    # The baseline would emit [1,2,3,4]: ids already ascending.
    assert got != [1, 2, 3, 4], "semantic ranking collapsed to the article_id order"


def test_a_user_without_a_vector_falls_back_to_the_tie_break(corpus):
    """Not an error: an unknown user is ranked by article_id, exactly like the
    accepted baseline, rather than by an arbitrary order."""
    row_of_article = score_fast.article_row_index(corpus, cast=int)
    frame = pl.DataFrame(
        {"impression_id": [1], "user_id": [999], "article_ids_inview": [[13, 11, 12]]},
        schema={
            "impression_id": pl.UInt32,
            "user_id": pl.UInt32,
            "article_ids_inview": pl.List(pl.Int32),
        },
    )
    ranks = score_fast.score_frame(frame, {}, np.zeros((0, 8)), corpus, row_of_article)
    # ids ascending: 11 -> 1, 12 -> 2, 13 -> 3, in the order given (13, 11, 12).
    assert ranks.to_list()[0] == [3, 1, 2]


# =============================================================================
# 4. Configuration
# =============================================================================


def test_there_is_no_lexical_term(corpus):
    """EB-NeRD is semantic-only because both its corpora selected alpha=0 on the
    validation split. Adding lexical here would override a measurement."""
    source = sp.__doc__ or ""
    assert "semantic only" in source.lower() or "Semantic only" in source
    assert not hasattr(sp, "MIND_ALPHA")
    assert not hasattr(sp, "ALPHA")


def test_the_history_path_is_the_test_split(corpus):
    assert sp.history_path().parts[-2:] == ("test", "history.parquet")
