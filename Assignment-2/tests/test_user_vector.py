"""Q3.4 — the user vector.

The failure guarded against: a vector that is well-formed, unit-length, and
built from the wrong clicks — a future one, or all of them, or in the wrong
order. Every such vector produces a plausible recall number and no error.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.retrieval import embed, user_vector


def _spec(dim: int = 8) -> config.EncoderSpec:
    return config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )


def _embeddings(article_ids: list[str], dim: int = 8, seed: int = 0):
    """Unit-norm vectors, one per article, deterministic per article_id."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(len(article_ids), dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return embed.ArticleEmbeddings(vectors=v, article_ids=list(article_ids), spec=_spec(dim))


def _impressions(rows):
    return pl.DataFrame(
        rows,
        schema={
            "impression_id": pl.Utf8,
            "user_id": pl.Utf8,
            "impression_time": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


def _history(rows):
    """Full-contract history frame — including `history_rank`.

    Both shapes are exercised deliberately. §1.16 records that 22 passing tests
    missed a 100% failure because every fixture was EB-NeRD-shaped; a
    timestamped-only fixture set here would repeat that exactly.
    """
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.Utf8,
            "article_id": pl.Utf8,
            "click_time": pl.Datetime("us", "UTC"),
            "history_rank": pl.Int32,
        },
        orient="row",
    )


def _t(day: int, hour: int = 12) -> datetime:
    return datetime(2019, 11, day, hour, tzinfo=UTC)


# =============================================================================
# Pooling arithmetic
# =============================================================================


def test_pooled_vector_is_unit_length():
    """A mean of unit vectors is not one — average two 60 degrees apart and you
    get length ~0.87. Unnormalised, a user with varied interests would score
    uniformly lower and it would look like the model serving them badly."""
    e = _embeddings([f"A{i}" for i in range(5)])
    out = user_vector.pool(e.vectors)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_mean_pooling_is_the_direction_of_the_average():
    e = _embeddings(["A", "B", "C"])
    out = user_vector.pool(e.vectors, rule="mean")
    want = e.vectors.mean(axis=0)
    want /= np.linalg.norm(want)
    assert np.allclose(out, want, atol=1e-6)


def test_pooling_one_click_returns_that_article_direction():
    e = _embeddings(["A"])
    assert np.allclose(user_vector.pool(e.vectors), e.vectors[0], atol=1e-6)


def test_recency_weighting_leans_towards_the_most_recent_click():
    """The ablation must actually differ from the headline. If `recency`
    produced the same vector as `mean`, D5's comparison would report a delta of
    zero as though it had measured one."""
    e = _embeddings(["A", "B", "C", "D"])
    mean = user_vector.pool(e.vectors, rule="mean")
    recent = user_vector.pool(e.vectors, rule="recency")
    assert not np.allclose(mean, recent, atol=1e-3)
    # vectors[0] is the most recent, so recency must sit closer to it
    assert float(recent @ e.vectors[0]) > float(mean @ e.vectors[0])


def test_recency_weights_decay_geometrically_and_mean_weights_do_not():
    w = user_vector.pooling_weights(4, rule="recency", decay=0.5)
    assert np.allclose(w, [1.0, 0.5, 0.25, 0.125])
    assert np.allclose(user_vector.pooling_weights(4, rule="mean"), [0.25] * 4)


def test_an_unknown_pooling_rule_is_refused():
    """Q3.4 makes the rule a config value. A typo must fail rather than silently
    fall back to the mean and report an ablation that never ran."""
    with pytest.raises(ValueError, match="unknown pooling rule"):
        user_vector.pooling_weights(3, rule="mena")


def test_empty_history_raises_rather_than_returning_a_zero_vector():
    """The zero vector's cosine with everything is 0/0. Whatever ordering came
    back would be floating-point noise wearing the shape of a result."""
    with pytest.raises(ValueError, match="at least one clicked-article vector"):
        user_vector.pool(np.zeros((0, 8), dtype=np.float32))


def test_exactly_cancelling_vectors_are_refused():
    v = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="zero or non-finite length"):
        user_vector.pool(v)


# =============================================================================
# Q3.4 — the boundary, on both dataset shapes
# =============================================================================


def test_user_embedding_respects_boundary_ebnerd_shaped():
    """A click AFTER the impression must not reach the vector. Manufactured,
    because natural data cannot exercise it."""
    e = _embeddings(["past", "future"])
    imps = _impressions([("i1", "u1", _t(5))])
    hist = _history(
        [
            ("u1", "past", _t(4), None),
            ("u1", "future", _t(6), None),  # after the impression
        ]
    )
    matrix, has = user_vector.user_vectors(imps, hist, e)
    assert has[0]
    assert np.allclose(matrix[0], e.vectors[0], atol=1e-6), "the future click leaked in"


def test_canary_click_at_exactly_the_impression_timestamp_is_excluded():
    """`strictly` before, on a case constructed to be unambiguous.

    A long-standing claim in this project was that real data never puts a click
    and an impression in the same microsecond. Measured, that is false --
    ebnerd_demo has 6,518 such pairs (0.0341%) and ebnerd_small 62,030 (0.0336%).
    This canary is still worth having: it isolates the boundary in a fixture
    where nothing else varies, so a failure names the cause. But it is no longer
    the ONLY way the distinction is observable -- the real-data test
    test_leakage.py::test_user_embedding_respects_boundary also fails under a
    `<=` mutation.
    """
    e = _embeddings(["earlier", "simultaneous"])
    imps = _impressions([("i1", "u1", _t(5))])
    hist = _history(
        [
            ("u1", "earlier", _t(4), None),
            ("u1", "simultaneous", _t(5), None),  # exactly the boundary
        ]
    )
    matrix, has = user_vector.user_vectors(imps, hist, e)
    assert has[0]
    assert np.allclose(matrix[0], e.vectors[0], atol=1e-6)


def test_mind_shaped_history_produces_a_vector():
    """MIND ships no click timestamps. Filtering on a null timestamp excludes
    everything — the bug §1.16 records, which showed up as 100% empty queries
    and would show up here as 100% cold start."""
    e = _embeddings(["a", "b"])
    imps = _impressions([("i1", "u1", _t(5))])
    hist = _history([("u1", "a", None, 0), ("u1", "b", None, 1)])
    matrix, has = user_vector.user_vectors(imps, hist, e)
    assert has[0], "MIND-shaped history produced no user vector"
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0, abs=1e-5)


def test_mind_recency_uses_history_rank_most_recent_first():
    """For MIND "recent" can only mean position. Asserting the direction proves
    rank 1 (the later click) outweighs rank 0."""
    e = _embeddings(["older", "newer"])
    imps = _impressions([("i1", "u1", _t(5))])
    hist = _history([("u1", "older", None, 0), ("u1", "newer", None, 1)])
    matrix, _ = user_vector.user_vectors(imps, hist, e, rule="recency")
    newer, older = e.vectors[1], e.vectors[0]
    assert float(matrix[0] @ newer) > float(matrix[0] @ older)


def test_the_history_cap_is_the_shared_config_value():
    """Same cap as BM25's query. If the semantic side pooled 639 clicks while
    the lexical side used 20, Q3.6 would be partly a comparison of windows."""
    assert user_vector.DEFAULT_LIMIT == config.QUERY_HISTORY_LIMIT

    ids = [f"A{i}" for i in range(30)]
    e = _embeddings(ids)
    # The impression is AFTER every click. The first version of this test put it
    # at day 20 with clicks running to day 30, so the "three most recent" were
    # correctly excluded as future and the assertion failed -- the fixture was
    # wrong, not the cap. Left as a comment because it is the same mistake this
    # module exists to prevent, made while writing its test.
    imps = _impressions([("i1", "u1", _t(30, 23))])
    hist = _history([("u1", a, _t(1 + i), None) for i, a in enumerate(ids)])

    capped, _ = user_vector.user_vectors(imps, hist, e, limit=3)
    # the three most recent are the last three clicked
    want = user_vector.pool(e.vectors[[29, 28, 27]])
    assert np.allclose(capped[0], want, atol=1e-6)


def test_pooling_goes_through_the_shared_selector(monkeypatch):
    """Structural, following test_query.py. The boundary must be applied by the
    one function that owns it, not re-derived here."""
    called = {"n": 0}
    real = user_vector.leakage.select_history_for

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(user_vector.leakage, "select_history_for", spy)
    e = _embeddings(["a"])
    imps = _impressions([("i1", "u1", _t(5))])
    hist = _history([("u1", "a", _t(4), None)])
    user_vector.user_vectors(imps, hist, e)
    assert called["n"] == 1


# =============================================================================
# Cold start, and the mask that reports it
# =============================================================================


def test_a_user_with_no_history_is_masked_not_given_a_vector():
    e = _embeddings(["a"])
    imps = _impressions([("i1", "stranger", _t(5))])
    hist = _history([("someone_else", "a", _t(4), None)])
    matrix, has = user_vector.user_vectors(imps, hist, e)
    assert not has[0]
    assert np.array_equal(matrix[0], np.zeros(8, dtype=np.float32))


def test_rows_stay_aligned_to_their_impressions():
    """The silent catastrophe: user vectors offset by one against impressions.
    Every norm is 1.0 and every count matches."""
    e = _embeddings(["a", "b"])
    imps = _impressions([("i1", "cold", _t(5)), ("i2", "warm", _t(5)), ("i3", "cold2", _t(5))])
    hist = _history([("warm", "b", _t(4), None)])
    matrix, has = user_vector.user_vectors(imps, hist, e)
    assert list(has) == [False, True, False]
    assert np.allclose(matrix[1], e.vectors[1], atol=1e-6)


def test_history_referencing_an_article_outside_the_corpus_is_skipped():
    """Contributing nothing is right; a placeholder would invent an interest."""
    e = _embeddings(["known"])
    imps = _impressions([("i1", "u1", _t(5))])
    hist = _history([("u1", "known", _t(4), None), ("u1", "missing", _t(3), None)])
    matrix, has = user_vector.user_vectors(imps, hist, e)
    assert has[0]
    assert np.allclose(matrix[0], e.vectors[0], atol=1e-6)


def test_every_returned_vector_is_unit_length_or_masked_zero():
    e = _embeddings([f"A{i}" for i in range(6)])
    imps = _impressions([(f"i{i}", f"u{i}", _t(5)) for i in range(4)])
    hist = _history(
        [("u0", "A0", _t(4), None), ("u1", "A1", _t(4), None), ("u1", "A2", _t(3), None)]
    )
    matrix, has = user_vector.user_vectors(imps, hist, e)
    for i, flagged in enumerate(has):
        norm = float(np.linalg.norm(matrix[i]))
        assert norm == pytest.approx(1.0 if flagged else 0.0, abs=1e-5)


def test_is_deterministic():
    e = _embeddings([f"A{i}" for i in range(5)])
    imps = _impressions([("i1", "u1", _t(9))])
    hist = _history([("u1", f"A{i}", _t(1 + i), None) for i in range(5)])
    a, _ = user_vector.user_vectors(imps, hist, e)
    b, _ = user_vector.user_vectors(imps, hist, e)
    assert np.array_equal(a, b)


# =============================================================================
# Against the committed fixture store, both datasets
# =============================================================================


def test_runs_against_the_fixture_store(store_root, dataset):
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id")
    ids = [str(a) for a in articles["article_id"].to_list()]
    e = _embeddings(ids)

    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    history = feature_store.read_table(store_root / "processed" / dataset / "history.parquet")

    matrix, has = user_vector.user_vectors(
        impressions.select("impression_id", "user_id", "impression_time"), history, e
    )
    assert matrix.shape == (len(impressions), 8)
    assert np.isfinite(matrix).all()
    for i, flagged in enumerate(has):
        if flagged:
            assert np.linalg.norm(matrix[i]) == pytest.approx(1.0, abs=1e-5)
