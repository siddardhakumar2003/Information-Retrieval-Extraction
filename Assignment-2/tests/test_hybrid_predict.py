"""The MIND hybrid submission.

This is assembly: every part it uses is tested elsewhere. What is new — and what
these tests cover — is the wiring, where the failures are silent rather than
loud:

- a blend weight that drifts from the one measured on the validation split
- a submission that is really pure semantic, or really the baseline, because one
  side was silent for most rows and nothing counted it
- a tie-break that differs from the offline one without that being deliberate
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from newsrec import config
from newsrec.eval import hybrid
from newsrec.retrieval import embed, user_vector
from newsrec.submit import hybrid_predict as hp


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
    rng = np.random.default_rng(4)
    v = rng.normal(size=(5, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return embed.ArticleEmbeddings(
        vectors=v, article_ids=["N1", "N2", "N3", "N4", "N5"], spec=_spec()
    )


# =============================================================================
# 1. The blend weight comes from the validation split and must not drift
# =============================================================================


def test_the_blend_weight_matches_what_was_selected_on_val():
    """MIND_ALPHA is a literal so a regenerated report cannot silently change a
    submission. That makes drift possible, so it is asserted instead."""
    measured = hp.alpha_from_report()
    if measured is None:
        pytest.skip("hybrid_ablation.json not built")
    assert hp.MIND_ALPHA == measured, (
        f"submission uses alpha={hp.MIND_ALPHA} but the val selection recorded "
        f"{measured}; one of the two is stale"
    )


def test_the_blend_weight_is_a_real_blend():
    """Not 0 (that is pure semantic) and not 1 (pure BM25, which lost on every
    corpus)."""
    assert 0.0 < hp.MIND_ALPHA < 1.0


def test_scoring_uses_the_shared_blend_not_a_local_copy(corpus):
    """`hybrid.blend` carries the per-impression normalisation. A second
    implementation here would let the submission and the measurement drift."""
    titles = {"N1": ["alpha"], "N2": ["beta"]}
    vocab = {"alpha": 0, "beta": 1}
    idf = np.array([1.0, 1.0])

    class Scorer:
        def score(self, term_ids, weights, rows):
            return np.array([10.0, 0.0][: rows.size], dtype=np.float64)

    scores, _, _ = hp.score_row(
        ["N1"],
        ["N1", "N2"],
        scorer=Scorer(),
        vocab=vocab,
        idf=idf,
        titles=titles,
        embeddings=corpus,
        doc_of={"N1": 0, "N2": 1},
        row_of_article={"N1": 0, "N2": 1},
        alpha=0.5,
    )
    lex = np.array([10.0, 0.0])
    sem = corpus.vectors[[0, 1]] @ user_vector.pool(corpus.vectors[[0]])
    assert scores == pytest.approx(hybrid.blend(lex, sem, 0.5))


# =============================================================================
# 2. Both signals must actually reach the ranking
# =============================================================================


def test_a_row_with_history_uses_both_signals(corpus):
    titles = {"N1": ["alpha"]}
    vocab = {"alpha": 0}
    idf = np.array([1.0])

    class Scorer:
        def score(self, term_ids, weights, rows):
            return np.ones(rows.size)

    _, had_lex, had_sem = hp.score_row(
        ["N1"],
        ["N2", "N3"],
        scorer=Scorer(),
        vocab=vocab,
        idf=idf,
        titles=titles,
        embeddings=corpus,
        doc_of={"N2": 1, "N3": 2},
        row_of_article={"N1": 0, "N2": 1, "N3": 2},
        alpha=hp.MIND_ALPHA,
    )
    assert had_lex and had_sem


def test_a_row_without_history_reports_both_signals_silent(corpus):
    """Cold start. Both sides derive from history, so both go quiet together —
    and the caller counts it rather than letting it pass unnoticed."""

    class Scorer:
        def score(self, term_ids, weights, rows):  # pragma: no cover - not reached
            raise AssertionError("should not be called for an empty query")

    scores, had_lex, had_sem = hp.score_row(
        [],
        ["N1", "N2"],
        scorer=Scorer(),
        vocab={},
        idf=np.array([]),
        titles={},
        embeddings=corpus,
        doc_of={"N1": 0, "N2": 1},
        row_of_article={"N1": 0, "N2": 1},
        alpha=hp.MIND_ALPHA,
    )
    assert not had_lex and not had_sem
    assert scores.tolist() == [0.0, 0.0], "a cold row must fall through to the tie-break"


def test_a_history_of_unknown_articles_is_treated_as_cold(corpus):
    class Scorer:
        def score(self, term_ids, weights, rows):
            return np.ones(rows.size)

    _, had_lex, had_sem = hp.score_row(
        ["N999"],
        ["N1"],
        scorer=Scorer(),
        vocab={},
        idf=np.array([]),
        titles={},
        embeddings=corpus,
        doc_of={"N1": 0},
        row_of_article={"N1": 0},
        alpha=hp.MIND_ALPHA,
    )
    assert not had_lex and not had_sem


def test_user_vector_is_none_rather_than_zeros_for_an_unusable_history(corpus):
    """A zero vector would score every candidate 0.0, which is indistinguishable
    from a genuine cosine of zero."""
    row_of = {a: i for i, a in enumerate(corpus.article_ids)}
    assert hp.user_vector_for([], corpus, row_of) is None
    assert hp.user_vector_for(["N999"], corpus, row_of) is None
    assert hp.user_vector_for(["N1"], corpus, row_of) is not None


def test_the_user_vector_takes_the_most_recent_clicks(corpus):
    """MIND history is oldest-first, so the cap must take the tail. Taking the
    head would build every vector from the user's oldest interests."""
    row_of = {a: i for i, a in enumerate(corpus.article_ids)}
    got = hp.user_vector_for(["N1", "N2", "N5"], corpus, row_of, limit=1)
    assert got == pytest.approx(user_vector.pool(corpus.vectors[[4]]), abs=1e-6)


def test_the_user_vector_is_unit_norm(corpus):
    row_of = {a: i for i, a in enumerate(corpus.article_ids)}
    got = hp.user_vector_for(["N1", "N2"], corpus, row_of)
    assert float(np.linalg.norm(got)) == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# 3. The tie-break
# =============================================================================


def test_ties_break_by_article_id_ascending():
    """No training-period popularity exists for MINDlarge_test, so the offline
    three-key tie-break is deliberately reduced to two."""
    order = hp._order(np.zeros(3), ["N30", "N10", "N20"])
    assert [["N30", "N10", "N20"][i] for i in order] == ["N10", "N20", "N30"]


def test_score_beats_the_tie_break():
    order = hp._order(np.array([0.0, 0.5]), ["N1", "N9"])
    assert list(order) == [1, 0], "a scored candidate must outrank an unscored one"


def test_a_cold_row_ranks_exactly_like_the_article_id_baseline():
    """Named because it is the intended behaviour, not an accident: for a user
    we know nothing about, the submission degrades to the baseline that scored
    0.5187 rather than to an arbitrary order."""
    from newsrec.submit import baseline

    candidates = ["N30", "N10", "N20"]
    hybrid_order = hp._order(np.zeros(3), candidates)
    assert list(hybrid_order) == list(baseline.baseline_order(candidates))


# =============================================================================
# 4. End to end on a hand-built behaviours file
# =============================================================================


@pytest.fixture
def behaviours(tmp_path, monkeypatch, corpus):
    """A two-row file: one warm, one cold."""
    import polars as pl

    src = tmp_path / "behaviors.tsv"
    src.write_text(
        "1\tU1\tt\tN1 N2\tN3 N4 N5\n2\tU2\tt\t\tN3 N4\n",
        encoding="utf-8",
    )
    articles = pl.DataFrame(
        {
            "article_id": corpus.article_ids,
            "title": [f"title {a}" for a in corpus.article_ids],
            "text": [f"title {a} body {a}" for a in corpus.article_ids],
        }
    )
    monkeypatch.setattr(hp.encode_test, "mind_articles", lambda *a, **k: articles)
    monkeypatch.setattr(hp, "_mind_titles", lambda *a, **k: articles.select("article_id", "title"))
    monkeypatch.setattr(hp.embed, "load", lambda *a, **k: corpus)
    return src


def test_the_written_file_has_one_line_per_row_and_validates(behaviours, tmp_path):
    got = hp.write_mind_hybrid(tmp_path, behaviours)
    assert got["lines"] == 2
    assert got["rank_integers"] == 5
    assert got["alpha"] == hp.MIND_ALPHA


def test_the_coverage_counts_are_reported(behaviours, tmp_path):
    """A submission where one signal was silent for most rows is a different
    system from the one measured, and the count is the only way to see it."""
    got = hp.write_mind_hybrid(tmp_path, behaviours)
    assert got["coverage"]["both_silent"] == 1, "the cold row"
    assert got["coverage"]["lexical_silent"] == 1
    assert got["coverage"]["semantic_silent"] == 1


def test_the_archive_holds_a_bare_prediction_txt(behaviours, tmp_path):
    hp.write_mind_hybrid(tmp_path, behaviours)
    with zipfile.ZipFile(tmp_path / "mind_submission.zip") as zf:
        assert zf.namelist() == ["prediction.txt"]


def test_the_hybrid_ranks_differently_from_the_baseline_on_a_warm_row(behaviours, tmp_path):
    """If it did not, the blend would be doing nothing and the submission would
    be the baseline under another name."""
    from newsrec.submit import baseline

    hp.write_mind_hybrid(tmp_path, behaviours)
    hybrid_lines = (tmp_path / "prediction.txt").read_text().splitlines()

    baseline.write_mind_baseline(tmp_path, behaviours)
    baseline_lines = (tmp_path / "prediction.txt").read_text().splitlines()

    assert hybrid_lines[1] == baseline_lines[1], "the cold row must match the baseline"
    assert hybrid_lines[0] != baseline_lines[0], "the warm row must not"


def test_a_row_with_no_candidates_raises(tmp_path, behaviours):
    src = tmp_path / "empty.tsv"
    src.write_text("1\tU1\tt\tN1\t\n", encoding="utf-8")
    with pytest.raises(hp.mind_test.MindTestError, match="no candidates"):
        hp.write_mind_hybrid(tmp_path, src)


def test_a_malformed_row_raises_rather_than_being_skipped(tmp_path, behaviours):
    """A skipped row shortens the file by one line, and the MIND scorer fills a
    missing line rather than rejecting it."""
    src = tmp_path / "short.tsv"
    src.write_text("1\tU1\tt\tN1\n", encoding="utf-8")
    with pytest.raises(hp.mind_test.MindTestError, match="fields"):
        hp.write_mind_hybrid(tmp_path, src)


def test_alpha_zero_reproduces_pure_semantic_ordering(behaviours, tmp_path, corpus):
    """The endpoint check, at submission level rather than in the ablation."""
    hp.write_mind_hybrid(tmp_path, behaviours, alpha=0.0)
    at_zero = (tmp_path / "prediction.txt").read_text()

    row_of = {a: i for i, a in enumerate(corpus.article_ids)}
    vector = hp.user_vector_for(["N1", "N2"], corpus, row_of)
    sem = corpus.vectors[[row_of[a] for a in ["N3", "N4", "N5"]]] @ vector
    from newsrec.submit import format as fmt

    expected = fmt.format_line(1, fmt.ranks_from_scores(hp._order(sem, ["N3", "N4", "N5"]), 3))
    assert at_zero.splitlines()[0] == expected
