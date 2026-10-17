"""Q4.4 — diversity, novelty and coverage.

The failure guarded against is a beyond-accuracy number that is arithmetically
fine and measures the wrong thing: a diversity that counts self-pairs, a novelty
that silently drops the 80% of articles unseen in training, a coverage whose
denominator is the candidate pool rather than the catalogue. Each of those
produces a plausible number, which is why each has a test with a value computed
by hand rather than a smoke test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from newsrec import config
from newsrec.eval import beyond_accuracy as ba
from newsrec.retrieval import embed


def _spec(dim: int = 4) -> config.EncoderSpec:
    return config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )


@pytest.fixture
def unit_corpus() -> embed.ArticleEmbeddings:
    """Four articles: two identical, one orthogonal, one opposite.

    Hand-chosen so every pairwise cosine is 0, 1 or -1 and the expected
    diversity of any subset can be written down.
    """
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # a
            [1.0, 0.0, 0.0, 0.0],  # b — identical to a
            [0.0, 1.0, 0.0, 0.0],  # c — orthogonal
            [-1.0, 0.0, 0.0, 0.0],  # d — opposite to a
        ],
        dtype=np.float32,
    )
    return embed.ArticleEmbeddings(vectors=vectors, article_ids=["a", "b", "c", "d"], spec=_spec())


# -----------------------------------------------------------------------------
# Diversity
# -----------------------------------------------------------------------------


def test_two_identical_articles_have_zero_diversity(unit_corpus):
    assert ba.diversity(["a", "b"], unit_corpus) == pytest.approx(0.0)


def test_two_orthogonal_articles_have_diversity_one(unit_corpus):
    assert ba.diversity(["a", "c"], unit_corpus) == pytest.approx(1.0)


def test_two_opposite_articles_have_diversity_two(unit_corpus):
    # cosine distance is 1 - cos, so opposite vectors give 2, not 1. The metric
    # is bounded by [0, 2] and the report must not be read as a fraction.
    assert ba.diversity(["a", "d"], unit_corpus) == pytest.approx(2.0)


def test_diversity_averages_unordered_pairs_and_excludes_self_pairs(unit_corpus):
    """Three articles give three pairs, not nine and not six.

    Including the self-pairs (similarity 1, distance 0) would divide by 6 and
    pull the answer toward zero; counting each pair twice would not change the
    mean, so only the self-pair error is visible here — which is the one that
    actually happens.
    """
    # pairs: (a,b)=0, (a,c)=1, (b,c)=1  -> mean 2/3
    assert ba.diversity(["a", "b", "c"], unit_corpus) == pytest.approx(2 / 3)


def test_diversity_is_undefined_for_fewer_than_two_embedded_articles(unit_corpus):
    assert ba.diversity([], unit_corpus) is None
    assert ba.diversity(["a"], unit_corpus) is None


def test_an_article_missing_from_the_embedding_table_is_skipped_not_zeroed(unit_corpus):
    """A zero row would read as maximally distant and inflate diversity."""
    assert ba.diversity(["a", "b", "absent"], unit_corpus) == pytest.approx(
        ba.diversity(["a", "b"], unit_corpus)
    )
    # And a single real article plus a missing one is still undefined.
    assert ba.diversity(["a", "absent"], unit_corpus) is None


def test_a_prebuilt_row_index_gives_the_same_answer_as_building_one(unit_corpus):
    row_of = ba.row_index(unit_corpus)
    assert ba.diversity(["a", "c", "d"], unit_corpus, row_of=row_of) == pytest.approx(
        ba.diversity(["a", "c", "d"], unit_corpus)
    )


# -----------------------------------------------------------------------------
# Novelty
# -----------------------------------------------------------------------------


def test_smoothed_probabilities_add_one_to_every_count_and_include_the_catalogue():
    popularity = {"a": 3, "b": 1}
    p, p_unseen = ba.smoothed_probabilities(popularity, catalogue_size=10)
    # total clicks 4, V 10 -> denominator 14
    assert p["a"] == pytest.approx(4 / 14)
    assert p["b"] == pytest.approx(2 / 14)
    assert p_unseen == pytest.approx(1 / 14)


def test_an_article_unseen_in_training_is_finite_and_maximally_novel():
    """The measured case: 68-80% of candidates were never clicked in training.

    Unsmoothed, -log2(0) is infinity and the average of any list containing one
    is infinity — so this is the test that would have caught shipping the
    protocol's literal definition.
    """
    # 3 training clicks and a 10-article catalogue -> denominator 13.
    p, p_unseen = ba.smoothed_probabilities({"a": 3}, catalogue_size=10)
    value = ba.novelty(["unseen"], p, p_unseen)
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log2(1 / 13))
    # And nothing seen in training can beat it.
    assert value > ba.novelty(["a"], p, p_unseen)


def test_a_more_popular_article_is_less_novel():
    p, p_unseen = ba.smoothed_probabilities({"hit": 100, "quiet": 1}, catalogue_size=50)
    assert ba.novelty(["hit"], p, p_unseen) < ba.novelty(["quiet"], p, p_unseen)


def test_novelty_is_the_mean_over_the_list():
    p, p_unseen = ba.smoothed_probabilities({"a": 3, "b": 1}, catalogue_size=10)
    expected = (-math.log2(4 / 14) - math.log2(2 / 14)) / 2
    assert ba.novelty(["a", "b"], p, p_unseen) == pytest.approx(expected)


def test_novelty_of_an_empty_list_is_undefined_rather_than_zero():
    p, p_unseen = ba.smoothed_probabilities({"a": 1}, catalogue_size=5)
    assert ba.novelty([], p, p_unseen) is None


def test_novelty_refuses_an_unsmoothed_zero_probability():
    """The canary for skipping the smoothing step.

    If a caller ever hands in raw frequencies, the right outcome is a loud
    failure — not an infinite mean that a report would print as `Infinity`.
    """
    with pytest.raises(ValueError, match="positive"):
        ba.novelty(["a"], {"a": 0.0}, p_unseen=0.0)


def test_novelty_unseen_fraction_all_seen_is_zero():
    assert ba.novelty_unseen_fraction(["a", "b"], {"a": 3, "b": 1}) == 0.0


def test_novelty_unseen_fraction_all_unseen_is_one():
    assert ba.novelty_unseen_fraction(["a", "b"], {}) == 1.0


def test_novelty_unseen_fraction_is_the_mean_over_the_list():
    # "a" seen, "b" and "c" unseen -> 2 of 3 unseen.
    assert ba.novelty_unseen_fraction(["a", "b", "c"], {"a": 5}) == pytest.approx(2 / 3)


def test_novelty_unseen_fraction_of_an_empty_list_is_undefined_rather_than_zero():
    assert ba.novelty_unseen_fraction([], {"a": 1}) is None


# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------


def test_coverage_denominator_is_the_catalogue_not_the_candidates():
    """Five distinct articles recommended out of a 100-article catalogue.

    Dividing by the number recommended would give 1.0 for any run at all.
    """
    lists = [["a", "b"], ["b", "c"], ["d", "e"]]
    assert ba.coverage(lists, catalogue_size=100) == pytest.approx(0.05)


def test_coverage_counts_each_article_once_however_often_it_is_recommended():
    lists = [["a"], ["a"], ["a"], ["a"]]
    assert ba.coverage(lists, catalogue_size=10) == pytest.approx(0.1)


def test_coverage_of_an_empty_run_is_zero():
    assert ba.coverage([], catalogue_size=10) == 0.0


def test_coverage_ci_reports_the_actual_coverage_not_the_resample_mean():
    """A bootstrap draw misses ~37% of impressions, so resampled unions are
    biased low. Reporting their mean as the headline would understate coverage
    systematically — a bias that grows with how many impressions there are.
    """
    lists = [[f"a{i}"] for i in range(200)]
    got = ba.coverage_ci(lists, catalogue_size=1000, n_resamples=50)
    assert got["mean"] == pytest.approx(ba.coverage(lists, 1000))
    assert got["mean"] == pytest.approx(0.2)
    # The interval sits below the point estimate, which is the bias made visible.
    assert got["ci_high"] <= got["mean"]
    assert got["n"] == 200


def test_coverage_ci_has_the_same_shape_as_every_other_metric():
    got = ba.coverage_ci([["a"], ["b"]], catalogue_size=10, n_resamples=20)
    assert set(got) == {"mean", "ci_low", "ci_high", "n"}


def test_coverage_ci_is_seeded_and_identical_across_runs():
    lists = [[f"a{i % 30}"] for i in range(120)]
    first = ba.coverage_ci(lists, catalogue_size=200, n_resamples=64)
    second = ba.coverage_ci(lists, catalogue_size=200, n_resamples=64)
    assert first == second


def test_coverage_ci_of_a_single_impression_collapses_rather_than_faking_spread():
    got = ba.coverage_ci([["a", "b"]], catalogue_size=10)
    assert got == {"mean": 0.2, "ci_low": 0.2, "ci_high": 0.2, "n": 1}


def test_coverage_ci_of_no_impressions_is_nan_not_zero():
    got = ba.coverage_ci([], catalogue_size=10)
    assert got["n"] == 0
    assert math.isnan(got["mean"])
