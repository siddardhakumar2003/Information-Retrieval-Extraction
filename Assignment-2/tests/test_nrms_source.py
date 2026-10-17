"""A2 Q3 — the bridge from the feature store to the model's tensors.

Separate from `tests/test_nrms.py` (model mechanics) because the risk is
different. Here nothing crashes when it goes wrong: a history row pointing at
the wrong article, or a score returned in a different order than the table it
describes, produces a plausible near-random metric with **no error anywhere**.
`submit/score_fast.py`'s docstring warns about the same class of bug, and A1 hit
it for real — a submission that scored byte-identically to the baseline while
its coverage guard read 1.0000 (`design_rationale.md` §1.21).

So these tests are about alignment: row `i` of a tensor describing impression
`i`, and a score vector matching the row order of the table it came from.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from newsrec import config
from newsrec.rerank import features, nrms

EMBED_DIM = 6


def _embeddings(article_ids: list[str]) -> np.ndarray:
    """Deterministic, distinct vectors — row j is recognisably article j."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((len(article_ids), EMBED_DIM)).astype(np.float32)


@pytest.fixture
def click_history_table() -> pl.DataFrame:
    """Shaped like A2 Q1's `user_click_history_features.parquet`: most recent
    click first, capped at `config.QUERY_HISTORY_LIMIT`."""
    return pl.DataFrame(
        {
            "impression_id": ["i1", "i2", "i3"],
            "user_id": ["u1", "u2", "u3"],
            "recent_article_ids": [
                ["a3", "a1"],  # two clicks, most recent first
                ["a0"],  # one click
                [],  # cold start
            ],
        }
    )


# =============================================================================
# History rows — alignment, ordering and the cap
# =============================================================================


def test_history_rows_align_to_the_impression_ids_they_were_asked_for():
    """Row i must describe impression i of the REQUESTED order, not of the
    table's storage order. Those differ as soon as a sample is drawn."""
    table = pl.DataFrame(
        {
            "impression_id": ["i1", "i2", "i3"],
            "recent_article_ids": [["a1"], ["a2"], ["a3"]],
        }
    )
    row_of = {"a1": 10, "a2": 20, "a3": 30}
    got = features.history_rows_by_impression(table, row_of)

    # Asked for in a different order than stored.
    assert got["i3"].tolist() == [30]
    assert got["i1"].tolist() == [10]
    assert got["i2"].tolist() == [20]


def test_history_rows_preserve_most_recent_first_order(click_history_table):
    """A2 Q1 stores `recent_article_ids` most-recent-first. Reversing it here
    would not raise and would not change any metric much — the user encoder is
    order-blind (see test_nrms.py) — but the *padding* positions would move,
    and a later model that did use order would be silently wrong."""
    row_of = {"a0": 0, "a1": 1, "a3": 3}
    got = features.history_rows_by_impression(click_history_table, row_of)
    assert got["i1"].tolist() == [3, 1], "most-recent-first order was not preserved"


def test_a_cold_start_impression_gets_an_empty_history_not_a_missing_key(
    click_history_table,
):
    """`_history_batch` reads this dict with `.get`, so a missing key and an
    empty array behave the same — but only one of them is a deliberate answer."""
    row_of = {"a0": 0, "a1": 1, "a3": 3}
    got = features.history_rows_by_impression(click_history_table, row_of)
    assert "i3" in got
    assert len(got["i3"]) == 0


def test_an_article_with_no_embedding_row_is_dropped_not_mapped_to_row_zero():
    """A missing id must not silently become row 0 — that would put a real,
    arbitrary article into someone's click history."""
    table = pl.DataFrame(
        {"impression_id": ["i1"], "recent_article_ids": [["known", "unknown", "known2"]]}
    )
    got = features.history_rows_by_impression(table, {"known": 5, "known2": 6})
    assert got["i1"].tolist() == [5, 6]


def test_history_is_capped_at_the_config_limit():
    many = [f"a{j}" for j in range(config.QUERY_HISTORY_LIMIT + 10)]
    table = pl.DataFrame({"impression_id": ["i1"], "recent_article_ids": [many]})
    row_of = {a: j for j, a in enumerate(many)}
    got = features.history_rows_by_impression(table, row_of)
    assert len(got["i1"]) == config.QUERY_HISTORY_LIMIT


# =============================================================================
# The batch tensors
# =============================================================================


def _source(with_categories: bool = False) -> nrms.ListwiseBatchSource:
    article_ids = ["a0", "a1", "a2", "a3", "a4"]
    return nrms.ListwiseBatchSource(
        groups=[("i1", "a0", ["a1", "a2", "a3", "a4"])],
        history_rows={"i1": np.array([3, 1], dtype=np.int64), "i2": np.array([], dtype=np.int64)},
        row_of_article={a: j for j, a in enumerate(article_ids)},
        embeddings=_embeddings(article_ids),
        category_of_row=np.array([0, 1, 1, 2, 0]) if with_categories else None,
        n_categories=3 if with_categories else 0,
    )


def test_the_history_batch_marks_real_clicks_and_pads_the_rest():
    source = _source()
    vectors, mask, categories = nrms._history_batch(source, ["i1", "i2"], "cpu")

    assert vectors.shape == (2, config.QUERY_HISTORY_LIMIT, EMBED_DIM)
    assert mask[0].tolist() == [True, True] + [False] * (config.QUERY_HISTORY_LIMIT - 2)
    assert not mask[1].any(), "the cold-start impression has no real clicks"
    assert categories is None, "no categories were supplied"

    # The real slots must hold the right articles' vectors.
    np.testing.assert_allclose(vectors[0, 0].numpy(), source.embeddings[3])
    np.testing.assert_allclose(vectors[0, 1].numpy(), source.embeddings[1])
    # Padding is zeros, and the mask is what makes that irrelevant.
    assert vectors[0, 2:].abs().max().item() == 0.0


def test_the_candidate_batch_puts_the_clicked_article_first():
    """`listwise_loss` takes column 0 as the correct class and needs no label
    tensor, so this ordering IS the label. Getting it wrong trains the model to
    prefer a non-clicked article and raises nothing."""
    source = _source()
    _, clicked, negatives = source.groups[0]
    candidates, _ = nrms._candidate_batch(source, [[clicked, *negatives]], "cpu")
    np.testing.assert_allclose(
        candidates[0, 0].numpy(), source.embeddings[source.row_of_article[clicked]]
    )
    assert candidates.shape == (1, 1 + config.NRMS_NEG_SAMPLES, EMBED_DIM)


def test_category_indices_are_supplied_only_when_the_source_has_them():
    plain = _source(with_categories=False)
    assert nrms._history_batch(plain, ["i1"], "cpu")[2] is None
    assert nrms._candidate_batch(plain, [["a0", "a1"]], "cpu")[1] is None

    aware = _source(with_categories=True)
    history_categories = nrms._history_batch(aware, ["i1"], "cpu")[2]
    assert history_categories is not None
    # i1's history is rows [3, 1] -> categories [2, 1].
    assert history_categories[0, :2].tolist() == [2, 1]


# =============================================================================
# predict's row-order contract — the alignment bug with no error message
# =============================================================================


def _tiny_model(category_aware: bool = False) -> nrms.NRMSModel:
    torch.manual_seed(0)
    net = nrms.NRMSNet(
        embedding_dim=EMBED_DIM,
        dim=8,
        heads=2,
        attention_dim=4,
        n_categories=3 if category_aware else 0,
        category_dim=2,
        dropout=0.0,
    )
    return nrms.NRMSModel(
        net=net,
        device="cpu",
        category_aware=category_aware,
        n_categories=3 if category_aware else 0,
        embedding_dim=EMBED_DIM,
    )


def _scoring_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": ["i1", "i1", "i1", "i2", "i2"],
            "article_id": ["a0", "a1", "a2", "a3", "a4"],
            "label": [1, 0, 0, 1, 0],
        }
    )


def test_predict_returns_one_score_per_row_in_table_order():
    model = _tiny_model()
    table = _scoring_table()
    scores = nrms.predict(model, _source(), table)
    assert scores.shape == (table.height,)
    assert np.isfinite(scores).all()


def test_predict_follows_the_table_when_its_rows_are_permuted():
    """The test that catches a returned-in-group-order score vector.

    Score the table, then score a row-permuted copy of the same table; each
    (impression, article) pair must keep its own score. A `predict` that
    returned scores in its own internal grouping order would pass
    `test_predict_returns_one_score_per_row_in_table_order` and fail here.
    """
    model = _tiny_model()
    table = _scoring_table()
    source = _source()

    first = nrms.predict(model, source, table)
    by_pair = dict(zip(zip(table["impression_id"], table["article_id"]), first, strict=True))

    shuffled = table[[3, 0, 4, 2, 1]]
    second = nrms.predict(model, source, shuffled)

    for (impression_id, article_id), score in zip(
        zip(shuffled["impression_id"], shuffled["article_id"]), second, strict=True
    ):
        assert score == pytest.approx(by_pair[(impression_id, article_id)], abs=1e-6)


def test_predict_gives_a_cold_start_impression_a_finite_score():
    """i2 has an empty history in `_source`, which is the NaN path."""
    model = _tiny_model()
    scores = nrms.predict(model, _source(), _scoring_table())
    assert np.isfinite(scores).all()


def test_predict_ignores_a_label_column_left_in_the_table():
    """The same contract `tests/test_rerank.py` asserts for GBDT and the MLP: a
    model must not be able to read the answer off the table it is scoring."""
    model = _tiny_model()
    source = _source()
    table = _scoring_table()
    with_label = nrms.predict(model, source, table)
    without = nrms.predict(model, source, table.drop("label"))
    np.testing.assert_allclose(with_label, without)
