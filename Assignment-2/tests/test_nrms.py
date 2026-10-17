"""A2 Q3 — the NRMS-style re-ranker's mechanics (a2_q3_plan.md).

These tests are about the two things that are easy to get wrong in an attention
model and invisible when wrong: **padding** and **cold start**. A model that
attends over its own padding still trains, still scores, and is simply worse;
a cold-start user with an all-padding history divides by a zero softmax
denominator and produces NaN, which propagates into the metric rather than
raising.

Everything here runs on CPU. Only `nrms.train` needs a GPU, by design
(`rerank/mlp.py:_require_cuda`), so the one test that trains is marked `gpu` and
is excluded from `make test` / `make test-full`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from newsrec import config
from newsrec.rerank import nrms

EMBED_DIM = 8  # tiny stand-in for the real 384; the mechanics do not care


def _encoder(category_aware: bool = False) -> nrms.NewsEncoder:
    return nrms.NewsEncoder(
        embedding_dim=EMBED_DIM,
        dim=16,
        n_categories=5 if category_aware else 0,
        category_dim=4,
        dropout=0.0,
    )


def _user_encoder() -> nrms.UserEncoder:
    return nrms.UserEncoder(dim=16, heads=4, attention_dim=8, dropout=0.0)


# =============================================================================
# AdditiveAttention — the learned weighted average that replaces
# config.USER_POOLING_DECAY
# =============================================================================


def test_additive_attention_weights_sum_to_one_and_ignore_padding():
    torch.manual_seed(0)
    attention = nrms.AdditiveAttention(dim=16, attention_dim=8)
    h = torch.randn(2, 5, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    _, weights = attention(h, mask)

    assert weights.shape == (2, 5)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    # A masked slot must get EXACTLY zero, not merely a small share. Masking
    # after the softmax instead of before leaves padding with probability mass
    # and renormalises nothing.
    assert weights[0, 3:].abs().max().item() == 0.0
    assert weights[1, 1:].abs().max().item() == 0.0


def test_additive_attention_is_not_a_uniform_mean():
    """If the weights were constant this would be A1's mean pool wearing an
    attention costume, and every other test here would still pass."""
    torch.manual_seed(0)
    attention = nrms.AdditiveAttention(dim=16, attention_dim=8)
    h = torch.randn(1, 6, 16) * 3.0
    _, weights = attention(h, torch.ones(1, 6, dtype=torch.bool))
    assert weights.std().item() > 1e-6, "attention weights are uniform"


# =============================================================================
# Padding — the silent degrader
# =============================================================================


def test_padding_positions_cannot_change_the_user_vector():
    """Extend a history with padding and the user vector must not move.

    This is the bug that costs accuracy without ever raising: the attention
    happily attends to whatever the padded slots contain.
    """
    torch.manual_seed(0)
    encoder = _user_encoder()
    encoder.eval()

    real = torch.randn(1, 3, 16)
    mask_short = torch.ones(1, 3, dtype=torch.bool)

    # Same three clicks, then two padded slots holding deliberately large junk.
    padded = torch.cat([real, torch.full((1, 2, 16), 99.0)], dim=1)
    mask_long = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        short = encoder(real, mask_short)
        long = encoder(padded, mask_long)

    torch.testing.assert_close(short, long, rtol=1e-5, atol=1e-6)


def test_a_cold_user_produces_a_finite_user_vector():
    """A user with no prior clicks has an all-padding history, so every
    attention weight is masked and the softmax denominator is zero. Left alone
    that is NaN, which reaches the score and then the reported metric."""
    torch.manual_seed(0)
    encoder = _user_encoder()
    encoder.eval()
    with torch.no_grad():
        vector = encoder(torch.zeros(1, 4, 16), torch.zeros(1, 4, dtype=torch.bool))
    assert torch.isfinite(vector).all(), "cold-start user produced NaN or inf"


def test_the_user_encoder_is_content_sensitive_not_a_uniform_mean():
    """The actual difference between this and A1's user vector.

    A1's `retrieval/user_vector.py` runs with `config.USER_POOLING = "mean"`:
    every past click gets weight 1/n, whatever it is. An attentive encoder's
    weights depend on the clicks' own content, so changing one click changes how
    much the others count. A mean pool would pass every other test in this file
    and fail this one.
    """
    torch.manual_seed(0)
    encoder = _user_encoder()
    encoder.eval()
    h = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)

    louder = h.clone()
    louder[0, 0] = louder[0, 0] * 5.0

    with torch.no_grad():
        original = encoder(h, mask)
        changed = encoder(louder, mask)

    assert not torch.allclose(original, changed, atol=1e-4), (
        "changing a click's content did not change the user vector — "
        "this is a fixed-weight pool, not attention"
    )


def test_the_user_encoder_discards_click_ORDER_and_that_is_the_architecture():
    """Measured 2026-09-16, and **not** a bug to fix.

    Multi-head self-attention with no positional encoding is a set operation,
    and so is additive-attention pooling. Published NRMS adds no position
    signal to the click history, so its user encoder is permutation-invariant
    by construction: reversing a user's click order gives a bit-identical user
    vector.

    This is pinned rather than merely noted because it is surprising, and
    because someone reading `test_..._is_content_sensitive...` above could
    reasonably assume order matters too and "fix" the model by adding
    positional encoding — which would stop it being a reproduction.

    The consequence is worth stating in the design note: A1's *other* pooling
    option (`"recency"`, a `0.9 ** i` decay by position) uses order and this
    does not. Attention buys content-awareness and **gives up** the recency
    signal. In a domain where articles decay in hours, that is a real trade,
    not a free upgrade. The headline baseline happens to use `"mean"`, which is
    also order-blind, so the comparison stays apples-to-apples — but a
    recency-pooled baseline would be testing something this model cannot do.
    """
    torch.manual_seed(0)
    encoder = _user_encoder()
    encoder.eval()
    h = torch.randn(1, 6, 16)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        original = encoder(h, mask)
        reversed_ = encoder(h.flip(dims=[1]), mask)
    torch.testing.assert_close(original, reversed_, rtol=1e-5, atol=1e-6)


# =============================================================================
# The ONE principled change (A2.Q3.2)
# =============================================================================


def test_the_variant_differs_from_the_baseline_in_one_declared_way():
    """ "One principled change" has to be checkable, not asserted in prose.

    The category-aware encoder must add the category embedding and its
    projection and NOTHING else — any other extra parameter means the ablation
    is measuring more than one thing.
    """
    plain = {name for name, _ in _encoder(category_aware=False).named_parameters()}
    aware = {name for name, _ in _encoder(category_aware=True).named_parameters()}

    assert plain <= aware, f"the variant dropped parameters: {sorted(plain - aware)}"
    added = aware - plain
    assert added, "the category-aware encoder is identical to the baseline"
    for name in added:
        assert "category" in name, f"unexpected extra parameter {name!r} in the variant"


def test_the_baseline_encoder_has_no_category_parameters():
    names = [name for name, _ in _encoder(category_aware=False).named_parameters()]
    assert not any("category" in n for n in names)


def test_category_ids_change_the_news_vector_only_when_category_aware():
    torch.manual_seed(0)
    vectors = torch.randn(1, 3, EMBED_DIM)
    cats_a = torch.zeros(1, 3, dtype=torch.long)
    cats_b = torch.ones(1, 3, dtype=torch.long)

    plain = _encoder(category_aware=False)
    plain.eval()
    with torch.no_grad():
        assert torch.allclose(plain(vectors, cats_a), plain(vectors, cats_b)), (
            "the baseline encoder is reading the category it is not supposed to see"
        )

    aware = _encoder(category_aware=True)
    aware.eval()
    with torch.no_grad():
        assert not torch.allclose(aware(vectors, cats_a), aware(vectors, cats_b)), (
            "the category-aware encoder ignores the category"
        )


# =============================================================================
# The listwise loss — the second half of the reproduction
# =============================================================================


def test_listwise_loss_is_invariant_to_a_constant_score_shift():
    """A softmax property. The obvious implementation bug — reducing over the
    wrong axis — breaks it, and otherwise produces a number that looks like a
    loss and trains to something wrong."""
    scores = torch.tensor([[2.0, 0.5, -1.0, 0.0, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5]])
    base = nrms.listwise_loss(scores)
    shifted = nrms.listwise_loss(scores + 7.25)
    torch.testing.assert_close(base, shifted, rtol=1e-5, atol=1e-6)


def test_listwise_loss_is_lowest_when_the_clicked_candidate_scores_highest():
    """Column 0 is the clicked candidate by construction (a2_q3_plan.md)."""
    good = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])
    assert nrms.listwise_loss(good).item() < nrms.listwise_loss(bad).item()


def test_listwise_loss_of_a_tie_is_log_of_the_list_length():
    """All-equal scores means a uniform softmax, so the loss is log(C). A
    closed form worth asserting: it pins the normalisation constant."""
    scores = torch.zeros(1, 5)
    expected = float(np.log(5))
    assert nrms.listwise_loss(scores).item() == pytest.approx(expected, abs=1e-6)


# =============================================================================
# Negative sampling
# =============================================================================


def _one_impression_table(n_candidates: int, n_clicks: int = 1) -> pl.DataFrame:
    labels = [1] * n_clicks + [0] * (n_candidates - n_clicks)
    return pl.DataFrame(
        {
            "impression_id": ["i1"] * n_candidates,
            "article_id": [f"a{j}" for j in range(n_candidates)],
            "label": labels,
        }
    )


def test_negative_sampling_never_draws_the_clicked_article():
    table = _one_impression_table(n_candidates=12)
    groups, padded = nrms.build_listwise_groups(table, seed=config.GLOBAL_SEED)
    assert len(groups) == 1
    positive, negatives = groups[0][1], groups[0][2]
    assert positive == "a0"
    assert "a0" not in negatives
    assert len(negatives) == config.NRMS_NEG_SAMPLES
    assert padded == 0


def test_padded_negatives_are_counted_not_hidden():
    """An impression with fewer non-clicked candidates than NRMS_NEG_SAMPLES is
    sampled with replacement. That is a measurement about the data — EB-NeRD's
    median impression has 9 candidates against MIND's 23 — so it is reported,
    not absorbed."""
    table = _one_impression_table(n_candidates=3)  # 1 click, 2 negatives, need 4
    groups, padded = nrms.build_listwise_groups(table, seed=config.GLOBAL_SEED)
    assert len(groups[0][2]) == config.NRMS_NEG_SAMPLES, "the list must still be full width"
    assert padded == 1, "the shortfall was not counted"


def test_an_impression_with_no_click_is_skipped():
    table = _one_impression_table(n_candidates=6, n_clicks=0)
    groups, _ = nrms.build_listwise_groups(table, seed=config.GLOBAL_SEED)
    assert groups == []


def test_negative_sampling_is_reproducible_under_a_fixed_seed():
    table = _one_impression_table(n_candidates=20)
    first, _ = nrms.build_listwise_groups(table, seed=123)
    second, _ = nrms.build_listwise_groups(table, seed=123)
    other, _ = nrms.build_listwise_groups(table, seed=124)
    assert first[0][2] == second[0][2]
    assert first[0][2] != other[0][2], "the seed is not being used"


# =============================================================================
# Constants come from config, not from literals at the call site
# =============================================================================


def test_history_length_comes_from_config_not_a_literal():
    """A1's query construction, A2 Q1's click-history table and this model must
    all mean the same twenty clicks. `nrms.py` must not re-declare it."""
    text = Path(nrms.__file__ or "").read_text(encoding="utf-8")
    assert "QUERY_HISTORY_LIMIT" in text
    assert "history_length" not in text or "config.QUERY_HISTORY_LIMIT" in text


def test_heads_divide_the_model_width():
    """torch's MultiheadAttention requires it; a bad pair fails at construction
    time with a message that does not mention config."""
    assert config.NRMS_DIM % config.NRMS_HEADS == 0


# =============================================================================
# A source with no listwise groups must still be usable as a val source
# =============================================================================


def test_a_source_with_no_groups_is_not_treated_as_absent():
    """The bug this file exists to prevent a repeat of, found 2026-09-16.

    `ListwiseBatchSource.__len__` returns the number of *training* groups, and a
    val/test source is built with `with_groups=False` — so it is legitimately
    empty by design, and therefore **falsy**. `train` selected its evaluation
    source with `val_rows or train_rows`, which silently substituted the
    TRAINING source: every val impression then had no history entry, so every
    val score collapsed to the cold-start constant, and early stopping ran on a
    metric that was identical for every model.

    Measured: `train()` reported `best_val_ndcg@10 = 0.434698` for both ablation
    variants, to six decimal places, while the models it returned actually
    scored 0.470 and 0.508. Nothing raised. The model selection was noise.

    `val_rows` is now a required argument, so there is no fallback to substitute.
    """
    empty = nrms.ListwiseBatchSource(groups=[], history_rows={"i1": np.array([1, 2])})
    assert empty is not None
    assert not empty, "a no-groups source is falsy — that is the trap"
    assert len(empty) == 0
    assert empty.history_rows, "but it still carries the histories that matter"


def test_train_seeds_torch_so_the_ablation_is_reproducible():
    """A graded delta whose sign is not reproducible is not a measurement.

    Measured: without seeding torch, `nrms_category_aware` minus
    `nrms_reproduction` on `mind_small` came out as -0.0002 on one run and
    +0.0047 on the next. Both were "not established", so the conclusion held —
    but Q1.13 already commits this repository to byte-identical rebuilds, and
    the sign of the headline comparison is not something to leave to the global
    RNG's state.
    """
    text = Path(nrms.__file__ or "").read_text(encoding="utf-8")
    assert "seeding.seed_everything()" in text
    assert "np.random.default_rng(config.GLOBAL_SEED)" in text


def test_train_requires_an_explicit_val_source():
    """No default, so `or` can never pick the wrong one again."""
    import inspect

    parameter = inspect.signature(nrms.train).parameters["val_rows"]
    assert parameter.default is inspect.Parameter.empty, (
        "val_rows has a default again; a falsy-but-present source can be "
        "silently replaced by the training source"
    )


# =============================================================================
# The GPU-only contract, checked without a GPU
# =============================================================================


def test_nrms_requires_cuda_and_does_not_fall_back_to_cpu(monkeypatch):
    """Same contract as `rerank/mlp.py`, and the same function enforcing it —
    one definition of the rule, not two that can drift."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        nrms.train(
            train_rows=nrms.ListwiseBatchSource.empty(),
            val_table=pl.DataFrame(),
            val_rows=nrms.ListwiseBatchSource.empty(),
            category_aware=False,
        )


# =============================================================================
# One real training step (needs a GPU)
# =============================================================================


@pytest.mark.gpu
def test_one_training_step_reduces_the_loss_on_a_memorisable_batch():
    """Not a quality claim — a wiring claim. If the gradient does not reach the
    encoders, the loss stays flat and every downstream number is noise."""
    torch.manual_seed(0)
    net = nrms.NRMSNet(
        embedding_dim=EMBED_DIM,
        dim=16,
        heads=4,
        attention_dim=8,
        n_categories=0,
        category_dim=4,
        dropout=0.0,
    ).cuda()
    history = torch.randn(4, 3, EMBED_DIM, device="cuda")
    mask = torch.ones(4, 3, dtype=torch.bool, device="cuda")
    candidates = torch.randn(4, 5, EMBED_DIM, device="cuda")

    optimiser = torch.optim.Adam(net.parameters(), lr=0.05)
    first = None
    for _ in range(20):
        optimiser.zero_grad()
        scores = net(history, mask, candidates)
        loss = nrms.listwise_loss(scores)
        loss.backward()
        optimiser.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first, f"loss did not fall: {first} -> {loss.item()}"
