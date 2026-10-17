"""A2 Q3 — an NRMS-style re-ranker: learned attention over the click history.

Design: `agent_docs/a2_q3_plan.md`. Read that file's "Background" section if the
architecture is not already familiar; this docstring assumes it.

## What this reproduces, and what it deliberately does not

NRMS (Wu et al., 2019) builds two vectors and scores their dot product: a *news*
vector per article, and a *user* vector from the articles the user clicked
before. Both are built with the same two mechanisms — multi-head self-attention
so the items weigh each other, then additive attention to pool them into one
vector.

**The user encoder here is the real thing.** It is the part carrying NRMS's
claim, and the claim is specifically against what Assignment 1 already built:
`retrieval/user_vector.py` pools past clicks with a *fixed* rule — a mean, or a
geometric decay where click `i` back gets weight `config.USER_POOLING_DECAY ** i`.
That rule weights by position only. It cannot notice that a user's nine football
clicks matter for a football candidate and their one recipe click does not, and
it is identical for every user. This module replaces it with weights the model
learns.

## What the attention does and does not buy — measured, not assumed

Found by running the tests, 2026-09-16, and it changes what this model can
claim:

**The user encoder is content-sensitive.** Change what one past click *is* and
the weights on all the others move. A mean pool cannot do that — it gives every
click `1/n` regardless.

**The user encoder is completely blind to click ORDER.** Multi-head
self-attention with no positional encoding is a set operation, and so is
additive-attention pooling. Published NRMS adds no position signal to the click
history, so reversing a user's clicks produces a bit-identical user vector.
`tests/test_nrms.py::test_the_user_encoder_discards_click_ORDER_and_that_is_the_architecture`
pins this so nobody "fixes" it by adding positional encoding, which would stop
this being a reproduction.

So attention is **not** a strict upgrade over A1's pooling. A1 shipped two
rules: `"mean"` (order-blind, fixed weights) and `"recency"` (a `0.9 ** i` decay
that uses order and nothing else). This model generalises the first and cannot
express the second. In a domain where articles decay within hours, giving up the
recency signal to gain content-awareness is a real trade.

It happens not to muddy the comparison here, because `config.USER_POOLING` is
`"mean"`, so the retrieval baseline this is measured against is order-blind too.
But a recency-pooled baseline would be testing something this architecture
structurally cannot do, and that is worth knowing before reading the result.

**The news encoder is deliberately NOT the paper's.** Published NRMS embeds a
title's words and runs word-level self-attention over them. Here, an article's
vector is the 384-dimensional sentence embedding A1 already computed
(`artifacts/embeddings/<dataset>/`, multilingual-e5-small, pinned by
`tests/test_encoder_pin.py`) put through one learned linear projection.

Three reasons, all in `a2_q3_plan.md` and all repeated in the design note
because a substituted component is not a detail:

1. A word-level encoder needs a corpus vocabulary, a document-frequency cut, an
   out-of-vocabulary token, a padded token matrix and a second set of padding
   masks. That is tokenisation engineering; the recommendation idea under test
   is untouched by it.
2. A learned word-embedding table over an English and a Danish corpus, plus
   token sequences for 120,000 articles, does not sit comfortably in this
   machine's 4096 MiB of VRAM beside the activations and the optimiser state.
3. It keeps the comparison against `rerank/mlp.py` a comparison of
   *architecture*. If this model also got raw text, a win would be ambiguous
   between "attention helps" and "more input information helps".

The reference hyperparameters are **not** known to us — see OQ-12. The values in
`config.py`'s A2 Q3 block are our declared choices, labelled as such.

## Training: listwise, which is the other half of the reproduction

`rerank/mlp.py` trains pointwise: one row, one binary label, and the loss never
compares two candidates from the same impression. Its docstring says so
explicitly, and says it is not reproducing NRMS's procedure.

This module trains listwise, as NRMS does: the clicked article plus
`config.NRMS_NEG_SAMPLES` non-clicked articles **from the same impression** are
scored together, and a softmax cross-entropy asks the model to pick the clicked
one. The model optimises the task rather than a proxy for it.

Same features, same data, pointwise against listwise — that contrast is much of
what Q3 has to say, and it costs nothing because both models exist.

## Two failure modes this module is shaped around

Both were designed for rather than discovered, because both are invisible when
they happen:

**Attending over padding.** Most users have fewer than
`config.QUERY_HISTORY_LIMIT` prior clicks, so a history is padded. If the mask
is applied after the softmax rather than before it, the padded slots keep a
share of the probability mass and the pooled vector is polluted. The model still
trains and still scores; it is just worse.
`tests/test_nrms.py::test_padding_positions_cannot_change_the_user_vector`.

**Cold start.** A user with zero prior clicks has an all-padding history, so
every weight is masked and the softmax denominator is zero — `NaN`, which
propagates into the score and then into the reported metric instead of raising.
A learned fallback vector covers that case.

## Leakage

None added. The history a model sees comes from A2 Q1's
`user_click_history_features.recent_article_ids`, which is already strictly
prior to the impression, already capped at `config.QUERY_HISTORY_LIMIT`, and
already gated at build time by `behaviour_features.check`. This module gathers
embedding rows for those ids and defines no second notion of "before".

That is deliberate: A2 Q1 measured what happens when `leakage.select_history_for`
is reused outside the bound its own docstring states — a
`users x impressions x clicks` join that materialised ~184,000,000 rows and
OOM-killed the build (`design_rationale.md` §5.1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import beyond_accuracy as ba
from newsrec.eval import metrics
from newsrec.rerank import features, seeding
from newsrec.rerank.mlp import _require_cuda
from newsrec.retrieval import embed

#: Where a padded candidate's score is sent so it cannot win a ranking. Finite
#: rather than -inf: a row of all -inf in a softmax is NaN, and padded rows do
#: occur (an impression with fewer candidates than the batch's widest).
PADDED_CANDIDATE_SCORE = -1.0e9


class AdditiveAttention(nn.Module):
    """A weighted average whose weights the model learns.

    `weights = softmax(tanh(W h + b) @ q)`, then `sum(weights * h)`. `W` and the
    query vector `q` are the learned parameters.

    This is the layer that replaces `config.USER_POOLING_DECAY`. The decay says
    "the click `i` steps back gets weight `0.9 ** i`", for every user and every
    candidate. This says "the model decides, from what the clicks are".

    The mask is applied **before** the softmax. Applying it after leaves padded
    slots holding probability mass that nothing renormalises away.
    """

    def __init__(self, dim: int, attention_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(dim, attention_dim)
        self.query = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`h` is [batch, length, dim]; `mask` is [batch, length], True = real.

        Returns `(pooled [batch, dim], weights [batch, length])`. The weights are
        returned so a test can assert they sum to one and are exactly zero on
        padding — a property that is otherwise only visible in the output.
        """
        logits = self.query(torch.tanh(self.project(h))).squeeze(-1)

        if mask is None:
            return (torch.softmax(logits, dim=1).unsqueeze(-1) * h).sum(dim=1), torch.softmax(
                logits, dim=1
            )

        # A row with no real entries at all (a cold-start user) would make the
        # softmax denominator zero and the result NaN. Softmax over the
        # unmasked logits first, then zero the padding and renormalise by a
        # clamped sum — which leaves an all-padding row as exactly zero weights
        # and a zero pooled vector, for the caller to replace.
        weights = torch.softmax(logits.masked_fill(~mask, -1.0e9), dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0e-12)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        return pooled, weights


class NewsEncoder(nn.Module):
    """One article -> one vector.

    A linear projection of the pinned sentence embedding, optionally with a
    learned category embedding concatenated in and projected back to `dim`.

    **That category embedding is A2 Q3's ONE principled change** (A2.Q3.2), and
    it is reached by a single constructor argument (`n_categories > 0`) so the
    two ablation variants differ by exactly one thing. A2 Q2's feature table
    already had `category_match`, but only as a 0/1 flag — "is the candidate's
    category one the user recently clicked" — which throws away which category,
    how much of the user's history sits in it, and whether two categories are
    related at all.
    """

    def __init__(
        self,
        embedding_dim: int,
        dim: int,
        n_categories: int = 0,
        category_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.category_aware = n_categories > 0
        self.project = nn.Linear(embedding_dim, dim)
        self.dropout = nn.Dropout(dropout)
        if self.category_aware:
            self.category_embedding = nn.Embedding(n_categories, category_dim)
            self.category_project = nn.Linear(dim + category_dim, dim)

    def forward(self, vectors: torch.Tensor, categories: torch.Tensor | None = None):
        """`vectors` is [..., embedding_dim]; `categories` is [...] of category
        row indices. A baseline encoder ignores `categories` even when given
        them, which a test asserts — otherwise the ablation is not isolating
        the change."""
        out = self.dropout(self.project(vectors))
        if not self.category_aware:
            return out
        if categories is None:
            raise ValueError("a category-aware NewsEncoder needs category indices")
        return self.category_project(torch.cat([out, self.category_embedding(categories)], dim=-1))


class UserEncoder(nn.Module):
    """The user's past clicks -> one interest vector.

    Multi-head self-attention so the clicks weigh each other, then
    `AdditiveAttention` to pool them. This is NRMS's mechanism and the reason
    this question exists.

    A user with no prior clicks gets `cold_start`, a learned vector, rather than
    the zero (and, without the clamp in `AdditiveAttention`, NaN) that an
    all-padding history would otherwise produce.
    """

    def __init__(self, dim: int, heads: int, attention_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"NRMS_HEADS={heads} must divide NRMS_DIM={dim}")
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.pool = AdditiveAttention(dim, attention_dim)
        self.cold_start = nn.Parameter(torch.zeros(dim))

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`history` is [batch, length, dim]; `mask` is [batch, length]."""
        # A fully-padded row makes MultiheadAttention emit NaN for that row
        # regardless of the mask, so tell it every slot is real and rely on the
        # pooling mask plus the cold-start substitution below. The attended
        # values for those rows are then discarded, not used.
        any_real = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~any_real] = True

        attended, _ = self.attention(
            history, history, history, key_padding_mask=~safe_mask, need_weights=False
        )
        pooled, _ = self.pool(attended, mask)
        return torch.where(any_real.unsqueeze(-1), pooled, self.cold_start.expand_as(pooled))


class NRMSNet(nn.Module):
    """News encoder + user encoder + a dot-product score.

    The news encoder's weights are **shared** between the history side and the
    candidate side, as in NRMS: an article should get the same vector whether it
    is something the user read or something we are considering showing them.
    """

    def __init__(
        self,
        embedding_dim: int,
        dim: int,
        heads: int,
        attention_dim: int,
        n_categories: int = 0,
        category_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.news = NewsEncoder(embedding_dim, dim, n_categories, category_dim, dropout)
        self.user = UserEncoder(dim, heads, attention_dim, dropout)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        history_categories: torch.Tensor | None = None,
        candidate_categories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scores, [batch, n_candidates].

        `history` is [batch, length, embedding_dim] and `candidates` is
        [batch, n_candidates, embedding_dim].
        """
        user_vector = self.user(self.news(history, history_categories), history_mask)
        candidate_vectors = self.news(candidates, candidate_categories)
        return torch.einsum("bd,bcd->bc", user_vector, candidate_vectors)


def listwise_loss(scores: torch.Tensor, candidate_mask: torch.Tensor | None = None):
    """Softmax cross-entropy over one impression's candidates, clicked at index 0.

    NRMS's training objective. `scores` is [batch, n_candidates] with **column 0
    holding the clicked candidate** — `build_listwise_groups` guarantees that
    ordering, so the target is a column of zeros and no label tensor is needed.

    Reducing over the wrong axis produces a number that looks like a loss and
    trains the model to something else, so
    `tests/test_nrms.py::test_listwise_loss_is_invariant_to_a_constant_score_shift`
    pins the softmax property that catches it.
    """
    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask, PADDED_CANDIDATE_SCORE)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return nn.functional.cross_entropy(scores, target)


def build_listwise_groups(
    table: pl.DataFrame, seed: int | None = None
) -> tuple[list[tuple[str, str, list[str]]], int]:
    """One `(impression_id, clicked_article, negatives)` per impression.

    Negatives are drawn **from the same impression's own non-clicked
    candidates** — that is what makes the loss a within-impression comparison
    rather than a comparison against the corpus at large.

    Returns the groups and the number of impressions that had **fewer** than
    `config.NRMS_NEG_SAMPLES` negatives available and so were sampled with
    replacement. That count is returned rather than swallowed: `eval.json`
    measures a median of 9 candidates per impression on EB-NeRD against 23 on
    MIND, and `a2_q2_insights.md` names short EB-NeRD candidate lists as an
    *unverified* hypothesis for why GBDT's pairwise objective failed there.
    Reporting the share makes it a measurement.

    An impression with no clicked candidate is skipped — there is no correct
    class for the softmax to point at. `recall.sample_impressions` already
    filters those out upstream, so this is a guard rather than a common path.
    """
    rng = np.random.default_rng(config.GLOBAL_SEED if seed is None else seed)
    n_negatives = config.NRMS_NEG_SAMPLES
    groups: list[tuple[str, str, list[str]]] = []
    padded = 0

    for (impression_id,), group in table.group_by("impression_id", maintain_order=True):
        labels = group["label"].to_numpy()
        article_ids = group["article_id"].to_list()
        clicked = [a for a, y in zip(article_ids, labels, strict=True) if y == 1]
        if not clicked:
            continue
        pool = [a for a, y in zip(article_ids, labels, strict=True) if y == 0]
        if not pool:
            # Every candidate was clicked: there is nothing to rank against.
            continue
        if len(pool) >= n_negatives:
            chosen = rng.choice(pool, size=n_negatives, replace=False)
        else:
            chosen = rng.choice(pool, size=n_negatives, replace=True)
            padded += 1
        groups.append((str(impression_id), clicked[0], [str(a) for a in chosen]))

    return groups, padded


@dataclass
class ListwiseBatchSource:
    """Everything `train` needs, assembled once so the training loop is plain.

    `history_rows` and `row_of_article` both index into the same `embeddings`
    matrix, so a model never holds a copy of the vectors.
    """

    groups: list[tuple[str, str, list[str]]] = field(default_factory=list)
    history_rows: dict[str, np.ndarray] = field(default_factory=dict)
    row_of_article: dict[str, int] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    category_of_row: np.ndarray | None = None
    n_categories: int = 0
    padded_negatives: int = 0

    @classmethod
    def empty(cls) -> ListwiseBatchSource:
        return cls()

    def __len__(self) -> int:
        return len(self.groups)


@dataclass
class NRMSModel:
    """A trained network plus everything needed to score a new table with it.

    The architecture widths are carried **on the model** rather than read back
    out of `config` when saving. `rerank/store.save` needs to record the shape
    this network actually has; reading `config.NRMS_DIM` at save time would
    write down whatever the config happens to say, which is a different thing
    as soon as the config changes or a caller builds a net with other widths
    (every test does). `tests/test_rerank_store.py` caught exactly that.
    """

    net: NRMSNet
    device: str
    category_aware: bool
    n_categories: int
    embedding_dim: int
    dim: int = config.NRMS_DIM
    heads: int = config.NRMS_HEADS
    attention_dim: int = config.NRMS_ATTENTION_DIM
    category_dim: int = config.NRMS_CATEGORY_DIM
    padded_negatives: int = 0
    epochs_trained: int = 0
    best_val_ndcg10: float = float("-inf")


@dataclass
class DatasetResources:
    """The per-dataset pieces every split's source shares.

    Loaded **once** per dataset and reused for train, val and test. Two reasons,
    and the second is the important one:

    - Cost. `embed.load` reads a 145 MB matrix and `row_index` builds a
      65,238-entry dict; doing that three times per variant is waste.
    - Correctness. `category_of_row` maps an embedding row to a *category
      index*, and those indices must mean the same thing in every split. Built
      per split, a category present in test but not in train would get a
      different integer in each, and the learned category embedding would be
      looked up at the wrong row — silently, with no error.
    """

    embeddings: embed.ArticleEmbeddings
    row_of_article: dict[str, int]
    category_of_row: np.ndarray
    n_categories: int
    click_history: pl.DataFrame


def load_resources(dataset: str, store_root: Path | None = None) -> DatasetResources:
    """Embeddings, the article->row index, the row->category index, and A2 Q1's
    click-history table for one dataset."""
    root = store_root or config.DATA_DIR
    store = root / "feature_store" / dataset

    embeddings = embed.load(embed.embeddings_dir(dataset))
    row_of_article = ba.row_index(embeddings)

    article_features = feature_store.read_table(store / "article_features.parquet")
    category_by_article = dict(
        zip(
            article_features["article_id"].to_list(),
            article_features["category"].to_list(),
            strict=True,
        )
    )
    # Index 0 is reserved for "no category on record", so a missing value is a
    # category of its own rather than colliding with a real one.
    names = sorted({c for c in category_by_article.values() if c is not None})
    index_of_category = {name: i + 1 for i, name in enumerate(names)}
    category_of_row = np.zeros(embeddings.n_articles, dtype=np.int64)
    for article_id, row in row_of_article.items():
        category = category_by_article.get(article_id)
        if category is not None:
            category_of_row[row] = index_of_category[category]

    return DatasetResources(
        embeddings=embeddings,
        row_of_article=row_of_article,
        category_of_row=category_of_row,
        n_categories=len(index_of_category) + 1,
        click_history=feature_store.read_table(store / "user_click_history_features.parquet"),
    )


def build_source(
    resources: DatasetResources,
    table: pl.DataFrame,
    *,
    seed: int | None = None,
    with_groups: bool = True,
) -> ListwiseBatchSource:
    """Everything `train`/`predict` need for one split's candidate table.

    `with_groups=False` for a table that is only ever scored (val, test): the
    listwise groups exist to be *trained* on, and sampling negatives for a
    table nobody fits is wasted work that also consumes the RNG.
    """
    impression_ids = set(table["impression_id"].to_list())
    click_history = resources.click_history.filter(
        pl.col("impression_id").is_in(list(impression_ids))
    )
    groups: list[tuple[str, str, list[str]]] = []
    padded = 0
    if with_groups:
        groups, padded = build_listwise_groups(table, seed=seed)

    return ListwiseBatchSource(
        groups=groups,
        history_rows=features.history_rows_by_impression(click_history, resources.row_of_article),
        row_of_article=resources.row_of_article,
        embeddings=resources.embeddings.vectors,
        category_of_row=resources.category_of_row,
        n_categories=resources.n_categories,
        padded_negatives=padded,
    )


def _history_batch(
    source: ListwiseBatchSource, impression_ids: list[str], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Padded history vectors, the real/padding mask, and category indices."""
    length = config.QUERY_HISTORY_LIMIT
    assert source.embeddings is not None
    dim = source.embeddings.shape[1]

    vectors = np.zeros((len(impression_ids), length, dim), dtype=np.float32)
    mask = np.zeros((len(impression_ids), length), dtype=bool)
    categories = np.zeros((len(impression_ids), length), dtype=np.int64)

    for i, impression_id in enumerate(impression_ids):
        rows = source.history_rows.get(impression_id)
        if rows is None or len(rows) == 0:
            continue
        rows = rows[:length]
        vectors[i, : len(rows)] = source.embeddings[rows]
        mask[i, : len(rows)] = True
        if source.category_of_row is not None:
            categories[i, : len(rows)] = source.category_of_row[rows]

    category_tensor = (
        torch.from_numpy(categories).to(device) if source.category_of_row is not None else None
    )
    return (
        torch.from_numpy(vectors).to(device),
        torch.from_numpy(mask).to(device),
        category_tensor,
    )


def _candidate_batch(
    source: ListwiseBatchSource, article_lists: list[list[str]], device: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert source.embeddings is not None
    dim = source.embeddings.shape[1]
    width = max(len(a) for a in article_lists)
    vectors = np.zeros((len(article_lists), width, dim), dtype=np.float32)
    categories = np.zeros((len(article_lists), width), dtype=np.int64)

    for i, articles in enumerate(article_lists):
        rows = np.array([source.row_of_article.get(a, 0) for a in articles], dtype=np.int64)
        vectors[i, : len(rows)] = source.embeddings[rows]
        if source.category_of_row is not None:
            categories[i, : len(rows)] = source.category_of_row[rows]

    category_tensor = (
        torch.from_numpy(categories).to(device) if source.category_of_row is not None else None
    )
    return torch.from_numpy(vectors).to(device), category_tensor


def _val_ndcg10(model: NRMSModel, source: ListwiseBatchSource, val_table: pl.DataFrame) -> float:
    """nDCG@10 over the val table — the selection and early-stopping metric.

    The same metric `gbdt.py` and `mlp.py` stop on, so the three options are
    comparable. Reading val to decide *when* to stop is not a leak: no weight is
    updated from a val row.
    """
    scores = predict(model, source, val_table)
    scored = val_table.with_columns(pl.Series("_s", scores))
    values = []
    for _, group in scored.group_by("impression_id", maintain_order=True):
        labels = group["label"].to_numpy()
        if metrics.degenerate_reason(labels) is not None:
            continue
        values.append(metrics.ranking_metrics(labels, group["_s"].to_numpy())["ndcg@10"])
    return float(np.mean(values)) if values else float("-inf")


def train(
    train_rows: ListwiseBatchSource,
    val_table: pl.DataFrame,
    *,
    val_rows: ListwiseBatchSource,
    category_aware: bool,
) -> NRMSModel:
    """Fit listwise on `train_rows`, stopping early on `val_table`'s nDCG@10.

    Same contract as `gbdt.train` / `mlp.train`: which split each argument holds
    is the caller's responsibility, and `nrms_ablation.py` is what enforces it.

    GPU only, and it **raises** rather than falling back — `_require_cuda` is
    imported from `rerank/mlp.py` so there is one definition of that rule.

    **`val_rows` is required and has no default.** It used to default to `None`
    with `val_rows or train_rows` behind it, and that was a real bug, measured
    2026-09-16. `ListwiseBatchSource.__len__` counts *training* groups, and a
    val source is built with `with_groups=False` — so it is legitimately empty,
    therefore falsy, and `or` silently handed the training source to the
    evaluation. Every val impression then had no history entry, every val score
    collapsed to the cold-start constant, and `best_val_ndcg@10` came out as
    **0.434698 for both ablation variants, identical to six decimal places**,
    while the models actually scored 0.470 and 0.508. Nothing raised; early
    stopping was choosing between epochs on noise.

    Requiring the argument removes the fallback that made the substitution
    possible. `tests/test_nrms.py::test_train_requires_an_explicit_val_source`
    keeps it required.
    """
    device = _require_cuda()

    # Seeded per call, not once at import, so both variants start from the same
    # weights and the only difference between them is the declared one. Also
    # asks the GPU for deterministic kernels: seeding alone left the ablation's
    # deltas moving between runs.
    seeding.seed_everything()

    embedding_dim = train_rows.embeddings.shape[1] if train_rows.embeddings is not None else 0
    net = NRMSNet(
        embedding_dim=embedding_dim,
        dim=config.NRMS_DIM,
        heads=config.NRMS_HEADS,
        attention_dim=config.NRMS_ATTENTION_DIM,
        n_categories=train_rows.n_categories if category_aware else 0,
        category_dim=config.NRMS_CATEGORY_DIM,
        dropout=config.NRMS_DROPOUT,
    ).to(device)

    model = NRMSModel(
        net=net,
        device=device,
        category_aware=category_aware,
        n_categories=train_rows.n_categories if category_aware else 0,
        embedding_dim=embedding_dim,
        padded_negatives=train_rows.padded_negatives,
    )

    optimiser = torch.optim.Adam(net.parameters(), lr=config.NRMS_LR)
    order = np.arange(len(train_rows.groups))
    rng = np.random.default_rng(config.GLOBAL_SEED)

    best_score = float("-inf")
    best_state = copy.deepcopy(net.state_dict())
    epochs_without_improvement = 0
    epochs_trained = 0

    for _ in range(config.NRMS_MAX_EPOCHS):
        net.train()
        rng.shuffle(order)
        for start in range(0, len(order), config.NRMS_BATCH_IMPRESSIONS):
            batch = [
                train_rows.groups[i] for i in order[start : start + config.NRMS_BATCH_IMPRESSIONS]
            ]
            if not batch:
                continue
            impression_ids = [g[0] for g in batch]
            # Column 0 is the clicked article — `listwise_loss` relies on it.
            candidate_lists = [[g[1], *g[2]] for g in batch]

            history, history_mask, history_categories = _history_batch(
                train_rows, impression_ids, device
            )
            candidates, candidate_categories = _candidate_batch(train_rows, candidate_lists, device)

            optimiser.zero_grad()
            scores = net(
                history,
                history_mask,
                candidates,
                history_categories if category_aware else None,
                candidate_categories if category_aware else None,
            )
            loss = listwise_loss(scores)
            loss.backward()
            optimiser.step()

        epochs_trained += 1
        # `val_rows`, never `val_rows or train_rows` — see this function's
        # docstring for what that cost.
        score = _val_ndcg10(model, val_rows, val_table)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.NRMS_EARLY_STOPPING_PATIENCE:
            break

    net.load_state_dict(best_state)
    model.epochs_trained = epochs_trained
    model.best_val_ndcg10 = best_score
    return model


def predict(model: NRMSModel, source: ListwiseBatchSource, table: pl.DataFrame) -> np.ndarray:
    """One score per row of `table`, in `table`'s row order.

    Row order is the contract. `nrms_ablation.py` hands these scores straight to
    `eval/harness.py`'s ordering, and a permuted return would produce a
    plausible near-random metric with no error anywhere — the failure
    `submit/score_fast.py`'s docstring warns about.
    """
    model.net.eval()
    out = np.zeros(table.height, dtype=np.float64)

    # Carry each row's ORIGINAL position through the grouping and scatter the
    # scores back to it. The obvious implementation — assume an impression's
    # rows are contiguous and write out[offset : offset + width] — is wrong for
    # any table whose rows are not already grouped, and was caught by
    # `tests/test_nrms_source.py::test_predict_follows_the_table_when_its_rows_are_permuted`.
    # `features.build_candidate_table` does happen to emit contiguous rows, so
    # this would have been invisible in production while quietly mis-scoring
    # any other caller.
    indexed = table.with_row_index("_row")
    impression_ids: list[str] = []
    candidate_lists: list[list[str]] = []
    row_positions: list[np.ndarray] = []

    for (impression_id,), group in indexed.group_by("impression_id", maintain_order=True):
        impression_ids.append(str(impression_id))
        candidate_lists.append([str(a) for a in group["article_id"].to_list()])
        row_positions.append(group["_row"].to_numpy())

    batch_size = config.NRMS_BATCH_IMPRESSIONS
    with torch.no_grad():
        for start in range(0, len(impression_ids), batch_size):
            stop = start + batch_size
            ids = impression_ids[start:stop]
            lists = candidate_lists[start:stop]

            history, history_mask, history_categories = _history_batch(source, ids, model.device)
            candidates, candidate_categories = _candidate_batch(source, lists, model.device)
            scores = (
                model.net(
                    history,
                    history_mask,
                    candidates,
                    history_categories if model.category_aware else None,
                    candidate_categories if model.category_aware else None,
                )
                .cpu()
                .numpy()
            )
            for i, rows in enumerate(row_positions[start:stop]):
                out[rows] = scores[i, : len(rows)]

    return out
