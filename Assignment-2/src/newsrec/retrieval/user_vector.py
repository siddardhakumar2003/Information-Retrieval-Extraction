"""The user's interest as a vector, pooled from what they clicked (q3_plan D5).

Requirements covered: Q3.4 — pooled from clicked-article embeddings, using only
clicks **strictly before** the impression, with the pooling rule a config value
rather than a constant.

## The one rule this module exists to keep

Same rule as `query.py`, one representation later: a user vector for an
impression may be built only from clicks recorded **strictly before** that
impression's timestamp.

And, as there, it is not re-implemented. `leakage.select_history_for` is called.
Q1's mutation testing established that a permissive selector broke no test --
though not for the reason long given: equal timestamps DO occur in EB-NeRD,
6,518 times in `ebnerd_demo` (0.0341%) and 62,030 in `ebnerd_small` (0.0336%).
The mutation survived because nothing looked at those rows, not because they
were absent. Q2 then found the other half of it the hard way -- MIND ships no
click timestamps at all, so the timestamp selector returns nothing and every
query is silently empty (design_rationale 1.16). Exactly one place in this
codebase decides what counts as legitimate history, and every stage goes through
it.

## Pooled from article vectors, not encoded from text

Q3.4 says "computed from clicked-article **embeddings**", so the user vector is
an average of vectors we already have. It is worth being explicit about the
alternative that is thereby excluded, because the encoder was configured for it:

*We could instead concatenate the user's recent titles into one string and
encode it* -- which is what `query.py` effectively does for BM25, and what
`config.ENCODER.query_prefix` ("query: ") exists for. Rejected, and not only
because Q3.4 names embeddings:

- **It does not fit.** 20 titles run to roughly 200 sub-word tokens against the
  128-token limit (q3_plan D2), so the oldest ~8 clicks would be silently
  truncated away -- a second, undeclared history cap on top of the declared one.
- **It costs an encode per impression** at serving time, where pooling is a few
  array additions over vectors computed once.
- **It would not be comparable.** BM25's query is 20 titles; a truncated 12-title
  encode would make Q3.6 partly a comparison of history windows.

**Consequence worth stating: `query_prefix` is currently unused.** Every vector
in play -- articles and users alike -- lives in "passage" space. That is
self-consistent, and for a symmetric similarity (document against document) it
is the right call; it is recorded here so nobody later "fixes" the user path by
adding a prefix that would put the two sides in different spaces.

## Why the vector is renormalised after pooling

The mean of several unit vectors is **not** a unit vector -- average two vectors
60 degrees apart and the result has length ~0.87. Left alone, a user with varied
interests would get a *shorter* vector than a single-minded one, and since the
ANN index ranks by inner product, that user's every similarity would be scaled
down. It would look like the model finding them harder to serve. Renormalising
makes the comparison about direction only, which is what cosine means.

## Cold start is the popularity list, exactly as in Q2

A user with no legitimate prior click has nothing to pool. `user_vectors`
reports them via a mask rather than inventing a vector, and the caller routes
them to `query.cold_start_candidates` -- the *same* training-period popularity
table BM25 falls back to (q2_plan D7). Both retrievers then behave identically
for those impressions, so a lexical-vs-semantic difference in Q3.6 cannot come
from the fallback.

The zero vector is refused rather than used. Its cosine with everything is 0/0,
and whatever ordering came back would be an artefact of floating-point noise.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import leakage
from newsrec.retrieval import embed

#: Same cap as BM25's query (q2_plan D6), read from the same constant. This is
#: load-bearing for Q3.6: EB-NeRD's median user has 88 history clicks and its
#: 95th percentile is 639, so pooling everything would average a user into the
#: corpus centroid -- and if the semantic side used 639 clicks while the lexical
#: side used 20, the comparison would be partly about the window rather than the
#: representation.
DEFAULT_LIMIT = config.QUERY_HISTORY_LIMIT


def pooling_weights(n: int, rule: str | None = None, decay: float | None = None) -> np.ndarray:
    """Weights for `n` clicks ordered most-recent-first.

    `mean` weights every click equally. `recency` applies a geometric decay, so
    the i-th most recent click is worth `decay ** i` -- at the default 0.9 over a
    20-click window the oldest click carries about 12% of the newest one's
    weight. A real gradient rather than a rounding difference, which is the point
    of running the ablation at all (q3_plan D5).

    "Most recent" means `click_time` for EB-NeRD and `history_rank` for MIND,
    which ships no click timestamps. `select_history_for` has already applied
    whichever key applies, so this function only sees position -- which is the
    only notion of recency defined for both datasets.
    """
    rule = config.USER_POOLING if rule is None else rule
    decay = config.USER_POOLING_DECAY if decay is None else decay

    if rule not in config.USER_POOLING_VARIANTS:
        raise ValueError(
            f"unknown pooling rule {rule!r}; expected one of {config.USER_POOLING_VARIANTS}. "
            f"The rule is a config value (Q3.4), so a typo here must fail rather than "
            f"silently fall back to the mean."
        )
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if rule == "mean":
        return np.full(n, 1.0 / n, dtype=np.float32)
    return (decay ** np.arange(n, dtype=np.float32)).astype(np.float32)


def pool(
    vectors: np.ndarray,
    rule: str | None = None,
    decay: float | None = None,
) -> np.ndarray:
    """One unit-length user vector from clicked-article vectors, most-recent-first.

    Raises on an empty input: "no history" is a *caller's* branch (the cold-start
    fallback), not something to paper over with a zero vector here.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(
            "pool() needs at least one clicked-article vector. An empty history is a "
            "cold-start impression and belongs on query.cold_start_candidates, not here."
        )

    weights = pooling_weights(len(vectors), rule, decay)
    pooled = weights @ vectors

    norm = float(np.linalg.norm(pooled))
    if norm == 0.0 or not np.isfinite(norm):
        # Mathematically possible if the clicked vectors cancel exactly. There is
        # no direction to report, and normalising would produce NaN, which sorts
        # unpredictably and would corrupt recall silently.
        raise ValueError(
            f"pooled user vector has zero or non-finite length ({norm}); it has no "
            f"direction, so cosine against it is undefined."
        )
    return (pooled / norm).astype(np.float32)


def user_vectors(
    impressions: pl.DataFrame,
    history: pl.DataFrame,
    embeddings: embed.ArticleEmbeddings,
    rule: str | None = None,
    limit: int | None = None,
    decay: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """User vectors for many impressions at once (Q3.4).

    Returns `(matrix, has_vector)`:

    - `matrix` is `(len(impressions), dim)`, row *i* for `impressions[i]`.
    - `has_vector` is a boolean mask; `False` marks a cold-start impression whose
      row is left as zeros and **must not be searched with**. The caller routes
      those to the popularity fallback.

    A mask rather than a shorter matrix, so the caller cannot lose track of which
    impression a row belongs to -- the same reasoning that keeps `article_ids`
    beside the embedding matrix in `embed.py`.

    The boundary is applied ONCE, to the whole frame, by `select_history_for`.
    Per-impression calls would re-run its join for every row; more importantly,
    one call means one place where the rule is applied rather than N.
    """
    limit = DEFAULT_LIMIT if limit is None else limit
    n, dim = len(impressions), embeddings.dim
    matrix = np.zeros((n, dim), dtype=np.float32)
    has_vector = np.zeros(n, dtype=bool)

    if n == 0 or history.is_empty():
        return matrix, has_vector

    # THE boundary. Not re-derived here; see the module docstring.
    past = leakage.select_history_for(
        history, impressions.select("impression_id", "user_id", "impression_time")
    )
    if past.is_empty():
        return matrix, has_vector

    # Already most-recent-first within each impression, so the cap is a
    # positional head per group rather than another sort.
    capped = past.with_columns(pl.int_range(pl.len()).over("impression_id").alias("_pos")).filter(
        pl.col("_pos") < limit
    )

    row_of = {imp: i for i, imp in enumerate(impressions["impression_id"].to_list())}
    vector_of = {a: i for i, a in enumerate(embeddings.article_ids)}

    # Group the (impression, article) pairs, preserving the recency order the
    # selector established -- `_pos` is that order, so it is the sort key.
    grouped: dict[str, list[int]] = {}
    for impression_id, article_id, _pos in capped.select(
        "impression_id", "article_id", "_pos"
    ).iter_rows():
        row = vector_of.get(article_id)
        if row is None:
            # History may reference an article the corpus does not carry.
            # Contributing nothing is right; a placeholder would invent an
            # interest the user never expressed.
            continue
        grouped.setdefault(impression_id, []).append(row)

    for impression_id, rows in grouped.items():
        i = row_of.get(impression_id)
        if i is None or not rows:
            continue
        matrix[i] = pool(embeddings.vectors[rows], rule=rule, decay=decay)
        has_vector[i] = True

    return matrix, has_vector
