"""Semantic scoring at submission scale (Q5.1, Q5.2).

Design: agent_docs/q5_plan.md

## Why this exists rather than reusing the Q4 harness

`eval/harness.py` scores one impression per loop iteration. That is the right
shape for 2,000 impressions and the wrong shape for 13,536,710: measured at
roughly 10 ms an impression, EB-NeRD's test set would take over 30 hours. The
arithmetic underneath is not the problem -- 205,925,868 dot products at dim 384
is about 158 GFLOP, minutes of real work. Essentially all of the cost is Python
loop overhead.

So this module keeps the *definition* of a semantic score identical and changes
only the shape of the computation: explode every candidate into one flat array,
gather, multiply, rank, implode. Nothing here decides what a score means.

## User vectors are computed per USER, not per impression

EB-NeRD's 13,536,710 test rows come from 807,677 distinct users -- a 17x saving
from pooling each user's history once.

**This is only valid for the submission path, and must not be back-ported into
`retrieval/user_vector.py`.** It works here because the test set ships one fixed
history per user (`article_id_fixed`), so a user's vector genuinely does not
vary between their impressions. In the offline path it does vary: history is
truncated at each impression's timestamp, so two impressions by one user
legitimately have different vectors, and caching per user would leak later
clicks into earlier impressions. The optimisation is a property of this data
layout, not an improvement to the algorithm.

## Leakage: a deliberate skip, not an omission

No timestamp filtering happens here. The test history is the pre-period history
the competition hands us -- it is, by construction, the state before the test
period begins -- so there is no boundary to enforce and no `click_time` to
enforce it against. This is the same structural argument MIND relies on
throughout (`data_schemas.md`), and it is recorded here rather than left to
inference, because "no leakage check in the submission path" should read as a
decision rather than as something nobody thought about.

## Ties, and why ranking goes through polars

`1/rank` requires distinct integer ranks, and float32 cosines tie more often
than intuition suggests. Ranking is therefore done on `(-score, article_id)`,
which is a strict total order and matches the serving tie-break the rest of the
pipeline uses (q2_plan D8).

`rank("ordinal")` returns, for each element, its position in the order -- which
IS the inverse permutation the submission format wants. There is no separate
inversion step and therefore no opportunity to write the permutation by
mistake, which is the single most dangerous bug available here: it produces a
well-formed file that scores near random with no error anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl

from newsrec import config
from newsrec.retrieval import embed, user_vector

#: Candidate rows scored per chunk. Chunking the FLAT array rather than the
#: impressions keeps memory predictable: candidate counts run from 1 to 250, so
#: a fixed number of impressions is a wildly variable amount of work. At dim 384
#: and float32, 2M rows is ~3 GB of gathered vectors per side.
SCORE_CHUNK = 2_000_000

#: Below any achievable cosine, so a candidate with no embedding sorts last
#: rather than being dropped -- dropping one would change that impression's
#: candidate count and break the positional contract with the truth file.
UNSCORABLE = np.float32(-2.0)


def user_vectors_from_fixed_history(
    history: pl.DataFrame,
    embeddings: embed.ArticleEmbeddings,
    limit: int | None = None,
    row_of_article: dict | None = None,
) -> tuple[dict[int | str, int], np.ndarray]:
    """Pool one vector per user from `article_id_fixed`.

    Returns `(row_of_user, matrix)`, where `matrix[row_of_user[u]]` is that
    user's unit-norm vector. Users whose history contains no embeddable article
    are absent from the mapping; the caller falls back for them.

    Pooling is `user_vector.pool`, unchanged, so the submission and the offline
    evaluation compute the same quantity from the same code. Only the selection
    of which clicks to pool differs, and that is the point above about fixed
    history.

    **`row_of_article` must be passed when the history's id type is not the
    embedding table's.** EB-NeRD stores `article_id_fixed` as `Int32` while the
    table keys on `str`, so the default index built here matches nothing and
    every user silently ends up without a vector -- a submission that is
    well-formed, accepted, and identical to the baseline.

    Found by running: a candidate-side coverage check read 1.0000 while this
    returned zero users, because the two halves resolve ids independently and
    only one of them was being checked.
    """
    limit = config.QUERY_HISTORY_LIMIT if limit is None else limit
    if row_of_article is None:
        row_of_article = {a: i for i, a in enumerate(embeddings.article_ids)}

    row_of_user: dict[int | str, int] = {}
    pooled: list[np.ndarray] = []
    for user_id, clicks in history.select("user_id", "article_id_fixed").iter_rows():
        if not clicks:
            continue
        # Most recent first, then capped -- the same ordering rule as the
        # offline path, applied to a history that is already chronological.
        recent = list(clicks)[-limit:][::-1]
        rows = [row_of_article[a] for a in recent if a in row_of_article]
        if not rows:
            continue
        row_of_user[user_id] = len(pooled)
        pooled.append(user_vector.pool(embeddings.vectors[rows]))

    if not pooled:
        return {}, np.zeros((0, embeddings.dim), dtype=np.float32)
    return row_of_user, np.vstack(pooled).astype(np.float32)


def score_chunk(
    article_rows: np.ndarray,
    user_rows: np.ndarray,
    article_vectors: np.ndarray,
    user_matrix: np.ndarray,
) -> np.ndarray:
    """Cosine of each (candidate, user) pair, as one flat array.

    `article_rows` and `user_rows` are parallel and already flattened, so this
    is a row-wise dot product rather than a matrix product -- we want the
    diagonal of `A @ U.T`, never the full outer product, which at 2M x 807k
    would be absurd.

    A row index of -1 marks a candidate with no embedding; those score
    `UNSCORABLE`. Both vector sets are unit-norm, so the dot product is the
    cosine and nothing is renormalised here -- an unnormalised vector reaching
    this point should fail loudly upstream, not be quietly rescued.
    """
    out = np.full(article_rows.size, UNSCORABLE, dtype=np.float32)
    known = (article_rows >= 0) & (user_rows >= 0)
    if not known.any():
        return out
    a = article_vectors[article_rows[known]]
    u = user_matrix[user_rows[known]]
    # No dtype= on the einsum. Forcing float32 there makes it REFUSE a float64
    # input rather than downcast it ("cannot cast ... according to the rule
    # 'safe'"), so a caller that built its user matrix in float64 -- which is
    # what numpy gives you for `ones(...) / sqrt(8)` -- gets a TypeError deep in
    # numpy instead of an answer. Compute in whatever precision arrived and let
    # the assignment into `out` do the narrowing.
    out[known] = np.einsum("ij,ij->i", a, u)
    return out


def rank_within_impressions(frame: pl.DataFrame) -> pl.Series:
    """Ranks 1..n per impression, best-first, as a list column.

    Expects `row_ix`, `score` and `article_id` columns on an exploded frame, in
    the original candidate order within each `row_ix`. Returns one list per
    impression, aligned to that order -- the inverse permutation the format
    wants.

    Sorting on `(-score, article_id)` rather than score alone: float32 cosines
    tie, and `1/rank` needs distinct ranks.
    """
    return (
        frame.with_columns((-pl.col("score")).alias("_neg"))
        .with_columns(
            pl.struct("_neg", "article_id")
            .rank("ordinal")
            .over("row_ix")
            .cast(pl.Int64)
            .alias("rank")
        )
        .group_by("row_ix", maintain_order=True)
        .agg(pl.col("rank"))
        .get_column("rank")
    )


def score_frame(
    frame: pl.DataFrame,
    row_of_user: dict[int | str, int],
    user_matrix: np.ndarray,
    embeddings: embed.ArticleEmbeddings,
    row_of_article: dict[object, int],
    candidate_column: str = "article_ids_inview",
    user_column: str = "user_id",
) -> pl.Series:
    """Ranks for one chunk of behaviour rows, as a list column.

    One `explode`, one gather, one row-wise dot product, one grouped rank. No
    per-impression Python anywhere on this path.
    """
    empty = int((frame[candidate_column].list.len() == 0).sum())
    if empty:
        # `explode` turns an empty list into one null row, so the impression
        # would come back with a single bogus candidate and a rank list of the
        # wrong length. The validator would catch it, but only after writing a
        # multi-gigabyte file, and the message would point at the symptom.
        raise ValueError(
            f"{empty} row(s) have an empty {candidate_column}; there is no "
            f"well-formed prediction line for an impression with no candidates"
        )

    indexed = frame.with_row_index("row_ix")
    flat = indexed.select("row_ix", user_column, candidate_column).explode(
        candidate_column, empty_as_null=True
    )

    article_rows = np.fromiter(
        (row_of_article.get(a, -1) for a in flat[candidate_column]),
        dtype=np.int64,
        count=len(flat),
    )
    user_rows = np.fromiter(
        (row_of_user.get(u, -1) for u in flat[user_column]),
        dtype=np.int64,
        count=len(flat),
    )

    scores = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, len(flat))
        scores[start:stop] = score_chunk(
            article_rows[start:stop],
            user_rows[start:stop],
            embeddings.vectors,
            user_matrix,
        )

    scored = flat.select("row_ix", pl.col(candidate_column).alias("article_id")).with_columns(
        pl.Series("score", scores)
    )
    return rank_within_impressions(scored)


def lines_for_frame(
    frame: pl.DataFrame,
    ranks: pl.Series,
    id_column: str = "impression_id",
) -> Iterator[str]:
    """`<impression_id> [r1,r2,...]`, one per row, in the frame's own order."""
    built = frame.select(
        (
            pl.col(id_column).cast(pl.Utf8)
            + pl.lit(" [")
            + ranks.cast(pl.List(pl.Utf8)).list.join(",").alias("body")
            + pl.lit("]")
        ).alias("line")
    )
    yield from built["line"].to_list()


def article_row_index(embeddings: embed.ArticleEmbeddings, cast: type | None = None) -> dict:
    """article_id -> embedding row, optionally casting the key type.

    EB-NeRD candidate ids are `Int32` in the parquet while our embedding tables
    key on whatever the feature store produced. The cast is explicit so a type
    mismatch shows up as a build-time choice rather than as every candidate
    silently missing its embedding -- which would score the whole submission
    `UNSCORABLE` and rank purely by article_id, i.e. the baseline, while looking
    like it ran.
    """
    if cast is None:
        return {a: i for i, a in enumerate(embeddings.article_ids)}
    return {cast(a): i for i, a in enumerate(embeddings.article_ids)}


def coverage(row_of_article: dict, candidate_ids: pl.Series) -> float:
    """Share of candidate slots whose article has an embedding.

    Reported before writing anything. The failure this exists to catch is a key
    type mismatch: 0.0 coverage produces a perfectly well-formed submission that
    is really the article_id baseline, and nothing else would reveal it.
    """
    if candidate_ids.len() == 0:
        return float("nan")
    hits = sum(1 for a in candidate_ids if a in row_of_article)
    return hits / candidate_ids.len()
