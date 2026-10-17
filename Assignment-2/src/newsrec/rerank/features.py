"""The candidate-level feature table the re-ranker trains and predicts on
(a2_q2_plan.md).

One row per (impression, candidate article), for the impression's OWN
candidate list — `eval/harness.py` already settled that this, not a
corpus-wide top-K, is what gets ranked (see that module's docstring): recall@K
over the whole corpus is ~2-3%, so a corpus shortlist has no positive at all
for most impressions.

## The one rule this module exists to keep

A row belongs to exactly the split its impression belongs to.
`recall.sample_impressions(impressions, split, size, seed)` — the same
function Q2's recall and Q3's semantic recall already share — does the
filtering, so "train" can never pick up a val/test impression by accident. Every
individual FEATURE value (BM25 score, semantic score, every A2 Q1 column) is
already leakage-safe by construction from A1/A2 Q1; the only new risk this
stage introduces is which ROWS end up in which table, which is what
`tests/test_rerank_features.py` checks.

## Reusing A1's own scorers, not re-deriving them

`bm25_score`/`semantic_score` come from `eval.harness._bm25_scores`/
`_semantic_scores` — the exact functions A1's `eval.json` was computed with.
Re-deriving them would risk a re-ranker that looks better than the baseline
partly because its "baseline" features were computed slightly differently, not
because ranking on top of them actually helps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval.harness import _bm25_scores, _semantic_scores
from newsrec.retrieval import embed, recall
from newsrec.retrieval import index as idx

#: The columns a model actually sees. One list, imported by `gbdt.py`,
#: `mlp.py` and `ablation.py`, so "what is a feature" is answered in one place
#: (exam_notes.md 4.2) — adding a feature means editing this list once, not
#: three model-specific ones that can silently drift apart.
FEATURE_COLUMNS: tuple[str, ...] = (
    "bm25_score",
    "semantic_score",
    "n_clicks_in_history",
    "recency_weighted_click_count",
    "popularity_train",
    "has_published_time",
    "session_click_count",
    "position_in_session",
    "category_match",
)

#: Columns kept alongside the features for bookkeeping (evaluation, grouping)
#: but never fed to a model as input.
ID_COLUMNS: tuple[str, ...] = ("impression_id", "article_id", "user_id", "dataset", "split")


def build_candidate_table(
    dataset: str,
    split: str,
    size: int | None = None,
    seed: int | None = None,
    store_root: Path | None = None,
) -> pl.DataFrame:
    """One row per (impression, candidate) for a sampled, split-filtered set of
    impressions. `label` is 1 for the clicked candidate(s), 0 otherwise —
    always present (even for "test"), since `sample_impressions` only keeps
    impressions with at least one click; callers that must not look at labels
    (a model at predict time) select `FEATURE_COLUMNS` explicitly rather than
    relying on the caller to drop the column.
    """
    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset
    store = root / "feature_store" / dataset

    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")
    sample = recall.sample_impressions(impressions, split, size, seed)
    if sample.is_empty():
        return _empty_table()

    users = sample["user_id"].unique().to_list()
    user_history = history.filter(pl.col("user_id").is_in(users))

    titles = feature_store.read_table(processed / "articles.parquet").select("article_id", "title")
    article_features = feature_store.read_table(store / "article_features.parquet")
    spec = config.tokeniser_for(dataset)
    bm25_index = idx.build(article_features.select("article_id", "text"), spec)
    bm25_scores = _bm25_scores(sample, user_history, titles, bm25_index, spec)

    embeddings = embed.load(embed.embeddings_dir(dataset))
    semantic_scores, _ = _semantic_scores(sample, user_history, embeddings)

    click_history = feature_store.read_table(store / "user_click_history_features.parquet")
    click_history_by_impression = {
        row["impression_id"]: row
        for row in click_history.select(
            "impression_id",
            "n_clicks_in_history",
            "recency_weighted_click_count",
            "recent_categories",
        ).iter_rows(named=True)
    }

    session_path = store / "session_features.parquet"
    session_by_impression: dict = {}
    if session_path.exists():
        sessions = feature_store.read_table(session_path)
        session_by_impression = {
            row["impression_id"]: row
            for row in sessions.select(
                "impression_id", "session_click_count", "position_in_session"
            ).iter_rows(named=True)
        }

    article_by_id = {
        row["article_id"]: row
        for row in article_features.select(
            "article_id", "category", "popularity_train", "has_published_time"
        ).iter_rows(named=True)
    }

    columns: dict[str, list] = {name: [] for name in (*ID_COLUMNS, "label", *FEATURE_COLUMNS)}

    for i, row in enumerate(sample.iter_rows(named=True)):
        impression_id = row["impression_id"]
        user_id = row["user_id"]
        candidates = row["candidate_ids"]
        clicked = set(row["clicked_ids"])
        bm25 = bm25_scores[i]
        sem = semantic_scores[i]

        ch = click_history_by_impression.get(impression_id)
        n_clicks = ch["n_clicks_in_history"] if ch else 0
        recency_weighted = ch["recency_weighted_click_count"] if ch else 0.0
        recent_categories = set(ch["recent_categories"]) if ch else set()

        sess = session_by_impression.get(impression_id)
        session_clicks = sess["session_click_count"] if sess else 0
        position = sess["position_in_session"] if sess else 0

        for j, article_id in enumerate(candidates):
            art = article_by_id.get(article_id)
            columns["impression_id"].append(impression_id)
            columns["article_id"].append(article_id)
            columns["user_id"].append(user_id)
            columns["dataset"].append(dataset)
            columns["split"].append(split)
            columns["label"].append(1 if article_id in clicked else 0)
            columns["bm25_score"].append(float(bm25[j]))
            columns["semantic_score"].append(float(sem[j]))
            columns["n_clicks_in_history"].append(n_clicks)
            columns["recency_weighted_click_count"].append(recency_weighted)
            columns["popularity_train"].append(art["popularity_train"] if art else 0)
            columns["has_published_time"].append(int(art["has_published_time"]) if art else 0)
            columns["session_click_count"].append(session_clicks)
            columns["position_in_session"].append(position)
            columns["category_match"].append(
                int(bool(art) and art["category"] in recent_categories)
            )

    return pl.DataFrame(columns).select(*ID_COLUMNS, "label", *FEATURE_COLUMNS)


def _empty_table() -> pl.DataFrame:
    schema = (
        {name: pl.Utf8 for name in ID_COLUMNS}
        | {"label": pl.Int64}
        | {name: pl.Float64 for name in FEATURE_COLUMNS}
    )
    return pl.DataFrame(schema=schema)


def group_sizes(table: pl.DataFrame) -> list[int]:
    """Candidates per impression, IN TABLE ROW ORDER — what LightGBM's
    `group` parameter needs. `build_candidate_table` never reorders rows away
    from per-impression contiguity, so a plain run-length count is correct;
    this function does not itself re-sort anything, which would silently make
    the groups wrong for a caller who assumed order was preserved.
    """
    return (
        table.select("impression_id")
        .group_by("impression_id", maintain_order=True)
        .agg(pl.len().alias("n"))["n"]
        .to_list()
    )


def history_rows_by_impression(
    click_history: pl.DataFrame,
    row_of_article: dict[str, int],
) -> dict[str, np.ndarray]:
    """impression_id -> embedding rows of that impression's prior clicks.

    The input is A2 Q1's `user_click_history_features.parquet` (or the subset of
    it a caller has already filtered), whose `recent_article_ids` column is
    already **strictly prior** to the impression, already **most-recent-first**,
    and already gated at build time by `behaviour_features.check`. This function
    only translates ids into rows of the embedding matrix.

    That is the whole point: A2 Q3's model adds **no new leakage surface**,
    because it defines no second notion of "before". A2 Q1 measured what
    happens when `leakage.select_history_for` is reused outside the bound its
    own docstring states — a `users x impressions x clicks` join that
    materialised ~184,000,000 rows and OOM-killed the build
    (`design_rationale.md` §5.1). Reading the table that function already
    produced costs nothing and inherits its guarantee.

    An article with no embedding row is **dropped**, not mapped to row 0.
    Defaulting would put a real, arbitrary article into someone's click history
    and nothing would raise.

    Caps at `config.QUERY_HISTORY_LIMIT` rather than re-declaring a length, so
    A1's query construction, A2 Q1's table and this model all mean the same
    twenty clicks.
    """
    limit = config.QUERY_HISTORY_LIMIT
    out: dict[str, np.ndarray] = {}
    for row in click_history.select("impression_id", "recent_article_ids").iter_rows(named=True):
        rows = [
            row_of_article[article_id]
            for article_id in (row["recent_article_ids"] or [])
            if article_id in row_of_article
        ]
        out[str(row["impression_id"])] = np.array(rows[:limit], dtype=np.int64)
    return out


def feature_matrix(table: pl.DataFrame) -> np.ndarray:
    """The model-facing view: FEATURE_COLUMNS only, as a float32 array. The
    single place a model's input is carved out of the full table, so a model
    can never accidentally see `label` or an id column."""
    return table.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
