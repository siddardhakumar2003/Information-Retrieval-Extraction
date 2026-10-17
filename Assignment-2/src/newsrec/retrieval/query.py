"""Turning a user's click history into a BM25 query (q2_plan.md D6, D7).

Requirements covered: Q2.5 (strict boundary), Q2.6 (cold-start fallback).

## The one rule this module exists to keep

A query for an impression may use only clicks recorded **strictly before** that
impression's timestamp. Everything else here is detail.

That rule is not re-implemented here. `leakage.select_history_before` is called,
because Q1's mutation testing established that a permissive selector broke no
test -- writing `<=` instead of `<` changed no number any test was looking at.

**Corrected 2026-08-22.** The *reason* long given for that was wrong. The claim
was that a click and an impression essentially never share a microsecond, so the
boundary case does not arise. Measured, it does: `ebnerd_demo` has **6,518**
(user, click, impression) triples where `click_time` is exactly
`impression_time` (0.0341% of 19.1M pairs) and `ebnerd_small` has **62,030**
(0.0336%). Rare, but not never. The mutation survived because no test at the
time *looked at those rows*, not because they did not exist -- and a later test
that does look (`test_leakage.py::test_user_embedding_respects_boundary`) fails
under the same mutation on real data.

The conclusion is unchanged and better supported: the defence is structural.
Exactly one place in the codebase writes that comparison, and every stage goes
through it.
`tests/test_query.py::test_query_construction_goes_through_the_shared_selector`
asserts the call actually happens.

## Titles in, title-plus-abstract out

The indexed *documents* are title + abstract (Q2.2). The *queries* are titles
only. That asymmetry is deliberate (D6): a query built from twenty full abstracts
would run to thousands of terms, cost time proportional to its length, and match
everything weakly. Short query against a richer document is the ordinary
information-retrieval arrangement.

## Why the history is capped

EB-NeRD users have a mean of **182.3** history clicks. Concatenating 182 titles
gives roughly 4,500 query terms, every one of which touches a posting list. D6
takes the most recent `QUERY_HISTORY_LIMIT` clicks instead: recency is the
strongest signal in news, where interest moves quickly. `limit` is a declared
parameter reported in the design note, never silently tuned.

## Popularity is computed from training rows, and only training rows

A cold-start user has no clicks, so there is no query to build. The fallback is
the most-clicked articles of the **training** period (D7). Computing popularity
over the whole log would be the `total_inviews` leak from Q9 wearing a different
hat: it would tell a training-time prediction which articles the publisher was
going to promote later.

The same table is D8's tie-break for equally-scored candidates. One table with
two consumers, rather than two tables and two chances to build it from the wrong
rows.
"""

from __future__ import annotations

from collections import Counter

import polars as pl

from newsrec import config
from newsrec.data import leakage
from newsrec.retrieval import tokenise

#: D6's cap, read from config rather than written here, so the ablation sweeps a
#: configured value and the design note can say which number produced which
#: result.
DEFAULT_LIMIT = config.QUERY_HISTORY_LIMIT

#: Which splits may contribute to the popularity table. Training only -- see the
#: module docstring. Exposed as a parameter *solely* so the leak canary can pass
#: the wrong value and prove the check bites; nothing in the pipeline overrides
#: it.
TRAINING_SPLITS: tuple[str, ...] = ("train",)


def build_query(
    impression: dict,
    history: pl.DataFrame,
    articles: pl.DataFrame,
    spec: config.TokeniserSpec,
    limit: int | None = None,
) -> list[str]:
    """The BM25 query terms for one impression.

    `impression` is a row dict carrying at least `impression_id`, `user_id` and
    `impression_time`. Returns terms in order, **with repetition preserved**: a
    word appearing in five of the user's clicked titles should weigh more than
    one appearing once (D6 option D, rejected).

    Returns `[]` for a user with no legitimate prior clicks. That is a defined
    outcome, not an error -- MIND has 3,238 training impressions with empty
    history (Q2.6). Callers route an empty query to `cold_start_candidates`.
    """
    limit = DEFAULT_LIMIT if limit is None else limit
    if history.is_empty():
        return []

    one = pl.DataFrame(
        {
            "impression_id": [impression["impression_id"]],
            "user_id": [impression["user_id"]],
            "impression_time": [impression["impression_time"]],
        },
        schema={
            "impression_id": pl.Utf8,
            "user_id": pl.Utf8,
            "impression_time": pl.Datetime("us", "UTC"),
        },
    )

    # THE boundary. Not re-derived here; see the module docstring.
    #
    # `select_history_for`, not `select_history_before`, because the two
    # datasets enforce the boundary differently and this stage must not know
    # which. EB-NeRD is bounded by `click_time < impression_time`; MIND has no
    # timestamps and is bounded structurally by the parser. Calling the
    # timestamp selector directly returned nothing at all for MIND, because null
    # compared with anything is null -- 100% empty queries, silently.
    past = leakage.select_history_for(history, one)
    if past.is_empty():
        return []

    # Already ordered most-recent-first by the selector, under whichever key
    # applies (click_time for EB-NeRD, history_rank for MIND), with article_id
    # breaking ties so the query is identical on every run (Q1.13). Taking the
    # head is all that is left to do.
    recent = past.head(limit)

    titles = dict(
        articles.select("article_id", "title")
        .drop_nulls("title")
        .iter_rows()  # (article_id, title)
    )

    terms: list[str] = []
    for article_id in recent["article_id"].to_list():
        title = titles.get(article_id)
        if title is None:
            # History may reference an article the corpus does not carry.
            # Contributing nothing is right; raising would stop a whole run over
            # one missing row, and injecting a placeholder would invent a term.
            continue
        terms.extend(tokenise.tokenise(title, spec))
    return terms


def build_queries(
    impressions: pl.DataFrame,
    history: pl.DataFrame,
    articles: pl.DataFrame,
    spec: config.TokeniserSpec,
    limit: int | None = None,
) -> dict[str, list[str]]:
    """`build_query` for many impressions at once — same rule, one pass.

    Returns `impression_id -> terms`, including empty lists for impressions with
    no legitimate history, so the caller can tell "cold start" from "absent".

    This exists for cost, not for a different definition. Calling `build_query`
    per impression re-runs the selector's join every time; over a recall
    evaluation that is the dominant cost. The boundary rule is identical --
    `select_history_for` is called ONCE with the whole frame, which is the shape
    it was designed for, and the per-impression cap is applied afterwards by
    grouping. Nothing here re-derives what counts as legitimate history.
    """
    limit = DEFAULT_LIMIT if limit is None else limit
    ids = impressions["impression_id"].to_list()
    out: dict[str, list[str]] = {i: [] for i in ids}
    if history.is_empty() or impressions.is_empty():
        return out

    past = leakage.select_history_for(
        history, impressions.select("impression_id", "user_id", "impression_time")
    )
    if past.is_empty():
        return out

    # The selector already ordered most-recent-first within each impression, so
    # the cap is a positional head per group rather than another sort.
    capped = past.with_columns(pl.int_range(pl.len()).over("impression_id").alias("_pos")).filter(
        pl.col("_pos") < limit
    )

    titles = dict(articles.select("article_id", "title").drop_nulls("title").iter_rows())
    cache: dict[str, list[str]] = {}

    for impression_id, article_id in capped.select("impression_id", "article_id").iter_rows():
        title = titles.get(article_id)
        if title is None:
            continue
        terms = cache.get(article_id)
        if terms is None:
            # One tokenisation per ARTICLE, not per (impression, article). A
            # popular article appears in thousands of users' histories.
            terms = tokenise.tokenise(title, spec)
            cache[article_id] = terms
        out[impression_id].extend(terms)
    return out


def training_popularity(
    impressions: pl.DataFrame,
    splits: tuple[str, ...] = TRAINING_SPLITS,
) -> dict[str, int]:
    """article_id -> click count, over `splits` only.

    Counts clicks (`clicked_ids`), not impressions shown. "Shown" is the
    publisher's decision and is itself a dataset-period aggregate; "clicked" is
    what users did, which is what a popularity prior should mean.

    `splits` exists so `tests/test_query.py` can build a deliberately leaked
    table and prove the difference is detectable. The pipeline never passes it.
    """
    if impressions.is_empty():
        return {}
    subset = impressions.filter(pl.col("split").is_in(list(splits)))
    if subset.is_empty():
        return {}

    counts: Counter[str] = Counter()
    for clicked in subset["clicked_ids"].to_list():
        if clicked:
            counts.update(clicked)
    return dict(counts)


def cold_start_candidates(
    impressions: pl.DataFrame,
    k: int,
    popularity: dict[str, int] | None = None,
) -> list[str]:
    """The fallback list for a user with no usable history (Q2.6, D7).

    Most-clicked-in-training first, `article_id` ascending as the tie-break so
    the list is a total order and identical on every run. Returns `[]` rather
    than raising when there is nothing to rank -- a corpus with no training
    clicks is degenerate but not an error.
    """
    if k <= 0:
        return []
    table = training_popularity(impressions) if popularity is None else popularity
    if not table:
        return []
    ranked = sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))
    return [article_id for article_id, _ in ranked[:k]]


def popularity_ranking(
    article_ids: list[str],
    popularity: dict[str, int],
    k: int,
) -> list[str]:
    """The whole corpus ranked by popularity — what BM25's cold path returns.

    `cold_start_candidates` ranks only articles that were clicked during
    training. That is the right answer to "what is popular", and the WRONG list
    to hand a retriever that must return K candidates: it runs out early. On a
    corpus where fewer than K articles were ever clicked in training, the
    semantic cold path returned a short list while BM25's zero-score tie-break
    kept going through the rest of the corpus, so the two retrievers were
    compared over different numbers of candidates.

    Found by `tests/test_compare.py::test_an_all_cold_sample_reports_a_zero_delta`
    on a fixture with 3 training-clicked articles and 7 in the corpus. The real
    corpora have 6,398 (MIND) and 1,673 (ebnerd_small) training-clicked
    articles, so at K=200 the two lists happened to agree -- the bug was
    invisible on real data and would have stayed invisible.

    `(-popularity, article_id)` reproduces BM25's ordering exactly: an empty
    query scores every article 0, so `top_k` sorts by the popularity tie-break
    and falls through to `doc_id`, which is `article_id` order because the
    feature store emits articles sorted. `tests/test_compare.py` pins that
    dependency.
    """
    if k <= 0:
        return []
    return sorted(article_ids, key=lambda a: (-popularity.get(a, 0), a))[:k]


def popularity_array(
    index_article_ids: list[str],
    popularity: dict[str, int],
) -> list[int]:
    """Popularity aligned to an index's doc_id order, for D8's tie-break.

    `bm25.top_k` takes the tie-break as a per-document array, so the mapping from
    article_id to doc_id happens once here rather than per query. Articles absent
    from the training period score 0, which is correct: unknown is not popular.
    """
    return [popularity.get(article_id, 0) for article_id in index_article_ids]
