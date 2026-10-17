"""Q2.7, Q2.8 — recall@K measurement.

The failure this file guards against is not a crash. It is a recall number that
is computed over the wrong pool, or averaged the wrong way, or quietly different
between runs — all of which produce a plausible figure that goes straight into
the design note.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.retrieval import bm25, query, recall, tokenise
from newsrec.retrieval import index as idx


@pytest.fixture
def report(store_root, dataset):
    return recall.evaluate_dataset(dataset, sample_size=25, store_root=store_root)


# =============================================================================
# D1 — the pool is the whole corpus, not the impression's candidate list
# =============================================================================


def test_retrieval_pool_is_the_entire_corpus(report, store_root, dataset):
    """q2_plan D1, the decision that makes the number mean anything.

    Retrieving from the impression's own candidate list would make recall@200
    exactly 1.0 for every EB-NeRD impression, which shows a mean of 11.5 and a
    maximum of 100 candidates. The metric would be arithmetic, not measurement.
    """
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    )
    assert report["corpus_articles"] == len(articles)
    assert report["pool"] == "full_corpus"


def test_pool_is_larger_than_any_impressions_candidate_list(report, store_root, dataset):
    """The same claim from the other direction, so a pool that silently shrank
    to the candidate list would fail even if `corpus_articles` were relabelled."""
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    widest = impressions.select(pl.col("candidate_ids").list.len().max()).item()
    assert report["corpus_articles"] > widest


# =============================================================================
# Determinism (the check the plan asks for by name)
# =============================================================================


def test_recall_is_measured_consistently(store_root, dataset, tmp_path):
    """Rebuild the index and remeasure: the report must be byte-identical.

    Q1.13 applies to every stage. If two runs of unchanged code disagree, then
    "the number moved" stops being evidence of anything, and no ablation in Q2.9
    or comparison in Q3.6 can be trusted.

    Compares serialised bytes rather than floats, because a difference of one
    unit in the last place is still a difference — and it is exactly the kind
    an unseeded bootstrap or an unstable sort would produce.
    """
    first = recall.run(
        datasets=(dataset,), out_dir=tmp_path / "a", sample_size=25, store_root=store_root
    )
    second = recall.run(
        datasets=(dataset,), out_dir=tmp_path / "b", sample_size=25, store_root=store_root
    )
    assert first.read_bytes() == second.read_bytes()


def test_sampled_impression_set_is_stable_and_fingerprinted(store_root, dataset):
    """Q3.5 requires semantic recall over the SAME impressions. Recording the
    seed makes that reproducible; the fingerprint makes it checkable."""
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    a = recall.sample_impressions(impressions, "test", 25)
    b = recall.sample_impressions(impressions, "test", 25)
    assert a["impression_id"].to_list() == b["impression_id"].to_list()
    assert recall.impression_set_fingerprint(a) == recall.impression_set_fingerprint(b)

    different = recall.sample_impressions(impressions, "test", 25, seed=config.GLOBAL_SEED + 1)
    if len(a) < len(
        impressions.filter((pl.col("split") == "test") & (pl.col("clicked_ids").list.len() > 0))
    ):
        assert recall.impression_set_fingerprint(different) != recall.impression_set_fingerprint(a)


# =============================================================================
# The metric's definition
# =============================================================================


def test_recall_values_are_proportions(report):
    for k, r in report["recall"].items():
        assert 0.0 <= r["mean"] <= 1.0, f"K={k} recall {r['mean']} outside [0, 1]"
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]


def test_recall_is_monotonic_in_k(report):
    """A larger K can only include more of what a smaller K found. If this ever
    fails, the top-K slice is not nested and the ordering is unstable."""
    values = [report["recall"][str(k)]["mean"] for k in config.RECALL_K_VALUES]
    assert values == sorted(values), f"recall not non-decreasing in K: {values}"


def test_impressions_with_no_click_are_excluded_and_counted(report, store_root, dataset):
    """Recall is undefined with an empty denominator (design_rationale 2.12).

    Excluded, never imputed as 0 — a zero would look like a measured failure to
    retrieve and would drag the average down for a reason unrelated to
    retrieval.
    """
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    test_split = impressions.filter(pl.col("split") == "test")
    expected = len(test_split.filter(pl.col("clicked_ids").list.len() == 0))
    assert report["impressions_undefined_excluded"] == expected
    assert report["impressions_evaluated"] <= len(test_split) - expected


def test_report_records_what_is_needed_to_reproduce_it(report):
    """A number nobody can regenerate is not a result (CLAUDE.md)."""
    for key in ("seed", "sample_size", "tokeniser", "impression_set_fingerprint"):
        assert key in report and report[key] is not None


def test_report_has_at_least_six_dataset_k_entries(tmp_path, store_root):
    """Q2.8 — both datasets, K in {50, 100, 200}."""
    dest = recall.run(
        datasets=("mind_small", "ebnerd_demo"),
        out_dir=tmp_path,
        sample_size=10,
        store_root=store_root,
    )
    data = json.loads(dest.read_text())
    entries = [(d, k) for d, v in data["datasets"].items() for k in v["recall"]]
    assert len(entries) >= 6
    for _, v in data["datasets"].items():
        for r in v["recall"].values():
            assert "ci_low" in r and "n" in r


# =============================================================================
# Cold start (Q2.6) — the fallback is the tie-break, not a second code path
# =============================================================================


def test_empty_query_retrieves_the_popularity_ranking(store_root, dataset):
    """D7's fallback and D8's tie-break are one mechanism.

    An impression with no history scores every article 0, so the order is
    decided entirely by the popularity tie-break. This asserts that what comes
    out really is the popularity list, rather than the two drifting apart into
    two definitions of "most popular".
    """
    processed = store_root / "processed" / dataset
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    impressions = feature_store.read_table(processed / "impressions.parquet")

    spec = config.tokeniser_for(dataset)
    index = idx.build(articles, spec)
    popularity = query.training_popularity(impressions)
    if not popularity:
        pytest.skip(f"{dataset} fixture has no training clicks to rank")

    tie_break = np.asarray(query.popularity_array(index.article_ids, popularity), dtype=np.int64)
    scores = bm25.score(index, [])  # the empty query
    top = bm25.top_k(scores, 3, tie_break=tie_break)
    retrieved = [index.article_ids[int(d)] for d in top]

    expected = query.cold_start_candidates(impressions, k=3)
    assert retrieved[: len(expected)] == expected[: len(retrieved)]


# =============================================================================
# End-to-end correctness on real data
# =============================================================================


@pytest.mark.realdata
def test_article_retrieves_itself_from_its_own_title(processed_store, feature_store_dir, dataset):
    """The strongest end-to-end check available without labels.

    Searching for a document's exact title must return that document first. If
    the postings, the idf table, the length normalisation or the top-K ordering
    were wrong, this would collapse — and unlike recall@K it has an unambiguous
    right answer, so a low score here means a bug rather than a hard task.

    Not 100% at rank 1 by design: the query is the title alone while the indexed
    document is title + abstract, so a longer document sharing the title's rare
    terms can legitimately outrank it.
    """
    spec = config.tokeniser_for(dataset)
    articles = feature_store.read_table(feature_store_dir / "article_features.parquet").select(
        "article_id", "text"
    )
    titles = dict(
        feature_store.read_table(processed_store / "articles.parquet")
        .select("article_id", "title")
        .drop_nulls("title")
        .iter_rows()
    )
    index = idx.build(articles, spec)
    idf = bm25.idf_array(index)

    rng = np.random.default_rng(config.GLOBAL_SEED)
    picks = rng.choice(index.n_docs, size=min(150, index.n_docs), replace=False)

    checked = first = top10 = 0
    for doc_id in picks:
        title = titles.get(index.article_ids[int(doc_id)])
        if not title:
            continue
        terms = tokenise.tokenise(title, spec)
        if not terms:
            continue
        checked += 1
        top = [int(d) for d in bm25.top_k(bm25.score(index, terms, idf=idf), 10)]
        first += top[0] == int(doc_id)
        top10 += int(doc_id) in top

    assert checked >= 50, "too few titles checked for the result to mean anything"
    assert first / checked >= 0.85, f"only {first}/{checked} articles ranked themselves first"
    assert top10 / checked >= 0.98, f"only {top10}/{checked} articles reached their own top-10"
