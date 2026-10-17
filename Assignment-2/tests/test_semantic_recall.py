"""Q3.5 — semantic recall@K, and the assertion that makes it comparable.

The failure guarded against is the one Q3.5 exists to prevent: two retrievers
scored on *different* impressions. Both numbers look entirely reasonable, their
difference means nothing, and every comparison built on it in Q3.6 inherits the
problem with no symptom anywhere.

The fixture corpus is tiny, so these tests prove the mechanism. The numbers that
go in the report are measured on the real corpora by `semantic.run()`.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.retrieval import ann, embed, query, recall, semantic

pytest.importorskip("faiss", reason="faiss-cpu is a declared dependency")


def _spec(dim: int = 16) -> config.EncoderSpec:
    return config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )


def _fake_encoder(dim: int = 16):
    """Deterministic per text, and CLUSTERED rather than uniform.

    Uniform random points in 16 dimensions are all roughly equidistant, so the
    ANN graph has nothing to get wrong and recall becomes an accident of
    tie-breaking. Real embeddings are clustered.
    """

    def encode(texts):
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            centre = np.zeros(dim, dtype=np.float32)
            centre[digest[8] % dim] = 3.0
            out[i] = centre + rng.normal(size=dim)
        return out

    return encode


@pytest.fixture
def prepared(store_root, dataset):
    """Embeddings + ANN index over the fixture corpus, built in memory."""
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    embeddings = embed.build(articles, spec=_spec(), encoder=_fake_encoder())
    return embeddings, ann.build(embeddings)


@pytest.fixture
def report(prepared, store_root, dataset):
    embeddings, index = prepared
    return semantic.evaluate_dataset(
        dataset,
        sample_size=25,
        store_root=store_root,
        embeddings=embeddings,
        index=index,
    )


# =============================================================================
# Q3.5 — the same impressions as Q2.8
# =============================================================================


def test_evaluates_the_identical_impression_set_as_bm25(store_root, dataset, report):
    """The requirement itself. `recall.sample_impressions` is imported rather
    than reimplemented, so this asserts the import is actually being used."""
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    expected = recall.sample_impressions(impressions, "test", 25)
    assert report["impression_set_fingerprint"] == recall.impression_set_fingerprint(expected)


def test_the_sample_is_drawn_with_the_seed_and_size_it_was_given(prepared, store_root, dataset):
    """Structural, because the fixture cannot show it any other way.

    The fixture store holds only 2 evaluable test impressions, so
    `sample_impressions` returns all of them and the seed never changes the
    answer. A mutation altering the seed therefore survived every behavioural
    test here -- caught by mutation testing, fixed by asserting the call itself
    rather than its effect on a corpus too small to have one.

    On the real store the seed decides which 2,000 of 73,152 impressions are
    evaluated, and Q3.5 requires that to be BM25's set exactly.
    """
    embeddings, index = prepared
    seen: list[tuple] = []
    real = recall.sample_impressions

    def spy(impressions, split, size, seed=None):
        seen.append((split, size, seed))
        return real(impressions, split, size, seed)

    import unittest.mock

    with unittest.mock.patch.object(semantic.recall, "sample_impressions", spy):
        semantic.evaluate_dataset(
            dataset,
            sample_size=25,
            seed=config.GLOBAL_SEED,
            store_root=store_root,
            embeddings=embeddings,
            index=index,
        )
    assert seen == [("test", 25, config.GLOBAL_SEED)], (
        f"sample drawn with {seen}, not the split/size/seed this stage was given"
    )


def test_a_fingerprint_mismatch_raises_rather_than_writing_a_report(
    prepared, store_root, dataset, tmp_path, monkeypatch
):
    """The canary, and the most important test in this file.

    Two retrievers scored on different impressions produce two plausible
    numbers whose difference is meaningless. Nothing about the output looks
    wrong, so the check has to be structural — and a check nobody has watched
    fail is not a check.
    """
    embeddings, index = prepared
    (tmp_path / recall.REPORT_NAME).write_text(
        json.dumps({"datasets": {dataset: {"impression_set_fingerprint": "deadbeefdeadbeef"}}})
    )
    monkeypatch.setattr(semantic.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(semantic.ann, "load", lambda *a, **k: index)

    with pytest.raises(semantic.ImpressionSetMismatch, match="SAME impressions"):
        semantic.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    assert not (tmp_path / semantic.REPORT_NAME).exists(), (
        "a report was written despite a mismatched impression set"
    )


def test_matching_fingerprints_are_accepted(prepared, store_root, dataset, tmp_path, monkeypatch):
    """The other direction — a guard that refuses everything is useless."""
    embeddings, index = prepared
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    good = recall.impression_set_fingerprint(recall.sample_impressions(impressions, "test", 25))
    (tmp_path / recall.REPORT_NAME).write_text(
        json.dumps({"datasets": {dataset: {"impression_set_fingerprint": good}}})
    )
    monkeypatch.setattr(semantic.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(semantic.ann, "load", lambda *a, **k: index)

    dest = semantic.run(
        datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root
    )
    assert json.loads(dest.read_text())["impression_set_matches_bm25"] is True


def test_no_bm25_report_present_is_not_silently_a_pass(
    prepared, store_root, dataset, tmp_path, monkeypatch
):
    """With nothing to compare against, the flag must say so rather than claim
    a match that was never checked."""
    embeddings, index = prepared
    monkeypatch.setattr(semantic.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(semantic.ann, "load", lambda *a, **k: index)
    dest = semantic.run(
        datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root
    )
    assert json.loads(dest.read_text())["impression_set_matches_bm25"] is False


# =============================================================================
# The metric's definition — identical to Q2.8 by construction
# =============================================================================


def test_recall_values_are_proportions(report):
    for k, r in report["recall"].items():
        assert 0.0 <= r["mean"] <= 1.0, f"K={k} recall {r['mean']} outside [0, 1]"
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]


def test_recall_is_monotonic_in_k(report):
    values = [report["recall"][str(k)]["mean"] for k in config.RECALL_K_VALUES]
    assert values == sorted(values), f"recall not non-decreasing in K: {values}"


def test_the_pool_is_the_whole_corpus(report, prepared):
    embeddings, _ = prepared
    assert report["corpus_articles"] == embeddings.n_articles
    assert report["pool"] == "full_corpus"


def test_impressions_with_no_click_are_excluded_and_counted(report, store_root, dataset):
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    test_split = impressions.filter(pl.col("split") == "test")
    expected = len(test_split.filter(pl.col("clicked_ids").list.len() == 0))
    assert report["impressions_undefined_excluded"] == expected


def test_report_records_what_is_needed_to_reproduce_it(report):
    for key in (
        "seed",
        "sample_size",
        "encoder",
        "encoder_revision",
        "embedding_fingerprint",
        "pooling",
        "ann_ef_search",
        "impression_set_fingerprint",
    ):
        assert key in report and report[key] is not None, key


# =============================================================================
# Cold start behaves exactly as BM25's does (D6)
# =============================================================================


def test_cold_start_impressions_are_served_the_bm25_popularity_fallback(
    prepared, store_root, dataset, monkeypatch
):
    """Both retrievers must behave identically for users with no history, so a
    Q3.6 difference cannot come from the fallback rather than the ranking."""
    embeddings, index = prepared
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    if not query.training_popularity(impressions):
        pytest.skip(f"{dataset} fixture has no training clicks to rank")

    # Force every impression cold by handing the stage an empty history.
    monkeypatch.setattr(
        semantic.user_vector,
        "user_vectors",
        lambda imps, *a, **k: (
            np.zeros((len(imps), embeddings.dim), dtype=np.float32),
            np.zeros(len(imps), dtype=bool),
        ),
    )
    out = semantic.evaluate_dataset(
        dataset, sample_size=25, store_root=store_root, embeddings=embeddings, index=index
    )
    assert out["cold_start_impressions"] == out["impressions_evaluated"]

    # Assert the recall is EXACTLY what the popularity list achieves, computed
    # independently here. The first version of this test only checked the value
    # was in [0, 1], which is also true of an empty candidate list -- so a
    # mutation replacing the fallback with `[]` survived it. Mutation testing
    # caught that; the fix is a stronger assertion, not a weaker mutation.
    sample = recall.sample_impressions(
        feature_store.read_table(store_root / "processed" / dataset / "impressions.parquet"),
        "test",
        25,
    )
    # popularity_ranking, not cold_start_candidates. The latter ranks only
    # articles CLICKED during training, so it runs out before K on a corpus
    # where few were -- while BM25's zero-score tie-break keeps going through
    # the rest of the corpus. The two retrievers were being compared over
    # different numbers of candidates. Found by tests/test_compare.py on a
    # fixture with 3 training-clicked articles and 7 in the corpus; invisible on
    # the real corpora, where the popularity table is larger than K.
    fallback = query.popularity_ranking(
        embeddings.article_ids,
        query.training_popularity(impressions),
        k=max(config.RECALL_K_VALUES),
    )
    assert fallback, "the fixture must have a non-empty popularity list for this to bite"

    # The value assertion below is vacuous whenever no clicked article happens
    # to sit in the popularity list -- both the real fallback and an empty one
    # then score 0.0. That is the case on this fixture, so a mutation replacing
    # the fallback with `[]` survived it. Assert the CALL as well: if the stage
    # stops consulting the ranking, this fails regardless of what the data
    # happens to make the numbers do.
    import unittest.mock

    calls: list[dict] = []
    # Bound BEFORE patching. `semantic.query` is the same module object as
    # `query`, so patching one patches both -- calling query.popularity_ranking
    # inside the spy made it call itself.
    real_fallback = query.popularity_ranking

    def spy(article_ids, popularity, k):
        out = real_fallback(article_ids, popularity, k=k)
        calls.append({"k": k, "returned": out})
        return out

    with unittest.mock.patch.object(semantic.query, "popularity_ranking", spy):
        semantic.evaluate_dataset(
            dataset,
            sample_size=25,
            store_root=store_root,
            embeddings=embeddings,
            index=index,
        )
    assert len(calls) == 1, "the cold-start fallback was never consulted"
    assert calls[0]["returned"], "the fallback list came back empty"
    assert calls[0]["k"] == max(config.RECALL_K_VALUES)
    for k in config.RECALL_K_VALUES:
        expected = float(
            np.mean(
                [
                    len(set(row["clicked_ids"]).intersection(fallback[:k]))
                    / len(row["clicked_ids"])
                    for row in sample.iter_rows(named=True)
                ]
            )
        )
        assert out["recall"][str(k)]["mean"] == pytest.approx(expected, abs=1e-9), (
            f"K={k}: cold-start recall is not the popularity fallback's"
        )


def test_cold_start_count_is_reported_not_averaged_away(report):
    assert "cold_start_impressions" in report
    assert report["cold_start_impressions"] <= report["impressions_evaluated"]


# =============================================================================
# Reproducibility
# =============================================================================


def test_the_report_is_byte_identical_across_runs(
    prepared, store_root, dataset, tmp_path, monkeypatch
):
    embeddings, index = prepared
    monkeypatch.setattr(semantic.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(semantic.ann, "load", lambda *a, **k: index)
    kw = dict(datasets=(dataset,), sample_size=25, store_root=store_root)
    a = semantic.run(out_dir=tmp_path / "a", **kw)
    b = semantic.run(out_dir=tmp_path / "b", **kw)
    assert a.read_bytes() == b.read_bytes()


def test_no_wall_clock_leaks_into_the_report(prepared, store_root, dataset, tmp_path, monkeypatch):
    """The mistake recall.py made and its determinism test caught."""
    embeddings, index = prepared
    monkeypatch.setattr(semantic.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(semantic.ann, "load", lambda *a, **k: index)
    dest = semantic.run(
        datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root
    )
    text = dest.read_text()
    assert "timing_seconds" not in text
    assert "_per_impression_recall" not in text


def test_report_has_at_least_six_dataset_k_entries(store_root, tmp_path, monkeypatch, request):
    """Q3.5 — both datasets, K in {50, 100, 200}."""
    built = {}
    for ds in ("mind_small", "ebnerd_demo"):
        articles = feature_store.read_table(
            store_root / "feature_store" / ds / "article_features.parquet"
        ).select("article_id", "text")
        e = embed.build(articles, spec=_spec(), encoder=_fake_encoder())
        built[ds] = (e, ann.build(e))

    monkeypatch.setattr(semantic.embed, "load", lambda d, *a, **k: built[_ds_of(d)][0])
    monkeypatch.setattr(semantic.ann, "load", lambda d, *a, **k: built[_ds_of(d)][1])

    dest = semantic.run(
        datasets=("mind_small", "ebnerd_demo"),
        out_dir=tmp_path,
        sample_size=10,
        store_root=store_root,
    )
    data = json.loads(dest.read_text())
    entries = [(d, k) for d, v in data["datasets"].items() for k in v["recall"]]
    assert len(entries) >= 6
    for v in data["datasets"].values():
        for r in v["recall"].values():
            assert "ci_low" in r and "n" in r


def _ds_of(path) -> str:
    """Recover the dataset name from the artefact directory a loader was given."""
    return str(path).rstrip("/").rsplit("/", 1)[-1]
