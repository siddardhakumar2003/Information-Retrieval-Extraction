"""Q3.3 — the ANN index, and the measurement that keeps it honest.

The failure guarded against: an approximate index that quietly returns worse
neighbours than it should. Nothing errors, every row index is a valid integer,
and semantic recall comes out low in a way that reads as "embeddings
underperform BM25" rather than "our graph is misconfigured".

The fixture corpus here is tiny, so these tests prove the *mechanism*. The
agreement number that goes in the report is measured on the real corpora by
`ann.run()` and written to `reports/metrics/ann_recall.json`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from newsrec import config
from newsrec.retrieval import ann, embed

faiss = pytest.importorskip("faiss", reason="faiss-cpu is a declared dependency")


def synthetic(n: int = 400, dim: int = 32, seed: int = 0) -> embed.ArticleEmbeddings:
    """A clustered unit-norm corpus.

    Clustered rather than uniform on purpose: uniform random points in 32
    dimensions are all roughly equidistant, so *any* index looks equally good
    and the agreement measurement cannot discriminate. Real embeddings are
    clustered, and clusters are what a graph walk can get lost between.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(8, dim))
    v = centres[rng.integers(0, 8, n)] + 0.35 * rng.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    spec = config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )
    return embed.ArticleEmbeddings(
        vectors=v.astype(np.float32),
        article_ids=[f"A{i}" for i in range(n)],
        spec=spec,
    )


@pytest.fixture
def corpus():
    return synthetic()


@pytest.fixture
def built(corpus):
    return ann.build(corpus)


# =============================================================================
# Q3.3 — agreement with brute force
# =============================================================================


def test_ann_matches_bruteforce_on_fixture(corpus, built):
    """Q3.3's named condition."""
    exact = ann.ExactIndex(corpus.vectors)
    queries = ann.sample_queries(corpus, size=100)
    recall = ann.agreement_with_exact(built, exact, queries, k=20)
    assert recall >= config.ANN_RECALL_FLOOR, f"ANN agreement {recall:.4f} below the floor"


def test_the_agreement_check_can_actually_fail(corpus):
    """The canary. A threshold nobody has watched fail is not a threshold.

    efSearch=1 cripples the graph walk deliberately. If this passes, the
    measurement is not sensitive to index quality and the Q3.3 number above
    means nothing.
    """
    crippled = ann.build(corpus, ef_search=1)
    exact = ann.ExactIndex(corpus.vectors)
    queries = ann.sample_queries(corpus, size=100)
    recall = ann.agreement_with_exact(crippled, exact, queries, k=20)
    assert recall < config.ANN_RECALL_FLOOR, (
        f"efSearch=1 still scored {recall:.4f}; the check cannot detect a bad index"
    )


def test_higher_ef_search_never_hurts_agreement(corpus):
    """More of the graph explored cannot find fewer true neighbours. If this
    fails, efSearch is not reaching the index."""
    exact = ann.ExactIndex(corpus.vectors)
    queries = ann.sample_queries(corpus, size=100)
    scores = [
        ann.agreement_with_exact(ann.build(corpus, ef_search=ef), exact, queries, k=20)
        for ef in (4, 16, 64, 256)
    ]
    assert scores == sorted(scores), f"agreement not non-decreasing in efSearch: {scores}"


def test_exact_index_is_actually_exact(corpus):
    """The ground truth has to be ground truth. Checked against a naive
    full-sort rather than against itself."""
    exact = ann.ExactIndex(corpus.vectors)
    q = corpus.vectors[:5]
    got = exact.search(q, k=10).indices
    want = np.argsort(-(q @ corpus.vectors.T), axis=1)[:, :10]
    assert np.array_equal(got, want)


def test_every_article_is_its_own_nearest_neighbour(corpus, built):
    """The strongest available end-to-end check, and the semantic analogue of
    Q2's self-retrieval test: a vector's closest point must be itself."""
    top = built.search(corpus.vectors, k=1).indices[:, 0]
    assert (top == np.arange(len(corpus.vectors))).mean() >= 0.99


# =============================================================================
# Inner product is cosine only for unit vectors
# =============================================================================


def test_unnormalised_vectors_are_refused(corpus):
    """Otherwise inner product ranks by magnitude as well as direction, and a
    longer article silently outranks a more similar one."""
    scaled = embed.ArticleEmbeddings(
        vectors=(corpus.vectors * 3.0).astype(np.float32),
        article_ids=corpus.article_ids,
        spec=corpus.spec,
    )
    with pytest.raises(ann.AnnMismatch, match="not unit length"):
        ann.build(scaled)


def test_scores_are_cosine_similarities(corpus, built):
    got = built.search(corpus.vectors[:3], k=5)
    for row, qi in enumerate(range(3)):
        for col, doc in enumerate(got.indices[row]):
            expected = float(corpus.vectors[qi] @ corpus.vectors[doc])
            assert got.scores[row, col] == pytest.approx(expected, abs=1e-4)


# =============================================================================
# Determinism (Q1.13)
# =============================================================================


def test_two_builds_produce_identical_results(corpus, tmp_path):
    """faiss parallelises graph construction, and concurrent inserts link nodes
    in a non-fixed order — so a multi-threaded build gives a different graph
    every run and a recall figure that moves when nothing changed. `build`
    pins one thread; this asserts the pinning works."""
    a, b = ann.build(corpus), ann.build(corpus)
    ann.save(a, tmp_path / "a")
    ann.save(b, tmp_path / "b")
    assert (tmp_path / "a" / ann.INDEX_FILE).read_bytes() == (
        tmp_path / "b" / ann.INDEX_FILE
    ).read_bytes()


def test_search_is_repeatable(corpus, built):
    q = ann.sample_queries(corpus, size=50)
    assert np.array_equal(built.search(q, 20).indices, built.search(q, 20).indices)


def test_build_restores_the_thread_count_it_borrowed(corpus):
    """`build` pins faiss to one thread. Leaking that would silently
    single-thread every later search in the process."""
    before = faiss.omp_get_max_threads()
    ann.build(corpus)
    assert faiss.omp_get_max_threads() == before


# =============================================================================
# Persistence and the fingerprint enforced on read
# =============================================================================


def test_round_trips_through_disk(corpus, built, tmp_path):
    ann.save(built, tmp_path)
    back = ann.load(tmp_path, embeddings=corpus)
    q = ann.sample_queries(corpus, size=20)
    assert np.array_equal(back.search(q, 10).indices, built.search(q, 10).indices)


def test_an_index_built_over_other_vectors_is_refused(corpus, built, tmp_path):
    """The canary that matters most here. Row numbers from the wrong graph are
    all valid integers pointing at the wrong articles."""
    ann.save(built, tmp_path)
    other_spec = config.EncoderSpec(
        model_id="test/other",
        revision="1" * 40,
        dim=corpus.dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )
    other = embed.ArticleEmbeddings(corpus.vectors, corpus.article_ids, other_spec)
    with pytest.raises(ann.AnnMismatch, match="built over"):
        ann.load(tmp_path, embeddings=other)


def test_a_row_count_mismatch_is_refused(corpus, built, tmp_path):
    ann.save(built, tmp_path)
    short = embed.ArticleEmbeddings(corpus.vectors[:-5], corpus.article_ids[:-5], corpus.spec)
    with pytest.raises(ann.AnnMismatch, match="article ids were supplied"):
        ann.load(tmp_path, embeddings=short)


def test_the_manifest_records_the_parameters(built, tmp_path):
    ann.save(built, tmp_path)
    meta = json.loads((tmp_path / ann.META_FILE).read_text())
    assert meta["m"] == config.ANN_HNSW_M
    assert meta["ef_construction"] == config.ANN_HNSW_EF_CONSTRUCTION
    assert meta["ef_search"] == config.ANN_HNSW_EF_SEARCH
    assert meta["embedding_fingerprint"]


# =============================================================================
# The report, and its refusal to pass a bad index
# =============================================================================


def test_run_refuses_to_write_a_passing_report_for_a_bad_index(corpus, tmp_path, monkeypatch):
    """Q3.3 asks for the approximation to be MEASURED. A measurement that
    cannot fail is a decoration, and a degraded index would depress semantic
    recall in a way that looks like the embeddings underperforming."""
    monkeypatch.setattr(config, "ANN_HNSW_EF_SEARCH", 1)
    monkeypatch.setattr(ann.embed, "load", lambda *a, **k: corpus)
    with pytest.raises(ann.AnnMismatch, match="below the"):
        ann.run(
            datasets=("mind_small",),
            out_dir=tmp_path,
            write_index=False,
            ef_values=(1,),
            sample_size=100,
        )
    report = json.loads((tmp_path / ann.REPORT_NAME).read_text())
    assert report["all_datasets_meet_floor"] is False


def test_run_writes_a_sweep_so_the_choice_is_evidenced(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(ann.embed, "load", lambda *a, **k: corpus)
    dest = ann.run(
        datasets=("mind_small",),
        out_dir=tmp_path,
        write_index=False,
        ef_values=(64, config.ANN_HNSW_EF_SEARCH),
        sample_size=100,
    )
    report = json.loads(dest.read_text())
    d = report["datasets"]["mind_small"]
    assert set(d["sweep"]) == {"64", str(config.ANN_HNSW_EF_SEARCH)}
    assert d["floor"] == config.ANN_RECALL_FLOOR
    assert d["query_source"] == "article_vectors"


def test_the_report_carries_no_wall_clock_anything(corpus, tmp_path, monkeypatch):
    """Timings belong in scale.json, where churn is the point."""
    monkeypatch.setattr(ann.embed, "load", lambda *a, **k: corpus)
    dest = ann.run(
        datasets=("mind_small",),
        out_dir=tmp_path,
        write_index=False,
        ef_values=(config.ANN_HNSW_EF_SEARCH,),
        sample_size=100,
    )
    text = dest.read_text()
    assert "timing_seconds" not in text
    # Not just the top-level key. A `seconds` field lived inside each sweep
    # entry for one commit and this test passed the whole time, because it
    # looked only for the name it expected. Assert on the PROPERTY -- no
    # wall-clock anywhere -- rather than on one spelling of it.
    assert "seconds" not in text, "a wall-clock field is still in the report"

    # And the property the absence exists to protect, checked directly.
    second = ann.run(
        datasets=("mind_small",),
        out_dir=tmp_path / "again",
        write_index=False,
        ef_values=(config.ANN_HNSW_EF_SEARCH,),
        sample_size=100,
    )
    assert dest.read_bytes() == second.read_bytes(), "two runs of unchanged input differ"


def test_sample_queries_is_seeded_and_stable(corpus):
    a = ann.sample_queries(corpus, size=50)
    b = ann.sample_queries(corpus, size=50)
    assert np.array_equal(a, b)


def test_ann_dir_is_per_dataset():
    assert ann.ann_dir("mind_small") != ann.ann_dir("ebnerd_demo")
