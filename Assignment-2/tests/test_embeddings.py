"""Q3.1, Q3.2 — article embeddings.

The failure guarded against is not a crash. It is a matrix of plausible numbers:
right shape, right dimension, unit norms, every coverage assertion green — and
produced by the wrong model, or with rows belonging to the wrong articles, or
from text this pipeline never indexed. Every one of those yields a recall figure
that looks exactly like a real one.

Most tests here inject a deterministic stand-in encoder rather than the real
510 MB model, because `make test` must pass in a clean clone with no network
(tests/conftest.py, OQ-9). A stand-in proves the plumbing and never the model, so
`test_the_real_pinned_encoder_produces_the_declared_dimension` loads the actual
weights and skips, naming what is missing, when they are not cached.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.retrieval import embed


def fake_encoder(dim: int = 8, seed: int = 0) -> embed.Encoder:
    """A stand-in that is a deterministic function OF THE TEXT.

    Hashing the text matters. An encoder returning `rng.normal(size=(n, dim))`
    would pass every shape, norm and coverage assertion in this file while
    ignoring its input entirely — so a bug that fed it the wrong strings, or the
    same string every time, would be invisible. Deriving each row from its own
    text means identical inputs give identical vectors and different inputs give
    different ones, which is the one property the real encoder has that these
    tests actually depend on.
    """

    def encode(texts):
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # hashlib, not the builtin hash(): PYTHONHASHSEED randomises string
            # hashing per process, so builtin hash() would make this stand-in
            # differ between runs -- and a helper whose docstring promises
            # determinism while not having it is worse than no helper.
            digest = hashlib.sha256(f"{seed}\x00{text}".encode()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            out[i] = rng.normal(size=dim)
        return out

    return encode


def spec(dim: int = 8) -> config.EncoderSpec:
    return config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )


@pytest.fixture
def articles(store_root, dataset) -> pl.DataFrame:
    return feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")


@pytest.fixture
def built(articles) -> embed.ArticleEmbeddings:
    return embed.build(articles, spec=spec(), encoder=fake_encoder())


# =============================================================================
# Q3.1 — coverage, no NaN, unit norm, dimension recorded
# =============================================================================


def test_coverage_and_norms(built, articles):
    """Q3.1's four conditions, together, because they fail together."""
    assert built.n_articles == len(articles)
    assert built.article_ids == [str(a) for a in articles["article_id"].to_list()]
    assert np.isfinite(built.vectors).all()
    norms = np.linalg.norm(built.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"norms range {norms.min()} to {norms.max()}"


def test_an_article_with_empty_text_still_gets_a_row(articles):
    """`ebnerd_small` has 19. Dropping them would make row positions disagree
    with the BM25 index, and Q3.6 compares the two row for row."""
    with_empty = pl.concat(
        [
            articles,
            pl.DataFrame(
                {"article_id": ["EMPTY-1", "EMPTY-2"], "text": ["", None]},
                schema=articles.schema,
            ),
        ]
    )
    out = embed.build(with_empty, spec=spec(), encoder=fake_encoder())
    assert out.n_articles == len(articles) + 2
    assert "EMPTY-1" in out.article_ids and "EMPTY-2" in out.article_ids


def test_row_order_matches_the_input_order_article_for_article(articles):
    """The silent catastrophe: vectors offset by one against their ids. Every
    norm is still 1.0 and every count still matches."""
    out = embed.build(articles, spec=spec(), encoder=fake_encoder())
    shuffled = articles.sort("article_id", descending=True)
    other = embed.build(shuffled, spec=spec(), encoder=fake_encoder())

    by_id = dict(zip(other.article_ids, other.vectors, strict=True))
    for article_id, vector in zip(out.article_ids, out.vectors, strict=True):
        assert np.allclose(by_id[article_id], vector), f"{article_id} moved between builds"


def test_duplicate_article_ids_are_refused(articles):
    doubled = pl.concat([articles, articles.head(1)])
    with pytest.raises(ValueError, match="duplicate article_id"):
        embed.build(doubled, spec=spec(), encoder=fake_encoder())


# =============================================================================
# The text is taken, not rebuilt (D3)
# =============================================================================


def test_the_embedded_string_is_the_feature_stores_text_column(articles):
    """Q2.2's contract, extended to Q3: if the semantic retriever saw richer text
    than the lexical one, Q3.6 would be measuring the text, not the
    representation."""
    seen: list[str] = []

    def recording(texts):
        seen.extend(texts)
        return fake_encoder()(texts)

    embed.build(articles, spec=spec(), encoder=recording)
    expected = [f"passage: {t or ''}" for t in articles["text"].to_list()]
    assert seen == expected


def test_build_refuses_a_frame_without_the_text_column(articles):
    with pytest.raises(ValueError, match=r"article_id.*text|text"):
        embed.build(articles.select("article_id"), spec=spec(), encoder=fake_encoder())


def test_the_passage_prefix_is_applied_and_is_not_the_query_prefix(articles):
    """The e5 model card requires it "even for non-English texts". Omitting it
    degrades the space quietly — unit norms, plausible recall, no error."""
    seen: list[str] = []

    def recording(texts):
        seen.extend(texts)
        return fake_encoder()(texts)

    embed.build(articles.head(3), spec=spec(), encoder=recording)
    assert all(t.startswith("passage: ") for t in seen)
    assert not any(t.startswith("query: ") for t in seen)


# =============================================================================
# Normalisation refuses what it cannot represent
# =============================================================================


def test_a_zero_vector_is_refused_not_silently_turned_into_nan(articles):
    """0/0 is NaN, a NaN score sorts unpredictably, and a corrupted recall number
    looks exactly like a real one."""

    def zeros(texts):
        return np.zeros((len(texts), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="zero length"):
        embed.build(articles, spec=spec(), encoder=zeros)


def test_non_finite_output_is_refused(articles):
    def nans(texts):
        out = fake_encoder()(texts)
        out[0, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        embed.build(articles, spec=spec(), encoder=nans)


def test_an_encoder_returning_the_wrong_row_count_is_refused(articles):
    """Silently zipping mismatched lengths would map every article after the gap
    to another article's vector."""

    def short(texts):
        return fake_encoder()(texts)[:-1]

    with pytest.raises(embed.EmbeddingMismatch, match="vectors for"):
        embed.build(articles, spec=spec(), encoder=short)


def test_an_encoder_returning_the_wrong_dimension_is_refused(articles):
    with pytest.raises(embed.EmbeddingMismatch, match="dimension"):
        embed.build(articles, spec=spec(dim=8), encoder=fake_encoder(dim=16))


# =============================================================================
# Persistence, and the fingerprint enforced ON READ (design_rationale 2.10)
# =============================================================================


def test_round_trips_through_disk_unchanged(built, tmp_path):
    embed.save(built, tmp_path)
    back = embed.load(tmp_path, expect=built.spec)
    assert back.article_ids == built.article_ids
    assert np.array_equal(back.vectors, built.vectors)
    assert back.dim == built.dim


def test_two_saves_are_byte_identical(built, tmp_path):
    """Q1.13 applies to every stage. If an unchanged rebuild differs, "the
    embeddings changed" stops being evidence of anything."""
    a, b = tmp_path / "a", tmp_path / "b"
    embed.save(built, a)
    embed.save(built, b)
    for name in (embed.VECTORS_FILE, embed.ARTICLE_IDS_FILE, embed.META_FILE):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_the_manifest_records_what_q31_requires(built, tmp_path):
    embed.save(built, tmp_path, device="cpu")
    meta = json.loads((tmp_path / embed.META_FILE).read_text())
    assert meta["dim"] == built.dim
    assert meta["model_id"] == built.spec.model_id
    assert meta["revision"] == built.spec.revision
    assert meta["n_articles"] == built.n_articles
    assert meta["normalised"] is True
    assert meta["device"] == "cpu"


def test_the_manifest_carries_no_wall_clock_anything(built, tmp_path):
    """The mistake recall.py made and its determinism test caught."""
    embed.save(built, tmp_path)
    meta = json.loads((tmp_path / embed.META_FILE).read_text())
    for key in meta:
        assert "time" not in key.lower() or key == "max_seq_length"
    assert "built_at" not in meta and "timestamp" not in meta


def test_embeddings_from_another_model_are_refused_on_read(built, tmp_path):
    """The canary. Dimensions can even agree — this is what catches it anyway."""
    embed.save(built, tmp_path)
    other = config.EncoderSpec(
        model_id="test/some-other-model",
        revision="1" * 40,
        dim=built.dim,
        max_seq_length=128,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )
    with pytest.raises(embed.EmbeddingMismatch, match="different model"):
        embed.load(tmp_path, expect=other)


def test_a_changed_sequence_length_is_a_different_fingerprint():
    """Re-encoding MIND at 256 changes 9.1% of its vectors (q3_plan D2's measured
    truncation rate) while model, revision and dimension all still match. That
    must not look like a cache hit."""
    a = spec()
    b = config.EncoderSpec(
        model_id=a.model_id,
        revision=a.revision,
        dim=a.dim,
        max_seq_length=256,
        passage_prefix=a.passage_prefix,
        query_prefix=a.query_prefix,
    )
    assert embed.fingerprint(a) != embed.fingerprint(b)


def test_a_changed_prefix_is_a_different_fingerprint():
    a = spec()
    b = config.EncoderSpec(
        model_id=a.model_id,
        revision=a.revision,
        dim=a.dim,
        max_seq_length=a.max_seq_length,
        passage_prefix="",
        query_prefix=a.query_prefix,
    )
    assert embed.fingerprint(a) != embed.fingerprint(b)


def test_a_stale_embedding_version_is_refused(built, tmp_path):
    embed.save(built, tmp_path)
    meta = json.loads((tmp_path / embed.META_FILE).read_text())
    meta["embedding_version"] = config.EMBEDDING_VERSION + 1
    (tmp_path / embed.META_FILE).write_text(json.dumps(meta))
    with pytest.raises(embed.EmbeddingMismatch, match="version"):
        embed.load(tmp_path, expect=built.spec)


def test_a_truncated_vectors_file_is_refused(built, tmp_path):
    """A correct manifest and the wrong number of rows — which the fingerprint
    check alone would pass."""
    embed.save(built, tmp_path)
    with (tmp_path / embed.VECTORS_FILE).open("wb") as fh:
        np.save(fh, built.vectors[:-1])
    with pytest.raises(embed.EmbeddingMismatch, match=r"claim .* articles but store"):
        embed.load(tmp_path, expect=built.spec)


def test_a_matching_artefact_loads(built, tmp_path):
    """The other direction. A guard that refuses everything passes every test
    above and is useless."""
    embed.save(built, tmp_path)
    assert embed.load(tmp_path, expect=built.spec).n_articles == built.n_articles


def test_embeddings_dir_is_per_dataset():
    assert embed.embeddings_dir("mind_small") != embed.embeddings_dir("ebnerd_demo")
    assert embed.embeddings_dir("mind_small").name == "mind_small"


# =============================================================================
# Q3.2 — device resolution, and the real model
# =============================================================================


def test_cpu_fallback_path(monkeypatch):
    """Q3.2: `make all` must complete in a grader's environment. "auto" resolves
    to CPU with no GPU, and never raises for its absence."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert embed.resolve_device("auto") == "cpu"
    assert embed.resolve_device("cpu") == "cpu"


def test_explicit_cuda_refuses_rather_than_downgrading_silently(monkeypatch):
    """A silent fallback would make a "GPU run" write CPU timings into
    scale.json with nothing to show the difference."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        embed.resolve_device("cuda")


def test_auto_prefers_cuda_when_present(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert embed.resolve_device("auto") == "cuda"


def test_an_unknown_device_is_refused():
    with pytest.raises(ValueError, match="device must be"):
        embed.resolve_device("tpu")


@pytest.mark.slow
def test_the_real_pinned_encoder_produces_the_declared_dimension():
    """The one test the stand-in cannot stand in for.

    Every other test here proves the plumbing around the encoder. This proves
    `config.ENCODER` describes the model that actually exists — that the pinned
    revision loads, reports 384 dimensions, and that two encodes of the same text
    agree. Skipped rather than failed when the weights are not cached, because
    `make test` must run without the network (OQ-9); `make test-full` runs it.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        encoder = embed.load_encoder(device="cpu")
    except (LocalEntryNotFoundError, OSError) as exc:
        pytest.skip(f"pinned encoder weights not cached; run `make download-models` ({exc})")

    out = encoder(["passage: hello world", "passage: goddag verden"])
    assert out.shape == (2, config.ENCODER.dim)
    assert np.isfinite(out).all()

    again = encoder(["passage: hello world"])
    assert np.allclose(out[0], again[0], atol=1e-5), "the same text encoded twice must agree"


# =============================================================================
# Space diagnostics (design_rationale 1.18) — an interpretation aid, not a metric
# =============================================================================


def test_diagnostics_reports_observations_for_every_dataset(tmp_path, monkeypatch, articles):
    built = embed.build(articles, spec=spec(), encoder=fake_encoder())
    monkeypatch.setattr(embed, "load", lambda *a, **k: built)
    dest = embed.run_diagnostics(
        datasets=("mind_small", "ebnerd_demo"), out_dir=tmp_path, sample_size=20
    )
    report = json.loads(dest.read_text())
    assert set(report["datasets"]) == {"mind_small", "ebnerd_demo"}
    for d in report["datasets"].values():
        for key in (
            "nearest_neighbour_similarity",
            "random_pair_similarity",
            "margin",
            "separation",
            "self_nn_rank1",
        ):
            assert key in d, key
        for band in ("nearest_neighbour_similarity", "random_pair_similarity"):
            assert set(d[band]) == {"mean", "p05", "p50", "p95"}


def test_diagnostics_carries_no_derived_concepts(tmp_path, monkeypatch, articles):
    """Observations only. `effective_rank_ratio` was proposed and dropped: if
    effective rank matters later it can be added then, on evidence, rather than
    shipped as a concept nobody asked a question about."""
    built = embed.build(articles, spec=spec(), encoder=fake_encoder())
    monkeypatch.setattr(embed, "load", lambda *a, **k: built)
    dest = embed.run_diagnostics(datasets=("mind_small",), out_dir=tmp_path, sample_size=20)
    assert "effective_rank_ratio" not in dest.read_text()


def test_separation_is_the_stated_subtraction(tmp_path, monkeypatch, articles):
    """`separation` must be derivable from two reported percentiles, so a reader
    can check it rather than trust it."""
    built = embed.build(articles, spec=spec(), encoder=fake_encoder())
    monkeypatch.setattr(embed, "load", lambda *a, **k: built)
    dest = embed.run_diagnostics(datasets=("mind_small",), out_dir=tmp_path, sample_size=20)
    d = json.loads(dest.read_text())["datasets"]["mind_small"]
    expected = d["nearest_neighbour_similarity"]["p05"] - d["random_pair_similarity"]["p95"]
    assert d["separation"] == pytest.approx(expected, abs=1e-4)


def test_nearest_neighbour_excludes_the_article_itself(tmp_path, monkeypatch, articles):
    """Every vector's best match is itself at 1.0. Left in, the whole
    distribution would collapse to 1.0 and measure nothing."""
    built = embed.build(articles, spec=spec(), encoder=fake_encoder())
    monkeypatch.setattr(embed, "load", lambda *a, **k: built)
    dest = embed.run_diagnostics(datasets=("mind_small",), out_dir=tmp_path, sample_size=20)
    d = json.loads(dest.read_text())["datasets"]["mind_small"]
    assert d["nearest_neighbour_similarity"]["mean"] < 0.999


def test_diagnostics_are_byte_identical_across_runs(tmp_path, monkeypatch, articles):
    built = embed.build(articles, spec=spec(), encoder=fake_encoder())
    monkeypatch.setattr(embed, "load", lambda *a, **k: built)
    a = embed.run_diagnostics(datasets=("mind_small",), out_dir=tmp_path / "a", sample_size=20)
    b = embed.run_diagnostics(datasets=("mind_small",), out_dir=tmp_path / "b", sample_size=20)
    assert a.read_bytes() == b.read_bytes()


# =============================================================================
# The fingerprint-based skip — three re-encodes' worth of evidence
# =============================================================================


def test_a_matching_artefact_is_reused_without_encoding(tmp_path, store_root, monkeypatch):
    """Make invalidates this stage on embed.py's mtime, which re-ran a
    24-minute encode three times to produce byte-identical vectors. The skip
    asks the honest question instead: does what is on disk match the config we
    would encode under?"""
    calls = {"n": 0}

    def counting(texts):
        calls["n"] += 1
        return fake_encoder()(texts)

    first = embed.run(
        datasets=("mind_small",),
        store_root=store_root,
        out_root=tmp_path,
        encoder=counting,
        spec=spec(),
        device="cpu",
    )
    assert first["mind_small"]["reused"] is False and calls["n"] == 1

    second = embed.run(
        datasets=("mind_small",),
        store_root=store_root,
        out_root=tmp_path,
        encoder=counting,
        spec=spec(),
        device="cpu",
    )
    assert second["mind_small"]["reused"] is True
    assert calls["n"] == 1, "the encoder ran again despite a matching fingerprint"


def test_force_re_encodes_even_when_the_fingerprint_matches(tmp_path, store_root):
    calls = {"n": 0}

    def counting(texts):
        calls["n"] += 1
        return fake_encoder()(texts)

    kw = dict(
        datasets=("mind_small",),
        store_root=store_root,
        out_root=tmp_path,
        encoder=counting,
        spec=spec(),
        device="cpu",
    )
    embed.run(**kw)
    out = embed.run(**kw, force=True)
    assert out["mind_small"]["reused"] is False
    assert calls["n"] == 2


def test_a_different_spec_is_not_a_cache_hit(tmp_path, store_root):
    """The skip must key on the fingerprint, not on the files existing."""
    embed.run(
        datasets=("mind_small",),
        store_root=store_root,
        out_root=tmp_path,
        encoder=fake_encoder(),
        spec=spec(),
        device="cpu",
    )
    other = config.EncoderSpec(
        model_id="test/stand-in",
        revision="0" * 40,
        dim=8,
        max_seq_length=256,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )
    out = embed.run(
        datasets=("mind_small",),
        store_root=store_root,
        out_root=tmp_path,
        encoder=fake_encoder(),
        spec=other,
        device="cpu",
    )
    assert out["mind_small"]["reused"] is False
