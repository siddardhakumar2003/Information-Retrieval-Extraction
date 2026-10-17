"""Q3.6 — the paired lexical-vs-semantic comparison.

The failure guarded against is a comparison that is not actually paired. Two
retrievers scored on different impressions, or slices that quietly drop rows,
produce deltas that look precise and mean nothing — and the report carries
confidence intervals, which makes them look *more* trustworthy rather than less.

The fixture corpus is tiny, so these tests prove the mechanism. The numbers in
the report are measured on the real corpora by `compare.run()`.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.retrieval import ann, compare, embed, recall, semantic

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
def prepared(store_root, dataset, monkeypatch):
    """Point embed.load / ann.load at in-memory artefacts built from the fixture."""
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    embeddings = embed.build(articles, spec=_spec(), encoder=_fake_encoder())
    index = ann.build(embeddings)
    monkeypatch.setattr(compare.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(compare.ann, "load", lambda *a, **k: index)
    return embeddings, index


@pytest.fixture
def report(prepared, store_root, dataset):
    return compare.evaluate_dataset(dataset, sample_size=25, store_root=store_root)


@pytest.fixture
def sample(store_root, dataset):
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    return recall.sample_impressions(impressions, "test", 25)


# =============================================================================
# 1. The fingerprint check fires against BOTH committed reports
# =============================================================================


def _write_reports(tmp_path, dataset, bm25_fp, semantic_fp):
    (tmp_path / recall.REPORT_NAME).write_text(
        json.dumps({"datasets": {dataset: {"impression_set_fingerprint": bm25_fp}}})
    )
    (tmp_path / semantic.REPORT_NAME).write_text(
        json.dumps({"datasets": {dataset: {"impression_set_fingerprint": semantic_fp}}})
    )


def test_both_committed_reports_are_checked_not_just_bm25(
    prepared, store_root, dataset, sample, tmp_path
):
    """Agreeing with BM25 while disagreeing with the semantic report would still
    make the pairing fictional. Both are checked, so both are tested."""
    good = recall.impression_set_fingerprint(sample)

    # BM25 agrees, semantic does not.
    _write_reports(tmp_path, dataset, good, "deadbeefdeadbeef")
    with pytest.raises(semantic.ImpressionSetMismatch, match="semantic"):
        compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)

    # Semantic agrees, BM25 does not.
    _write_reports(tmp_path, dataset, "deadbeefdeadbeef", good)
    with pytest.raises(semantic.ImpressionSetMismatch, match="bm25"):
        compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)


def test_matching_fingerprints_are_accepted(prepared, store_root, dataset, sample, tmp_path):
    """The other direction — a guard that refuses everything is useless."""
    good = recall.impression_set_fingerprint(sample)
    _write_reports(tmp_path, dataset, good, good)
    dest = compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    assert json.loads(dest.read_text())["impression_sets_match_committed_reports"] is True


# =============================================================================
# 2. A mismatch raises rather than producing a wrong number silently
# =============================================================================


def test_a_mismatch_raises_and_writes_nothing(prepared, store_root, dataset, tmp_path):
    """The point of the check. A written report is a report someone will quote,
    and its deltas would be differences between unrelated rows."""
    _write_reports(tmp_path, dataset, "deadbeefdeadbeef", "deadbeefdeadbeef")
    with pytest.raises(semantic.ImpressionSetMismatch, match="pairing is not real"):
        compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    assert not (tmp_path / compare.REPORT_NAME).exists(), (
        "a comparison was written despite a mismatched impression set"
    )


def test_the_error_names_which_report_disagreed(prepared, store_root, dataset, sample, tmp_path):
    """ "Mismatch" alone tells whoever hits this nothing about what to rebuild."""
    good = recall.impression_set_fingerprint(sample)
    _write_reports(tmp_path, dataset, good, "deadbeefdeadbeef")
    with pytest.raises(semantic.ImpressionSetMismatch) as excinfo:
        compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    message = str(excinfo.value)
    assert dataset in message and "deadbeefdeadbeef" in message and good in message


# =============================================================================
# 3. Cold-start delta is EXACTLY zero
# =============================================================================


def test_the_two_cold_start_paths_return_the_identical_ranking(store_root, dataset):
    """The property that makes a zero cold-start delta *mean* something.

    BM25 reaches its cold-start ranking indirectly: an empty query scores every
    article 0, so `top_k` orders by its tie-break, which is training-period
    popularity. The semantic side calls `query.cold_start_candidates` directly.
    Two different code paths, and Q3.6 reports their difference as a slice — so
    if they diverge, the report shows a non-zero delta that looks like a finding
    about retrieval and is actually a bug.

    Written after the first version of this test failed for the wrong reason:
    forcing only the semantic side cold left BM25 with its history, which is not
    cold start at all. The claim worth testing is that the two RANKINGS agree.

    Verified on the real corpora too (mind_small and ebnerd_small both agree at
    K=200) — but note the dependency the next test pins.
    """
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    popularity = compare.query.training_popularity(impressions)
    if not popularity:
        pytest.skip(f"{dataset} fixture has no training clicks to rank")

    index = compare.idx.build(articles, config.tokeniser_for(dataset))
    tie_break = np.asarray(
        compare.query.popularity_array(index.article_ids, popularity), dtype=np.int64
    )
    k = min(50, index.n_docs)

    lexical_cold = [
        index.article_ids[int(d)]
        for d in compare.bm25.top_k(compare.bm25.score(index, []), k, tie_break=tie_break)
    ]
    semantic_cold = compare.query.cold_start_candidates(impressions, k=k, popularity=popularity)

    assert lexical_cold[: len(semantic_cold)] == semantic_cold[: len(lexical_cold)], (
        "the two cold-start paths disagree; the cold_start slice would report a "
        "non-zero delta that is a bug rather than a finding"
    )


def test_the_cold_start_equivalence_depends_on_index_ordering(store_root, dataset):
    """Pinning an implicit dependency the test above rests on.

    BM25's tie-break falls through to `doc_id` — the article's POSITION in the
    index — while `cold_start_candidates` falls through to `article_id`. Those
    two orders coincide only because the feature store emits articles sorted by
    `article_id`. If that ever changed, the two cold paths would silently
    diverge for every article with equal popularity, which is most of them.
    """
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id")
    ids = [str(a) for a in articles["article_id"].to_list()]
    assert ids == sorted(ids), (
        "the feature store no longer emits articles sorted by article_id, so "
        "BM25's doc_id tie-break and cold_start_candidates' article_id tie-break "
        "no longer agree — see test_the_two_cold_start_paths_return_the_identical_ranking"
    )


def test_an_all_cold_sample_reports_a_zero_delta(prepared, store_root, dataset, monkeypatch):
    """The slice-level consequence, forced on BOTH sides.

    EB-NeRD has 0 cold-start impressions in the real sample and the fixture has
    none either, so this has to be manufactured or it asserts nothing.
    """
    empty_history = feature_store.read_table(
        store_root / "processed" / dataset / "history.parquet"
    ).head(0)
    real_read = feature_store.read_table

    def read(path, *a, **k):
        table = real_read(path, *a, **k)
        return empty_history if path.name == "history.parquet" else table

    monkeypatch.setattr(compare.feature_store, "read_table", read)
    out = compare.evaluate_dataset(dataset, sample_size=25, store_root=store_root)

    cold = out["slices"]["cold_start"]
    assert cold["n"] == out["impressions_evaluated"], "not every impression came out cold"
    for k in config.RECALL_K_VALUES:
        cell = cold["k"][str(k)]
        assert cell["delta"] == 0.0, (
            f"K={k}: cold-start delta is {cell['delta']}, so the two retrievers "
            f"no longer serve the same fallback"
        )
        assert cell["lexical"] == cell["semantic"]
        assert cell["winner"] == "neither"


def test_cold_start_slice_exists_even_when_empty(report):
    """An absent slice and an empty slice look the same in a report unless the
    key is always present."""
    assert "cold_start" in report["slices"]
    assert "warm" in report["slices"]


# =============================================================================
# 4. Slice masks partition — no overlap, no gaps
# =============================================================================


def test_head_and_tail_partition_the_sample(store_root, dataset, sample):
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    history = feature_store.read_table(store_root / "processed" / dataset / "history.parquet")
    slices = compare.build_slices(sample, history, impressions)

    head, tail = slices["article_head"], slices["article_tail"]
    assert not (head & tail).any(), "an impression is in both head and tail"
    assert (head | tail).all(), "an impression is in neither head nor tail"
    assert head.sum() + tail.sum() == len(sample)


def test_cold_and_warm_partition_the_sample(report):
    cold, warm = report["slices"]["cold_start"], report["slices"]["warm"]
    assert cold["n"] + warm["n"] == report["impressions_evaluated"]


def test_history_quartiles_do_not_overlap_unless_the_distribution_is_degenerate(
    store_root, dataset, sample
):
    """Short is <= p25 and long is >= p75, so they overlap only when the
    quartiles coincide — which happens when nearly every user has the same
    history length. Asserted rather than assumed, because a silent overlap
    would double-count impressions across two reported slices.
    """
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    history = feature_store.read_table(store_root / "processed" / dataset / "history.parquet")
    lengths = compare.history_lengths(sample, history)
    q1, q3 = np.percentile(lengths, [25, 75])
    slices = compare.build_slices(sample, history, impressions)

    overlap = slices["history_short"] & slices["history_long"]
    if q1 < q3:
        assert not overlap.any(), "history quartile slices overlap despite q25 < q75"
    else:
        assert overlap.all() == slices["history_short"].all()


def test_history_lengths_are_aligned_to_sample_row_order(store_root, dataset, sample):
    """Off-by-one alignment here would slice on the wrong user's history and
    every reported number would still look reasonable."""
    history = feature_store.read_table(store_root / "processed" / dataset / "history.parquet")
    lengths = compare.history_lengths(sample, history)
    counts = dict(history.group_by("user_id").len().iter_rows())
    for i, user in enumerate(sample["user_id"].to_list()):
        assert lengths[i] == counts.get(user, 0)


def test_head_articles_carry_about_half_the_test_clicks(store_root, dataset):
    """D8 chose test-period popularity because it splits near 50/50, unlike
    training-period popularity which put 85-98.6% in one bucket."""
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    head = compare.head_articles(impressions)
    test_split = impressions.filter(pl.col("split") == "test")
    clicked = [a for c in test_split["clicked_ids"].to_list() if c for a in c]
    if not clicked:
        pytest.skip(f"{dataset} fixture has no test-split clicks")
    covered = sum(1 for a in clicked if a in head)
    assert covered >= len(clicked) / 2


# =============================================================================
# The pairing itself
# =============================================================================


def test_delta_equals_the_difference_of_the_two_reported_means(report):
    """The paired mean difference and the difference of means are the same
    number; if they disagree, the two arms were not aligned row for row."""
    for k in config.RECALL_K_VALUES:
        cell = report["overall"][str(k)]
        assert cell["delta"] == pytest.approx(cell["semantic"] - cell["lexical"], abs=1e-6)


def test_a_system_compared_against_itself_has_exactly_zero_delta():
    """The sharpest check on the pairing. Identical inputs must give exactly
    0.0, not merely something small — a near-zero mean would also come from
    subtracting two independently-ordered arrays of the same values."""
    values = [0.0, 0.5, 1.0, 0.25, 0.75] * 4
    out = compare._paired(values, values)
    assert out["delta"] == 0.0
    assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0
    assert out["established"] is False
    assert out["winner"] == "neither"


def test_established_means_the_interval_excludes_zero_and_nothing_more(report):
    for cell in report["overall"].values():
        expected = cell["ci_low"] > 0.0 or cell["ci_high"] < 0.0
        assert cell["established"] is expected
        assert cell["ci_low"] <= cell["delta"] <= cell["ci_high"]


def test_the_winner_label_follows_the_sign_only_when_established(report):
    for cell in report["overall"].values():
        if not cell["established"]:
            assert cell["winner"] == "neither"
        else:
            assert cell["winner"] == ("semantic" if cell["delta"] > 0 else "lexical")


def test_an_empty_slice_is_reported_as_empty_not_as_zero(report):
    """A slice with no impressions has no delta. Reporting 0.0 would look like
    a measured tie."""
    out = compare._paired([1.0, 0.0], [1.0, 0.0], mask=np.zeros(2, dtype=bool))
    assert out["n"] == 0 and out["winner"] == "empty_slice"
    assert "delta" not in out


# =============================================================================
# Reproducibility
# =============================================================================


def test_the_report_is_byte_identical_across_runs(prepared, store_root, dataset, sample, tmp_path):
    good = recall.impression_set_fingerprint(sample)
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        _write_reports(tmp_path / sub, dataset, good, good)
    kw = dict(datasets=(dataset,), sample_size=25, store_root=store_root)
    a = compare.run(out_dir=tmp_path / "a", **kw)
    b = compare.run(out_dir=tmp_path / "b", **kw)
    assert a.read_bytes() == b.read_bytes()


def test_no_wall_clock_leaks_into_the_report(prepared, store_root, dataset, sample, tmp_path):
    good = recall.impression_set_fingerprint(sample)
    _write_reports(tmp_path, dataset, good, good)
    dest = compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    assert "timing_seconds" not in dest.read_text()


def test_the_report_records_the_slice_definitions_and_the_leakage_note(
    prepared, store_root, dataset, sample, tmp_path
):
    """A reader seeing test-period popularity in a leakage-sensitive project is
    right to ask; the answer belongs in the artefact, not only in code."""
    good = recall.impression_set_fingerprint(sample)
    _write_reports(tmp_path, dataset, good, good)
    dest = compare.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    data = json.loads(dest.read_text())
    assert set(data["slice_definitions"]) >= {
        "history_short",
        "history_long",
        "article_head",
        "article_tail",
        "cold_start",
        "warm",
    }
    assert "analysis-time descriptor" in data["slice_note"]
    assert data["paired"] is True
