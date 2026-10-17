"""Q4.8 — the evaluation harness.

The failure guarded against here is a number that is arithmetically correct and
describes the wrong thing: a ranking that is really the publisher's list order,
two methods scored over different candidate sets, or a report whose impressions
are not the ones Q2.8 measured.

The fixture corpus is tiny, so these prove the mechanism. The numbers in
`reports/metrics/eval.json` are measured on the real corpora by `harness.run()`.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import harness, metrics
from newsrec.retrieval import embed, recall, semantic


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
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    embeddings = embed.build(articles, spec=_spec(), encoder=_fake_encoder())
    monkeypatch.setattr(harness.embed, "load", lambda *a, **k: embeddings)
    return embeddings


@pytest.fixture
def report(prepared, store_root, dataset):
    return harness.evaluate_dataset(dataset, sample_size=25, store_root=store_root)


@pytest.fixture
def sample(store_root, dataset):
    impressions = feature_store.read_table(
        store_root / "processed" / dataset / "impressions.parquet"
    )
    return recall.sample_impressions(impressions, "test", 25)


# =============================================================================
# 1. What is ranked: the candidate list, not the corpus
# =============================================================================


def test_metrics_are_computed_over_the_impressions_own_candidates(report, sample, prepared):
    """D1. If the harness ranked the corpus instead, the reported candidate
    counts would be the catalogue size for every impression."""
    lengths = sample["candidate_ids"].list.len().to_numpy()
    assert report["candidates_per_impression"]["median"] == int(np.median(lengths))
    assert report["candidates_per_impression"]["max"] == int(lengths.max())
    assert report["catalogue_articles"] == prepared.n_articles
    assert report["candidates_per_impression"]["max"] <= prepared.n_articles


def test_every_method_scores_exactly_the_same_candidate_lists(report):
    """Two methods averaged over different lists cannot be compared.

    Both blocks are built from one pass over one sample, so the row counts must
    agree metric for metric.
    """
    blocks = report["methods"]
    assert set(blocks) == set(harness.METHODS)
    for name in metrics.RANKING_METRICS:
        ns = {m: b["overall"][name]["n"] for m, b in blocks.items()}
        assert len(set(ns.values())) == 1, f"{name} averaged over different rows: {ns}"


# =============================================================================
# 2. The serving tie-break
# =============================================================================


def test_the_tie_break_is_popularity_then_article_id():
    """All scores equal, so the tie-break decides the entire ranking.

    This is the cold-start case: without the tie-break the order would be
    whatever sequence the publisher wrote the candidates in.
    """
    order = harness._serving_order(
        scores=np.zeros(4),
        popularity=np.array([1.0, 9.0, 9.0, 0.0]),
        article_ids=["d", "c", "b", "a"],
    )
    # popularity 9 first (b before c on article_id), then 1, then 0
    assert [["d", "c", "b", "a"][i] for i in order] == ["b", "c", "d", "a"]


def test_score_beats_popularity():
    """Popularity is the tie-break, not a co-equal signal. An article with a
    real score outranks a more popular one with none."""
    order = harness._serving_order(
        scores=np.array([0.0, 0.1]),
        popularity=np.array([1000.0, 0.0]),
        article_ids=["popular", "relevant"],
    )
    assert list(order) == [1, 0]


def test_ranked_scores_reproduce_the_order_with_no_ties():
    order = np.array([2, 0, 1])
    got = harness._ranked_scores(order)
    assert list(np.argsort(-got, kind="stable")) == list(order)
    assert len(set(got.tolist())) == got.size, "ties survived into the metric input"


def test_both_methods_rank_a_cold_start_impression_identically(report):
    """Both cold paths fall back to the same popularity prior, so a difference
    between them on the cold slice would be a bug, not a finding.

    Measured on real data: MIND's 62 cold-start impressions score AUC 0.4786
    under both methods, to every reported digit.
    """
    cold = {m: b["slices"]["cold_start"] for m, b in report["methods"].items()}
    if cold["bm25"]["n"] == 0:
        pytest.skip("no cold-start impressions in this fixture")
    for name in metrics.RANKING_METRICS:
        values = {m: c[name]["mean"] for m, c in cold.items()}
        assert len(set(values.values())) == 1, f"cold-start {name} differs: {values}"


# =============================================================================
# 3. Unscorable and missing candidates
# =============================================================================


def test_an_unembeddable_candidate_ranks_last_rather_than_being_dropped():
    """Dropping it would change the denominator and make the two methods score
    different lists; scoring it 0 would put it above genuinely negative
    cosines."""
    assert harness.UNSCORABLE < -1.0
    order = harness._serving_order(
        scores=np.array([-0.9, harness.UNSCORABLE, 0.5]),
        popularity=np.zeros(3),
        article_ids=["negative", "unknown", "good"],
    )
    assert list(order) == [2, 0, 1]


def test_a_candidate_absent_from_the_bm25_corpus_scores_zero(
    store_root, dataset, sample, monkeypatch
):
    """ "No evidence", the same as a candidate matching no query term — not a
    crash and not a silent drop."""
    from newsrec.retrieval import index as idx

    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    titles = feature_store.read_table(
        store_root / "processed" / dataset / "articles.parquet"
    ).select("article_id", "title")
    history = feature_store.read_table(store_root / "processed" / dataset / "history.parquet")
    spec = config.tokeniser_for(dataset)

    # Drop half the corpus so many candidates are unknown to the index.
    index = idx.build(articles.head(max(1, len(articles) // 2)), spec)
    scores = harness._bm25_scores(sample, history, titles, index, spec)

    assert len(scores) == len(sample)
    for row, got in zip(sample.iter_rows(named=True), scores, strict=True):
        assert got.shape == (len(row["candidate_ids"]),)
        assert np.all(np.isfinite(got))


# =============================================================================
# 4. Q4.3 — degenerate impressions are counted, not silently dropped
# =============================================================================


def test_summarise_counts_the_undefined_values_it_excluded():
    """The count must be real, not a constant.

    Every impression in the real sample is non-degenerate, so `n_excluded` is 0
    everywhere and a hard-coded 0 would agree with every observed value. This
    hands `_summarise` a list that *does* contain undefined entries, which is
    the only way the field can be shown to work at all.
    """
    got = harness._summarise([0.5, None, 0.7, None, None])
    assert got["n"] == 2
    assert got["n_excluded"] == 3
    assert got["mean"] == pytest.approx(0.6)


def test_summarise_counts_exclusions_within_a_slice_not_the_whole_sample():
    mask = np.array([True, True, False, False])
    got = harness._summarise([0.5, None, None, 0.9], mask)
    assert got["n"] == 1
    assert got["n_excluded"] == 1, "counted exclusions from outside the slice"


def test_degenerate_impressions_are_counted_and_excluded(report):
    """On real data there are zero of these; the field must still exist so a
    reader can see it was checked rather than assumed."""
    for block in report["methods"].values():
        assert "degenerate_impressions" in block
        counted = sum(block["degenerate_impressions"].values())
        excluded = block["overall"]["auc"]["n_excluded"]
        assert counted == excluded, "degenerate count and AUC exclusions disagree"
        assert block["overall"]["auc"]["n"] + excluded == report["impressions_evaluated"]


# =============================================================================
# 5. The AUC verdict
# =============================================================================


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.51, 0.60, "better_than_random"),
        (0.40, 0.49, "worse_than_random"),
        (0.4822, 0.5105, "indistinguishable_from_random"),  # ebnerd_demo's real interval
        (0.50, 0.60, "indistinguishable_from_random"),  # touching 0.5 is not clearing it
    ],
)
def test_the_auc_verdict_reads_the_interval_not_the_mean(low, high, expected):
    mean = (low + high) / 2
    assert harness.auc_verdict({"mean": mean, "ci_low": low, "ci_high": high}) == expected


def test_an_undefined_auc_has_an_undefined_verdict():
    nan = float("nan")
    assert harness.auc_verdict({"mean": nan, "ci_low": nan, "ci_high": nan}) == "undefined"


def test_a_mean_above_half_with_a_straddling_interval_is_not_called_better(report):
    """The failure this field exists to prevent: 0.4956 printed next to
    [0.4822, 0.5105] reads as "nearly 0.5, close enough" unless the report says
    outright that it is not distinguishable from random."""
    for block in report["methods"].values():
        auc = block["overall"]["auc"]
        if auc["ci_low"] <= 0.5 <= auc["ci_high"]:
            assert auc["verdict"] == "indistinguishable_from_random"


# =============================================================================
# 6. The impression-set fingerprint guard
# =============================================================================


def _write_bm25_report(tmp_path, dataset, fingerprint):
    (tmp_path / recall.REPORT_NAME).write_text(
        json.dumps({"datasets": {dataset: {"impression_set_fingerprint": fingerprint}}})
    )


def test_a_fingerprint_mismatch_raises_rather_than_writing_a_wrong_report(
    prepared, store_root, dataset, sample, tmp_path
):
    _write_bm25_report(tmp_path, dataset, "deadbeefdeadbeef")
    with pytest.raises(semantic.ImpressionSetMismatch):
        harness.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    assert not (tmp_path / harness.REPORT_NAME).exists(), "wrote a report it should have refused"


def test_a_matching_fingerprint_is_accepted(prepared, store_root, dataset, sample, tmp_path):
    """A guard that refuses everything is useless."""
    _write_bm25_report(tmp_path, dataset, recall.impression_set_fingerprint(sample))
    dest = harness.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    written = json.loads(dest.read_text())
    assert written["datasets"][dataset]["impression_set_fingerprint"] == (
        recall.impression_set_fingerprint(sample)
    )


# =============================================================================
# 7. Determinism and the report shape
# =============================================================================


def test_the_report_is_byte_identical_across_runs(prepared, store_root, dataset, sample, tmp_path):
    """A seeded bootstrap and a stable tie-break mean a rerun changes nothing.

    Byte comparison rather than field comparison: it also catches a wall-clock
    value or an unordered dict sneaking into the report.
    """
    _write_bm25_report(tmp_path, dataset, recall.impression_set_fingerprint(sample))
    kwargs = dict(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    first = harness.run(**kwargs).read_bytes()
    second = harness.run(**kwargs).read_bytes()
    assert first == second


def test_the_report_carries_no_wall_clock_anything(prepared, store_root, dataset, sample, tmp_path):
    """Timings belong in scale.json. A duration in the report makes every
    rebuild differ and destroys the determinism check above."""
    _write_bm25_report(tmp_path, dataset, recall.impression_set_fingerprint(sample))
    dest = harness.run(datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root)
    assert "seconds" not in dest.read_text()


def test_a_degenerate_slice_is_flagged_next_to_its_own_numbers(report):
    """Not in a footnote at the top of the report: a reader who scrolls to a
    slice with an AUC of 0.85 needs to see there that n=8."""
    for block in report["methods"].values():
        for name, sl in block["slices"].items():
            share = sl["share"]
            if 0 < share < harness.DEGENERATE_SLICE_SHARE:
                assert "degenerate_slice" in sl, f"{name} at share {share} is not flagged"
            if sl["n"] == 0:
                assert "empty_slice" in sl


def test_slice_block_flags_a_tiny_slice_and_its_huge_complement():
    """Both ends of the flag, on a manufactured mask.

    The committed report is checked in `test_reports.py`, but that reads a file
    on disk — removing the flagging code would not change it. This exercises the
    code path directly, which is the only way the rule can be shown to work.

    8 of 2,000 is the real measured share of `article_head_train`.
    """
    per_impression = {
        "ranking": {name: [0.5] * 2000 for name in metrics.RANKING_METRICS},
        "diversity": [0.1] * 2000,
        "novelty": [1.0] * 2000,
        "novelty_unseen_fraction": [1.0] * 2000,
        "top_lists": [["a"]] * 2000,
    }
    tiny = np.zeros(2000, dtype=bool)
    tiny[:8] = True

    head = harness._slice_block("article_head_train", tiny, per_impression, 100)
    assert head["n"] == 8
    assert head["share"] == 0.004
    assert "degenerate_slice" in head
    assert "ci_warning" in head, "n=8 is far below MIN_TRUSTWORTHY_N"

    tail = harness._slice_block("article_tail_train", ~tiny, per_impression, 100)
    assert "degenerate_slice" in tail, "the 99.6% complement is equally uninformative"


def test_slice_block_does_not_flag_an_even_split():
    """A guard that fires on everything says nothing."""
    per_impression = {
        "ranking": {name: [0.5] * 2000 for name in metrics.RANKING_METRICS},
        "diversity": [0.1] * 2000,
        "novelty": [1.0] * 2000,
        "novelty_unseen_fraction": [1.0] * 2000,
        "top_lists": [["a"]] * 2000,
    }
    half = np.zeros(2000, dtype=bool)
    half[:1000] = True
    block = harness._slice_block("article_head", half, per_impression, 100)
    assert "degenerate_slice" not in block
    assert "ci_warning" not in block


def test_an_empty_slice_is_labelled_rather_than_omitted():
    """EB-NeRD has 0 cold-start impressions. The slice must still appear, or a
    reader cannot tell "none" from "not measured"."""
    per_impression = {
        "ranking": {name: [0.5] * 10 for name in metrics.RANKING_METRICS},
        "diversity": [0.1] * 10,
        "novelty": [1.0] * 10,
        "novelty_unseen_fraction": [1.0] * 10,
        "top_lists": [["a"]] * 10,
    }
    block = harness._slice_block("cold_start", np.zeros(10, dtype=bool), per_impression, 100)
    assert block["n"] == 0
    assert "empty_slice" in block
    assert "degenerate_slice" not in block, "an empty slice is empty, not degenerate"


def test_novelty_smoothing_is_recorded_so_the_number_is_reproducible(report):
    smoothing = report["novelty_smoothing"]
    assert smoothing["scheme"] == "add_one"
    assert smoothing["vocabulary"] == report["catalogue_articles"]
    assert smoothing["training_clicks"] >= 0
