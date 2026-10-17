"""The lexical+semantic blend ablation.

The failure guarded against is a blend that reads the wrong scores. Unlike most
bugs in this repo it would not produce an obviously broken number — it would
produce a smooth, plausible alpha curve with a nice interior optimum, and we
would report a finding about a computation we never performed.

The defence is the endpoint check: alpha=1.0 must reproduce BM25's committed AUC
and alpha=0.0 must reproduce semantic's, to floating-point precision. Those are
the same computation over the same impressions, so anything else means the wiring
is wrong. Tested here on fixtures, and asserted by `run` on real data.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import harness, hybrid
from newsrec.retrieval import embed


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
    monkeypatch.setattr(hybrid.embed, "load", lambda *a, **k: embeddings)
    monkeypatch.setattr(harness.embed, "load", lambda *a, **k: embeddings)
    return embeddings


@pytest.fixture
def report(prepared, store_root, dataset):
    return hybrid.evaluate_dataset(dataset, store_root=store_root)


# =============================================================================
# 1. minmax
# =============================================================================


def test_minmax_maps_to_the_unit_interval():
    got = hybrid.minmax(np.array([2.0, 4.0, 6.0]))
    assert got.tolist() == [0.0, 0.5, 1.0]


def test_an_all_equal_vector_becomes_zeros_not_halves():
    """The cold-start case: every candidate scores 0.

    Mapping to zeros leaves the blend at exactly the other method's ranking, so
    the popularity tie-break still decides. Mapping to 0.5 would be arithmetically
    tidy and would silently give a constant vector a vote.
    """
    assert hybrid.minmax(np.zeros(4)).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert hybrid.minmax(np.full(3, 7.0)).tolist() == [0.0, 0.0, 0.0]


def test_minmax_is_invariant_to_scale_and_offset():
    """This is what makes alpha mean 'how much do I trust each signal' rather
    than 'which score happens to be numerically larger'."""
    base = np.array([1.0, 3.0, 2.0])
    assert hybrid.minmax(base) == pytest.approx(hybrid.minmax(base * 1000 + 50))


# =============================================================================
# 2. The endpoint check — the load-bearing test
# =============================================================================


def test_the_endpoints_reproduce_the_pure_methods(report, prepared, store_root, dataset):
    """alpha=1 IS bm25 and alpha=0 IS semantic, computed the same way.

    Compared against `harness.evaluate_dataset` on the same fixture rather than
    against a committed report, so the test runs in a clean clone.
    """
    other = harness.evaluate_dataset(
        dataset, sample_size=config.RECALL_SAMPLE_SIZE, store_root=store_root
    )
    for alpha, method in (("1.0", "bm25"), ("0.0", "semantic")):
        assert report["alpha"][alpha]["auc"] == pytest.approx(
            other["methods"][method]["overall"]["auc"]["mean"], abs=1e-9
        ), f"alpha={alpha} does not reproduce {method}"


def test_run_raises_when_an_endpoint_disagrees_with_eval_json(
    prepared, store_root, dataset, tmp_path
):
    """The guard must bite. A blend reading the wrong scores would still produce
    a smooth curve, so this is the only thing standing between a wiring bug and
    a reported finding."""
    (tmp_path / "eval.json").write_text(
        json.dumps(
            {
                "datasets": {
                    dataset: {
                        "methods": {
                            "bm25": {"overall": {"auc": {"mean": 0.123456}}},
                            "semantic": {"overall": {"auc": {"mean": 0.654321}}},
                        }
                    }
                }
            }
        )
    )
    with pytest.raises(hybrid.EndpointMismatch, match="endpoints"):
        hybrid.run(datasets=(dataset,), out_dir=tmp_path, store_root=store_root)


def test_run_without_eval_json_records_that_it_could_not_check(
    prepared, store_root, dataset, tmp_path
):
    """Absent evidence is reported, never silently treated as agreement."""
    dest = hybrid.run(datasets=(dataset,), out_dir=tmp_path, store_root=store_root)
    written = json.loads(dest.read_text())
    assert written["endpoint_check"]["checked"] is False
    assert "not built" in written["endpoint_check"]["reason"]


# =============================================================================
# 3. The sweep and its deltas
# =============================================================================


def test_every_alpha_is_reported_with_an_interval(report):
    assert set(report["alpha"]) == {str(a) for a in hybrid.ALPHAS}
    for entry in report["alpha"].values():
        for field in ("auc", "ci_low", "ci_high", "n"):
            assert field in entry


def test_the_deltas_are_paired_against_semantic_not_against_each_other(report):
    """A delta of exactly zero at alpha=0 would mean it was paired against
    itself; the field is absent there instead, which is the honest shape."""
    assert "delta_vs_semantic" not in report["alpha"]["0.0"]
    for alpha in hybrid.ALPHAS:
        if alpha == 0.0:
            continue
        assert "delta_vs_semantic" in report["alpha"][str(alpha)]


def test_the_delta_at_alpha_one_matches_the_bm25_minus_semantic_gap(report):
    """Consistency between the sweep and its own endpoints."""
    delta = report["alpha"]["1.0"]["delta_vs_semantic"]["mean"]
    gap = report["alpha"]["1.0"]["auc"] - report["alpha"]["0.0"]["auc"]
    assert delta == pytest.approx(gap, abs=1e-6)


def test_established_requires_the_interval_to_exclude_zero(report):
    for alpha in hybrid.ALPHAS:
        if alpha == 0.0:
            continue
        d = report["alpha"][str(alpha)]["delta_vs_semantic"]
        excludes = d["ci_low"] > 0.0 or d["ci_high"] < 0.0
        assert d["established"] is excludes


def test_the_best_interior_alpha_is_not_an_endpoint(report):
    """The endpoints are the pure methods; calling one of them 'the best blend'
    would be a category error."""
    assert 0.0 < report["best_interior_alpha"] < 1.0


def test_hybrid_beats_semantic_requires_an_established_gain(report):
    best = report["alpha"][str(report["best_interior_alpha"])]["delta_vs_semantic"]
    expected = best["established"] and best["mean"] > 0
    assert report["hybrid_beats_semantic"] is expected


# =============================================================================
# 4. The spread diagnostics
# =============================================================================


def test_score_spread_reports_every_diagnostic(report):
    spread = report["bm25_score_spread"]
    assert set(spread) == {
        "zero_score_share",
        "within_impression_cv",
        "top_vs_rest_gap",
        "tied_share",
        "top1_is_a_click",
    }
    assert 0.0 <= spread["zero_score_share"] <= 1.0
    assert 0.0 <= spread["tied_share"] <= 1.0


def test_a_flat_score_vector_has_no_spread_and_a_spiky_one_does():
    """The quantity the Danish/English split is explained by.

    Hand-built rather than measured: a flat vector must report CV 0 and a spiky
    one must report much more, or the diagnostic cannot distinguish the two
    cases it exists to distinguish.
    """
    sample = pl.DataFrame(
        {
            "candidate_ids": [["a", "b", "c", "d"], ["e", "f", "g", "h"]],
            "clicked_ids": [["a"], ["e"]],
        }
    )
    flat = hybrid.score_spread(sample, [np.full(4, 5.0), np.full(4, 5.0)])
    spiky = hybrid.score_spread(
        sample, [np.array([50.0, 1.0, 1.0, 1.0]), np.array([50.0, 1.0, 1.0, 1.0])]
    )
    assert flat["within_impression_cv"] == 0.0
    assert spiky["within_impression_cv"] > 1.0
    assert spiky["top_vs_rest_gap"] > flat["top_vs_rest_gap"]


def test_zero_scores_and_ties_are_counted():
    sample = pl.DataFrame({"candidate_ids": [["a", "b", "c", "d"]], "clicked_ids": [["a"]]})
    got = hybrid.score_spread(sample, [np.array([0.0, 0.0, 3.0, 4.0])])
    assert got["zero_score_share"] == 0.5
    assert got["tied_share"] == 0.5  # the two zeros are tied with each other


def test_top1_is_a_click_reads_the_labels():
    sample = pl.DataFrame(
        {"candidate_ids": [["a", "b"], ["c", "d"]], "clicked_ids": [["a"], ["d"]]}
    )
    # First impression: top score is 'a', which was clicked. Second: 'c', not clicked.
    got = hybrid.score_spread(sample, [np.array([9.0, 1.0]), np.array([9.0, 1.0])])
    assert got["top1_is_a_click"] == 0.5


# =============================================================================
# 5. The report
# =============================================================================


def test_the_report_states_the_answer_and_refuses_to_bury_it(
    prepared, store_root, dataset, tmp_path
):
    dest = hybrid.run(datasets=(dataset,), out_dir=tmp_path, store_root=store_root)
    written = json.loads(dest.read_text())
    assert "Selecting alpha on TEST" in written["answer"]
    assert "Selecting alpha on VAL" in written["answer"]
    assert written["bootstrap"]["paired"] is True


def test_the_report_carries_no_wall_clock(prepared, store_root, dataset, tmp_path):
    dest = hybrid.run(datasets=(dataset,), out_dir=tmp_path, store_root=store_root)
    assert "seconds" not in dest.read_text()


def test_the_report_is_byte_identical_across_runs(prepared, store_root, dataset, tmp_path):
    kwargs = dict(datasets=(dataset,), out_dir=tmp_path, store_root=store_root)
    assert hybrid.run(**kwargs).read_bytes() == hybrid.run(**kwargs).read_bytes()


# =============================================================================
# 6. The normalisation is the design decision, so it gets its own tests
#
# Found by mutation: deleting the normalisation passed every test above,
# because minmax is monotonic and the endpoints are therefore identical with or
# without it. Only an interior alpha can tell the difference.
# =============================================================================


def test_the_blend_balances_signals_of_wildly_different_scale():
    """BM25 runs to tens; cosine lives in [-1, 1].

    Blended raw, a 50/50 split is not a balanced opinion — it is BM25 with
    rounding error. Normalised, each signal gets an equal vote, which is what
    alpha is supposed to mean.
    """
    lex = np.array([100.0, 0.0])
    sem = np.array([0.1, 0.9])

    got = hybrid.blend(lex, sem, 0.5)
    assert got.tolist() == [0.5, 0.5], "each signal should contribute equally at alpha=0.5"

    raw = 0.5 * lex + 0.5 * sem
    assert np.argmax(raw) == 0, "unnormalised, the larger scale decides"
    assert got[0] == got[1], "normalised, neither signal wins on scale alone"


def test_an_interior_blend_differs_from_the_unnormalised_one():
    """The mutation that survived: `alpha*lex + (1-alpha)*sem` with no minmax.

    It agrees at both endpoints and disagrees everywhere between, so only an
    interior alpha can catch it.
    """
    lex = np.array([50.0, 10.0, 30.0])
    sem = np.array([0.2, 0.9, 0.4])
    alpha = 0.5

    normalised = hybrid.blend(lex, sem, alpha)
    unnormalised = alpha * lex + (1.0 - alpha) * sem
    assert list(np.argsort(-normalised)) != list(np.argsort(-unnormalised))


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_the_endpoints_are_blind_to_normalisation(alpha):
    """Documents WHY the tests above were needed.

    At an endpoint the blend is a monotonic transform of one score vector, so
    the ranking is identical normalised or not. Any test that only checks
    endpoints cannot see the normalisation at all.
    """
    lex = np.array([50.0, 10.0, 30.0])
    sem = np.array([0.2, 0.9, 0.4])
    normalised = hybrid.blend(lex, sem, alpha)
    unnormalised = alpha * lex + (1.0 - alpha) * sem
    assert list(np.argsort(-normalised)) == list(np.argsort(-unnormalised))


def test_the_blend_recovers_each_pure_signal_at_its_endpoint():
    lex = np.array([50.0, 10.0, 30.0])
    sem = np.array([0.2, 0.9, 0.4])
    assert hybrid.blend(lex, sem, 1.0) == pytest.approx(hybrid.minmax(lex))
    assert hybrid.blend(lex, sem, 0.0) == pytest.approx(hybrid.minmax(sem))


# =============================================================================
# 7. Selecting alpha on val rather than on the split we report
# =============================================================================


def test_val_selection_chooses_from_val_and_reports_on_test(prepared, store_root, dataset):
    """The point of the exercise: the choice and the number come from different
    splits, so the reported gain is not partly an artefact of having peeked."""
    got = hybrid.select_on_val(dataset, store_root=store_root)
    assert got["selected_on"] == "val"
    assert got["selected_alpha"] in {0.0, *[a for a in hybrid.ALPHAS if 0.0 < a < 1.0]}
    assert set(got["val_alpha_curve"]) == {
        str(a) for a in [0.0, *[a for a in hybrid.ALPHAS if 0.0 < a < 1.0]]
    }


def test_pure_bm25_is_never_selectable(prepared, store_root, dataset):
    """alpha=1.0 is excluded from selection: the question is whether adding
    lexical to semantic helps, not whether to abandon semantic entirely. Q4
    already answered the latter, and the answer was no on all three corpora."""
    got = hybrid.select_on_val(dataset, store_root=store_root)
    assert got["selected_alpha"] != 1.0
    assert "1.0" not in got["val_alpha_curve"]


def test_selecting_alpha_zero_means_no_blend_and_says_so(prepared, store_root, dataset):
    """If no blend beats pure semantic on val, the honest selection is none.

    Measured on the real corpora, both EB-NeRD variants land here independently.
    """
    got = hybrid.select_on_val(dataset, store_root=store_root)
    if got["selected_alpha"] == 0.0:
        assert got["test_delta_vs_semantic"] is None
        assert "semantic-only" in got["verdict"]
        assert got["test_auc"] == got["test_semantic_auc"]


def test_a_selected_blend_reports_a_paired_delta_on_test(prepared, store_root, dataset):
    got = hybrid.select_on_val(dataset, store_root=store_root)
    if got["selected_alpha"] != 0.0:
        delta = got["test_delta_vs_semantic"]
        assert delta is not None
        assert delta["mean"] == pytest.approx(got["test_auc"] - got["test_semantic_auc"], abs=1e-6)
        expected = delta["established"] and delta["mean"] > 0
        assert ("beats semantic on test" in got["verdict"]) is expected


def test_the_report_carries_both_selections_and_distinguishes_them(
    prepared, store_root, dataset, tmp_path
):
    dest = hybrid.run(datasets=(dataset,), out_dir=tmp_path, store_root=store_root)
    written = json.loads(dest.read_text())
    assert "val_selected" in written["datasets"][dataset]
    assert "Selecting alpha on TEST" in written["answer"]
    assert "Selecting alpha on VAL" in written["answer"]
    assert "peeked" in written["selection_note"]


def test_val_selection_can_be_switched_off_for_speed(prepared, store_root, dataset, tmp_path):
    dest = hybrid.run(
        datasets=(dataset,), out_dir=tmp_path, store_root=store_root, with_val_selection=False
    )
    assert "val_selected" not in json.loads(dest.read_text())["datasets"][dataset]
