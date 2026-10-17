"""Q2.9 — the tokenisation ablation report.

The failure guarded against is not a crash. It is an ablation that looks
rigorous and is not: two variants scored on different impression sets, or a
"delta" that is the difference of two unpaired means, or a report that claims
to cover both variants while quietly covering one.
"""

from __future__ import annotations

import json

import pytest

from newsrec import config
from newsrec.retrieval import ablation


@pytest.fixture
def report(store_root, dataset):
    return ablation.compare_variants(dataset, sample_size=25, store_root=store_root)


# =============================================================================
# Coverage — Q2.9's checkable condition
# =============================================================================


def test_both_variants_are_scored(report, dataset):
    """Q2.9: the stemmed variant is *scored as a logged ablation*, not silently
    swapped in. A report carrying only the headline variant would satisfy no
    part of that."""
    expected = {v.name for v in config.tokeniser_variants_for(dataset)}
    assert set(report["variants"]) == expected
    assert len(expected) >= 2, "an ablation with one arm is not an ablation"


def test_the_danish_variant_pair_is_the_naive_and_the_stemmed_one(report, dataset):
    """D4 made the Danish baseline deliberately naive so the ablation has
    something to show. If the baseline ever gained a stemmer, the ablation would
    still run and would compare two stemmed variants — reporting a near-zero
    delta that reads as "stemming does not help"."""
    if config.DATASETS[dataset].language != "da":
        pytest.skip("Danish-specific claim")
    baseline = report["variants"][report["baseline_variant"]]["spec"]
    assert baseline["stemmer"] is None and baseline["stopwords"] is None
    others = [v for n, v in report["variants"].items() if n != report["baseline_variant"]]
    assert any(v["spec"]["stemmer"] == "danish" for v in others)


def test_every_variant_reports_every_k(report):
    for name, v in report["variants"].items():
        assert {int(k) for k in v["recall"]} == set(config.RECALL_K_VALUES), name


def test_report_has_both_datasets_and_a_delta_per_variant_and_k(tmp_path, store_root):
    dest = ablation.run(
        datasets=("mind_small", "ebnerd_demo"),
        out_dir=tmp_path,
        sample_size=10,
        store_root=store_root,
    )
    data = json.loads(dest.read_text())
    assert set(data["datasets"]) == {"mind_small", "ebnerd_demo"}
    for d in data["datasets"].values():
        non_baseline = set(d["variants"]) - {d["baseline_variant"]}
        assert set(d["delta_vs_baseline"]) == non_baseline
        for per_k in d["delta_vs_baseline"].values():
            assert {int(k) for k in per_k} == set(config.RECALL_K_VALUES)


# =============================================================================
# The pairing — the property that makes the delta mean anything
# =============================================================================


def test_variants_are_scored_on_the_identical_impression_set(report):
    """Without this the per-impression subtraction compares unrelated rows and
    produces a difference that looks measured and is noise."""
    assert report["impression_set_fingerprint"]


def test_a_variant_compared_against_itself_has_exactly_zero_delta(store_root, dataset):
    """The sharpest available check on the pairing.

    Two runs of the SAME spec differ in nothing, so every per-impression
    difference must be exactly 0.0 — not merely small. A mean near zero would
    also be produced by subtracting two independently-sorted arrays of the same
    values, which is precisely the bug this guards against.
    """
    spec = config.tokeniser_for(dataset)
    twin = config.TokeniserSpec(
        name="twin",
        language=spec.language,
        lowercase=spec.lowercase,
        stopwords=spec.stopwords,
        stemmer=spec.stemmer,
    )
    out = ablation.compare_variants(
        dataset, variants=(spec, twin), sample_size=25, store_root=store_root
    )
    for k, d in out["delta_vs_baseline"]["twin"].items():
        assert d["mean"] == 0.0, f"K={k} delta {d['mean']} for a variant identical to the baseline"
        assert d["ci_low"] == 0.0 and d["ci_high"] == 0.0
        assert d["significant"] is False


def test_delta_mean_equals_the_difference_of_the_two_reported_means(report):
    """The paired mean difference and the difference of means are the same
    number; if they disagree, the two arms were not aligned row for row."""
    base = report["variants"][report["baseline_variant"]]["recall"]
    for name, per_k in report["delta_vs_baseline"].items():
        for k, d in per_k.items():
            expected = report["variants"][name]["recall"][k]["mean"] - base[k]["mean"]
            assert d["mean"] == pytest.approx(expected, abs=1e-12)


def test_mismatched_impression_sets_are_refused_not_reported(store_root, dataset, monkeypatch):
    """Prove the guard bites. A check that has never been seen to fire is not a
    check (CLAUDE.md)."""
    from newsrec.retrieval import recall as recall_mod

    real = recall_mod.evaluate_dataset
    calls = {"n": 0}

    def wobbly(*args, **kwargs):
        out = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1:  # the second variant "sampled" something else
            out["impression_set_fingerprint"] = "deadbeefdeadbeef"
        return out

    monkeypatch.setattr(ablation.recall, "evaluate_dataset", wobbly)
    with pytest.raises(ValueError, match="different impression sets"):
        ablation.compare_variants(dataset, sample_size=25, store_root=store_root)


# =============================================================================
# Significance is reported honestly
# =============================================================================


def test_significant_means_the_interval_excludes_zero_and_nothing_more(report):
    for per_k in report["delta_vs_baseline"].values():
        for k, d in per_k.items():
            expected = d["ci_low"] > 0.0 or d["ci_high"] < 0.0
            assert d["significant"] is expected, k
            assert d["ci_low"] <= d["mean"] <= d["ci_high"], k


# =============================================================================
# Reproducibility
# =============================================================================


def test_the_report_is_byte_identical_across_runs(store_root, dataset, tmp_path):
    """Q1.13 applies to every stage. If a re-run moves the delta, "the stemmer
    changed the number" stops being evidence of anything."""
    a = ablation.run(
        datasets=(dataset,), out_dir=tmp_path / "a", sample_size=25, store_root=store_root
    )
    b = ablation.run(
        datasets=(dataset,), out_dir=tmp_path / "b", sample_size=25, store_root=store_root
    )
    assert a.read_bytes() == b.read_bytes()


def test_no_wall_clock_timings_leak_into_the_report(store_root, dataset, tmp_path):
    """The specific mistake `recall.run` made and its determinism test caught."""
    dest = ablation.run(
        datasets=(dataset,), out_dir=tmp_path, sample_size=25, store_root=store_root
    )
    text = dest.read_text()
    assert "timing_seconds" not in text
    assert "per_impression_recall" not in text, "2,000 raw floats per K are not a summary"


def test_variant_list_is_derived_from_config_not_hard_coded(dataset):
    """`tokeniser_variants_for` exists so a variant added to
    `TOKENISER_VARIANTS` is ablated automatically. A second hard-coded list
    would have to be remembered, and forgetting it produces a report that
    silently covers less than it claims."""
    variants = config.tokeniser_variants_for(dataset)
    language = config.DATASETS[dataset].language
    assert {v.name for v in variants} == {
        n for n, s in config.TOKENISER_VARIANTS.items() if s.language == language
    }
    assert variants[0].name == config.DEFAULT_TOKENISER[language], "headline must come first"
