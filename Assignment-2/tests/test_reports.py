"""Q4.6, Q4.8 — conditions on the committed report itself.

Every other test in this repo checks code. These check the artefact a grader
actually opens: `reports/metrics/eval.json`. A harness that computes intervals
correctly but writes a report missing half of them still fails Q4.6, and no
unit test of the harness would notice.

The report is committed, so these run in a clean clone with no `data/`.
"""

from __future__ import annotations

import json
import math

import pytest

from newsrec import config
from newsrec.eval import harness, metrics, slices

REPORT = config.METRICS_DIR / harness.REPORT_NAME

#: Every metric the report must carry for each (method, dataset, slice).
EXPECTED_METRICS: tuple[str, ...] = (
    *metrics.RANKING_METRICS,
    "diversity",
    "novelty",
    "coverage",
)


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT} not built; run `make eval`")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _metric_cells(report: dict):
    """Yield (path, cell) for every metric dict in the report."""
    for dataset, block in report["datasets"].items():
        for method, mblock in block["methods"].items():
            for name in EXPECTED_METRICS:
                yield f"{dataset}/{method}/overall/{name}", mblock["overall"][name]
            for slice_name, sl in mblock["slices"].items():
                for name in EXPECTED_METRICS:
                    yield f"{dataset}/{method}/{slice_name}/{name}", sl[name]


# =============================================================================
# Q4.6 — every metric has an interval
# =============================================================================


def test_every_metric_has_ci(report):
    """The named Q4.6 condition: a metric without an interval is a failure.

    Checked structurally over every cell rather than spot-checked, because the
    failure mode is one metric quietly emitted as a bare float — which reads
    perfectly well and is exactly what a reader would then quote.
    """
    seen = 0
    for path, cell in _metric_cells(report):
        assert isinstance(cell, dict), f"{path} is not a metric object"
        for field in ("mean", "ci_low", "ci_high", "n"):
            assert field in cell, f"{path} has no {field}"
        seen += 1
    assert seen > 0, "no metric cells found; the report shape has changed"


def test_every_per_impression_interval_brackets_its_mean(report):
    """True of every metric that is an average over impressions.

    Coverage is excluded and checked separately below: it is not an average, so
    this invariant genuinely does not hold for it.
    """
    for path, cell in _metric_cells(report):
        if path.endswith("/coverage"):
            continue
        if cell["n"] == 0:
            assert math.isnan(cell["mean"]), f"{path} has n=0 but a real mean"
            continue
        assert cell["ci_low"] <= cell["mean"] <= cell["ci_high"], f"{path} mean outside its CI"


def test_coverage_sits_at_or_above_its_own_interval(report):
    """Coverage's interval is biased low, and that is expected rather than a bug.

    A bootstrap draw misses about 37% of impressions, so every resampled union
    covers less of the catalogue than the real sample does. The reported mean is
    the actual coverage; the interval comes from the resamples and therefore
    sits below it.

    Asserting the *direction* rather than skipping the check: if coverage ever
    started reporting the resample mean instead, the interval would bracket it
    and this test would fail — which is the mistake worth catching.
    """
    seen = 0
    for path, cell in _metric_cells(report):
        if not path.endswith("/coverage") or cell["n"] == 0:
            continue
        assert cell["ci_low"] <= cell["ci_high"] <= cell["mean"] + 1e-9, path
        seen += 1
    assert seen > 0, "no coverage cells found"


def test_the_bootstrap_settings_are_recorded(report):
    """A CI is only reproducible if the reader knows how it was made."""
    boot = report["bootstrap"]
    assert boot["n_resamples"] == config.BOOTSTRAP_N
    assert boot["ci"] == config.BOOTSTRAP_CI
    assert boot["resampled_over"] == "impressions"
    assert boot["min_trustworthy_n"] > 0


def test_small_slices_carry_a_ci_warning(report):
    """Measured, a 95% interval over a sparse metric covers ~0.85 at n=100.

    The numbers stay in the report; the confidence claim does not.
    """
    for dataset, block in report["datasets"].items():
        for method, mblock in block["methods"].items():
            for name, sl in mblock["slices"].items():
                if 0 < sl["n"] < report["bootstrap"]["min_trustworthy_n"]:
                    assert "ci_warning" in sl, f"{dataset}/{method}/{name} n={sl['n']} unflagged"


# =============================================================================
# Q4.8 — the grid
# =============================================================================


def test_eval_json_covers_method_dataset_grid(report):
    """The named Q4.8 condition: at least four (method, dataset) blocks.

    Both retrievers on both datasets is the requirement; we run three datasets,
    so six.
    """
    pairs = {
        (dataset, method)
        for dataset, block in report["datasets"].items()
        for method in block["methods"]
    }
    assert len(pairs) >= 4, f"only {len(pairs)} (method, dataset) blocks: {sorted(pairs)}"

    datasets = {d for d, _ in pairs}
    assert any(d.startswith("mind") for d in datasets), "no MIND dataset"
    assert any(d.startswith("ebnerd") for d in datasets), "no EB-NeRD dataset"
    for method in harness.METHODS:
        assert {d for d, m in pairs if m == method} == datasets, (
            f"{method} was not run on every dataset"
        )


def test_every_block_carries_the_full_metric_set(report):
    for dataset, block in report["datasets"].items():
        for method, mblock in block["methods"].items():
            missing = set(EXPECTED_METRICS) - set(mblock["overall"])
            assert not missing, f"{dataset}/{method} is missing {sorted(missing)}"


def test_both_slice_families_are_reported(report):
    """Q4.5 asks for at least one slice; both cuts are reported and labelled."""
    for dataset, block in report["datasets"].items():
        for method, mblock in block["methods"].items():
            got = set(mblock["slices"])
            expected = set(slices.TRAINING_PERIOD_SLICES) | set(slices.TEST_PERIOD_SLICES)
            assert got == expected, f"{dataset}/{method} slices: {sorted(got)}"


# =============================================================================
# The report says plainly what the numbers mean
# =============================================================================


def test_the_auc_verdict_is_stated_for_every_block(report):
    """0.4956 beside [0.4822, 0.5105] reads as "close enough to 0.5" unless the
    report says outright that it is not distinguishable from random."""
    allowed = {
        "better_than_random",
        "worse_than_random",
        "indistinguishable_from_random",
        "undefined",
    }
    for dataset, block in report["datasets"].items():
        for method, mblock in block["methods"].items():
            verdict = mblock["overall"]["auc"].get("verdict")
            assert verdict in allowed, f"{dataset}/{method} verdict: {verdict!r}"


def test_an_interval_straddling_one_half_is_never_called_better_than_random(report):
    for path, cell in _metric_cells(report):
        if not path.endswith("/auc") or "verdict" not in cell:
            continue
        if cell["ci_low"] <= 0.5 <= cell["ci_high"]:
            assert cell["verdict"] == "indistinguishable_from_random", path


def test_the_training_period_head_tail_degeneracy_is_stated_not_buried(report):
    """Q4.5 requires a training-period cut; the data makes it degenerate.

    Both facts belong in the report — the requirement is met and the result is
    labelled unreliable. This asserts the warning exists at the top level AND
    beside the slice's own numbers.
    """
    warning = report["training_period_head_tail_warning"]
    assert "DEGENERATE" in warning
    assert "Q4.5" in warning

    # Unconditional, not "if it happens to be small": on the real corpora this
    # cut IS degenerate (measured 0.4% / 99.6%), so every block must carry both
    # the share and the flag. A conditional assertion here would pass even if
    # the flagging were removed entirely.
    for dataset, block in report["datasets"].items():
        for method, mblock in block["methods"].items():
            head = mblock["slices"]["article_head_train"]
            tail = mblock["slices"]["article_tail_train"]
            assert head["share"] < harness.DEGENERATE_SLICE_SHARE, (
                f"{dataset}/{method} head_train share is {head['share']}; if this cut has "
                f"stopped being degenerate, the warning text must be rewritten"
            )
            assert "degenerate_slice" in head, f"{dataset}/{method} head_train unflagged"
            assert "degenerate_slice" in tail, f"{dataset}/{method} tail_train unflagged"
            assert "ci_warning" in head, f"{dataset}/{method} head_train has no CI warning"


def test_the_test_period_slices_are_labelled_as_analysis_descriptors(report):
    """An assessor scanning for leakage will see future-derived popularity here
    and is right to ask. The report answers before they do."""
    note = report["slice_note"]
    assert "TEST-period" in note
    assert "never model inputs" in note


def test_the_report_distinguishes_recall_from_ranking(report):
    """The two pools answer different questions and the report must say so, or
    a reader will compare AUC here against recall@K there."""
    assert "candidate_ids" in report["ranked_pool"]
    assert "corpus" in report["pool_note"]


def test_the_report_carries_no_wall_clock(report):
    """Timings live in scale.json. A duration here would make every rebuild
    differ and destroy the determinism guarantee."""
    assert "seconds" not in REPORT.read_text(encoding="utf-8")


def test_the_impression_sets_match_the_bm25_recall_report(report):
    """A reader comparing eval.json's AUC against bm25_recall.json's recall is
    entitled to assume both describe the same impressions."""
    other = config.METRICS_DIR / "bm25_recall.json"
    if not other.exists():
        pytest.skip("bm25_recall.json not built")
    committed = json.loads(other.read_text(encoding="utf-8"))["datasets"]
    for dataset, block in report["datasets"].items():
        if dataset in committed:
            assert (
                block["impression_set_fingerprint"]
                == (committed[dataset]["impression_set_fingerprint"])
            ), dataset


# =============================================================================
# Q6.5 — the 10x analysis is grounded in measured cost, not estimates
# =============================================================================


def test_scale_json_has_measured_stages():
    """The named Q6.5 condition.

    Q6.5 says the scale analysis must be "grounded in **measured** per-stage
    time and memory ... not estimates". This asserts the measurements exist and
    are real numbers, per stage and per dataset — the design note's §4 table is
    only as good as this file.
    """
    path = config.METRICS_DIR / "scale.json"
    if not path.exists():
        pytest.skip("scale.json not built; run the pipeline")
    blob = json.loads(path.read_text(encoding="utf-8"))

    required = {"embed", "bm25_recall", "ann", "eval"}
    missing = required - set(blob)
    assert not missing, f"scale.json has no timings for {sorted(missing)}"

    for stage in sorted(required):
        per_dataset = blob[stage]
        assert per_dataset, f"{stage} has no datasets"
        for dataset, values in per_dataset.items():
            numbers = [v for v in values.values() if isinstance(v, (int, float))]
            assert numbers, f"{stage}/{dataset} records no numeric measurement"
            assert all(v >= 0 for v in numbers), f"{stage}/{dataset} has a negative duration"
            assert any(v > 0 for v in numbers), (
                f"{stage}/{dataset} is all zeros, which is not a measurement"
            )


def test_scale_json_covers_both_datasets():
    """A 10x analysis resting on one corpus would not support the note's claim
    that the pipeline behaves differently on the two."""
    path = config.METRICS_DIR / "scale.json"
    if not path.exists():
        pytest.skip("scale.json not built")
    blob = json.loads(path.read_text(encoding="utf-8"))
    for stage in ("embed", "bm25_recall"):
        datasets = set(blob.get(stage, {}))
        assert any(d.startswith("mind") for d in datasets), f"{stage}: no MIND"
        assert any(d.startswith("ebnerd") for d in datasets), f"{stage}: no EB-NeRD"


# =============================================================================
# Q2.9 — the tokenisation ablation covers both variants on both languages
# =============================================================================


def test_tokenisation_ablation_has_both_variants_per_language():
    """The named Q2.9 condition.

    Q2.9 asks for language-aware tokenisation *declared and compared*. A report
    with only the baseline would satisfy "we tokenise per language" while
    proving nothing about whether the choice matters.
    """
    path = config.METRICS_DIR / "bm25_tokenisation_ablation.json"
    if not path.exists():
        pytest.skip("bm25_tokenisation_ablation.json not built")
    blob = json.loads(path.read_text(encoding="utf-8"))

    for dataset, block in blob["datasets"].items():
        variants = set(block["variants"])
        assert len(variants) >= 2, f"{dataset} has only {variants}"
        language = block["language"]
        assert any(v.startswith(language) for v in variants), (
            f"{dataset} is {language} but its variants are {sorted(variants)}"
        )
        assert any("stem" in v for v in variants), f"{dataset} has no stemmed variant"

    danish = [d for d, b in blob["datasets"].items() if b["language"] == "da"]
    assert danish, "no Danish dataset in the ablation"


def test_the_tokenisation_ablation_is_paired_and_reports_significance():
    """A delta without an interval cannot say whether the variant mattered."""
    path = config.METRICS_DIR / "bm25_tokenisation_ablation.json"
    if not path.exists():
        pytest.skip("bm25_tokenisation_ablation.json not built")
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["paired"] is True
    for dataset, block in blob["datasets"].items():
        for variant, ks in block["delta_vs_baseline"].items():
            for k, cell in ks.items():
                for field in ("mean", "ci_low", "ci_high", "significant"):
                    assert field in cell, f"{dataset}/{variant}/K={k} has no {field}"
