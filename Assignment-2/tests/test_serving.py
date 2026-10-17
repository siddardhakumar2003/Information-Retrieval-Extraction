"""A2 Q4 — index memory, p99 latency, cost and the 10x argument.

Design: `agent_docs/a2_q4_plan.md`.

The measurements themselves need a built store, so the tests that read the
report skip without one. What is checkable everywhere is the *shape* of the
claim: percentiles that are actually ordered, stage times that account for the
total, a candidate K that comes from config, and — the one that matters most —
a cost figure that cannot be quoted without naming its source.

That last one is not pedantry. A1's design note carried an unsourced 0.4397
until the number-traceability checker caught it, and a plausible invented
instance price is the same failure with better camouflage (OQ-13).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from newsrec import config
from newsrec.eval import serving

REPORT = config.METRICS_DIR / serving.REPORT_NAME


def _fake_latency(mean_ms: float = 10.0, p99_ms: float = 25.0) -> dict:
    return {
        "device": "cpu",
        "batch_size": 1,
        "candidate_k": config.SERVING_CANDIDATE_K,
        "measured_requests": config.SERVING_BENCH_REQUESTS,
        "total": {
            "mean_ms": mean_ms,
            "p50_ms": mean_ms,
            "p95_ms": p99_ms * 0.8,
            "p99_ms": p99_ms,
            "max_ms": p99_ms * 2,
        },
        "stages": {s: {"p99_ms": 1.0} for s in serving.STAGES},
    }


# =============================================================================
# Percentiles
# =============================================================================


def test_percentiles_are_ordered():
    values = np.random.default_rng(0).gamma(2.0, 0.005, size=5000)
    got = serving._percentiles(values)
    assert got["p50_ms"] <= got["p95_ms"] <= got["p99_ms"] <= got["max_ms"]


def test_percentiles_are_reported_in_milliseconds():
    """Seconds in, milliseconds out. A unit slip here would put the SLA
    comparison off by 1000x and still look like a plausible number."""
    got = serving._percentiles(np.full(100, 0.05))  # 50 ms
    assert got["mean_ms"] == pytest.approx(50.0)
    assert got["p99_ms"] == pytest.approx(50.0)


def test_the_mean_does_not_stand_in_for_the_tail():
    """Why A2 Q4.2 asks for p99 and why A1's means were not enough.

    970 requests at 1 ms and 30 at 500 ms: the mean is ~16 ms and the p99 is
    500 ms. A system reported at its mean looks like it meets a 100 ms SLA and
    misses it for 3% of users.

    (The tail here is 3%, not 1%: with exactly 1% slow requests the 99th
    percentile lands *on* the boundary between the two populations and comes
    out near the mean by coincidence — which is itself a decent reminder that a
    percentile at the edge of a sample is an interpolation, not a measurement.)
    """
    values = np.concatenate([np.full(970, 0.001), np.full(30, 0.5)])
    got = serving._percentiles(values)
    assert got["mean_ms"] == pytest.approx(15.97, abs=0.01)
    assert got["p99_ms"] == pytest.approx(500.0)
    assert got["p99_ms"] > got["mean_ms"] * 5


# =============================================================================
# The cost model — the anti-invention guard
# =============================================================================


def test_no_currency_figure_is_reported_without_a_cited_price():
    """`config.SERVING_COST_USD_PER_HOUR` is deliberately unset (OQ-13)."""
    cost = serving.cost_model(_fake_latency())
    assert cost["usd_per_1000_queries"] is None
    assert "OQ-13" in cost["why_no_currency_figure"]


def test_core_seconds_is_reported_without_needing_a_price():
    """The measured answer to "what does a thousand queries cost", in the unit
    that needs no price at all: 10 ms x 1000 = 10 core-seconds."""
    cost = serving.cost_model(_fake_latency(mean_ms=10.0))
    assert cost["core_seconds_per_1000_queries"] == pytest.approx(10.0)


def test_a_price_without_a_source_still_reports_no_currency_figure(monkeypatch):
    """Half an assumption is not an assumption. A number with no citation is
    exactly what the design-note checker cannot trace."""
    monkeypatch.setattr(config, "SERVING_COST_USD_PER_HOUR", 0.42)
    monkeypatch.setattr(config, "SERVING_COST_SOURCE", "")
    assert serving.cost_model(_fake_latency())["usd_per_1000_queries"] is None


def test_a_cited_price_does_produce_a_currency_figure(monkeypatch):
    """The other half of the guard: it must not be permanently off."""
    monkeypatch.setattr(config, "SERVING_COST_USD_PER_HOUR", 3.6)
    monkeypatch.setattr(config, "SERVING_COST_SOURCE", "example price page, 2026-09-16")
    cost = serving.cost_model(_fake_latency(mean_ms=10.0))
    assert cost["usd_per_1000_queries"] is not None
    assert cost["assumptions"]["price_source"]


def test_the_cost_model_names_every_assumption():
    """A back-of-envelope estimate is only checkable if the envelope is shown."""
    cost = serving.cost_model(_fake_latency())
    for key in ("cores", "batch_size", "candidate_k", "device", "sla_p99_ms"):
        assert key in cost["assumptions"], key


def test_the_sla_verdict_follows_the_measured_p99():
    assert serving.cost_model(_fake_latency(p99_ms=25.0))["sla"]["met"] is True
    assert serving.cost_model(_fake_latency(p99_ms=250.0))["sla"]["met"] is False


# =============================================================================
# Config, not literals
# =============================================================================


def test_candidate_k_comes_from_config():
    """A2 Q2 states K ~ 100-200. The benchmark must measure the K the rest of
    the system would use, not one typed at the call site."""
    assert 100 <= config.SERVING_CANDIDATE_K <= 200
    assert serving.cost_model(_fake_latency())["assumptions"]["candidate_k"] == (
        config.SERVING_CANDIDATE_K
    )


def test_the_stage_list_is_shared():
    """One definition of "the stages", so the report, the breakdown and the
    sum-to-total check cannot disagree."""
    assert serving.STAGES[0] == "user_pooling"
    assert serving.STAGES[-1] == "order"
    assert "model_score" in serving.STAGES
    assert len(set(serving.STAGES)) == len(serving.STAGES)


# =============================================================================
# The 10x block
# =============================================================================


def test_the_ten_times_block_names_the_keys_its_claims_came_from():
    """Every claim in the design note has to be traceable to a measurement.
    A block of prose with no key references is an assertion."""
    memory = {
        "resident_bytes": {"embedding_matrix_exact": 200_000_000},
        "on_disk_bytes": {"bm25_index": 10_000_000},
        "articles": 65_238,
    }
    block = serving.scaling_10x(_fake_latency(), memory)
    assert block["from_keys"], "no measured keys named"
    for entry in block["binding_constraint_order"]:
        assert entry["from_key"], f"{entry['what']} names no measured key"
        assert entry["why"], f"{entry['what']} has no reasoning"


def test_ten_times_the_embedding_matrix_is_compared_against_the_real_vram_budget():
    """4096 MiB is this machine's actual GPU. 200 MB x 10 = 2 GB fits; 500 MB
    x 10 = 5 GB does not, and the report must say which."""
    small = serving.scaling_10x(
        _fake_latency(), {"resident_bytes": {"embedding_matrix_exact": 200_000_000}}
    )
    large = serving.scaling_10x(
        _fake_latency(), {"resident_bytes": {"embedding_matrix_exact": 500_000_000}}
    )
    assert small["embedding_matrix_at_10x_fits_in_4gb_vram"] is True
    assert large["embedding_matrix_at_10x_fits_in_4gb_vram"] is False


# =============================================================================
# The written report
# =============================================================================


def _report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT.name} not built; run `make serving`")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_the_report_covers_both_datasets():
    report = _report()
    assert set(report["datasets"]) >= set(config.HEADLINE_DATASETS)


def test_stage_timings_sum_to_the_total_within_tolerance():
    """An unaccounted stage is a hidden cost. If the six stages explain much
    less than the measured total, something expensive is happening between them
    and the report is describing the wrong thing."""
    report = _report()
    for dataset, block in report["datasets"].items():
        share = block["latency"]["unaccounted_share"]
        assert share <= serving.STAGE_SUM_TOLERANCE, (
            f"{dataset}: {share:.1%} of the measured request time is not attributed "
            f"to any of {list(serving.STAGES)}"
        )


def test_warmup_requests_are_excluded():
    report = _report()
    for dataset, block in report["datasets"].items():
        latency = block["latency"]
        assert latency["warmup_requests"] == config.SERVING_WARMUP_REQUESTS, dataset
        assert latency["measured_requests"] <= config.SERVING_BENCH_REQUESTS, dataset


def test_the_measured_percentiles_are_ordered_on_every_dataset():
    report = _report()
    for dataset, block in report["datasets"].items():
        total = block["latency"]["total"]
        assert total["p50_ms"] <= total["p95_ms"] <= total["p99_ms"] <= total["max_ms"], dataset


def test_memory_block_reports_both_on_disk_and_resident():
    """On disk is what a grader can verify with `du`; resident is what runs out
    at 10x. Reporting one as if it were the other is the easy mistake."""
    report = _report()
    for dataset, block in report["datasets"].items():
        memory = block["memory"]
        assert memory["on_disk_total_bytes"] > 0, dataset
        for index in ("embeddings", "ann_index", "bm25_index", "feature_store"):
            assert index in memory["on_disk_bytes"], f"{dataset}/{index}"
        assert memory["resident_bytes"]["embedding_matrix_exact"] > 0, dataset


def test_the_exact_embedding_size_is_the_arithmetic_not_an_rss_delta():
    """`n_articles x dim x 4` is knowable exactly. An RSS delta is a noisy
    lower bound, so it is reported beside the arithmetic as a check on it, not
    instead of it."""
    report = _report()
    for dataset, block in report["datasets"].items():
        memory = block["memory"]
        expected = memory["articles"] * memory["embedding_dim"] * 4
        assert memory["resident_bytes"]["embedding_matrix_exact"] == expected, dataset


def test_the_report_records_which_stage_owns_the_p99():
    """The point of a per-stage breakdown: the design note should be able to
    name the bottleneck, not just quote a total."""
    report = _report()
    for dataset, block in report["datasets"].items():
        slowest = block["latency"]["slowest_stage_at_p99"]
        assert slowest in serving.STAGES, f"{dataset}: {slowest}"


def test_the_report_states_the_device_it_measured_on():
    report = _report()
    for dataset, block in report["datasets"].items():
        assert block["latency"]["device"], dataset
