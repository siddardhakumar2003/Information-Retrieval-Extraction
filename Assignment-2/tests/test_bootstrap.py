"""Q4.6, Q4.7 — the bootstrap interval, validated rather than assumed.

Every number in this repo since Q2.8 has carried an interval from
`bootstrap_ci`, and until this file existed none of them had been checked. That
is the failure mode worth naming: a bootstrap that is subtly wrong still returns
a tight, confident-looking interval, and nothing downstream notices. A wrong
interval is worse than no interval, because it converts "we do not know" into a
claim.

The headline test is `test_ci_coverage_on_known_distribution`: draw many samples
from a distribution whose true mean we chose, and count how often the 95%
interval actually contains it. Nominal 95% coverage means about 95% of the time.

**Three distributions, not one**, because percentile-bootstrap coverage depends
on the shape of what is being resampled:

- **normal** — the easy case; failing here means the implementation is broken.
- **exponential** — strongly right-skewed, where a percentile bootstrap is known
  to undercover. Passing on normal alone would prove very little.
- **recall_shaped** — 97% zeros with a small positive tail, which is what our
  per-impression recall@K values actually look like. This is the distribution we
  bootstrap in practice, so it is the one whose coverage we actually need.

**Three sample sizes**, because coverage depends on how much data the interval
is built from: the full 2,000-impression sample, a 500-row quartile slice, and a
100-row sample that is there to fail. Measured over 2,000 trials per cell, the
recall-shaped coverage is 0.938 / 0.942 / 0.846 — trustworthy at both sizes we
report at, and badly overconfident below roughly 200, while a normal
distribution at the same n=100 still covers 0.958. That bound is the finding
this file exists to surface; see
`test_coverage_collapses_on_small_samples_and_that_bound_is_recorded`.

`normal` and `exponential` are checked at the slice size and `recall_shaped` at
all three: the first two exist to probe sensitivity to *shape*, which does not
need the largest sample, while the third is the distribution we actually publish
intervals over and so is checked at every size we publish at. `make eval`
measures the full grid regardless — the trimming is only to keep `make test`
fast enough to run on every change.

The distributions, the sizes and the measurement live in
`newsrec.eval.bootstrap`, not here, because `make eval` writes the measured
coverage into `eval.json`. A number that exists only inside an assertion cannot
be quoted in the design note.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from newsrec import config
from newsrec.eval import bootstrap
from newsrec.eval.bootstrap import bootstrap_ci

#: Trials per measurement. Coverage is itself estimated from a finite number of
#: trials and so has its own standard error: at p=0.95 and 300 trials that is
#: sqrt(.95*.05/300) = 1.3 percentage points. Every tolerance below is several
#: of those wide, so a passing run is evidence and a failing run is not noise.
TRIALS = 300
#: Resamples per interval. Below config.BOOTSTRAP_N=1000 purely for runtime;
#: `test_coverage_is_stable_at_the_configured_resample_count` pins that this
#: does not change what is being validated.
RESAMPLES = 200
#: The size our headline numbers are computed at.
FULL = config.RECALL_SAMPLE_SIZE
#: A quartile slice of it — the size our per-slice numbers are computed at.
SLICE = config.RECALL_SAMPLE_SIZE // 4


# -----------------------------------------------------------------------------
# Q4.7 — the named requirement
# -----------------------------------------------------------------------------


def test_ci_coverage_on_known_distribution():
    """Nominal 95% intervals contain the true mean about 95% of the time.

    The lower bound is what matters: an interval that covers too rarely is
    overconfident, and every "established" and "not established" verdict in the
    Q2 and Q3 reports rests on it not being.
    """
    got = bootstrap.coverage_of("normal", SLICE, trials=TRIALS, n_resamples=RESAMPLES)
    assert 0.92 <= got <= 0.98, f"normal coverage {got:.3f}"


def test_ci_coverage_on_a_skewed_distribution():
    got = bootstrap.coverage_of("exponential", SLICE, trials=TRIALS, n_resamples=RESAMPLES)
    assert 0.92 <= got <= 0.98, f"exponential coverage {got:.3f}"


def test_ci_coverage_on_the_recall_shaped_distribution_we_actually_bootstrap():
    """The distribution our published intervals are actually computed over.

    Mostly zeros with a sparse positive tail — the case a normal-only validation
    would have declared fine.
    """
    got = bootstrap.coverage_of("recall_shaped", FULL, trials=TRIALS, n_resamples=RESAMPLES)
    assert 0.92 <= got <= 0.98, f"recall-shaped coverage {got:.3f}"


def test_ci_coverage_holds_at_the_slice_sized_sample():
    """A quartile slice — 500 impressions — still covers at close to nominal.

    Worth its own test because it is the size most of our *sliced* verdicts are
    computed at, and because it is not obvious in advance: a percentile
    bootstrap over a variable that is zero 97% of the time might plausibly have
    fallen apart here. Measured, it does not (0.933-0.945 across seeds).
    """
    got = bootstrap.coverage_of("recall_shaped", SLICE, trials=TRIALS, n_resamples=RESAMPLES)
    assert 0.92 <= got <= 0.98, f"slice-sized coverage {got:.3f}"


def test_coverage_collapses_on_small_samples_and_that_bound_is_recorded():
    """Where the interval stops meaning what it says.

    Measured over 2,000 trials per size, recall-shaped coverage is 0.938 at
    n=2,000, 0.942 at n=500, 0.892 at n=200 and 0.846 at n=100. So the interval
    is trustworthy at the sizes we report at, and materially overconfident below
    roughly 200 — at n=100 a "95%" interval misses the true mean about one time
    in six.

    It is the **interaction** of sparsity and small n, not small n alone: at the
    same n=100 the normal distribution still covers 0.958. A sample of 100
    impressions that are 97% zeros contains only about three non-zero values, so
    a large share of the resamples contain none at all and the interval collapses
    onto zero. That is why the flag is on the metric's sample size and not on
    some general notion of "small".

    That is why `MIN_TRUSTWORTHY_N` exists and why the harness flags any slice
    below it rather than quietly printing a confident-looking interval. This
    test asserts the collapse is real, because a future change that fixed it
    would make the flag unnecessary and a future change that worsened it would
    make the flag insufficient — both should force a decision here.
    """
    small = bootstrap.coverage_of("recall_shaped", 100, trials=TRIALS, n_resamples=RESAMPLES)
    assert small < 0.90, f"expected visible undercoverage at n=100, got {small:.3f}"
    assert bootstrap.MIN_TRUSTWORTHY_N == 500
    assert 100 in bootstrap.VALIDATION_SIZES, "the failing size must stay in the report"


def test_coverage_is_stable_at_the_configured_resample_count():
    """RESAMPLES=200 is a runtime shortcut, not a different method.

    If coverage at the configured BOOTSTRAP_N differed materially from coverage
    at 200, the tests above would be validating something the pipeline does not
    run. Few trials — this checks agreement of the verdicts, not the level.
    """
    rng = np.random.default_rng(config.GLOBAL_SEED + 3000)
    agree = 0
    for trial in range(60):
        sample = rng.normal(loc=3.0, scale=2.0, size=200)
        short = bootstrap_ci(sample, n_resamples=RESAMPLES, seed=trial)
        full = bootstrap_ci(sample, n_resamples=config.BOOTSTRAP_N, seed=trial)
        agree += (short["ci_low"] <= 3.0 <= short["ci_high"]) == (
            full["ci_low"] <= 3.0 <= full["ci_high"]
        )
    assert agree >= 57


def test_validate_coverage_reports_every_distribution_at_every_reported_size():
    rows = bootstrap.validate_coverage(trials=40, n_resamples=100)
    assert {(r["distribution"], r["sample_size"]) for r in rows} == {
        (name, n) for name in bootstrap.SYNTHETIC_DISTRIBUTIONS for n in bootstrap.VALIDATION_SIZES
    }
    assert all(r["nominal"] == config.BOOTSTRAP_CI for r in rows)
    assert all(0.0 <= r["measured"] <= 1.0 for r in rows)


def test_the_validation_is_seeded_and_repeats_exactly():
    """A coverage figure that moves between runs cannot be quoted in a report."""
    first = bootstrap.coverage_of("normal", 200, trials=40, n_resamples=100)
    second = bootstrap.coverage_of("normal", 200, trials=40, n_resamples=100)
    assert first == second


def test_the_recall_shaped_draw_is_mostly_zero_as_its_name_claims():
    """Guards the fixture itself.

    If `recall_shaped` silently became dense, the sparse case would stop being
    tested and every test above would still pass.
    """
    sample = bootstrap.recall_shaped(np.random.default_rng(0), 10_000)
    assert 0.02 <= (sample > 0).mean() <= 0.04
    assert sample.min() == 0.0


# -----------------------------------------------------------------------------
# The interval is the interval it says it is
#
# Coverage tests cannot tell "correct" from "conservative": an interval computed
# at 97.5% instead of 95% covers the true mean about 97% of the time, which sits
# inside any tolerance loose enough not to be flaky. Found by mutation — widening
# the tails passed every test above. These two checks pin the width itself.
# -----------------------------------------------------------------------------


def test_the_endpoints_are_the_requested_percentiles_of_the_resample_distribution():
    """Recomputed independently, with the same seed, to the same numbers.

    This is the test that fails when the tail fraction is wrong in either
    direction — the one failure mode a coverage measurement is blind to.
    """
    values = np.random.default_rng(7).normal(size=400)
    got = bootstrap_ci(values, n_resamples=RESAMPLES, ci=0.95, seed=99)

    rng = np.random.default_rng(99)
    idx = rng.integers(0, values.size, size=(RESAMPLES, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])

    assert got["ci_low"] == pytest.approx(float(low))
    assert got["ci_high"] == pytest.approx(float(high))


def test_the_width_matches_the_analytic_normal_interval():
    """For a large normal sample the bootstrap should reproduce 2 x 1.96 x SE.

    A cross-check against theory rather than against our own code, so an error
    shared by both the implementation and the test above still shows up here.
    """
    values = np.random.default_rng(8).normal(loc=0.0, scale=1.0, size=4000)
    got = bootstrap_ci(values, n_resamples=config.BOOTSTRAP_N, seed=5)
    expected = 2 * 1.96 * values.std(ddof=1) / math.sqrt(values.size)
    assert (got["ci_high"] - got["ci_low"]) == pytest.approx(expected, rel=0.10)


def test_the_interval_narrows_roughly_as_one_over_root_n():
    """A bootstrap that ignored the sample size would give a constant width.

    Sixteen times the data should roughly quarter the width. "Roughly" is a
    factor-of-two band, which still fails hard for a width that does not move.
    """
    rng = np.random.default_rng(config.GLOBAL_SEED)
    small = bootstrap_ci(rng.normal(size=100), n_resamples=RESAMPLES)
    large = bootstrap_ci(rng.normal(size=1600), n_resamples=RESAMPLES)
    ratio = (small["ci_high"] - small["ci_low"]) / (large["ci_high"] - large["ci_low"])
    assert 2.0 <= ratio <= 8.0, f"width ratio {ratio:.2f}, expected about 4"


def test_a_99_percent_interval_is_wider_than_a_95_percent_one():
    values = np.random.default_rng(config.GLOBAL_SEED).normal(size=300)
    at95 = bootstrap_ci(values, n_resamples=RESAMPLES, ci=0.95)
    at99 = bootstrap_ci(values, n_resamples=RESAMPLES, ci=0.99)
    assert (at99["ci_high"] - at99["ci_low"]) > (at95["ci_high"] - at95["ci_low"])


def test_the_reported_mean_is_the_sample_mean_not_the_resample_mean():
    """The resample mean estimates the same thing but is not identical.

    Reporting it would make the headline number depend on the seed, so a rerun
    would change the metric and not just the interval.
    """
    values = np.random.default_rng(0).normal(size=500)
    got = bootstrap_ci(values, n_resamples=RESAMPLES)
    assert got["mean"] == pytest.approx(float(values.mean()))


def test_the_interval_brackets_the_sample_mean():
    values = np.random.default_rng(1).exponential(size=400)
    got = bootstrap_ci(values, n_resamples=RESAMPLES)
    assert got["ci_low"] <= got["mean"] <= got["ci_high"]


def test_zero_variance_input_gives_a_zero_width_interval():
    got = bootstrap_ci([0.25] * 50, n_resamples=RESAMPLES)
    assert got["mean"] == got["ci_low"] == got["ci_high"] == 0.25


# -----------------------------------------------------------------------------
# Reproducibility (Q1.13) and the degenerate inputs
# -----------------------------------------------------------------------------


def test_the_interval_is_identical_across_calls():
    values = np.random.default_rng(2).normal(size=200)
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_a_different_seed_moves_the_interval_but_not_the_mean():
    """Pins that the seed is actually used.

    A hard-coded internal seed would make these equal and every determinism test
    would still pass.
    """
    values = np.random.default_rng(3).normal(size=200)
    a = bootstrap_ci(values, seed=1, n_resamples=RESAMPLES)
    b = bootstrap_ci(values, seed=2, n_resamples=RESAMPLES)
    assert a["mean"] == b["mean"]
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_the_defaults_come_from_config():
    """Changing config.GLOBAL_SEED must change the interval.

    Otherwise the module documents a configured seed it does not read.
    """
    values = np.random.default_rng(4).normal(size=200)
    assert bootstrap_ci(values) == bootstrap_ci(values, seed=config.GLOBAL_SEED)
    assert bootstrap_ci(values) != bootstrap_ci(values, seed=config.GLOBAL_SEED + 1)


def test_an_empty_input_is_nan_not_zero():
    got = bootstrap_ci([])
    assert got["n"] == 0
    assert all(math.isnan(got[k]) for k in ("mean", "ci_low", "ci_high"))


def test_a_single_observation_collapses_rather_than_faking_spread():
    got = bootstrap_ci([0.7])
    assert got == {"mean": 0.7, "ci_low": 0.7, "ci_high": 0.7, "n": 1}


def test_n_is_the_number_of_observations_not_the_number_of_resamples():
    got = bootstrap_ci(np.zeros(37), n_resamples=RESAMPLES)
    assert got["n"] == 37
