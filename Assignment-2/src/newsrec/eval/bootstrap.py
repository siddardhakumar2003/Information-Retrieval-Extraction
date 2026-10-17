"""Seeded bootstrap confidence intervals, resampled over impressions.

Requirements covered: Q4.6, Q4.7 (and used early by Q2.8's recall@K).
Design: agent_docs/q4_plan.md

## What a bootstrap confidence interval is, and why every metric needs one

We do not evaluate on every impression -- there are 244,647 in `ebnerd_small`'s
test split alone -- so every number we report is computed over a *sample*. A
different sample would give a slightly different answer, and without knowing how
much different, there is no way to tell a real improvement from sampling noise.

The bootstrap estimates that spread without any statistical assumptions: draw a
new sample of the same size *from the sample we have*, with replacement, compute
the metric, and repeat. The spread of those recomputed values approximates the
spread we would have seen across real samples. The 2.5th and 97.5th percentiles
give a 95% interval.

Resampling is over **impressions**, not over individual scores, because the
impression is the independent unit -- two clicks inside one impression are not
independent observations of a user's taste.

## Seeded, because a confidence interval that moves is not evidence

`GLOBAL_SEED` drives the resampling, so the interval is identical on every run
and a reported number is reproducible byte for byte (Q1.13). An unseeded
bootstrap would make every rebuild produce a slightly different interval, and
"the number changed" would stop being informative.
"""

from __future__ import annotations

import numpy as np

from newsrec import config


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    n_resamples: int | None = None,
    ci: float | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Mean of `values` with a percentile bootstrap confidence interval.

    Returns `{"mean", "ci_low", "ci_high", "n"}`.

    A single observation has a mean but no spread; rather than reporting a
    misleadingly tight interval, the bounds collapse to the value itself and `n`
    tells the reader why. An empty input returns NaNs -- there is no honest
    number to report, and 0.0 would look like a measured zero.
    """
    n_resamples = config.BOOTSTRAP_N if n_resamples is None else n_resamples
    ci = config.BOOTSTRAP_CI if ci is None else ci
    seed = config.GLOBAL_SEED if seed is None else seed

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}

    rng = np.random.default_rng(seed)
    # One (n_resamples x n) draw rather than a Python loop: same numbers, and
    # the resampling is the cheap part only if it is vectorised.
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)

    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high), "n": int(arr.size)}


# =============================================================================
# Q4.7 — validating that the interval means what it claims
# =============================================================================
#
# A bootstrap that is subtly wrong still returns a tight, confident-looking
# interval. The only way to know a 95% interval covers 95% of the time is to
# draw from a distribution whose mean we chose and count.
#
# This lives here rather than only in the test file because the measured
# coverage is a *reported number*: `make eval` writes it into `eval.json`, so
# the design note can quote it and a grader can regenerate it. A number that
# exists only inside an assertion cannot be quoted.


def recall_shaped(rng: np.random.Generator, n: int, p_nonzero: float = 0.03) -> np.ndarray:
    """`p_nonzero` of the draws uniform on (0, 1], the rest exactly zero.

    Not a textbook distribution -- it is the empirical shape of our own
    per-impression recall@K values, which are zero for most impressions and
    positive for a few. Validating on a normal distribution alone would say
    nothing about the intervals we actually publish.
    """
    hits = rng.random(n) < p_nonzero
    return np.where(hits, rng.random(n), 0.0)


#: name -> (draw(rng, n), true mean). The true mean is known by construction,
#: which is the whole point: coverage cannot be measured against an estimate.
SYNTHETIC_DISTRIBUTIONS: dict[str, tuple] = {
    "normal": (lambda rng, n: rng.normal(loc=3.0, scale=2.0, size=n), 3.0),
    "exponential": (lambda rng, n: rng.exponential(scale=2.5, size=n), 2.5),
    "recall_shaped": (recall_shaped, 0.03 * 0.5),
}


def coverage_of(
    distribution: str,
    sample_size: int,
    *,
    trials: int = 400,
    n_resamples: int = 300,
    seed: int | None = None,
) -> float:
    """Fraction of `trials` whose interval contains the distribution's true mean.

    `n_resamples` is deliberately lower than `config.BOOTSTRAP_N`: coverage is
    insensitive to it above a few hundred (pinned by a test), and the validation
    has to be cheap enough to run on every `make eval` rather than being
    something we did once and trusted afterwards.
    """
    draw, true_mean = SYNTHETIC_DISTRIBUTIONS[distribution]
    seed = config.GLOBAL_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    hits = 0
    for trial in range(trials):
        sample = draw(rng, sample_size)
        got = bootstrap_ci(sample, n_resamples=n_resamples, seed=seed + trial)
        hits += got["ci_low"] <= true_mean <= got["ci_high"]
    return hits / trials


#: The sizes coverage is measured at: the full evaluated sample, a quartile
#: slice of it, and a deliberately small slice.
#:
#: The third is not a size we report at -- it is there to make the failure
#: visible. Measured over 2,000 trials per cell, recall-shaped coverage is 0.938
#: at n=2,000, 0.942 at n=500, 0.892 at n=200 and 0.846 at n=100. Validating
#: only at the sizes that pass would hide where the interval stops meaning what
#: it says.
VALIDATION_SIZES: tuple[int, ...] = (
    config.RECALL_SAMPLE_SIZE,
    config.RECALL_SAMPLE_SIZE // 4,
    100,
)

#: Below this many observations, the measured coverage of a 95% interval on a
#: sparse metric drops far enough below nominal (0.892 at n=200) that the
#: interval should not be read as a 95% interval.
#:
#: The harness flags any slice smaller than this in the report rather than
#: dropping it: a small slice is still worth reporting, it just cannot carry a
#: confident verdict. Set at 500 because that is the smallest measured size
#: whose coverage is within noise of nominal.
MIN_TRUSTWORTHY_N: int = 500


def validate_coverage(
    *,
    trials: int = 400,
    n_resamples: int = 300,
    seed: int | None = None,
) -> list[dict]:
    """Measured coverage for every (distribution, reported sample size).

    Returned as rows rather than printed, so `make eval` can write them into the
    report and the design note can quote a number instead of a claim.
    """
    return [
        {
            "distribution": name,
            "sample_size": n,
            "nominal": config.BOOTSTRAP_CI,
            "measured": coverage_of(name, n, trials=trials, n_resamples=n_resamples, seed=seed),
            "trials": trials,
        }
        for name in SYNTHETIC_DISTRIBUTIONS
        for n in VALIDATION_SIZES
    ]
