"""Intra-list diversity, novelty, catalogue coverage.

Requirements covered: Q4.4
Design: agent_docs/q4_plan.md D4, D6

## Why these are reported at all

`eval_protocol.md` records that the EB-NeRD challenge deliberately builds its
test set to probe beyond-accuracy behaviour. A system tuned only for AUC is
being measured on axes it never optimised, so a diversity regression here is a
finding rather than noise.

All three are computed over each impression's **top-10 ranked candidates** --
the same ranking the accuracy metrics score, not a separate retrieval.

## Diversity

Mean pairwise cosine *distance* (`1 - cosine similarity`) between the article
embeddings of the top 10. Higher means a less repetitive slate. Because the
stored vectors are unit-norm (enforced on read by `embed.load`), the cosine is a
plain dot product and no renormalisation happens here -- if a vector were not
normalised, the right outcome is a loud failure upstream, not a silent rescue.

Fewer than two articles with embeddings gives no pair to compare and returns
`None`, excluded from the average the way a degenerate impression is.

## Novelty needs smoothing, and only measurement showed that

`eval_protocol.md` defines novelty as mean self-information `-log2(p(article))`
with `p` measured on the **training period only** -- measuring popularity on the
evaluation period would be the future-leak Q9 forbids.

Applied literally, the metric is undefined for most articles. Measured on the
evaluated candidate lists, the share never clicked in training is **80.4%**
(`mind_small`), **80.9%** (`ebnerd_demo`) and **68.0%** (`ebnerd_small`), and
`p = 0` gives `-log2(0) = inf`. News is fresh; most articles shown in the test
period simply did not exist during training.

So `p` is add-one (Laplace) smoothed: `p = (clicks + 1) / (total + V)`, with `V`
the catalogue size. Every article gets a finite value and an unseen article
comes out maximally novel, which is what "novel" ought to mean. The rejected
alternatives are in q4_plan D4: excluding unseen articles throws away 68-80% of
the data and turns novelty into a statement about the popular tail only, and
flooring at the smallest observed probability picks a constant that moves with
the dataset so cross-dataset numbers stop comparing.

The consequence is that the absolute value is not comparable to an unsmoothed
novelty from another paper. `V` is written into the report so the number is
reproducible, and the design note reports novelty **as a comparison between our
two retrievers on identical data**, never as an absolute.

## Coverage is not a per-impression quantity

Diversity and novelty are per-impression values that `bootstrap_ci` can average
and resample directly. Coverage -- the fraction of the catalogue appearing in
*any* top-10 -- is a property of the whole run: one impression contributes a set,
not a number. `coverage_ci` therefore does its own resampling, drawing whole
impressions with replacement and recomputing the union each time, from the same
`GLOBAL_SEED` and the same `BOOTSTRAP_N` as everything else.

That interval means something slightly different from the others -- "how much
would coverage move if we had drawn a different set of impressions" rather than
"how much would this average move" -- and the report says so inline.
"""

from __future__ import annotations

import numpy as np

from newsrec import config
from newsrec.retrieval.embed import ArticleEmbeddings

#: How many ranked articles each beyond-accuracy metric looks at.
#: `eval_protocol.md` fixes 10 for all three, so it is one constant rather than
#: three call sites that could drift.
TOP_N: int = 10


def row_index(embeddings: ArticleEmbeddings) -> dict[str, int]:
    """article_id -> row in `embeddings.vectors`.

    Built once per dataset by the caller and handed to `diversity`. Rebuilding
    it per impression would be a 97,753-entry dict constructed 2,000 times.
    """
    return {a: i for i, a in enumerate(embeddings.article_ids)}


def diversity(
    article_ids: list[str],
    embeddings: ArticleEmbeddings,
    row_of: dict[str, int] | None = None,
) -> float | None:
    """Mean pairwise cosine distance among `article_ids`, or `None` if undefined.

    Articles absent from the embedding table are skipped rather than treated as
    a zero vector: a missing article is unknown, and an all-zero row would read
    as maximally distant from everything and inflate the score.
    """
    row_of = row_index(embeddings) if row_of is None else row_of
    rows = [row_of[a] for a in article_ids if a in row_of]
    if len(rows) < 2:
        return None

    vectors = embeddings.vectors[rows]
    # Unit-norm rows, so the Gram matrix is the pairwise cosine similarity.
    sims = vectors @ vectors.T
    n = len(rows)
    # Off-diagonal upper triangle only: each unordered pair once, self-pairs
    # never (a self-similarity of 1 would drag every score toward 0 by 1/n).
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sims[iu].mean())


def smoothed_probabilities(
    popularity: dict[str, int],
    catalogue_size: int,
) -> tuple[dict[str, float], float]:
    """Add-one smoothed click probabilities, and the probability of an unseen article.

    Returns `(p_by_article, p_unseen)`. The second is what the great majority of
    candidates get, so it is returned explicitly rather than recomputed per
    lookup.
    """
    total = sum(popularity.values())
    denominator = float(total + catalogue_size)
    if denominator <= 0:
        # No training clicks and no catalogue: nothing to be novel relative to.
        return {}, float("nan")
    p_unseen = 1.0 / denominator
    return {a: (c + 1) / denominator for a, c in popularity.items()}, p_unseen


def novelty(
    article_ids: list[str],
    p_by_article: dict[str, float],
    p_unseen: float,
) -> float | None:
    """Mean `-log2(p)` over `article_ids`, with smoothed `p`.

    Takes the probability table rather than the raw counts so the smoothing is
    computed once per dataset instead of once per impression, and so a test can
    hand in a table it wrote by hand.
    """
    if not article_ids:
        return None
    ps = np.array([p_by_article.get(a, p_unseen) for a in article_ids], dtype=np.float64)
    if not np.all(ps > 0):
        raise ValueError("novelty needs strictly positive probabilities; smoothing was skipped")
    return float(-np.log2(ps).mean())


def novelty_unseen_fraction(
    article_ids: list[str],
    popularity: dict[str, int],
) -> float | None:
    """Fraction of `article_ids` with zero training-period clicks.

    A simpler, bounded-[0,1] novelty proxy alongside `novelty`'s self-information
    score: instead of weighting *how* rare each article is, it only asks
    whether it was ever clicked in training at all. Coarser than `novelty` (it
    cannot distinguish "rare" from "never seen"), but easier to read at a
    glance and comparable across differently-sized catalogues, which the
    self-information score by design is not (q4_plan.md D4).

    Uses the same `popularity` counts `novelty` is smoothed from, so this adds
    no new leakage surface: still training-period-only, still nothing from the
    evaluation window.
    """
    if not article_ids:
        return None
    unseen = sum(1 for a in article_ids if popularity.get(a, 0) == 0)
    return unseen / len(article_ids)


def coverage(top_lists: list[list[str]], catalogue_size: int) -> float:
    """Fraction of the catalogue appearing in any top-10.

    Bounded by what appears in candidate lists at all, not by the catalogue, so
    a low number here is expected and is not a defect -- the harness reports the
    number of distinct articles alongside it so a reader can see the ceiling.
    """
    if catalogue_size <= 0:
        return float("nan")
    seen: set[str] = set()
    for row in top_lists:
        seen.update(row)
    return len(seen) / catalogue_size


def coverage_ci(
    top_lists: list[list[str]],
    catalogue_size: int,
    *,
    n_resamples: int | None = None,
    ci: float | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """`coverage` with a bootstrap interval over impressions.

    Same shape as `bootstrap.bootstrap_ci` -- `{mean, ci_low, ci_high, n}` -- so
    the report writer treats it like every other metric. `mean` is the coverage
    of the actual sample, not the mean of the resamples: a resampled union is
    biased low (a bootstrap draw misses ~37% of the impressions), and reporting
    that as the headline would understate coverage systematically. The interval
    still comes from the resamples, which is what it is for.
    """
    n_resamples = config.BOOTSTRAP_N if n_resamples is None else n_resamples
    ci = config.BOOTSTRAP_CI if ci is None else ci
    seed = config.GLOBAL_SEED if seed is None else seed

    n = len(top_lists)
    point = coverage(top_lists, catalogue_size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if n == 1:
        return {"mean": point, "ci_low": point, "ci_high": point, "n": 1}

    sets = [set(row) for row in top_lists]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    values = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        seen: set[str] = set()
        for i in draws[b]:
            seen |= sets[i]
        values[b] = len(seen) / catalogue_size

    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return {"mean": point, "ci_low": float(low), "ci_high": float(high), "n": n}
