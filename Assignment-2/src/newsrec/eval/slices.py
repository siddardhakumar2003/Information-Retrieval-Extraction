"""Cold-start vs warm users and head vs tail articles (Q4.5).

Design: agent_docs/q4_plan.md D5

## What a slice is here

A boolean mask over the evaluated impressions. Every metric is computed once
over all impressions and again over each mask, so a slice never changes how a
number is produced -- only which rows it is averaged over. Nothing in a slice
definition reaches a score, a ranking or a model input.

That distinction matters for `article_head` below, which is derived from
test-period clicks. It labels which bucket an already-computed result is
reported in; every number was computed before the slice existed. An assessor
scanning for leakage will see future-derived popularity in this report and is
right to ask, so it is said here in so many words.

## Cold-start vs warm

Membership is the user's history length, which is training-period data, so Q4.5
is satisfied without argument.

`cold_start` is "no usable prior clicks at all" rather than "fewer than a
threshold", because measurement left no room for anything else: q3_plan D8
found that on both EB-NeRD corpora even "fewer than 5 clicks" selects **0 of
2,000** impressions -- the median EB-NeRD user has 88. A tunable threshold would
be a knob with one meaningful setting. MIND does have a genuine cold set (3,238
train impressions with empty history), so the slice is informative there and
empty on EB-NeRD, which is a fact about the data and is reported as such.

The quartile slices carry the load on EB-NeRD: `history_short` and
`history_long` are the bottom and top quartiles of **this dataset's own**
history distribution, so they compare positions within each population rather
than absolute click counts.

## Head vs tail, reported twice, because Q4.5 and the data disagree

Q4.5 requires slice membership computed from training-period data only. Applied
to head/tail that produces a degenerate cut: measured, **85.0% of MIND and 98.6%
of `ebnerd_demo` test impressions contain no article that was ever clicked
during training**, because news is fresh and test-period articles mostly did not
exist yet. The "tail" bucket becomes "almost everything" at a ratio of up to
1:70, and a comparison between a 1.4% bucket and a 98.6% bucket is not a
comparison.

Rather than pick one and hide the problem, both cuts are reported and labelled:

- `article_head_train` / `article_tail_train` -- the **Q4.5-compliant** cut,
  from training-period clicks, with its degeneracy stated in the report next to
  the numbers.
- `article_head` / `article_tail` -- the q3_plan D8 cut, from test-period
  clicks, which splits impressions near 50/50 (59.7% / 48.7% / 51.0%) and is
  where the interesting result lives.

An impression is "head" if **any** article it clicked is a head article. The
alternative -- all clicked articles -- would put a user who clicked one popular
and one obscure story in the tail bucket, which reads the slice as a statement
about the user rather than about what was clicked.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from newsrec.retrieval import compare

#: Slices whose membership uses only training-period data, so a reader can tell
#: at a glance which numbers satisfy Q4.5 literally.
TRAINING_PERIOD_SLICES: tuple[str, ...] = (
    "cold_start",
    "warm",
    "history_short",
    "history_long",
    "article_head_train",
    "article_tail_train",
)

#: Slices whose membership uses test-period clicks. Analysis descriptors, never
#: model inputs -- see the module docstring.
TEST_PERIOD_SLICES: tuple[str, ...] = ("article_head", "article_tail")


def build(
    sample: pl.DataFrame,
    history: pl.DataFrame,
    impressions: pl.DataFrame,
) -> dict[str, np.ndarray]:
    """name -> boolean mask over `sample` rows, in report order.

    `compare.build_slices` supplies the four q3_plan D8 slices unchanged, so
    Q3.6 and Q4 slice the same rows the same way. This adds the cold/warm pair
    Q4.5 names and the training-period head/tail cut Q4.5 mandates.
    """
    masks = dict(compare.build_slices(sample, history, impressions))

    lengths = compare.history_lengths(sample, history)
    masks["cold_start"] = lengths == 0
    masks["warm"] = lengths > 0

    train_head = compare.head_articles(impressions, split="train")
    touches = np.array(
        [bool(train_head.intersection(c or [])) for c in sample["clicked_ids"].to_list()]
    )
    masks["article_head_train"] = touches
    masks["article_tail_train"] = ~touches

    order = (*TRAINING_PERIOD_SLICES, *TEST_PERIOD_SLICES)
    return {name: masks[name] for name in order}


def partition_check(masks: dict[str, np.ndarray], n_rows: int) -> None:
    """Raise unless each head/tail and cold/warm pair partitions the sample.

    A slice pair that overlaps double-counts impressions, and one with a gap
    silently drops them; both produce per-slice numbers that look fine and do
    not reconstruct the overall number. Cheap to check, invisible when wrong.

    The quartile pair is deliberately NOT checked: `history_short` and
    `history_long` are the bottom and top quartiles and are meant to leave the
    middle half out.
    """
    for a, b in (
        ("cold_start", "warm"),
        ("article_head", "article_tail"),
        ("article_head_train", "article_tail_train"),
    ):
        both = masks[a] & masks[b]
        neither = ~(masks[a] | masks[b])
        if both.any() or neither.any():
            raise ValueError(
                f"slices {a}/{b} do not partition the sample: "
                f"{int(both.sum())} in both, {int(neither.sum())} in neither, of {n_rows}"
            )
