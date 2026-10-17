"""Reading `MINDlarge_test/behaviors.tsv` for submission (Q5.1).

Design: agent_docs/q5_plan.md

## Why this is not `data/mind.py`

`data/mind.py` parses MIND into the canonical schema: split, labels, history,
the feature store. None of that applies here. `MINDlarge_test` has **no labels**
-- it is the file the leaderboard scores against -- so it has no split, cannot
be evaluated offline, and must not enter the feature store, where every table
carries a `split` column and the leakage tests assert properties of it.

Registering it in `config.DATASETS` would ripple through every split and leakage
test for a corpus that has neither. So the submission path reads what it needs
and nothing else: the impression id and the candidate list.

## Streamed, because the file is 1.4 GB

2,370,727 rows. `polars.read_csv` on a 1.4 GB TSV where one column is a
space-separated list would materialise the whole thing to build the list column.
A line-at-a-time reader costs nothing here -- we need two fields and we write one
line out per line in.

## The row order IS the contract

OQ-1 records that the line count and `impression_id` order must match the truth
file exactly. We do not have the truth file, so the assumption -- stated rather
than buried -- is that it covers `behaviors.tsv` in `behaviors.tsv` order. That
is how MIND submissions are conventionally built, and the trivial-baseline
upload OQ-6 requires is what would expose it if wrong.

So nothing here sorts, groups or de-duplicates. Rows come out in file order.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from newsrec import config

#: Tab-separated: id, user, time, history, impressions. The last field holds
#: space-separated candidates, WITHOUT the `-0`/`-1` labels the train and dev
#: files carry -- this is the unlabelled test file.
N_COLUMNS = 5


class MindTestError(RuntimeError):
    """The test file is not shaped the way the submission writer expects."""


def behaviors_path(raw_root: Path | None = None) -> Path:
    root = config.RAW_DIR if raw_root is None else raw_root
    return root / "MIND" / "MINDlarge_test" / "behaviors.tsv"


def read_rows(path: Path | None = None) -> Iterator[tuple[str, list[str]]]:
    """Yield `(impression_id, candidate_ids)` in file order.

    Raises on a malformed row rather than skipping it. A skipped row would
    shorten the prediction file by one line, and the MIND scorer fills a missing
    line with all-1 ranks and scores it -- so the submission would come back
    with a real number and a silent hole in it.
    """
    path = behaviors_path() if path is None else path
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != N_COLUMNS:
                raise MindTestError(
                    f"{path.name} line {lineno}: {len(fields)} tab-separated fields, "
                    f"expected {N_COLUMNS}"
                )
            impression_id, candidates = fields[0], fields[4].split()
            if not impression_id:
                raise MindTestError(f"{path.name} line {lineno}: empty impression_id")
            if not candidates:
                raise MindTestError(
                    f"{path.name} line {lineno} (impression {impression_id}): no candidates. "
                    f"There is no well-formed line to write for an empty candidate list."
                )
            # Labels would mean we were handed the dev file by mistake, and the
            # candidate ids would silently carry a "-0" suffix.
            if "-" in candidates[0]:
                raise MindTestError(
                    f"{path.name} line {lineno}: candidate {candidates[0]!r} carries a label. "
                    f"This looks like a labelled dev file, not MINDlarge_test."
                )
            yield impression_id, candidates
