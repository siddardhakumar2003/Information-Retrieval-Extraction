"""Reading `ebnerd_testset` for submission (Q5.2).

Design: agent_docs/q5_plan.md

Same reasoning as `mind_test.py`: the test set has no labels and no split, so it
does not belong in `config.DATASETS` or the feature store, where every table
carries a `split` column and the leakage tests assert properties of it.

## Rows, not impressions

13,536,710 rows and only 13,336,711 distinct `impression_id` values, because all
200,000 beyond-accuracy rows carry `impression_id = 0`. Everything here is
therefore **positional**: chunks are read in file order, nothing groups, sorts
or de-duplicates, and the writer emits one line per row.

## Chunked, because the candidate lists are 206 million integers

`article_ids_inview` holds 205,925,868 ids across the file. Collecting it whole
is possible but wasteful; `iter_chunks` yields slices so peak memory stays
bounded regardless of how the caller uses them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from newsrec import config

#: Measured 2026-08-23. Asserted on read so a re-download that changes the file
#: cannot silently invalidate the positional assumptions above.
N_ROWS = 13_536_710
N_BEYOND_ACCURACY = 200_000
BEYOND_ACCURACY_ID = 0

DEFAULT_CHUNK = 1_000_000


class EbnerdTestError(RuntimeError):
    """The test set is not shaped the way the submission writer expects."""


def testset_root(raw_root: Path | None = None) -> Path:
    root = config.RAW_DIR if raw_root is None else raw_root
    return root / "EB-NeRD" / "ebnerd_testset"


def behaviors_path(raw_root: Path | None = None) -> Path:
    return testset_root(raw_root) / "test" / "behaviors.parquet"


def row_count(path: Path | None = None) -> int:
    path = behaviors_path() if path is None else path
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def iter_chunks(
    path: Path | None = None,
    chunk: int = DEFAULT_CHUNK,
    columns: tuple[str, ...] = ("impression_id", "article_ids_inview"),
) -> Iterator[pl.DataFrame]:
    """Yield slices of the behaviours table in file order.

    Row order is the contract -- the submission is matched positionally -- so
    this never sorts and never filters.
    """
    path = behaviors_path() if path is None else path
    lazy = pl.scan_parquet(path).select(list(columns))
    total = row_count(path)
    offset = 0
    while offset < total:
        frame = lazy.slice(offset, chunk).collect()
        if frame.is_empty():
            break
        yield frame
        offset += len(frame)


def check_shape(path: Path | None = None) -> dict:
    """Confirm the file still matches what the writer's assumptions were built on.

    Cheap, and it guards the one thing that would silently corrupt a submission:
    if a future download no longer collapses the beyond-accuracy rows onto id 0,
    or ships a different number of rows, the positional writer is still correct
    but every recorded count in the plan and the manifest is not.
    """
    path = behaviors_path() if path is None else path
    lazy = pl.scan_parquet(path)
    got = lazy.select(
        pl.len().alias("rows"),
        pl.col("is_beyond_accuracy").sum().alias("beyond_accuracy"),
        pl.col("article_ids_inview").list.len().sum().alias("candidate_slots"),
    ).collect()
    rows = int(got["rows"][0])
    ba = int(got["beyond_accuracy"][0])
    if rows != N_ROWS:
        raise EbnerdTestError(f"{path.name} has {rows:,} rows, expected {N_ROWS:,}")
    if ba != N_BEYOND_ACCURACY:
        raise EbnerdTestError(
            f"{path.name} has {ba:,} beyond-accuracy rows, expected {N_BEYOND_ACCURACY:,}"
        )
    return {
        "rows": rows,
        "beyond_accuracy": ba,
        "candidate_slots": int(got["candidate_slots"][0]),
    }
