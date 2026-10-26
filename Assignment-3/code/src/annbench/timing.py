"""Timing helpers: batched vs one-query-at-a-time search, repeated and
reduced with the median (per the assignment: "repeat each timing a few times
and take the median").
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - faiss is a hard requirement at runtime
    faiss = None


def set_num_threads(n: int) -> None:
    """Pin FAISS (and therefore its BLAS backend) to n threads, so single-query/
    single-thread QPS numbers are measured consistently rather than silently
    benefiting from multi-core parallelism."""
    if faiss is not None:
        faiss.omp_set_num_threads(n)


def time_batch(search_fn: Callable[[np.ndarray], object], queries: np.ndarray, repeats: int = 5) -> float:
    """Median wall-clock seconds to search *all* queries in one batched call."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        search_fn(queries)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def time_single_query(
    search_fn: Callable[[np.ndarray], object], queries: np.ndarray, repeats: int = 3
) -> float:
    """Median seconds-per-query when searching one query at a time (a separate
    call per query), repeated `repeats` times over the whole query set."""
    per_repeat_totals = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(queries.shape[0]):
            search_fn(queries[i : i + 1])
        per_repeat_totals.append(time.perf_counter() - start)
    median_total = float(np.median(per_repeat_totals))
    return median_total / queries.shape[0]


def queries_per_second_single(
    search_fn: Callable[[np.ndarray], object], queries: np.ndarray, repeats: int = 3
) -> float:
    """QPS measured one query at a time (what Part 2 reports)."""
    seconds_per_query = time_single_query(search_fn, queries, repeats=repeats)
    return 1.0 / seconds_per_query if seconds_per_query > 0 else float("inf")
