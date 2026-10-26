"""Shared benchmarking core: build an index, sweep its search parameter, and
measure recall/QPS on the tune/final query splits. Used by both
scripts/part2_benchmark.py and scripts/part3_scaling.py so every part
measures things exactly the same way.

Methodology (per the assignment's query split):
  - "tune" split (first 5,000 queries): for every build x search-param
    combination, run a *batched* (cheap) search to check recall coverage
    spans low -> ~0.95+. This is a sanity check on the grid, not a selection
    step that leaks into reported numbers.
  - "final" split (last 5,000 queries): the same grid is re-evaluated here —
    recall (batched search) and queries-per-second (one query at a time,
    single thread, median of a few repeats) — and *only these* numbers are
    written to results.csv / plotted / tabulated.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .grids import search_grid_for, search_param_name
from .indexes import ExactIndex, HNSWIndex, IVFFlatIndex, IVFPQIndex
from .plotting import pareto_frontier, qps_at_recall
from .recall import mean_recall_at_k
from .timing import queries_per_second_single, set_num_threads

K = 10


def build_index(index_type: str, train: np.ndarray, build_params: dict):
    cls = {"hnsw": HNSWIndex, "ivf_flat": IVFFlatIndex, "ivf_pq": IVFPQIndex}[index_type]
    index = cls()
    index.build(train, **build_params)
    return index


def benchmark_index_type(
    index_type: str,
    dataset_name: str,
    train: np.ndarray,
    tune,
    final,
    quick: bool,
    build_grid: list[dict],
    timing_repeats: int,
    part: int,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Returns (results_rows, (recall, qps) points for the Pareto plot)."""
    rows: list[dict] = []
    points: list[tuple[float, float]] = []
    param_name = search_param_name(index_type)

    for build_params in build_grid:
        print(f"  [{index_type}] building with {build_params} ...")
        index = build_index(index_type, train, build_params)
        size_bytes = index.size_bytes()
        bytes_per_vec = size_bytes / train.shape[0]
        search_values = search_grid_for(index_type, quick, build_params)

        # --- tune split: cheap batched recall-coverage check only ---
        tune_recalls = []
        for value in search_values:
            index.set_search_params(**{param_name: value})
            ids, dists = index.search(tune.queries, K)
            tune_recalls.append(mean_recall_at_k(tune.distances[:, :K], dists, k=K))
        max_tune_recall = max(tune_recalls) if tune_recalls else 0.0
        if max_tune_recall < 0.95:
            print(
                f"    [note] {index_type} {build_params} did not reach recall 0.95 on the "
                f"tune split (max={max_tune_recall:.3f}) within the swept {param_name} range "
                f"{search_values}."
            )

        # --- final split: the numbers that get reported ---
        set_num_threads(1)
        for value in search_values:
            index.set_search_params(**{param_name: value})
            ids, dists = index.search(final.queries, K)
            recall_final = mean_recall_at_k(final.distances[:, :K], dists, k=K)
            qps = queries_per_second_single(
                lambda q: index.search(q, K), final.queries, repeats=timing_repeats
            )
            points.append((recall_final, qps))
            rows.append(
                {
                    "part": part,
                    "dataset": dataset_name,
                    "index_type": index_type,
                    "split": "final",
                    "build_params": str(build_params),
                    "search_param_name": param_name,
                    "search_param_value": value,
                    "recall_at_10": recall_final,
                    "qps_single_thread": qps,
                    "build_time_s": index.build_time_s,
                    "index_size_bytes": size_bytes,
                    "bytes_per_vector": bytes_per_vec,
                    "n_train": train.shape[0],
                    "n_queries": final.queries.shape[0],
                    "notes": "",
                }
            )

    return rows, points


def benchmark_exact(
    dataset_name: str, train: np.ndarray, final, timing_repeats: int, part: int, notes: str = ""
) -> tuple[dict, float]:
    index = ExactIndex()
    index.build(train)
    set_num_threads(1)
    ids, dists = index.search(final.queries, K)
    recall_final = mean_recall_at_k(final.distances[:, :K], dists, k=K)
    qps = queries_per_second_single(lambda q: index.search(q, K), final.queries, repeats=timing_repeats)
    size_bytes = index.size_bytes()
    row = {
        "part": part,
        "dataset": dataset_name,
        "index_type": "exact",
        "split": "final",
        "build_params": "{}",
        "search_param_name": "",
        "search_param_value": "",
        "recall_at_10": recall_final,
        "qps_single_thread": qps,
        "build_time_s": index.build_time_s,
        "index_size_bytes": size_bytes,
        "bytes_per_vector": size_bytes / train.shape[0],
        "n_train": train.shape[0],
        "n_queries": final.queries.shape[0],
        "notes": notes,
    }
    return row, qps


RECALL_TARGETS_FOR_QPS_TABLE = [0.90, 0.95]
RECALL_TARGET_FOR_COST_TABLE = 0.95


def write_qps_table(
    out_path: Path, curves: dict[str, list[tuple[float, float]]], recall_targets: list[float] = RECALL_TARGETS_FOR_QPS_TABLE
) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index_type"] + [f"qps_at_recall_{t:.2f}" for t in recall_targets])
        for index_type, points in curves.items():
            frontier = pareto_frontier(points)
            row = [index_type] + [
                qps_at_recall(frontier, t) if qps_at_recall(frontier, t) is not None else "not reached"
                for t in recall_targets
            ]
            writer.writerow(row)
    print(f"  wrote {out_path}")


def write_cost_table(
    out_path: Path,
    curves: dict[str, list[tuple[float, float]]],
    all_rows: list[dict],
    recall_target: float = RECALL_TARGET_FOR_COST_TABLE,
) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index_type", "build_time_s", "index_size_bytes", "bytes_per_vector", "achieved_recall"])
        for index_type in curves:
            # among rows for this index_type that reach >= recall_target, take the
            # one with the lowest build_time (ties broken by smallest size) as
            # a representative "at this recall" config.
            candidates = [
                r for r in all_rows if r["index_type"] == index_type and r["recall_at_10"] >= recall_target
            ]
            if not candidates:
                writer.writerow([index_type, "not reached", "not reached", "not reached", "not reached"])
                continue
            best = min(candidates, key=lambda r: (r["build_time_s"], r["index_size_bytes"]))
            writer.writerow(
                [index_type, best["build_time_s"], best["index_size_bytes"], best["bytes_per_vector"], best["recall_at_10"]]
            )
    print(f"  wrote {out_path}")
