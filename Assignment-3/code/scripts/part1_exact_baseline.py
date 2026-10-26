#!/usr/bin/env python3
"""Part 1: exact baseline.

For each dataset:
  - build FAISS IndexFlatL2 over `train`
  - verify its results agree with the ground truth stored in the .hdf5 file
    (using our own tie-aware recall@10, expecting ~1.0)
  - time single-query vs batched search on the held-out final 5,000 queries,
    both pinned to 1 thread (isolates batching overhead amortization) and
    batched with the default thread count (shows the full practical effect)
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annbench.data import load_dataset, split_queries
from annbench.indexes import ExactIndex
from annbench.io_utils import append_results
from annbench.paths import RESULTS_CSV, ensure_output_dirs
from annbench.recall import mean_recall_at_k
from annbench.timing import queries_per_second_single, set_num_threads, time_batch

DATASETS = ["sift", "glove"]
K = 10


def run_for_dataset(name: str, quick: bool) -> list[dict]:
    print(f"\n=== Part 1: {name} ===")
    dataset = load_dataset(name, quick=quick)
    _tune, final = split_queries(dataset)

    index = ExactIndex()
    index.build(dataset.train)
    print(f"  built exact index: n={dataset.train.shape[0]} d={dataset.train.shape[1]} "
          f"build_time={index.build_time_s:.2f}s")

    # --- verify against the file's own stored ground truth ---
    ids, dists = index.search(final.queries, K)
    verify_recall = mean_recall_at_k(final.distances[:, :K], dists, k=K)
    print(f"  recall@10 of exact search vs stored ground truth: {verify_recall:.4f} (expect ~1.0)")

    # --- timing: single query at a time vs one batched call ---
    set_num_threads(1)
    single_qps_1thread = queries_per_second_single(lambda q: index.search(q, K), final.queries, repeats=3)
    batch_time_1thread = time_batch(lambda q: index.search(q, K), final.queries, repeats=5)
    batch_qps_1thread = final.queries.shape[0] / batch_time_1thread

    n_cores = multiprocessing.cpu_count()
    set_num_threads(n_cores)
    batch_time_default = time_batch(lambda q: index.search(q, K), final.queries, repeats=5)
    batch_qps_default = final.queries.shape[0] / batch_time_default
    set_num_threads(1)

    print(f"  single-query QPS (1 thread):        {single_qps_1thread:,.1f}")
    print(f"  batched QPS (1 thread):              {batch_qps_1thread:,.1f}")
    print(f"  batched QPS (default, {n_cores} threads): {batch_qps_default:,.1f}")
    print(
        "  why batch is faster even at 1 thread: a single search() call amortizes fixed "
        "Python/FAISS call overhead across all queries and lets FAISS use blocked, "
        "cache-friendly matrix operations (BLAS-style distance computation) instead of "
        "repeating that overhead per query; the extra jump from 1 thread to "
        f"{n_cores} threads comes from FAISS parallelising the batched distance "
        "computation itself across cores, which a single query (one row) cannot benefit "
        "from as much."
    )

    rows = [
        {
            "part": 1,
            "dataset": name,
            "index_type": "exact",
            "split": "final",
            "build_params": "{}",
            "search_param_name": "",
            "search_param_value": "",
            "recall_at_10": verify_recall,
            "qps_single_thread": single_qps_1thread,
            "build_time_s": index.build_time_s,
            "index_size_bytes": index.size_bytes(),
            "bytes_per_vector": index.size_bytes() / dataset.train.shape[0],
            "n_train": dataset.train.shape[0],
            "n_queries": final.queries.shape[0],
            "notes": "single-query search, 1 thread",
        },
        {
            "part": 1,
            "dataset": name,
            "index_type": "exact",
            "split": "final",
            "build_params": "{}",
            "search_param_name": "",
            "search_param_value": "",
            "recall_at_10": verify_recall,
            "qps_single_thread": batch_qps_1thread,
            "build_time_s": index.build_time_s,
            "index_size_bytes": index.size_bytes(),
            "bytes_per_vector": index.size_bytes() / dataset.train.shape[0],
            "n_train": dataset.train.shape[0],
            "n_queries": final.queries.shape[0],
            "notes": "batched search, 1 thread",
        },
        {
            "part": 1,
            "dataset": name,
            "index_type": "exact",
            "split": "final",
            "build_params": "{}",
            "search_param_name": "",
            "search_param_value": "",
            "recall_at_10": verify_recall,
            "qps_single_thread": batch_qps_default,
            "build_time_s": index.build_time_s,
            "index_size_bytes": index.size_bytes(),
            "bytes_per_vector": index.size_bytes() / dataset.train.shape[0],
            "n_train": dataset.train.shape[0],
            "n_queries": final.queries.shape[0],
            "notes": f"batched search, {n_cores} threads (default)",
        },
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a 100K-vector train subset")
    args = parser.parse_args()

    ensure_output_dirs()
    all_rows: list[dict] = []
    for name in DATASETS:
        all_rows.extend(run_for_dataset(name, quick=args.quick))

    append_results(all_rows, RESULTS_CSV)
    print(f"\nPart 1 done. Appended {len(all_rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
