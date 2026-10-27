#!/usr/bin/env python3
"""Part 4 (self-generated data): curse of dimensionality.

Self-generated uniform-random synthetic data (annbench.synthetic), dataset
size held fixed, dimensionality D varied: 32 (control), 64, 128, 256 -- the
classic setting where pairwise distances concentrate as D grows (the
assignment's own named example). HNSW and IVF-Flat are each evaluated at ONE
fixed build+search configuration (not swept to a recall target, unlike
Parts 1-3) -- the point is to show recall degrading as D grows under a
constant search-time budget. Exact search is included as the unaffected
reference (it never prunes, so distance concentration doesn't hurt it).

Hypothesis (see ../hypothesis.md, committed before this script was run):
recall@10 at fixed HNSW ef / IVF-Flat nprobe degrades monotonically as D
increases, because nearest/farthest-neighbour distance contrast shrinks
with dimensionality, weakening distance-based pruning.

No tune/final query split here (unlike Parts 1-3): there's no search-param
sweep/selection happening -- HNSW/IVF-Flat each use one fixed, pre-decided
config chosen independently of this run's data -- so there's nothing to
protect against leakage; all generated queries are used directly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annbench.benchmark import build_index
from annbench.indexes import ExactIndex
from annbench.io_utils import append_results
from annbench.paths import FIGURES_DIR, RESULTS_CSV, ensure_output_dirs
from annbench.recall import mean_recall_at_k
from annbench.synthetic import generate_uniform
from annbench.timing import queries_per_second_single, set_num_threads

PART = 4
DATASET = "synthetic_dim"
K = 10

DIMENSIONS = [32, 64, 128, 256]
FULL_N_TRAIN = 50_000
FULL_N_QUERIES = 2_000
QUICK_N_TRAIN = 10_000
QUICK_N_QUERIES = 500

HNSW_PARAMS = {"M": 16, "ef_construction": 200}
HNSW_EF = 40
IVF_FLAT_PARAMS = {"nlist": 100}
IVF_FLAT_NPROBE = 8


def run_for_dimension(d: int, quick: bool, timing_repeats: int) -> list[dict]:
    n_train = QUICK_N_TRAIN if quick else FULL_N_TRAIN
    n_queries = QUICK_N_QUERIES if quick else FULL_N_QUERIES
    print(f"\n=== Part 4 (dimensionality): d={d}, n_train={n_train:,} ===")

    train, queries = generate_uniform(n=n_train, d=d, n_queries=n_queries)

    # Ground truth: brute-force search IS the ground truth for self-generated data.
    exact = ExactIndex()
    exact.build(train)
    set_num_threads(1)
    gt_ids, gt_dists = exact.search(queries, K)
    exact_qps = queries_per_second_single(lambda q: exact.search(q, K), queries, repeats=timing_repeats)
    exact_recall = mean_recall_at_k(gt_dists, gt_dists, k=K)  # trivially 1.0, included for the plot/table

    rows = [
        {
            "part": PART,
            "dataset": DATASET,
            "index_type": "exact",
            "split": "all",
            "build_params": "{}",
            "search_param_name": "",
            "search_param_value": "",
            "recall_at_10": exact_recall,
            "qps_single_thread": exact_qps,
            "build_time_s": exact.build_time_s,
            "index_size_bytes": exact.size_bytes(),
            "bytes_per_vector": exact.size_bytes() / n_train,
            "n_train": n_train,
            "n_queries": n_queries,
            "notes": f"d={d}",
        }
    ]

    hnsw = build_index("hnsw", train, HNSW_PARAMS)
    hnsw.set_search_params(ef=HNSW_EF)
    hnsw_ids, hnsw_dists = hnsw.search(queries, K)
    hnsw_recall = mean_recall_at_k(gt_dists, hnsw_dists, k=K)
    hnsw_qps = queries_per_second_single(lambda q: hnsw.search(q, K), queries, repeats=timing_repeats)
    rows.append(
        {
            "part": PART,
            "dataset": DATASET,
            "index_type": "hnsw",
            "split": "all",
            "build_params": str(HNSW_PARAMS),
            "search_param_name": "ef",
            "search_param_value": HNSW_EF,
            "recall_at_10": hnsw_recall,
            "qps_single_thread": hnsw_qps,
            "build_time_s": hnsw.build_time_s,
            "index_size_bytes": hnsw.size_bytes(),
            "bytes_per_vector": hnsw.size_bytes() / n_train,
            "n_train": n_train,
            "n_queries": n_queries,
            "notes": f"d={d}",
        }
    )

    ivf_flat = build_index("ivf_flat", train, IVF_FLAT_PARAMS)
    ivf_flat.set_search_params(nprobe=IVF_FLAT_NPROBE)
    ivf_ids, ivf_dists = ivf_flat.search(queries, K)
    ivf_recall = mean_recall_at_k(gt_dists, ivf_dists, k=K)
    ivf_qps = queries_per_second_single(lambda q: ivf_flat.search(q, K), queries, repeats=timing_repeats)
    rows.append(
        {
            "part": PART,
            "dataset": DATASET,
            "index_type": "ivf_flat",
            "split": "all",
            "build_params": str(IVF_FLAT_PARAMS),
            "search_param_name": "nprobe",
            "search_param_value": IVF_FLAT_NPROBE,
            "recall_at_10": ivf_recall,
            "qps_single_thread": ivf_qps,
            "build_time_s": ivf_flat.build_time_s,
            "index_size_bytes": ivf_flat.size_bytes(),
            "bytes_per_vector": ivf_flat.size_bytes() / n_train,
            "n_train": n_train,
            "n_queries": n_queries,
            "notes": f"d={d}",
        }
    )

    print(f"  recall@10: exact={exact_recall:.3f} hnsw={hnsw_recall:.3f} ivf_flat={ivf_recall:.3f}")
    return rows


def plot_recall_vs_dimension(dimensions: list[int], recall_by_index: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for index_type, recalls in recall_by_index.items():
        ax.plot(dimensions, recalls, marker="o", label=index_type)
    ax.set_xlabel("dimensionality (d)")
    ax.set_ylabel("recall@10 (fixed search-time budget)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Part 4: recall vs dimensionality at a fixed search budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use smaller n_train/n_queries")
    args = parser.parse_args()

    ensure_output_dirs()
    timing_repeats = 1 if args.quick else 3

    all_rows: list[dict] = []
    recall_by_index: dict[str, list[float]] = {"exact": [], "hnsw": [], "ivf_flat": []}

    for d in DIMENSIONS:
        rows = run_for_dimension(d, quick=args.quick, timing_repeats=timing_repeats)
        all_rows.extend(rows)
        for row in rows:
            recall_by_index[row["index_type"]].append(row["recall_at_10"])

    fig_path = FIGURES_DIR / "part4_dimensionality_recall.png"
    plot_recall_vs_dimension(DIMENSIONS, recall_by_index, fig_path)
    print(f"\nwrote {fig_path}")

    table_path = FIGURES_DIR / "part4_dimensionality_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "d",
                "index_type",
                "recall_at_10",
                "qps_single_thread",
                "build_time_s",
                "index_size_bytes",
                "bytes_per_vector",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            d_value = int(row["notes"].split("=")[1])
            writer.writerow(
                {
                    "d": d_value,
                    "index_type": row["index_type"],
                    "recall_at_10": row["recall_at_10"],
                    "qps_single_thread": row["qps_single_thread"],
                    "build_time_s": row["build_time_s"],
                    "index_size_bytes": row["index_size_bytes"],
                    "bytes_per_vector": row["bytes_per_vector"],
                }
            )
    print(f"wrote {table_path}")

    append_results(all_rows, RESULTS_CSV)
    print(f"\nPart 4 done. Appended {len(all_rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
