#!/usr/bin/env python3
"""Part 3 (Scaling experiment): on SIFT, repeat the Part 2 benchmark at
100K, 300K and 1M vectors and see how the cost of reaching recall 0.95 grows
for exact search, HNSW and IVF-Flat (IVF-PQ is excluded — the assignment's
Scaling option names only these three).

Hypothesis (see ../hypothesis.md, committed before this script was run):
exact search's QPS should degrade roughly linearly with N; HNSW's QPS at
recall 0.95 should degrade much more slowly (closer to log N); IVF-Flat,
with nlist scaled ~sqrt(N) (same grid formula as Part 2), should fall in
between, degrading roughly with sqrt(N).

Reuses annbench.benchmark (same build/sweep/measure logic as Part 2) and
annbench.grids (same build-grid formulas, so nlist/M choices are generated
identically at every scale point) — this script only adds the loop over
train sizes and the scaling-specific plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annbench.benchmark import benchmark_exact, benchmark_index_type
from annbench.data import load_dataset, split_queries
from annbench.grids import hnsw_build_grid, ivf_flat_build_grid
from annbench.io_utils import append_results
from annbench.paths import FIGURES_DIR, RESULTS_CSV, ensure_output_dirs
from annbench.plotting import pareto_frontier, qps_at_recall

PART = 3
DATASET = "sift"
RECALL_TARGET = 0.95
INDEX_TYPES = ["hnsw", "ivf_flat"]  # + exact, handled separately (no sweep)

FULL_TRAIN_SIZES = [100_000, 300_000, 1_000_000]
QUICK_TRAIN_SIZES = [20_000, 50_000, 100_000]


def run_for_train_size(train_size: int, quick: bool) -> tuple[list[dict], dict[str, float | None]]:
    print(f"\n=== Part 3 (scaling): sift, n_train={train_size:,} ===")
    dataset = load_dataset(DATASET, max_train=train_size)
    tune, final = split_queries(dataset)
    n, d = dataset.train.shape
    timing_repeats = 1 if quick else 3

    all_rows: list[dict] = []
    qps_at_095: dict[str, float | None] = {}

    exact_row, exact_qps = benchmark_exact(
        DATASET, dataset.train, final, timing_repeats, part=PART, notes="Part 3 scaling experiment"
    )
    all_rows.append(exact_row)
    qps_at_095["exact"] = exact_qps  # exact recall is always 1.0, so it trivially satisfies >= 0.95

    build_grids = {
        "hnsw": hnsw_build_grid(quick),
        "ivf_flat": ivf_flat_build_grid(n, quick),
    }
    for index_type in INDEX_TYPES:
        rows, points = benchmark_index_type(
            index_type, DATASET, dataset.train, tune, final, quick, build_grids[index_type], timing_repeats, part=PART
        )
        all_rows.extend(rows)
        frontier = pareto_frontier(points)
        qps_at_095[index_type] = qps_at_recall(frontier, RECALL_TARGET)

    return all_rows, qps_at_095


def plot_scaling(train_sizes: list[int], qps_by_index: dict[str, list[float | None]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for index_type, qps_values in qps_by_index.items():
        xs = [n for n, q in zip(train_sizes, qps_values) if q is not None]
        ys = [q for q in qps_values if q is not None]
        if xs:
            ax.plot(xs, ys, marker="o", label=index_type)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_train (SIFT)")
    ax.set_ylabel("queries per second at recall@10 >= 0.95 (single thread)")
    ax.set_title("Part 3: cost of reaching recall 0.95 vs dataset size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use smaller train sizes and a smaller grid")
    args = parser.parse_args()

    ensure_output_dirs()
    train_sizes = QUICK_TRAIN_SIZES if args.quick else FULL_TRAIN_SIZES

    all_rows: list[dict] = []
    qps_by_index: dict[str, list[float | None]] = {"exact": [], "hnsw": [], "ivf_flat": []}
    cost_rows: list[dict] = []

    for train_size in train_sizes:
        rows, qps_at_095 = run_for_train_size(train_size, quick=args.quick)
        all_rows.extend(rows)
        for index_type in qps_by_index:
            qps_by_index[index_type].append(qps_at_095.get(index_type))
        # one cost-table row per (train_size, index_type), reusing the rows
        # just produced for this train_size only
        for index_type in ["exact", "hnsw", "ivf_flat"]:
            candidates = [r for r in rows if r["index_type"] == index_type and r["recall_at_10"] >= RECALL_TARGET]
            if candidates:
                best = min(candidates, key=lambda r: (r["build_time_s"], r["index_size_bytes"]))
                cost_rows.append({"n_train": train_size, **{k: best[k] for k in ["index_type", "build_time_s", "index_size_bytes", "bytes_per_vector", "recall_at_10"]}})
            else:
                cost_rows.append(
                    {
                        "n_train": train_size,
                        "index_type": index_type,
                        "build_time_s": "not reached",
                        "index_size_bytes": "not reached",
                        "bytes_per_vector": "not reached",
                        "recall_at_10": "not reached",
                    }
                )

    fig_path = FIGURES_DIR / "part3_scaling_qps_at_recall95.png"
    plot_scaling(train_sizes, qps_by_index, fig_path)
    print(f"\nwrote {fig_path}")

    import csv

    cost_table_path = FIGURES_DIR / "part3_scaling_cost_table.csv"
    with open(cost_table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_train", "index_type", "build_time_s", "index_size_bytes", "bytes_per_vector", "recall_at_10"]
        )
        writer.writeheader()
        writer.writerows(cost_rows)
    print(f"wrote {cost_table_path}")

    append_results(all_rows, RESULTS_CSV)
    print(f"\nPart 3 done. Appended {len(all_rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
