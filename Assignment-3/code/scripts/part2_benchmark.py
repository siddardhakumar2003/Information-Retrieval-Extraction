#!/usr/bin/env python3
"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ against the exact baseline on
both datasets. The actual build/sweep/measure logic lives in
annbench.benchmark (shared with scripts/part3_scaling.py); this script just
drives it with Part 2's grids and datasets and produces Part 2's plots/tables.
See annbench.benchmark's docstring for the tune/final query-split methodology.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annbench.benchmark import benchmark_exact, benchmark_index_type, write_cost_table, write_qps_table
from annbench.data import load_dataset, split_queries
from annbench.grids import hnsw_build_grid, ivf_flat_build_grid, ivf_pq_build_grid
from annbench.io_utils import append_results
from annbench.paths import FIGURES_DIR, RESULTS_CSV, ensure_output_dirs
from annbench.plotting import plot_recall_vs_qps

PART = 2
DATASETS = ["sift", "glove"]


def run_for_dataset(name: str, quick: bool) -> list[dict]:
    print(f"\n=== Part 2: {name} ===")
    dataset = load_dataset(name, quick=quick)
    tune, final = split_queries(dataset)
    n, d = dataset.train.shape
    timing_repeats = 1 if quick else 3

    all_rows: list[dict] = []

    exact_row, exact_qps = benchmark_exact(
        name, dataset.train, final, timing_repeats, part=PART, notes="exact baseline for Part 2 plot/tables"
    )
    all_rows.append(exact_row)

    curves: dict[str, list[tuple[float, float]]] = {}
    build_grids = {
        "hnsw": hnsw_build_grid(quick),
        "ivf_flat": ivf_flat_build_grid(n, quick),
        "ivf_pq": ivf_pq_build_grid(n, d, quick),
    }
    for index_type, build_grid in build_grids.items():
        rows, points = benchmark_index_type(
            index_type, name, dataset.train, tune, final, quick, build_grid, timing_repeats, part=PART
        )
        all_rows.extend(rows)
        curves[index_type] = points

    # --- plot ---
    fig_path = FIGURES_DIR / f"part2_recall_vs_qps_{name}.png"
    plot_recall_vs_qps(curves, fig_path, title=f"Recall@10 vs QPS — {name}", exact_qps=exact_qps)
    print(f"  wrote {fig_path}")

    # --- tables ---
    write_qps_table(FIGURES_DIR / f"part2_qps_at_recall_{name}.csv", curves)
    write_cost_table(FIGURES_DIR / f"part2_cost_at_recall95_{name}.csv", curves, all_rows)

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a 100K-vector train subset and a smaller grid")
    args = parser.parse_args()

    ensure_output_dirs()
    all_rows: list[dict] = []
    for name in DATASETS:
        all_rows.extend(run_for_dataset(name, quick=args.quick))

    append_results(all_rows, RESULTS_CSV)
    print(f"\nPart 2 done. Appended {len(all_rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
