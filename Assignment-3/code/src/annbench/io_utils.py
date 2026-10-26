"""Shared results.csv bookkeeping so every scripts/partN_*.py writes rows into
one accumulated file with a consistent schema, without clobbering rows other
scripts already wrote."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Stable column order; any row dict may omit columns that don't apply to it
# (e.g. "m_subvectors" only applies to ivf_pq) and they're filled with NaN.
RESULTS_COLUMNS = [
    "part",
    "dataset",
    "index_type",
    "split",
    "build_params",
    "search_param_name",
    "search_param_value",
    "recall_at_10",
    "qps_single_thread",
    "build_time_s",
    "index_size_bytes",
    "bytes_per_vector",
    "n_train",
    "n_queries",
    "notes",
]


def append_results(rows: list[dict], csv_path: Path) -> None:
    """Append rows to results.csv, replacing any existing rows with the same
    (part, dataset) pair. This makes every scripts/partN_*.py idempotent on
    re-run (running run_all.sh twice regenerates, rather than duplicates, its
    own rows) while still accumulating rows across different parts/datasets
    written by other scripts."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    for col in RESULTS_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    new_df = new_df[RESULTS_COLUMNS]

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        new_keys = set(zip(new_df["part"], new_df["dataset"]))
        keep_mask = ~existing.apply(lambda r: (r["part"], r["dataset"]) in new_keys, axis=1)
        combined = pd.concat([existing[keep_mask], new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
