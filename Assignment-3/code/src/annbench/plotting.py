"""Recall-vs-QPS trade-off curves: the Pareto-optimal frontier (max achievable
QPS for a given recall or better) over all swept build x search configurations
for an index, plus the plot that overlays each index's frontier."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Given (recall, qps) points, return the upper envelope sorted by recall
    ascending: for any recall value, the best (highest) QPS among points with
    recall >= that value is on, or dominates, this curve."""
    if not points:
        return []
    # Sort by recall descending, then keep a running max QPS so we only retain
    # points that are not dominated by a higher-recall-and-at-least-as-fast point.
    sorted_desc = sorted(points, key=lambda p: (-p[0], -p[1]))
    frontier_desc: list[tuple[float, float]] = []
    best_qps_so_far = -1.0
    for recall, qps in sorted_desc:
        if qps > best_qps_so_far:
            frontier_desc.append((recall, qps))
            best_qps_so_far = qps
    return list(reversed(frontier_desc))  # ascending recall


def qps_at_recall(frontier: list[tuple[float, float]], target_recall: float) -> float | None:
    """Given an ascending-recall Pareto frontier, return the QPS of the cheapest
    (highest-QPS) point that reaches at least target_recall, or None if the
    index never reaches it in the swept range."""
    for recall, qps in frontier:
        if recall >= target_recall:
            return qps
    return None


def plot_recall_vs_qps(
    curves: dict[str, list[tuple[float, float]]],
    out_path: Path,
    title: str,
    exact_qps: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for index_name, points in curves.items():
        frontier = pareto_frontier(points)
        if not frontier:
            continue
        recalls, qpss = zip(*frontier)
        ax.plot(recalls, qpss, marker="o", label=index_name)

    if exact_qps is not None:
        ax.axhline(exact_qps, color="gray", linestyle="--", label="exact (brute force)")

    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second (single thread, one query at a time)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
