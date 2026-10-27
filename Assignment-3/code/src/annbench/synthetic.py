"""Seeded synthetic data generator for Part 4.

generate_uniform draws train/query points uniformly at random from [low, high]^d
-- the classic setting for demonstrating the curse of dimensionality: as d
grows, pairwise distances between uniform random points concentrate (the
ratio of nearest to farthest neighbour distance approaches 1), which is
exactly the regime the assignment names as its example ("with uniform random
vectors, nearest neighbours become barely closer than everything else as the
dimension grows").

Seeded via np.random.default_rng, so the same arguments always produce the
same arrays -- required for run_all.sh's reproducibility.
"""

from __future__ import annotations

import numpy as np


def generate_uniform(
    n: int,
    d: int,
    n_queries: int,
    low: float = -1.0,
    high: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = rng.uniform(low, high, size=(n, d)).astype(np.float32)
    queries = rng.uniform(low, high, size=(n_queries, d)).astype(np.float32)
    return train, queries
