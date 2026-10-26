"""Build/search parameter grids for HNSW, IVF-Flat and IVF-PQ.

Shared between scripts/part2_benchmark.py (full sweep on both datasets) and
scripts/part3_scaling.py (same grids, applied at several SIFT train sizes) —
keeping the grid logic in one place means both experiments use exactly the
same build-config methodology (e.g. IVF-Flat's nlist always scaled ~sqrt(N)),
so comparisons across parts stay apples-to-apples.
"""

from __future__ import annotations

import numpy as np


def hnsw_build_grid(quick: bool) -> list[dict]:
    if quick:
        return [{"M": 16, "ef_construction": 100}]
    return [
        {"M": 8, "ef_construction": 200},
        {"M": 16, "ef_construction": 200},
        {"M": 32, "ef_construction": 200},
    ]


def hnsw_search_grid(quick: bool) -> list[int]:
    if quick:
        return [10, 20, 40, 80]
    return [10, 20, 40, 80, 160, 320, 640]


def _ivf_nlist_candidates(n: int, quick: bool) -> list[int]:
    if quick:
        return [100]
    base = max(int(round(np.sqrt(n))), 50)
    candidates = sorted({max(50, base // 2), base, base * 2})
    return candidates


def ivf_flat_build_grid(n: int, quick: bool) -> list[dict]:
    return [{"nlist": nlist} for nlist in _ivf_nlist_candidates(n, quick)]


def ivf_search_grid(quick: bool, nlist: int) -> list[int]:
    candidates = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    return sorted({min(p, nlist) for p in candidates})


def pq_m_subvector_grid(d: int) -> list[int]:
    """Pick 1-2 PQ sub-vector counts that divide d, targeting ~8-16 dim
    sub-vectors (a typical IVF-PQ regime)."""
    divisors = sorted(m for m in range(1, d + 1) if d % m == 0)
    target_ms = sorted({max(1, d // 16), max(1, d // 8)})
    chosen: list[int] = []
    for t in target_ms:
        best = min(divisors, key=lambda m: abs(m - t))
        if best not in chosen:
            chosen.append(best)
    return chosen


def ivf_pq_build_grid(n: int, d: int, quick: bool) -> list[dict]:
    nlists = _ivf_nlist_candidates(n, quick)[:1] if quick else _ivf_nlist_candidates(n, quick)[::2] or [
        _ivf_nlist_candidates(n, quick)[0]
    ]
    m_options = pq_m_subvector_grid(d)[:1] if quick else pq_m_subvector_grid(d)
    return [{"nlist": nlist, "m_subvectors": m, "nbits": 8} for nlist in nlists for m in m_options]


def search_param_name(index_type: str) -> str:
    return "ef" if index_type == "hnsw" else "nprobe"


def search_grid_for(index_type: str, quick: bool, build_params: dict) -> list[int]:
    if index_type == "hnsw":
        return hnsw_search_grid(quick)
    return ivf_search_grid(quick, build_params["nlist"])
