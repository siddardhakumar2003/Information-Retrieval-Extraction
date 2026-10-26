from __future__ import annotations

import faiss
import numpy as np

from .base import ANNIndex


class ExactIndex(ANNIndex):
    """Brute-force exact search (FAISS IndexFlatL2) — the reference baseline."""

    name = "exact"

    def _build(self, train: np.ndarray, **build_params) -> None:
        d = train.shape[1]
        self._index = faiss.IndexFlatL2(d)
        self._index.add(train)

    def set_search_params(self, **search_params) -> None:
        pass  # exact search has no tunable search-time parameter

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sq_dists, ids = self._index.search(queries, k)
        return ids, np.sqrt(np.maximum(sq_dists, 0.0))

    def _save(self, path: str) -> None:
        faiss.write_index(self._index, path)
