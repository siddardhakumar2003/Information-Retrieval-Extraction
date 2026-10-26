from __future__ import annotations

import hnswlib
import numpy as np

from .base import ANNIndex


class HNSWIndex(ANNIndex):
    """hnswlib HNSW index. Build params: M, ef_construction. Search param: ef."""

    name = "hnsw"

    def _build(self, train: np.ndarray, M: int = 16, ef_construction: int = 200) -> None:
        n, d = train.shape
        self._index = hnswlib.Index(space="l2", dim=d)
        self._index.init_index(max_elements=n, M=M, ef_construction=ef_construction)
        self._index.add_items(train, np.arange(n))
        self._ef = max(ef_construction, 10)
        self._index.set_ef(self._ef)

    def set_search_params(self, ef: int | None = None, **_ignored) -> None:
        if ef is not None:
            self._ef = ef
            self._index.set_ef(ef)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # hnswlib raises if ef < k; clamp defensively so low-recall sweeps don't crash.
        if self._ef < k:
            self.set_search_params(ef=k)
        ids, sq_dists = self._index.knn_query(queries, k=k)
        return ids, np.sqrt(np.maximum(sq_dists, 0.0))

    def _save(self, path: str) -> None:
        self._index.save_index(path)
