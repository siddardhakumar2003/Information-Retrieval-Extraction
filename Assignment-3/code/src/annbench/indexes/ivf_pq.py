from __future__ import annotations

import faiss
import numpy as np

from .base import ANNIndex
from ..gpu_utils import train_and_add


class IVFPQIndex(ANNIndex):
    """FAISS IndexIVFPQ. Build params: nlist, m_subvectors, nbits. Search param: nprobe.

    Training (IVF k-means + PQ codebooks) runs on GPU if one is available
    (see gpu_utils.train_and_add); search always stays on CPU, single-thread,
    per the assignment's methodology.
    """

    name = "ivf_pq"

    def _build(self, train: np.ndarray, nlist: int = 100, m_subvectors: int = 8, nbits: int = 8) -> None:
        d = train.shape[1]
        if d % m_subvectors != 0:
            raise ValueError(f"m_subvectors={m_subvectors} must divide dimensionality d={d}")
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m_subvectors, nbits, faiss.METRIC_L2)
        self._index = train_and_add(cpu_index, train)
        self._index.nprobe = 1

    def set_search_params(self, nprobe: int | None = None, **_ignored) -> None:
        if nprobe is not None:
            self._index.nprobe = nprobe

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sq_dists, ids = self._index.search(queries, k)
        dists = np.sqrt(np.maximum(sq_dists, 0.0))
        dists[ids == -1] = np.inf  # too few probes found fewer than k candidates
        return ids, dists

    def _save(self, path: str) -> None:
        faiss.write_index(self._index, path)
