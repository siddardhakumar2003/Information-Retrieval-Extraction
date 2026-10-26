"""Common interface every index wrapper (exact, HNSW, IVF-Flat, IVF-PQ)
implements, so benchmarking code (sweep.py, scripts/*) is written once and
works against any of them."""

from __future__ import annotations

import os
import tempfile
import time
from abc import ABC, abstractmethod

import numpy as np


class ANNIndex(ABC):
    name: str

    def __init__(self) -> None:
        self.build_time_s: float = 0.0
        self.build_params: dict = {}

    @abstractmethod
    def _build(self, train: np.ndarray, **build_params) -> None: ...

    def build(self, train: np.ndarray, **build_params) -> None:
        self.build_params = dict(build_params)
        start = time.perf_counter()
        self._build(train, **build_params)
        self.build_time_s = time.perf_counter() - start

    @abstractmethod
    def set_search_params(self, **search_params) -> None:
        """Set/update search-time parameters (e.g. efSearch, nprobe) in place."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Batched search. Returns (ids, distances), both shape (n_queries, k)."""

    @abstractmethod
    def _save(self, path: str) -> None: ...

    def size_bytes(self) -> int:
        """Serialise the index to a temp file and return its size on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.bin")
            self._save(path)
            return os.path.getsize(path)
