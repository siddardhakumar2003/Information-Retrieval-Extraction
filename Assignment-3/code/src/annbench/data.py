"""Loading SIFT / GloVe-100 ann-benchmarks datasets.

Each .hdf5 file has four top-level datasets: ``train`` (vectors to index),
``test`` (10,000 query vectors), ``neighbors`` and ``distances`` (ground-truth
top-100 nearest neighbours per query, as stored by ann-benchmarks).

GloVe-100 ships with angular (cosine) ground truth. Per the assignment, we
L2-normalise every GloVe vector on load, after which Euclidean distance ranks
vectors identically to cosine distance (||a-b||^2 = 2 - 2*cos(a,b) for unit
vectors), so every index downstream can just use Euclidean distance.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .paths import DATA_DIR

DATASET_URLS = {
    "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove": "http://ann-benchmarks.com/glove-100-angular.hdf5",
}
DATASET_FILENAMES = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}
QUICK_TRAIN_SIZE = 100_000
TUNE_QUERIES = 5_000


@dataclass
class Dataset:
    name: str
    train: np.ndarray  # (N, D) float32
    test: np.ndarray  # (10000, D) float32
    neighbors: np.ndarray  # (10000, >=10) int32, ground-truth neighbour ids into train
    distances: np.ndarray  # (10000, >=10) float32, ground-truth distances


def ensure_dataset(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download the dataset's .hdf5 file into data_dir if it isn't there already."""
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASET_URLS)}")

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / DATASET_FILENAMES[name]
    if path.exists() and path.stat().st_size > 0:
        return path

    url = DATASET_URLS[name]
    print(f"[data] {path.name} not found, downloading from {url} ...")
    tmp_path = path.with_suffix(".hdf5.part")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.rename(path)
    print(f"[data] downloaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_dataset(
    name: str, data_dir: Path = DATA_DIR, quick: bool = False, max_train: int | None = None
) -> Dataset:
    """Load a dataset, normalising GloVe vectors and optionally truncating train.

    `quick=True` truncates train to QUICK_TRAIN_SIZE (100K) — a convenience
    wrapper equivalent to `max_train=QUICK_TRAIN_SIZE`. `max_train` is the
    general form, used e.g. by Part 3's scaling experiment to truncate to
    arbitrary sizes (100K, 300K, 1M, ...). When `max_train` is at or above
    the full train size, no truncation happens and the file's own stored
    ground truth is used as-is.
    """
    path = ensure_dataset(name, data_dir)
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"][:], dtype=np.float32)
        test = np.asarray(f["test"][:], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"][:], dtype=np.int64)
        distances = np.asarray(f["distances"][:], dtype=np.float32)

    if name == "glove":
        train = _l2_normalize_rows(train)
        test = _l2_normalize_rows(test)
        # Ground-truth ids are unaffected by normalisation (monotonic distance
        # transform), but distances must be recomputed in the normalised space.
        distances = _recompute_gt_distances(train, test, neighbors)

    if quick:
        max_train = QUICK_TRAIN_SIZE

    if max_train is not None and max_train < train.shape[0]:
        train = train[:max_train]
        neighbors, distances = _recompute_ground_truth(train, test, k=neighbors.shape[1])

    return Dataset(name=name, train=train, test=test, neighbors=neighbors, distances=distances)


def _recompute_gt_distances(train: np.ndarray, test: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Recompute ground-truth distances (in the now-normalised space) for the stored neighbour ids."""
    nq, k = neighbors.shape
    out = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        pts = train[neighbors[i]]
        out[i] = np.linalg.norm(pts - test[i], axis=1)
    return out


def _recompute_ground_truth(train: np.ndarray, test: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force ground truth for a truncated (--quick) train set, since the
    file's stored neighbours assume the full train array."""
    import faiss

    d = train.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(train)
    distances_sq, neighbors = index.search(test, k)
    return neighbors.astype(np.int64), np.sqrt(np.maximum(distances_sq, 0.0)).astype(np.float32)


@dataclass
class QuerySplit:
    queries: np.ndarray
    neighbors: np.ndarray
    distances: np.ndarray


def split_queries(dataset: Dataset, tune_size: int = TUNE_QUERIES) -> tuple[QuerySplit, QuerySplit]:
    """Split the query set into a tuning half (first `tune_size`) and a final
    reporting half (the rest), with matching ground truth slices."""
    tune = QuerySplit(
        queries=dataset.test[:tune_size],
        neighbors=dataset.neighbors[:tune_size],
        distances=dataset.distances[:tune_size],
    )
    final = QuerySplit(
        queries=dataset.test[tune_size:],
        neighbors=dataset.neighbors[tune_size:],
        distances=dataset.distances[tune_size:],
    )
    return tune, final
