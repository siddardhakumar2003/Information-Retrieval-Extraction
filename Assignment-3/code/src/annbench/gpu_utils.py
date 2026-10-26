"""Optional GPU acceleration for the slow part of IVF-Flat/IVF-PQ: k-means
training during build(). Search is never run on GPU — every reported
recall/QPS number still comes from plain CPU, single-thread search, per the
assignment's methodology. This only speeds up the full (non-quick) run's
build step when a GPU + faiss-gpu are available; it's a no-op fallback to
plain CPU training everywhere else (e.g. your local faiss-cpu setup).
"""

from __future__ import annotations

import os

import faiss

# Opt-in: only try GPU training when explicitly enabled (set by run_ada.sh on
# the GPU node). Keeps local faiss-cpu runs from ever touching this path.
_GPU_ENABLED = os.environ.get("ANNBENCH_USE_GPU", "0") == "1"


def gpu_available() -> bool:
    return _GPU_ENABLED and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0


def train_and_add(cpu_index, train, gpu_id: int = 0):
    """Train + add `train` into `cpu_index`, using the GPU if available, and
    always return a CPU index (so search/size/save stay unchanged downstream).
    Falls back to CPU training on any GPU-path error (e.g. an unsupported
    nlist/m_subvectors combination for this faiss-gpu build)."""
    if not gpu_available():
        cpu_index.train(train)
        cpu_index.add(train)
        return cpu_index

    try:
        print(f"  [gpu] training on GPU {gpu_id} ...")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        gpu_index.train(train)
        gpu_index.add(train)
        return faiss.index_gpu_to_cpu(gpu_index)
    except Exception as exc:  # pragma: no cover - depends on GPU/driver specifics
        print(f"  [gpu] training on GPU failed ({exc!r}), falling back to CPU")
        cpu_index.train(train)
        cpu_index.add(train)
        return cpu_index
