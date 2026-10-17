"""One call that fixes every random source a trainer uses.

Called at the top of each `train()`, so every model starts from the same
weights, batches and dropout masks. Seeding alone is not enough on a GPU:
some kernels add numbers in a different order each run, so we also ask torch
for the deterministic versions. Speed cost is small at this data size.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from newsrec import config

SEED: int = config.GLOBAL_SEED


def seed_everything() -> None:
    # Must be in the environment before the first cuBLAS call of the process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
