"""Every random source the trainers touch is fixed, so a retrain is repeatable.

Before this, only NRMS seeded torch, the MLP seeded nothing, and no trainer
asked the GPU for deterministic kernels — so the same code gave different
numbers on different runs (one verdict flipped: ebnerd_small MLP).
"""

from __future__ import annotations

import os
import random

import numpy as np
import polars as pl
import pytest
import torch

from newsrec import config
from newsrec.rerank import features, gbdt, mlp, seeding


def test_seeding_makes_python_numpy_and_torch_repeat():
    seeding.seed_everything()
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seeding.seed_everything()
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_seeding_asks_torch_for_deterministic_kernels():
    seeding.seed_everything()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_the_seed_is_the_one_global_seed():
    assert seeding.SEED == config.GLOBAL_SEED


def test_lightgbm_is_seeded_and_deterministic_from_config():
    params = config.GBDT_FIXED_PARAMS
    assert params["seed"] == config.GLOBAL_SEED
    assert params["deterministic"] is True


def test_lightgbm_trains_on_cpu_because_its_gpu_mode_is_not_repeatable():
    """Measured: two identical runs on LightGBM's OpenCL GPU backend gave
    different trees on both EB-NeRD datasets (it sums floats in a different
    order each run, and LightGBM warns of this). CPU with `deterministic=True`
    is exact."""
    assert config.GBDT_FIXED_PARAMS["device_type"] == "cpu"


def _table(n_impressions: int = 20, per: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    schema = (
        {name: pl.Utf8 for name in features.ID_COLUMNS}
        | {"label": pl.Int64}
        | {name: pl.Float64 for name in features.FEATURE_COLUMNS}
    )
    rows = []
    for i in range(n_impressions):
        for j in range(per):
            row = {name: "x" for name in features.ID_COLUMNS}
            row["impression_id"] = f"i{i}"
            row["article_id"] = f"a{i}_{j}"
            row["label"] = 1 if j == 0 else 0
            for name in features.FEATURE_COLUMNS:
                row[name] = float(rng.normal())
            rows.append(row)
    return pl.DataFrame(rows, schema=schema, orient="row")


def _skip_without_gpu():
    if not torch.cuda.is_available():
        pytest.skip("gpu marker: no CUDA device on this machine")


@pytest.mark.gpu
def test_mlp_trained_twice_gives_identical_scores(monkeypatch):
    _skip_without_gpu()
    monkeypatch.setattr(config, "MLP_MAX_EPOCHS", 20)
    table = _table()
    a = mlp.predict(mlp.train(table, table, params={"hidden_dim": 8}), table)
    b = mlp.predict(mlp.train(table, table, params={"hidden_dim": 8}), table)
    assert np.array_equal(a, b)


def test_gbdt_trained_twice_gives_identical_scores(monkeypatch):
    monkeypatch.setattr(config, "GBDT_MAX_ROUNDS", 20)
    table = _table()
    a = gbdt.predict(gbdt.train(table, table, params={"num_leaves": 7}), table)
    b = gbdt.predict(gbdt.train(table, table, params={"num_leaves": 7}), table)
    assert np.array_equal(a, b)
