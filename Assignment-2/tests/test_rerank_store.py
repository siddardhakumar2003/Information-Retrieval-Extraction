"""A2 Q3.8 — a trained model survives a save/load round trip, or refuses.

Nothing in the repository persisted a trained model before this: A2 Q2's
`rerank/ablation.py` trains in memory and discards, which is fine for writing a
report and useless for Q5, where the submission path has to score 13.5 million
rows with the model the report selected.

The risk this file is about is not "saving fails loudly". It is **scoring with
the wrong model silently** — a stale file left behind by an earlier feature set,
loaded without complaint, producing plausible numbers for a model nobody chose.
So `load` refuses a mismatch rather than trusting the filename.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch

from newsrec import config
from newsrec.rerank import features, mlp, nrms, store

EMBED_DIM = 6


def _mlp_model() -> mlp.MLPModel:
    torch.manual_seed(0)
    net = mlp._Net(n_features=len(features.FEATURE_COLUMNS), hidden_dim=8)
    return mlp.MLPModel(
        net=net,
        # float32 to match what `mlp.train` produces: it takes the mean/std of
        # `features.feature_matrix`, which is float32.
        feature_mean=np.arange(len(features.FEATURE_COLUMNS), dtype=np.float32),
        feature_std=np.ones(len(features.FEATURE_COLUMNS), dtype=np.float32),
        device="cpu",
    )


def _mlp_table() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    n = 12
    data = {
        "impression_id": ["i1"] * 6 + ["i2"] * 6,
        "article_id": [f"a{j}" for j in range(n)],
        "user_id": ["u1"] * n,
        "dataset": ["mind_small"] * n,
        "split": ["test"] * n,
        "label": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    }
    for name in features.FEATURE_COLUMNS:
        data[name] = rng.standard_normal(n)
    return pl.DataFrame(data)


def _nrms_model(category_aware: bool = False) -> nrms.NRMSModel:
    torch.manual_seed(0)
    net = nrms.NRMSNet(
        embedding_dim=EMBED_DIM,
        dim=8,
        heads=2,
        attention_dim=4,
        n_categories=3 if category_aware else 0,
        category_dim=2,
        dropout=0.0,
    )
    return nrms.NRMSModel(
        net=net,
        device="cpu",
        category_aware=category_aware,
        n_categories=3 if category_aware else 0,
        embedding_dim=EMBED_DIM,
        dim=8,
        heads=2,
        attention_dim=4,
        category_dim=2,
        best_val_ndcg10=0.4242,
    )


def _nrms_source() -> nrms.ListwiseBatchSource:
    article_ids = [f"a{j}" for j in range(12)]
    rng = np.random.default_rng(7)
    return nrms.ListwiseBatchSource(
        groups=[],
        history_rows={"i1": np.array([3, 1]), "i2": np.array([], dtype=np.int64)},
        row_of_article={a: j for j, a in enumerate(article_ids)},
        embeddings=rng.standard_normal((len(article_ids), EMBED_DIM)).astype(np.float32),
    )


# =============================================================================
# The round trip — the only property that actually matters
# =============================================================================


def test_a_loaded_mlp_scores_identically_to_the_one_that_was_saved(tmp_path):
    model = _mlp_model()
    table = _mlp_table()
    before = mlp.predict(model, table)

    store.save(model, tmp_path / "mlp_best")
    loaded = store.load(tmp_path / "mlp_best")

    np.testing.assert_allclose(mlp.predict(loaded, table), before, rtol=0, atol=0)


def test_a_loaded_nrms_scores_identically_to_the_one_that_was_saved(tmp_path):
    model = _nrms_model()
    source = _nrms_source()
    table = _mlp_table()
    before = nrms.predict(model, source, table)

    store.save(model, tmp_path / "nrms_reproduction")
    loaded = store.load(tmp_path / "nrms_reproduction")

    np.testing.assert_allclose(nrms.predict(loaded, source, table), before, rtol=0, atol=0)


def test_the_category_aware_variant_round_trips_as_category_aware(tmp_path):
    """Load must rebuild the architecture, not a default one. A model saved
    with the category embedding and reloaded without it would score with
    randomly-initialised weights in place of the trained ones."""
    store.save(_nrms_model(category_aware=True), tmp_path / "nrms_category_aware")
    loaded = store.load(tmp_path / "nrms_category_aware")
    assert loaded.category_aware
    assert any("category" in name for name, _ in loaded.net.named_parameters())


# =============================================================================
# The refusals — scoring with the wrong model is the failure worth preventing
# =============================================================================


def test_load_refuses_a_stale_version(tmp_path):
    dest = tmp_path / "mlp_best"
    store.save(_mlp_model(), dest)

    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    meta["version"] = meta["version"] + 1
    (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(store.StaleModelError, match="version"):
        store.load(dest)


def test_load_refuses_a_changed_feature_list(tmp_path):
    """`FEATURE_COLUMNS` is the model's input contract. If it changes, the saved
    weights expect a different matrix — and `feature_matrix` would hand over one
    of the right width with the columns meaning different things."""
    dest = tmp_path / "mlp_best"
    store.save(_mlp_model(), dest)

    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    meta["feature_columns"] = ["something", "else", "entirely"]
    (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(store.StaleModelError, match="feature"):
        store.load(dest)


def test_load_refuses_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "never_trained")


def test_saving_records_the_selection_metric_and_the_version(tmp_path):
    """A grader reading `artifacts/rerank/` should be able to tell which model
    was chosen and on what, without rerunning anything."""
    dest = tmp_path / "nrms_reproduction"
    store.save(_nrms_model(), dest)
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))

    assert meta["kind"] == "nrms"
    assert meta["version"] == config.NRMS_VERSION
    assert meta["feature_columns"] == list(features.FEATURE_COLUMNS)
    assert meta["best_val_ndcg@10"] == pytest.approx(0.4242)


def test_the_model_directory_holds_no_training_data(tmp_path):
    """Only weights and metadata. `artifacts/` is gitignored, but a saved copy
    of a 20,000-impression table would also be a 100 MB surprise in a directory
    nobody expects one in."""
    dest = tmp_path / "mlp_best"
    store.save(_mlp_model(), dest)
    assert {p.name for p in dest.iterdir()} == {"model.pt", "meta.json"}
