"""Save and reload a trained re-ranker (A2 Q3.8).

Design: `agent_docs/a2_q3_plan.md`.

## Why this exists

Nothing in the repository persisted a trained model before A2 Q3. A2 Q2's
`rerank/ablation.py` trains in memory, writes its numbers and discards the
weights — which is exactly right for producing a report, and useless for Q5,
where the submission path has to score 2.37 million MIND rows and 13.5 million
EB-NeRD rows with *the model the report selected*. Retraining at submission time
would mean the submitted system is not provably the evaluated one.

## The failure this guards against

Not "saving fails". Saving either works or raises. The dangerous case is
**loading the wrong model quietly**: a directory left behind by an earlier
feature set or an earlier architecture, read without complaint, scoring 13.5
million rows with weights nobody chose. The numbers look plausible, nothing
raises, and the mistake is invisible.

So `meta.json` records the architecture, the version and the feature contract,
and `load` **raises** on a mismatch. `config.RERANK_VERSION` /
`config.NRMS_VERSION` are the manual invalidation levers — bump one when the
training logic changes in a way that makes old weights meaningless, the same
discipline `config.STORE_VERSION` and `config.BEHAVIOURAL_FEATURE_VERSION`
already use for the feature store.

Models live under `config.RERANK_MODEL_DIR` (inside `artifacts/`), which is
gitignored and matched by `config.FORBIDDEN_PATH_PATTERNS`, so a `*.pt` cannot
reach git even by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from newsrec import config
from newsrec.rerank import features, mlp, nrms

META_NAME = "meta.json"
WEIGHTS_NAME = "model.pt"


class StaleModelError(RuntimeError):
    """A saved model does not match the code that is trying to load it.

    Raised rather than warned: the alternative is scoring a submission with the
    wrong weights, which produces a complete, well-formed, wrong answer.
    """


def model_dir(dataset: str, variant: str) -> Path:
    """Where one dataset's one variant lives."""
    return config.RERANK_MODEL_DIR / dataset / variant


def save(model: mlp.MLPModel | nrms.NRMSModel, dest: Path) -> Path:
    """Write the weights and the metadata needed to rebuild the architecture.

    Only those two files. A saved copy of the training table would also be a
    100 MB surprise in a directory nobody expects one in.
    """
    dest.mkdir(parents=True, exist_ok=True)

    if isinstance(model, nrms.NRMSModel):
        meta = {
            "kind": "nrms",
            "version": config.NRMS_VERSION,
            "feature_columns": list(features.FEATURE_COLUMNS),
            "category_aware": bool(model.category_aware),
            "n_categories": int(model.n_categories),
            "embedding_dim": int(model.embedding_dim),
            # From the MODEL, not from config: this must describe the network
            # actually being saved. Reading config here records whatever the
            # config currently says, which is a different architecture as soon
            # as it changes or a caller builds other widths.
            "dim": int(model.dim),
            "heads": int(model.heads),
            "attention_dim": int(model.attention_dim),
            "category_dim": int(model.category_dim),
            "epochs_trained": int(model.epochs_trained),
            "padded_negatives": int(model.padded_negatives),
            "best_val_ndcg@10": float(model.best_val_ndcg10),
        }
        torch.save(model.net.state_dict(), dest / WEIGHTS_NAME)
    else:
        meta = {
            "kind": "mlp",
            "version": config.RERANK_VERSION,
            "feature_columns": list(features.FEATURE_COLUMNS),
            "hidden_dim": int(model.net.layers[0].out_features),
            # The normalisation statistics are part of the trained model, not a
            # property of whatever table is being scored next — recomputing them
            # from a test table would leak its distribution into scoring.
            "feature_mean": [float(v) for v in model.feature_mean],
            "feature_std": [float(v) for v in model.feature_std],
        }
        torch.save(model.net.state_dict(), dest / WEIGHTS_NAME)

    (dest / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", "utf-8")
    return dest


def _check(meta: dict, dest: Path) -> None:
    expected_version = config.NRMS_VERSION if meta["kind"] == "nrms" else config.RERANK_VERSION
    if meta.get("version") != expected_version:
        raise StaleModelError(
            f"{dest} holds a {meta['kind']} model at version {meta.get('version')}, "
            f"but the code is at version {expected_version}. Retrain rather than "
            f"scoring with weights fitted under different logic."
        )
    saved_columns = meta.get("feature_columns")
    if saved_columns != list(features.FEATURE_COLUMNS):
        raise StaleModelError(
            f"{dest} was trained on a different feature list.\n"
            f"  saved: {saved_columns}\n"
            f"  now:   {list(features.FEATURE_COLUMNS)}\n"
            "The weights expect the old matrix; `feature_matrix` would hand over "
            "one of a different shape or, worse, the same shape with the columns "
            "meaning different things."
        )


def load(dest: Path, device: str = "cpu") -> mlp.MLPModel | nrms.NRMSModel:
    """Rebuild a saved model, or raise.

    `device` defaults to CPU because *inference* has no reason to require a GPU
    — `eval/serving.py` benchmarks single-request latency on CPU deliberately.
    That does not contradict `mlp._require_cuda`, which governs **training**.
    """
    meta_path = dest / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"no saved model at {dest} (expected {META_NAME})")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _check(meta, dest)
    state = torch.load(dest / WEIGHTS_NAME, map_location=device, weights_only=True)

    if meta["kind"] == "nrms":
        net = nrms.NRMSNet(
            embedding_dim=meta["embedding_dim"],
            dim=meta["dim"],
            heads=meta["heads"],
            attention_dim=meta["attention_dim"],
            n_categories=meta["n_categories"] if meta["category_aware"] else 0,
            category_dim=meta["category_dim"],
            dropout=0.0,  # inference
        )
        net.load_state_dict(state)
        net.to(device).eval()
        return nrms.NRMSModel(
            net=net,
            device=device,
            category_aware=meta["category_aware"],
            n_categories=meta["n_categories"],
            embedding_dim=meta["embedding_dim"],
            dim=meta["dim"],
            heads=meta["heads"],
            attention_dim=meta["attention_dim"],
            category_dim=meta["category_dim"],
            padded_negatives=meta.get("padded_negatives", 0),
            epochs_trained=meta.get("epochs_trained", 0),
            best_val_ndcg10=meta.get("best_val_ndcg@10", float("-inf")),
        )

    net = mlp._Net(n_features=len(features.FEATURE_COLUMNS), hidden_dim=meta["hidden_dim"])
    net.load_state_dict(state)
    net.to(device).eval()
    return mlp.MLPModel(
        net=net,
        # float32, because that is what training produces: `mlp.train` takes the
        # mean/std of `features.feature_matrix`, which is float32. Restoring
        # them as float64 promotes the normalised input and `_Net`'s float32
        # weights then reject it ("mat1 and mat2 must have the same dtype").
        # Caught by tests/test_rerank_store.py's round trip.
        feature_mean=np.asarray(meta["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(meta["feature_std"], dtype=np.float32),
        device=device,
    )
