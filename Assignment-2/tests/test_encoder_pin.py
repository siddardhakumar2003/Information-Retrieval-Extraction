"""Q3.2 — the encoder is pinned, and the pin is verified rather than trusted.

The failure guarded against is slow and invisible: weights on the Hugging Face
Hub get replaced, a bare model id resolves to whatever the default branch points
at today, and two runs months apart produce different embeddings and different
recall with nothing in the diff to show why.

None of these tests reach the network. They assert the *shape* of the pin and
that the verification code refuses what it should; the real download is exercised
by `make download-models`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from newsrec import config

# scripts/ is a directory of entrypoints, not an installed package. Same
# approach as tests/test_ai_usage.py and tests/test_git_hygiene.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_models

# =============================================================================
# The pin itself
# =============================================================================


def test_revision_is_an_immutable_commit_sha_not_a_branch():
    """`revision="main"` downloads successfully and pins nothing. It is the
    mistake this whole mechanism exists to catch, and it is one word away from
    the correct value."""
    assert re.fullmatch(r"[0-9a-f]{40}", config.ENCODER.revision), (
        f"ENCODER.revision {config.ENCODER.revision!r} is not a 40-char commit sha"
    )


def test_encoder_declares_what_the_artefact_manifest_has_to_record():
    """Q3.1 requires the dimension in the manifest; Q3.2 the encoder and its
    budget. Neither can be recorded if it is not declared."""
    e = config.ENCODER
    assert e.model_id and e.dim > 0
    assert e.max_seq_length == 128, "q3_plan D2 fixes seq 128; changing it changes every number"


def test_e5_prefixes_are_declared_and_distinct():
    """The e5 model card: "Each input text should start with 'query: ' or
    'passage: ', even for non-English texts."

    Omitting them, or using one for both sides, degrades the embedding space
    quietly — the vectors are still unit-norm, the recall is still a plausible
    small number, and nothing errors.
    """
    e = config.ENCODER
    assert e.query_prefix != e.passage_prefix
    assert e.query_prefix.endswith(" ") and e.passage_prefix.endswith(" "), (
        "the prefix includes its trailing space; without it the first real token is mangled"
    )
    if "e5" in e.model_id:
        assert e.query_prefix == "query: " and e.passage_prefix == "passage: "


# =============================================================================
# The verification refuses what it should
# =============================================================================


def _spec(revision: str) -> config.EncoderSpec:
    e = config.ENCODER
    return config.EncoderSpec(
        model_id=e.model_id,
        revision=revision,
        dim=e.dim,
        max_seq_length=e.max_seq_length,
        passage_prefix=e.passage_prefix,
        query_prefix=e.query_prefix,
    )


def test_a_snapshot_from_a_different_revision_is_refused(tmp_path, monkeypatch):
    """The canary. Simulates the real divergence: the download SUCCEEDS, but
    what landed came from another commit — which is what happens when the pin is
    a branch name, or when a cache was populated earlier under a different
    revision.

    Asserting the message names both shas, because "revision mismatch" alone
    tells whoever hits this nothing about which one to fix.
    """
    wrong = "b" * 40
    snapshot = tmp_path / wrong
    snapshot.mkdir()
    for name in download_models.NEEDED:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot), raising=False
    )
    with pytest.raises(SystemExit) as excinfo:
        download_models.download(_spec("a" * 40))

    message = str(excinfo.value)
    assert "a" * 40 in message and wrong in message


def test_an_incomplete_snapshot_is_refused(tmp_path, monkeypatch):
    """A half-populated cache must fail loudly, not load a model missing its
    tokeniser and produce garbage vectors."""
    revision = "c" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")  # and nothing else

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot), raising=False
    )
    with pytest.raises(SystemExit, match="incomplete snapshot"):
        download_models.download(_spec(revision))


def test_a_matching_snapshot_is_accepted(tmp_path, monkeypatch):
    """The other direction. A guard that rejects everything passes both tests
    above and is useless."""
    revision = "d" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    for name in download_models.NEEDED:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot), raising=False
    )
    assert download_models.download(_spec(revision)) == snapshot


def test_the_download_fetches_only_what_the_encoder_loads():
    """The repo also ships onnx/, openvino/ and a pytorch_model.bin duplicating
    model.safetensors — roughly 2x the bytes for files nothing here opens."""
    assert "model.safetensors" in download_models.NEEDED
    assert not any(
        f.startswith(("onnx/", "openvino/")) or f == "pytorch_model.bin"
        for f in download_models.NEEDED
    )
    # sentence-transformers reads the pooling config to know it is mean-pooling.
    assert "1_Pooling/config.json" in download_models.NEEDED


# =============================================================================
# Pooling configuration (Q3.4 requires it to be a config value)
# =============================================================================


def test_user_pooling_is_a_config_value_with_both_arms_registered():
    assert config.USER_POOLING in config.USER_POOLING_VARIANTS
    assert len(config.USER_POOLING_VARIANTS) >= 2, "an ablation with one arm is not an ablation"
    assert 0.0 < config.USER_POOLING_DECAY < 1.0


def test_ann_search_parameter_is_above_the_measured_failing_default():
    """efSearch=64 — faiss's default — measured 0.9188 against exact search,
    BELOW Q3.3's 0.95 floor. Shipping the default would have failed the
    requirement; only the sweep in q3_plan's "Measured scale" revealed it."""
    assert config.ANN_HNSW_EF_SEARCH > 64
    assert config.ANN_RECALL_FLOOR == 0.95
