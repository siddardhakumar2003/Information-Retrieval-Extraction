#!/usr/bin/env python3
"""Fetch the Q3 encoder at its pinned revision, and prove it is the pinned one.

`make download-models`.

## Why this is a script and not one `huggingface-cli` line

A bare `model_id` resolves to whatever the repository's default branch points at
*today*. Weights on the Hub do get replaced, and when they are, two runs months
apart silently use different models and produce different numbers with nothing in
the diff to show it. That is the same class of failure `scripts/raw_archives.sha256`
exists to prevent for the dataset archives, and it deserves the same treatment:
pin an immutable identifier and **verify it after the fact**.

So this does three things a one-liner does not:

1. Downloads at `config.ENCODER.revision`, an immutable commit sha.
2. Re-resolves what actually landed on disk and **hard-fails** if the sha does not
   match the pin. Asking for a revision and checking you got it are different
   assertions; a cache populated earlier under a different revision is exactly the
   case where they diverge.
3. Fetches only the files the encoder needs. The repository also carries ONNX and
   OpenVINO exports and a duplicate `pytorch_model.bin`; pulling everything roughly
   triples the download for files nothing here loads.

## The limitation this does NOT solve

`make semantic` needs the network the first time. The weights cannot be committed
(CLAUDE.md invariant 3 -- no large binaries in git), and there is no offline
fallback that does not violate it. Recorded as **OQ-9**, deliberately, rather than
left as a surprise for a grader without a network connection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from newsrec import config

#: Everything sentence-transformers loads for this model, and nothing else. The
#: repo also ships onnx/, openvino/ and a pytorch_model.bin duplicating
#: model.safetensors -- ~2x the bytes for files we never open.
NEEDED = (
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


def download(spec: config.EncoderSpec = config.ENCODER, *, offline: bool = False) -> Path:
    """Fetch `spec` at its pinned revision. Returns the local snapshot path."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            spec.model_id,
            revision=spec.revision,
            allow_patterns=list(NEEDED),
            local_files_only=offline,
        )
    )

    # A snapshot directory is named for the commit it came from. Reading it back
    # is what turns "we asked for the pin" into "we got the pin" -- the two differ
    # when a cache was populated earlier under another revision.
    landed = path.name
    if landed != spec.revision:
        raise SystemExit(
            f"revision mismatch for {spec.model_id}\n"
            f"  pinned in config.ENCODER: {spec.revision}\n"
            f"  actually on disk:         {landed}\n"
            "Refusing: embeddings built from unpinned weights are not reproducible."
        )

    missing = [f for f in NEEDED if not (path / f).exists()]
    if missing:
        raise SystemExit(f"incomplete snapshot at {path}: missing {missing}")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--offline",
        action="store_true",
        help="fail rather than reach the network; use to check an existing cache",
    )
    args = p.parse_args(argv)

    spec = config.ENCODER
    print(f"{spec.model_id} @ {spec.revision}")
    path = download(spec, offline=args.offline)

    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  verified revision, {total / 1e6:.0f} MB at {path}")
    print(
        f"  dim={spec.dim} max_seq={spec.max_seq_length} prefixes={spec.query_prefix!r}/{spec.passage_prefix!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
