"""Q1.13 — building twice from identical input gives identical bytes."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from newsrec import config
from newsrec.data import feature_store


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _raw_for(dataset: str, fixture_raw_root: Path) -> Path:
    return fixture_raw_root / ("MIND" if dataset == "mind_small" else dataset)


def test_rebuild_is_byte_identical(tmp_path, dataset, fixture_raw_root):
    """Non-determinism here makes every downstream comparison meaningless: an
    ablation delta cannot be distinguished from build noise.

    Covers the whole store, the manifest included. That is the strict version --
    a wall-clock `built_at` field would force this test to exclude the very file
    describing the build, so none is written.
    """
    raw = _raw_for(dataset, fixture_raw_root)
    first = tmp_path / "a"
    second = tmp_path / "b"
    feature_store.build_dataset(dataset, raw_root=raw, root=first)
    feature_store.build_dataset(dataset, raw_root=raw, root=second)

    a, b = _digests(first), _digests(second)
    assert sorted(a) == sorted(b), "the two builds produced different files"
    differing = [name for name in a if a[name] != b[name]]
    assert not differing, f"non-deterministic output: {differing}"


def test_rebuild_over_an_existing_store_is_stable(tmp_path, dataset, fixture_raw_root):
    """Rebuilding in place must not drift.

    `make data` re-runs into a directory that already holds a store, so
    idempotency in place is the case that actually occurs -- and it is a
    different code path from building into an empty directory.
    """
    raw = _raw_for(dataset, fixture_raw_root)
    root = tmp_path / "store"
    feature_store.build_dataset(dataset, raw_root=raw, root=root)
    before = _digests(root)
    feature_store.build_dataset(dataset, raw_root=raw, root=root)
    assert _digests(root) == before


def test_config_fingerprint_changes_with_boundaries(dataset, monkeypatch):
    """A stale artefact built under different split boundaries must be
    detectable, or editing a boundary and forgetting to rebuild yields numbers
    that silently belong to the old configuration."""
    original = feature_store.config_fingerprint(dataset)

    spec = config.DATASETS[dataset].split
    shifted = config.SplitSpec(
        val_start=spec.val_start + timedelta(hours=1),
        test_start=spec.test_start,
        rationale="shifted for the test",
    )
    patched = config.DatasetSpec(
        key=config.DATASETS[dataset].key,
        language=config.DATASETS[dataset].language,
        raw_subdir=config.DATASETS[dataset].raw_subdir,
        split=shifted,
        headline=config.DATASETS[dataset].headline,
        unavailable_columns=config.DATASETS[dataset].unavailable_columns,
    )
    monkeypatch.setitem(config.DATASETS, dataset, patched)

    assert feature_store.config_fingerprint(dataset) != original


def test_reading_a_store_built_under_another_config_is_refused(
    tmp_path, dataset, fixture_raw_root, monkeypatch
):
    """The fingerprint has to be ENFORCED, not merely recorded.

    Without this, a store built under old boundaries reads back cleanly and
    every downstream number silently belongs to a configuration that is no
    longer in the repo.
    """
    raw = _raw_for(dataset, fixture_raw_root)
    root = tmp_path / "store"
    feature_store.build_dataset(dataset, raw_root=raw, root=root)

    spec = config.DATASETS[dataset].split
    original = config.DATASETS[dataset]
    patched = config.DatasetSpec(
        key=original.key,
        language=original.language,
        raw_subdir=original.raw_subdir,
        split=config.SplitSpec(
            val_start=spec.val_start + timedelta(hours=1),
            test_start=spec.test_start,
            rationale="shifted after the build",
        ),
        headline=original.headline,
        unavailable_columns=original.unavailable_columns,
    )
    monkeypatch.setitem(config.DATASETS, dataset, patched)

    with pytest.raises(feature_store.StoreError, match="different configuration"):
        feature_store.read_processed(dataset, root / "processed")
