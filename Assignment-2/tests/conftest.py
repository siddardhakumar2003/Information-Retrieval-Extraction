"""Shared fixtures.

Two rules shape everything here:

1. **`make test` must pass in a clean clone with no `data/` and no `Data/`.**
   That is the state a grader clones into. So the default fixtures resolve to
   the tiny synthetic samples committed under `tests/fixtures/`, and nothing in
   the default run touches a real dataset.

2. **Every pipeline test runs against BOTH datasets.** Every question in the
   assignment requires both, so the `dataset` fixture is parametrised rather
   than leaving per-dataset coverage to whoever remembers to add it.

`--realdata` switches the same tests onto the real processed store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The pipeline does not exist yet. Tests assert real conditions but skip with
#: this reason, so the suite is a contract rather than a memory. Delete the
#: guard as each stage lands — the assertion below it is already written.
PIPELINE_NOT_BUILT = "pipeline stage not implemented yet (see agent_docs/q1_plan.md)"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--realdata",
        action="store_true",
        default=False,
        help="run against the real processed store under data/ instead of tests/fixtures/",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "canary: proves another test is capable of failing")


@pytest.fixture(scope="session")
def use_realdata(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--realdata"))


@pytest.fixture(params=["mind_small", "ebnerd_demo"])
def dataset(request: pytest.FixtureRequest) -> str:
    """Every pipeline test runs against both datasets.

    ebnerd_demo rather than ebnerd_small: same schema and same code path, small
    enough to keep the default run under 30 s. Headline numbers come from
    ebnerd_small (OQ-3), but schema and leakage properties are dataset-shape
    properties, not size properties.
    """
    return str(request.param)


@pytest.fixture
def fixture_dir(dataset: str) -> Path:
    d = FIXTURES / dataset
    if not d.is_dir():
        pytest.skip(f"no committed fixture for {dataset}")
    return d


@pytest.fixture(scope="session")
def fixture_raw_root(tmp_path_factory) -> Path:
    """The committed fixtures, laid out the way each dataset actually ships.

    MIND is copied as raw TSV; EB-NeRD is materialised from its committed JSON
    into Parquet. Session-scoped because building a store per test would
    dominate the runtime of a suite that must finish in under 30 s.
    """
    import shutil
    import subprocess
    import sys

    root = tmp_path_factory.mktemp("raw")

    shutil.copytree(FIXTURES / "mind_small", root / "MIND")

    ebnerd_root = root / "ebnerd_demo"
    result = subprocess.run(
        [sys.executable, str(FIXTURES / "build_fixtures.py"), "ebnerd_demo", str(ebnerd_root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return root


@pytest.fixture(scope="session")
def fixture_store(tmp_path_factory, fixture_raw_root: Path) -> Path:
    """Build both datasets' stores from the committed fixtures, once."""
    from newsrec.data import feature_store

    out = tmp_path_factory.mktemp("store")
    for key, raw in (
        ("mind_small", fixture_raw_root / "MIND"),
        ("ebnerd_demo", fixture_raw_root / "ebnerd_demo"),
    ):
        feature_store.build_dataset(key, raw_root=raw, root=out)
    return out


@pytest.fixture
def store_root(fixture_store: Path, use_realdata: bool) -> Path:
    """Root containing processed/ and feature_store/."""
    if use_realdata:
        from newsrec import config

        if not config.PROCESSED_DIR.is_dir():
            pytest.skip("--realdata given but data/processed/ does not exist; run `make data`")
        return config.DATA_DIR
    return fixture_store


@pytest.fixture
def processed_store(dataset: str, store_root: Path) -> Path:
    """Where the canonical tables for `dataset` live."""
    store = store_root / "processed" / dataset
    if not store.is_dir():
        pytest.skip(f"no processed store for {dataset} at {store}")
    return store


@pytest.fixture
def feature_store_dir(dataset: str, store_root: Path) -> Path:
    store = store_root / "feature_store" / dataset
    if not store.is_dir():
        pytest.skip(f"no feature store for {dataset} at {store}")
    return store


@pytest.fixture(scope="session")
def leakage_canary() -> dict:
    """The deliberately corrupt fixture from Q9.6.

    Loadable without the pipeline, because the canary test must be runnable as
    soon as a leakage checker exists — it is what proves the checker works.
    """
    return json.loads((FIXTURES / "leakage_canary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
