"""Q8 — the git path policy as a test, not a promise.

Every one of these runs TODAY. That is the point: hooks do not clone, so the
guarantee has to be reproducible from a bare checkout, and a test is the form
that travels.

Provenance-trailer tests used to live here. The trailer system was removed on
2026-08-21; AI authorship is now stated in `ai_usage/AI_USAGE.md` and evidenced
by the verbatim records under `ai_usage/sessions/` and `ai_usage/transcripts/`.
The path policy below is unrelated to that and is unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_no_data_files as hygiene


def test_no_forbidden_paths_in_history():
    """Q8.2 — nothing forbidden has EVER been committed, on any ref."""
    assert hygiene.check_history() == []


def test_commit_messages_non_trivial():
    """Q8.1 — 'commit frequently with meaningful messages'."""
    assert hygiene.check_messages() == []


# ---------------------------------------------------------------------------
# The path policy itself. Both directions, because a matcher that rejects
# everything passes every "is it blocked?" test and breaks the repo.
# ---------------------------------------------------------------------------

FORBIDDEN = [
    "data/processed/mind_small/articles.parquet",
    "Data/MIND/MINDsmall_train/news.tsv",
    "Data/EB-NeRD/ebnerd_demo.zip",
    "artifacts/bm25/mind_small/index.bin",
    "src/newsrec/__pycache__/cli.cpython-312.pyc",
    "model.pt",
    "checkpoint.ckpt",
    "embeddings.npy",
]

ALLOWED = [
    # The regression that motivated this list: an unanchored `data/*` pattern
    # matched src/newsrec/data/, which would forbid committing the entire Q1
    # pipeline package. Anchoring is load-bearing, so it is pinned here.
    "src/newsrec/data/mind.py",
    "src/newsrec/data/__init__.py",
    "src/newsrec/data/feature_store.py",
    # Required deliverables that happen to match a forbidden pattern.
    "predictions/mind_dev.txt.gz",
    "predictions/SHA256SUMS",
    "ai_usage/raw/abc-123.jsonl.gz",
    "tests/fixtures/mind_small/news.tsv",
    # Ordinary committed files.
    "reports/metrics/eval.json",
    "design_note/figures/codabench_mind.png",
    "CLAUDE.md",
]


@pytest.mark.parametrize("path", FORBIDDEN)
def test_forbidden_paths_are_rejected(path):
    assert hygiene.is_forbidden(path) is not None, f"{path} should be blocked by Q8"


@pytest.mark.parametrize("path", ALLOWED)
def test_allowed_paths_are_not_rejected(path):
    assert hygiene.is_forbidden(path) is None, f"{path} must be committable"


def test_hygiene_script_runs_standalone():
    """It must work under a bare python3 with nothing installed — that is the
    environment a pre-commit hook and a fresh CI checkout actually provide."""
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check_no_data_files.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr


def test_ci_runs_the_hygiene_check(repo_root):
    """Q8.4 — the rule survives cloning.

    A pre-commit hook protects only the machine it is installed on. The same
    check must run in CI, from a fresh checkout, or the guarantee is local to
    whoever set their hooks up.
    """
    workflow = repo_root / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "no .github/workflows/ci.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "scripts/check_no_data_files.py" in text, (
        "CI does not run the data-hygiene check, so the rule is enforced only by a local hook"
    )
