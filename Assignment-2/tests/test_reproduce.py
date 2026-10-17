"""Q1.12, Q7.2 — the one-command rebuild actually rebuilds.

Two claims that sound similar and are not.

**Q1.12** is that `make clean-derived && make data` regenerates every processed
artefact *from `Data/` alone*. `test_determinism.py` already proves two builds
agree and that rebuilding in place is stable; neither would notice a pipeline
that quietly needed a previous build's output to produce the next one. Deleting
the store first is what tests that.

**Q7.2** is that the documented command works from a clean clone. What is
checked here is that the clone genuinely starts clean, that the README's command
is real, and that Make can resolve the whole graph there. What is **not**
checked is executing it: the real pipeline needs `Data/` (tens of GB, not in
git) and hours of encoding. That limit is stated rather than papered over — a
test whose name promises a clean-clone reproduce and quietly asserts less is
worse than one that says what it did.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from newsrec.data import feature_store

REPO = Path(__file__).resolve().parents[1]
README_COMMAND = "make clean-derived && make all"


def _digests(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _raw_for(dataset: str, fixture_raw_root: Path) -> Path:
    return fixture_raw_root / ("MIND" if dataset == "mind_small" else dataset)


# =============================================================================
# Q1.12 — clean, then rebuild from raw alone
# =============================================================================


def test_one_command_rebuild_on_fixtures(tmp_path, dataset, fixture_raw_root):
    """The named Q1.12 condition, as a delete-and-rebuild cycle.

    Distinct from the determinism tests: this removes the derived tree entirely
    between the two builds, so a pipeline that had come to depend on its own
    previous output — a cached vocabulary, an id map, a manifest read back in —
    would fail here and pass there.
    """
    raw = _raw_for(dataset, fixture_raw_root)
    root = tmp_path / "store"

    feature_store.build_dataset(dataset, raw_root=raw, root=root)
    before = _digests(root)
    assert before, "the first build produced nothing"

    # What `make clean-derived` does to the derived tree.
    shutil.rmtree(root)
    assert not root.exists()

    feature_store.build_dataset(dataset, raw_root=raw, root=root)
    after = _digests(root)

    assert sorted(after) == sorted(before), (
        "the rebuild produced a different set of files: "
        f"missing {sorted(set(before) - set(after))}, extra {sorted(set(after) - set(before))}"
    )
    differing = [name for name in before if before[name] != after[name]]
    assert not differing, f"rebuilt artefacts differ from the originals: {differing}"


def test_the_rebuild_leaves_the_raw_zone_untouched(tmp_path, dataset, fixture_raw_root):
    """`clean-derived` keeps `Data/`, and the build must not write into it.

    A pipeline that wrote a cache beside the raw files would make the raw zone
    mutable, and "regenerates from Data/ alone" would stop being checkable.
    """
    raw = _raw_for(dataset, fixture_raw_root)
    before = _digests(raw)
    feature_store.build_dataset(dataset, raw_root=raw, root=tmp_path / "store")
    assert _digests(raw) == before, "the build modified the raw zone"


def test_clean_derived_removes_the_derived_tree_and_keeps_raw():
    """Checked against the recipe itself, since running it would delete the
    working store this suite is otherwise using."""
    recipe = (REPO / "Makefile").read_text(encoding="utf-8")
    body = recipe.split("clean-derived:", 1)[1].split("\n.PHONY", 1)[0]
    assert "rm -rf" in body
    assert "data" in body
    assert "$(RAW)" not in body.split("rm -rf", 1)[1].split("\n", 1)[0], (
        "clean-derived would delete the raw zone"
    )


# =============================================================================
# Q7.2 — the documented command, from a clean clone
# =============================================================================


def test_the_readme_documents_the_reproduce_command():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert README_COMMAND in text, f"README does not document {README_COMMAND!r}"


@pytest.fixture(scope="module")
def clean_clone(tmp_path_factory) -> Path:
    """A fresh `git clone` of this repository, as a grader would receive it."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    dest = tmp_path_factory.mktemp("clone") / "repo"
    result = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"git clone failed: {result.stderr.strip()[:200]}")
    return dest


def test_clean_clone_readme_command(clean_clone):
    """The named Q7.2 condition, to the extent it can be checked without data.

    Proven here: the clone contains no `data/` and no `Data/`, the README's
    command is present, and Make resolves the entire target graph in that clone.

    NOT proven here: that running it end to end succeeds. That needs the raw
    archives, which are deliberately not in git, and hours of CPU. `make -n`
    resolving the graph is the strongest claim available offline, and the gap is
    named rather than hidden.
    """
    assert not (clean_clone / "data").exists(), "the clone carries a derived store"
    assert not (clean_clone / "Data").exists(), "the clone carries raw data"
    assert not (clean_clone / "artifacts").exists(), "the clone carries built artefacts"

    text = (clean_clone / "README.md").read_text(encoding="utf-8")
    assert README_COMMAND in text

    for target in ("clean-derived", "all"):
        result = subprocess.run(
            ["make", "-n", target],
            cwd=clean_clone,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"`make -n {target}` does not resolve in a clean clone:\n{result.stderr.strip()[:400]}"
        )


def test_the_clone_carries_the_numbers_a_grader_needs_without_rebuilding(clean_clone):
    """Every metric report is committed, so the design note's claims can be
    checked without running anything."""
    metrics = clean_clone / "reports" / "metrics"
    assert metrics.is_dir(), "no reports/metrics in the clone"
    names = {p.name for p in metrics.glob("*.json")}
    required = {"eval.json", "bm25_recall.json", "semantic_recall.json", "ablations.json"}
    assert required <= names, f"missing from the clone: {sorted(required - names)}"


def test_the_clone_can_check_the_design_note_offline(clean_clone):
    """Q6.6's checker must work for a grader, not just for us.

    Skipped in the personal/portfolio copy of this repository, where
    `design_note/` is deliberately untracked (`.gitignore`, commit 26e4c15) so
    the graded-only deliverables are not pushed there. A clone of THAT remote
    therefore has the checker but not the note, and there is nothing for the
    checker to read.

    The skip is conditioned on the note being absent from the clone rather than
    on a flag, so in the graded repository — where the note IS committed — this
    test runs and must pass. A blanket skip would have quietly retired the one
    check that proves a grader can verify the note's numbers without a rebuild.
    """
    script = clean_clone / "scripts" / "check_design_note.py"
    assert script.exists()
    if not (clean_clone / "design_note" / "design_note.md").exists():
        pytest.skip(
            "design_note/ is untracked in this personal copy (see .gitignore); "
            "the note is committed in the graded repository, where this runs"
        )
    result = subprocess.run(
        [str(Path(REPO / ".venv" / "bin" / "python")), str(script), "--check-numbers"],
        cwd=clean_clone,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"the number check fails in a clean clone:\n{result.stdout}\n{result.stderr}"
    )
