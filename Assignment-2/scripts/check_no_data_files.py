#!/usr/bin/env python3
"""Enforce the Q8 path policy: no data and no large binaries in git.

Why this is a script and not only a pre-commit hook: **git hooks do not clone.**
A grader who clones this repo gets zero hooks, so a guarantee that lives only in
`.pre-commit-config.yaml` is a guarantee about my machine, not about the
submission. The hook calls into this file, `make check-no-data-files` calls it,
`tests/test_git_hygiene.py` calls it, and CI calls it. One implementation, four
callers, and the one that matters runs in the grader's copy.

Modes:

    --staged     reject forbidden or oversized paths in the index (pre-commit
                 stage; catches `git add -f`, which .gitignore alone does not)
    --history    no forbidden path anywhere in `git log --all --numstat`
    --messages   no trivial commit subjects (Q8.1)

With no mode flags, runs --history and --messages.

Stdlib only, and it bootstraps `src/` onto sys.path itself, so it runs under a
bare `python3` with nothing installed -- which is what a pre-commit hook and a
fresh CI checkout actually have.

## History

This replaces `scripts/check_git_hygiene.py`, which additionally enforced a
`Provenance:` / `Session-Id:` / `Co-Authored-By:` trailer on every commit. That
trailer system was removed deliberately on 2026-08-21; AI authorship is now
stated directly in `ai_usage/AI_USAGE.md` and evidenced by the verbatim session
records under `ai_usage/sessions/` and `ai_usage/transcripts/`. The path policy
below is unrelated to that decision and is unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from newsrec import config

#: Subjects that carry no information. A commit history is evidence of how the
#: work proceeded (Q8.1); "wip" is not evidence.
TRIVIAL_SUBJECTS = {"wip", "fix", "fixes", "update", "updates", "changes", "stuff", "temp", "."}
MIN_SUBJECT_LEN = 15


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


# =============================================================================
# Path policy
# =============================================================================


def _matches(path: str, patterns: tuple[str, ...]) -> str | None:
    """Return the pattern that matches ``path``, or None.

    A leading slash anchors the pattern to the repo root; otherwise it is also
    tried at any depth. Both forms are needed and neither is sufficient alone:

      * `__pycache__/*` must match `src/newsrec/__pycache__/x.pyc`, so
        unanchored patterns are tried against `*/{pat}` too;
      * `/data/*` must NOT match `src/newsrec/data/mind.py`. Matching that
        pattern at any depth would forbid committing the entire Q1 pipeline
        package -- which is exactly what the first version of this function did.
    """
    for pat in patterns:
        if pat.startswith("/"):
            if fnmatch(path, pat.lstrip("/")):
                return pat
        elif fnmatch(path, pat) or fnmatch(path, f"*/{pat}"):
            return pat
    return None


def is_forbidden(path: str) -> str | None:
    """Return the violated pattern, or None if the path is allowed.

    Exemptions win over patterns: they are the deliverables that Q7 requires and
    that happen to match a Q8 pattern (gzipped predictions, the raw AI-usage
    capture, the tiny test fixtures). Each is enumerated in config, never
    inferred, so widening the exemption set is a visible edit.
    """
    if _matches(path, config.FORBIDDEN_PATH_EXEMPTIONS):
        return None
    return _matches(path, config.FORBIDDEN_PATH_PATTERNS)


# =============================================================================
# Modes
# =============================================================================


def check_staged() -> list[str]:
    """Pre-commit: what is about to enter the history."""
    problems: list[str] = []
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    for path in filter(None, out.splitlines()):
        if pattern := is_forbidden(path):
            problems.append(
                f"{path}: forbidden by Q8 (matches {pattern!r}). "
                f"Data and build artefacts never enter git. "
                f"If this really is a deliverable, add it to "
                f"config.FORBIDDEN_PATH_EXEMPTIONS with a reason."
            )

        blob = REPO_ROOT / path
        if not blob.is_file():
            continue
        # Exempted paths are required deliverables whose size is not a choice --
        # ai_usage/raw/<session>.jsonl.gz is the verbatim chat export Q7.4 asks
        # for, and a long session exceeds 1 MiB even gzipped. They get a higher
        # but still finite ceiling, so nothing unbounded ever enters git.
        exempt = _matches(path, config.FORBIDDEN_PATH_EXEMPTIONS) is not None
        limit = config.MAX_EXEMPT_FILE_BYTES if exempt else config.MAX_COMMITTED_FILE_BYTES
        if (size := blob.stat().st_size) > limit:
            problems.append(
                f"{path}: {size / 1024:.0f} KB exceeds the {limit / 1024:.0f} KB limit "
                f"({'exempt-deliverable' if exempt else 'Q8'})."
            )
    return problems


def check_history() -> list[str]:
    """Q8.2 -- nothing forbidden has EVER been committed, on any ref.

    Checking the working tree is not enough: a file deleted in a later commit is
    still in the history, still in every clone, and still counts.
    """
    problems: list[str] = []
    out = git("log", "--all", "--numstat", "--format=%H", "--no-renames")
    seen: set[str] = set()
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue  # a commit header line, not a numstat row
        path = parts[2]
        if path in seen:
            continue
        seen.add(path)
        if pattern := is_forbidden(path):
            problems.append(
                f"{path}: present in git history, forbidden by Q8 (matches {pattern!r})"
            )
    return problems


def _commits() -> list[tuple[str, str]]:
    """(sha, subject) for every commit reachable from any ref."""
    sep, rec = "\x1f", "\x1e"
    out = git("log", "--all", f"--format=%H{sep}%s{rec}")
    commits = []
    for chunk in out.split(rec):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        sha, subject = chunk.split(sep, 1)
        commits.append((sha, subject))
    return commits


def check_messages() -> list[str]:
    """Q8.1 -- 'commit frequently with meaningful messages'."""
    problems = []
    for sha, subject in _commits():
        stripped = subject.strip().rstrip(".").lower()
        if stripped in TRIVIAL_SUBJECTS:
            problems.append(f"{sha[:8]}: trivial subject {subject!r}")
        elif len(subject.strip()) < MIN_SUBJECT_LEN:
            problems.append(f"{sha[:8]}: subject {subject!r} is under {MIN_SUBJECT_LEN} characters")
    return problems


# =============================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--staged", action="store_true")
    p.add_argument("--history", action="store_true")
    p.add_argument("--messages", action="store_true")
    args = p.parse_args(argv)

    selected = any([args.staged, args.history, args.messages])

    checks: list[tuple[str, list[str]]] = []
    if args.staged:
        checks.append(("staged paths (Q8.2/Q8.3)", check_staged()))
    if args.history or not selected:
        checks.append(("history (Q8.2)", check_history()))
    if args.messages or not selected:
        checks.append(("commit messages (Q8.1)", check_messages()))

    failed = False
    for name, problems in checks:
        if problems:
            failed = True
            print(f"FAIL  {name}", file=sys.stderr)
            for problem in problems:
                print(f"      {problem}", file=sys.stderr)
        else:
            print(f"ok    {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
