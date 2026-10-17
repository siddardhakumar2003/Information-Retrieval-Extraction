#!/usr/bin/env python3
"""Re-run the verifiers named in agent_docs/assignment_requirements.md.

The checklist is only useful if it cannot drift from the repo. Every row names
the artefact that proves it — a pytest node id, a script, or a file that must
exist — and this reads those names back out of the markdown and checks each one
actually resolves. A row whose verifier does not exist is reported as MISSING
rather than quietly counted as done, which is the failure mode a hand-maintained
checklist always has.

    python scripts/check_requirements.py            # summary table
    python scripts/check_requirements.py --run      # also execute pytest verifiers
    python scripts/check_requirements.py --question Q2
    python scripts/check_requirements.py --checklist agent_docs/a2_requirements.md

Stdlib only; bootstraps src/ itself, so it runs before `make setup`.

## Two checklists, one checker

Assignment 2 has its own checklist with its own row ids (`A2.Q3.1`), because A1's
is a submitted deliverable whose output should not change. `--checklist` picks
which file to read, and the row pattern accepts both id shapes — so the A2 rows
are checked by the same code path that has been checking A1's, rather than by a
second, subtly different reader.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = REPO_ROOT / "agent_docs" / "assignment_requirements.md"
A2_CHECKLIST = REPO_ROOT / "agent_docs" / "a2_requirements.md"

#: Accepts both `Q2.3` (A1) and `A2.Q3.1` (A2). The optional prefix is a group
#: of its own so `Row.question` can strip only the trailing `.N`.
ROW_RE = re.compile(r"^\|\s*((?:A2\.)?Q\d+\.\d+)\s*\|(.+?)\|(.+?)\|\s*([☐◐☑])\s*\|\s*$")
PYTEST_NODE_RE = re.compile(r"(tests/[\w/]+\.py)(::\w+)?")
SCRIPT_RE = re.compile(r"(scripts/[\w_]+\.py)")
#: `reports` before `report` so the metrics directory still matches in full;
#: `report/` is where the Moodle deliverables (the PDF, the screenshots) live,
#: and without it a row naming one read as "manual" — which is how an absent
#: deliverable passes for a judgement call.
PATH_RE = re.compile(
    r"`?((?:reports|report|predictions|design_note|ai_usage|\.github)/[\w./*-]+)`?"
)


@dataclass
class Row:
    rid: str
    condition: str
    verifier: str
    status: str

    @property
    def question(self) -> str:
        """`Q2.3` -> `Q2`; `A2.Q3.1` -> `A2.Q3`.

        Split off the trailing `.N` rather than taking the first segment, or
        every A2 row lands in one bucket called "A2".
        """
        return self.rid.rsplit(".", 1)[0]


def parse_rows(checklist: Path | None = None) -> list[Row]:
    checklist = CHECKLIST if checklist is None else checklist
    if not checklist.exists():
        sys.exit(f"missing {checklist.relative_to(REPO_ROOT)}")
    rows = []
    for line in checklist.read_text(encoding="utf-8").splitlines():
        if m := ROW_RE.match(line):
            rows.append(Row(m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4)))
    return rows


def resolve(row: Row) -> tuple[str, str]:
    """Return (state, detail) for a row's verifier.

    States: ok (the verifier exists), MISSING (it is named but absent),
    manual (no machine-checkable verifier — reported, never silently passed).
    """
    targets: list[Path] = []
    for m in PYTEST_NODE_RE.finditer(row.verifier):
        targets.append(REPO_ROOT / m.group(1))
    for m in SCRIPT_RE.finditer(row.verifier):
        targets.append(REPO_ROOT / m.group(1))
    for m in PATH_RE.finditer(row.verifier):
        targets.append(REPO_ROOT / m.group(1))

    if not targets:
        return "manual", row.verifier

    missing = [t for t in targets if "*" not in str(t) and not t.exists()]
    if missing:
        return "MISSING", ", ".join(str(p.relative_to(REPO_ROOT)) for p in missing)
    return "ok", ", ".join(str(p.relative_to(REPO_ROOT)) for p in targets)


def run_pytest_verifiers(rows: list[Row]) -> int:
    nodes = sorted(
        {
            m.group(0)
            for row in rows
            for m in PYTEST_NODE_RE.finditer(row.verifier)
            if (REPO_ROOT / m.group(1)).exists()
        }
    )
    if not nodes:
        print("\nno pytest verifiers exist yet — nothing to run")
        return 0
    print(f"\nrunning {len(nodes)} pytest verifier(s)...")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *nodes], cwd=REPO_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="execute the pytest verifiers that exist")
    p.add_argument("--question", help="restrict to one question, e.g. Q2 or A2.Q3")
    p.add_argument(
        "--checklist",
        type=Path,
        default=CHECKLIST,
        help="which checklist to read (default: A1's; agent_docs/a2_requirements.md for A2)",
    )
    args = p.parse_args(argv)

    checklist = args.checklist if args.checklist.is_absolute() else REPO_ROOT / args.checklist
    print(f"checklist: {checklist.relative_to(REPO_ROOT)}")
    rows = parse_rows(checklist)
    if args.question:
        rows = [r for r in rows if r.question.lower() == args.question.lower()]
        if not rows:
            sys.exit(f"no rows for {args.question}")

    counts: dict[str, int] = {}
    current_q = None
    for row in rows:
        if row.question != current_q:
            current_q = row.question
            print(f"\n{current_q}")
        state, detail = resolve(row)
        counts[state] = counts.get(state, 0) + 1
        print(f"  {row.status} {row.rid:6s} {state:8s} {detail}")

    total = len(rows)
    done = sum(1 for r in rows if r.status == "☑")
    print(f"\n{total} conditions — {done} verified, {total - done} outstanding")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if counts.get("MISSING"):
        print("\nMISSING means the checklist names a verifier that does not exist.")
        print("That is the point: the checklist cannot silently drift from the repo.")

    if args.run:
        return run_pytest_verifiers(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
