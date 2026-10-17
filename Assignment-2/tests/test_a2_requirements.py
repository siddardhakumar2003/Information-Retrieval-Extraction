"""A2's requirements checklist cannot drift from the repository.

`agent_docs/a2_requirements.md` turns A2.pdf into rows, each naming the artefact
that proves it. These tests check the *checker*: a parser that silently skipped
every A2 row would report "0 conditions, 0 outstanding" and look like success,
which is the one failure mode a hand-maintained checklist always has.

The same argument `tests/test_design_note.py` makes about the page counter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_requirements as checker

A2_CHECKLIST = REPO / "agent_docs" / "a2_requirements.md"
A1_CHECKLIST = REPO / "agent_docs" / "assignment_requirements.md"


# =============================================================================
# The parser — A2 row ids carry an `A2.` prefix that A1's never did
# =============================================================================


def test_the_row_pattern_matches_an_a2_row_id():
    """`| A2.Q3.1 | ... | ... | ☐ |` must parse.

    A1's pattern was `Q\\d+\\.\\d+`, which does not match `A2.Q3.1`. Without
    this, every A2 row is skipped and the summary reads zero conditions.
    """
    line = "| A2.Q3.1 | NRMS is reproduced on both datasets | `tests/test_nrms.py` | ☐ |"
    match = checker.ROW_RE.match(line)
    assert match is not None, "the A2 row id did not parse"
    assert match.group(1) == "A2.Q3.1"


def test_the_row_pattern_still_matches_an_a1_row_id():
    """A1's checklist is a submitted deliverable; widening must not break it."""
    line = "| Q2.3 | BM25 retrieves top-K | `tests/test_bm25_scoring.py` | ☑ |"
    match = checker.ROW_RE.match(line)
    assert match is not None
    assert match.group(1) == "Q2.3"


def test_an_a2_row_groups_under_its_own_question():
    """`A2.Q3.1` belongs to question `A2.Q3`, not to `A2`.

    `Row.question` split on the FIRST dot, which put every A2 row in one bucket
    called "A2" and made `--question A2.Q3` impossible.
    """
    row = checker.Row("A2.Q3.1", "condition", "verifier", "☐")
    assert row.question == "A2.Q3"
    assert checker.Row("Q2.3", "c", "v", "☑").question == "Q2"


# =============================================================================
# The checklist itself
# =============================================================================


def test_a_file_under_report_is_a_checkable_verifier():
    """`report/A2_Report.pdf` must resolve as MISSING or ok, never as "manual".

    The path pattern listed `reports/` (the metrics directory) but not `report/`
    (the Moodle deliverables directory), so a row naming a report file read as
    "no machine-checkable verifier" — which is how an absent deliverable passes
    for a judgement call.
    """
    row = checker.Row("A2.Q7.3", "the report exists", "`report/A2_Report.pdf`", "☐")
    state, _ = checker.resolve(row)
    assert state != "manual", "a named file must be checked, not treated as manual"


def test_the_metrics_directory_still_resolves():
    """Widening the pattern must not shadow `reports/`."""
    row = checker.Row("Q4.8", "eval report", "`reports/metrics/eval.json`", "☑")
    state, detail = checker.resolve(row)
    assert state == "ok", detail
    assert "reports/metrics/eval.json" in detail


def test_the_a2_checklist_exists():
    """CLAUDE.md says to read the requirements checklist before starting work.
    For A2 there has to be one to read."""
    assert A2_CHECKLIST.exists(), f"missing {A2_CHECKLIST.relative_to(REPO)}"


def test_every_a2_row_parses():
    rows = checker.parse_rows(A2_CHECKLIST)
    assert rows, "no rows parsed out of the A2 checklist"
    text = A2_CHECKLIST.read_text(encoding="utf-8")
    # Every line that LOOKS like a row must actually have parsed as one, or the
    # checklist quietly under-reports itself.
    looks_like_a_row = [
        line
        for line in text.splitlines()
        if line.startswith("| A2.Q") and not line.startswith("| A2.Q | ")
    ]
    assert len(rows) == len(looks_like_a_row), (
        f"{len(looks_like_a_row)} lines look like rows but {len(rows)} parsed"
    )


def test_every_a2_question_from_the_pdf_has_at_least_one_row():
    """A2.pdf has Q1-Q9. A question with no row is work nobody tracked."""
    rows = checker.parse_rows(A2_CHECKLIST)
    questions = {row.question for row in rows}
    expected = {f"A2.Q{n}" for n in range(1, 10)}
    assert expected <= questions, f"no rows for {sorted(expected - questions)}"


def test_every_a2_row_has_a_status_symbol():
    rows = checker.parse_rows(A2_CHECKLIST)
    for row in rows:
        assert row.status in {"☐", "◐", "☑"}, f"{row.rid} has status {row.status!r}"


def test_no_claimed_a2_row_names_a_verifier_that_does_not_exist():
    """A row claimed as done or in progress must name something that exists.

    ☐ rows are allowed to name a *planned* verifier — that is how the checklist
    records what is still owed, and `make a2-requirements` prints those as
    MISSING for exactly that purpose. But ☑ and ◐ are claims, and a claim whose
    evidence is absent is the drift this file exists to prevent.
    """
    rows = checker.parse_rows(A2_CHECKLIST)
    broken = []
    for row in rows:
        if row.status == "☐":
            continue
        state, detail = checker.resolve(row)
        if state == "MISSING":
            broken.append(f"{row.rid} ({row.status}): {detail}")
    assert not broken, "claimed but the verifier is absent:\n  " + "\n  ".join(broken)


def test_outstanding_a2_rows_are_reported_rather_than_hidden():
    """The ☐ rows are the remaining work. This test does not fail on them; it
    fails if the checklist has stopped tracking any, which would mean either
    the assignment is finished or the file has quietly been emptied."""
    rows = checker.parse_rows(A2_CHECKLIST)
    outstanding = [r.rid for r in rows if r.status == "☐"]
    done = [r.rid for r in rows if r.status == "☑"]
    assert done, "no row is verified — the checklist is not tracking finished work"
    print(f"\n{len(done)} verified, {len(outstanding)} outstanding: {outstanding}")


def test_no_ticked_a2_row_lacks_a_machine_checkable_verifier():
    """☑ means "verified by the thing it names". A row with no named artefact
    cannot be ticked — it can only be ◐ or ☐ with the gap stated."""
    rows = checker.parse_rows(A2_CHECKLIST)
    hand_waved = [
        row.rid for row in rows if row.status == "☑" and checker.resolve(row)[0] == "manual"
    ]
    assert not hand_waved, f"ticked with no verifier: {hand_waved}"


# =============================================================================
# Canary — prove the tests above are capable of failing
# =============================================================================


@pytest.mark.canary
def test_a_row_naming_a_nonexistent_test_resolves_as_missing():
    row = checker.Row("A2.Q9.9", "invented", "`tests/test_does_not_exist.py`", "☑")
    state, _ = checker.resolve(row)
    assert state == "MISSING"


@pytest.mark.canary
def test_a_checklist_with_no_rows_is_reported_not_passed(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("# nothing here\n", encoding="utf-8")
    assert checker.parse_rows(empty) == []
