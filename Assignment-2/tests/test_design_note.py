"""Q6.1 and Q6.6 — the design note is short enough and its numbers are real.

Both conditions name `scripts/check_design_note.py`. These tests check the
checker: a page counter that reads a word count, or a number tracer that matches
everything, would pass silently and prove nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_design_note as checker
import render_design_note as renderer

# =============================================================================
# Q6.1 — the page limit, asserted on the built PDF
# =============================================================================


def test_design_note_page_count():
    """The named Q6.1 condition."""
    if not checker.PDF.exists():
        pytest.skip("design_note.pdf not built; run `make design-note`")
    pages, error = checker.check_pages()
    assert error is None, error
    assert 0 < pages <= checker.MAX_PAGES


def test_a_missing_pdf_fails_rather_than_passing_vacuously(tmp_path):
    """An absent artefact must not read as a satisfied condition."""
    pages, error = checker.check_pages(tmp_path / "nothing.pdf")
    assert pages == 0
    assert error and "does not exist" in error


def test_the_limit_is_four_pages():
    assert checker.MAX_PAGES == 4


# =============================================================================
# Q6.6 — every number traceable
# =============================================================================


def test_every_number_in_the_design_note_is_traceable():
    """The named Q6.6 condition.

    Caught a real violation on its first run: the note quoted bootstrap coverage
    figures that lived only in a test docstring, so `make eval` could not
    regenerate them. They are now written to eval.json.
    """
    problems = checker.check_numbers()
    assert not problems, f"numbers with no source: {problems}"


def test_the_tracer_rejects_an_invented_number(tmp_path):
    """A checker that matches everything proves nothing."""
    note = tmp_path / "note.md"
    note.write_text("The system scored 0.123456789 on every corpus.\n", encoding="utf-8")
    assert checker.check_numbers(note) == ["0.123456789"]


def test_the_tracer_accepts_a_number_that_is_in_a_report(tmp_path):
    """The other direction — a checker that rejects everything is equally useless."""
    import json

    real = json.loads((checker.METRICS / "eval.json").read_text(encoding="utf-8"))
    auc = real["datasets"]["mind_small"]["methods"]["semantic"]["overall"]["auc"]["mean"]
    note = tmp_path / "note.md"
    note.write_text(f"Semantic reached {auc:.4f} AUC.\n", encoding="utf-8")
    assert checker.check_numbers(note) == []


def test_a_decimal_is_not_split_into_a_phantom_integer(tmp_path):
    """The bug that produced 30 false failures on the first run.

    `\\b\\d{4,}\\b` matches the fractional part of 0.5555, so the checker
    reported a "5555" no report contains. Decimals must be consumed first.
    """
    got = checker.find_numbers("value 0.5555 and count 12,345")
    assert "0.5555" in got
    assert "5555" not in got
    assert "12,345" in got


def test_the_allowlist_entries_all_carry_a_reason():
    """An allowlist without reasons is a way to silence the checker."""
    for value, reason in checker.ALLOWED.items():
        assert reason.strip(), f"{value} is allowlisted with no reason"
        assert len(reason) > 15, f"{value}: reason too thin to audit ({reason!r})"


# =============================================================================
# The renderer refuses what it cannot convert
# =============================================================================


def test_the_renderer_raises_on_an_unsupported_construct():
    """Silently dropping a construct would measure the page count of a document
    nobody wrote."""
    for markdown in ("```\ncode\n```\n", "> quoted\n", "#### too deep\n"):
        with pytest.raises(renderer.UnsupportedMarkdown):
            renderer.to_html(markdown)


def test_the_renderer_converts_a_table_rather_than_dropping_it():
    html = renderer.to_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<table>" in html and "<th>a</th>" in html and "<td>2</td>" in html


def test_the_renderer_escapes_html_in_the_source():
    html = renderer.to_html("A <script>alert(1)</script> paragraph.\n")
    assert "&lt;script&gt;" in html


def test_the_real_note_converts_without_raising():
    """If the note grows a construct the renderer cannot handle, this fails here
    rather than producing a silently truncated PDF."""
    renderer.to_html(checker.NOTE.read_text(encoding="utf-8"))
