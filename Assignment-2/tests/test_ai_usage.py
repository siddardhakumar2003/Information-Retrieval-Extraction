"""Q7.4, Q7.7, Q7.8 — the AI usage log.

Mostly runnable today, because the capture pipeline exists even though the
research pipeline does not.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import extract_ai_log as ail

AI_USAGE = REPO / "ai_usage"


def test_ai_usage_complete():
    """Q7.4 — all prompts and chat exports present.

    The AI-vs-human marking is no longer a generated artefact: the provenance
    trailer system was removed on 2026-08-21, and AI authorship is stated
    directly in AI_USAGE.md and evidenced by sessions/ and transcripts/.
    """
    assert (AI_USAGE / "AI_USAGE.md").is_file(), "hand-written index is missing"
    assert (AI_USAGE / "index.csv").is_file()
    for sub in ("sessions", "transcripts", "raw"):
        d = AI_USAGE / sub
        assert d.is_dir() and any(d.iterdir()), f"ai_usage/{sub}/ is empty"


def test_every_session_has_both_files():
    """The two-file contract: a prompt log without its transcript is not an
    audit record, and a transcript without its prompt log is not readable."""
    sessions = {p.stem for p in (AI_USAGE / "sessions").glob("*.md")}
    transcripts = {p.stem for p in (AI_USAGE / "transcripts").glob("*.md")}
    assert sessions == transcripts, f"mismatch: {sessions ^ transcripts}"


def test_index_rows_match_session_files():
    with (AI_USAGE / "index.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "index.csv has no sessions"
    for row in rows:
        stem = f"{row['date']}_{row['topic']}"
        assert (AI_USAGE / "sessions" / f"{stem}.md").is_file(), f"no session file for {stem}"
        assert int(row["n_prompts"]) > 0


INJECTED = re.compile(
    r"<(ide_opened_file|ide_selection|system-reminder|command-name|command-message"
    r"|command-args|local-command-caveat|local-command-stdout)>",
    re.IGNORECASE,
)


def test_no_injected_content_in_session_prompts():
    """Q7.8 — the prompt log contains only what a human typed.

    This is the graded artefact for prompt quality. Injected IDE state and
    slash-command echoes appearing here would misrepresent it.
    """
    for path in (AI_USAGE / "sessions").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        # The explanatory blockquote names these tags; ignore that one line.
        body = "\n".join(ln for ln in text.splitlines() if not ln.startswith(">"))
        found = INJECTED.findall(body)
        assert not found, f"{path.name} contains injected content: {set(found)}"


def test_block_level_filtering_keeps_real_prompt():
    """The regression that motivated per-block filtering.

    Claude Code mixes injected context into the SAME message as typed text. A
    per-message filter would delete the real prompt along with the wrapper.
    """
    mixed = "<ide_opened_file>some file</ide_opened_file>"
    assert ail.clean_prompt_block(mixed) == ""

    real = "Read the PDF and then do the thing."
    assert ail.clean_prompt_block(real) == real
    assert ail.clean_prompt_block(f"{real}\n<system-reminder>noise</system-reminder>") == real


def test_harness_markers_are_not_prompts():
    assert ail.clean_prompt_block("[Request interrupted by user for tool use]") == ""


def test_redaction():
    assert ail.redact("/home/piyush/x", "/home/piyush") == "~/x"
    assert "REDACTED" in ail.redact("token: abcdef1234567890", "")
    assert "REDACTED" in ail.redact("use sk-abcdefghijklmnopqrst now", "")


def test_ai_log_is_idempotent_and_current(tmp_path):
    """Q7.7 — re-running produces no diff.

    Deliberately runs against a FROZEN COPY of the session files, not the live
    directory. The session being logged grows while the test runs, so asserting
    idempotency against the live source would be asserting something false and
    would fail at random.
    """
    proj = ail.project_dir_for(REPO)
    if not proj.is_dir():
        pytest.skip("no Claude Code session directory on this machine")

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for src in proj.glob("*.jsonl"):
        (frozen / src.name).write_bytes(src.read_bytes())

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    for out in (out_a, out_b):
        assert ail.main(["--projects-dir", str(frozen), "--out", str(out)]) == 0

    files_a = sorted(p.relative_to(out_a) for p in out_a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(out_b) for p in out_b.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), f"{rel} is not stable"


def test_generated_output_is_canonical():
    """Generated files must already satisfy the repo's whitespace hooks.

    Otherwise trailing-whitespace rewrites them and `make ai-log` regenerates
    them the original way, so the pair churn against each other forever.
    """
    for path in [*(AI_USAGE / "sessions").glob("*.md"), AI_USAGE / "index.csv"]:
        text = path.read_text(encoding="utf-8")
        assert "\r" not in text, f"{path.name} contains CRLF"
        assert not any(ln != ln.rstrip() for ln in text.splitlines()), (
            f"{path.name} has trailing whitespace"
        )
