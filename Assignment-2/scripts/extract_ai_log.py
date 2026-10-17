#!/usr/bin/env python3
"""Build ai_usage/ from Claude Code's session JSONL (Q7.4).

    make ai-log        -> everything below

Outputs
-------
    ai_usage/index.csv                          one row per session
    ai_usage/sessions/<date>_<topic>.md         MY PROMPTS ONLY, in order
    ai_usage/transcripts/<date>_<topic>.md      full transcript + tool calls
    ai_usage/raw/<session-uuid>.jsonl.gz        verbatim source, for audit

`AI_USAGE.md` is hand-written and is never touched by this script.

Two files per session, deliberately: `sessions/` is the graded prompt-quality
artefact and must be readable as a record of how the work was directed, so it
contains nothing but what the human typed. `transcripts/` is the audit record and
contains everything. One combined file would serve neither — the prompts would be
buried in tool output, and the audit trail would be incomplete.

Idempotency (Q7.7)
------------------
Re-running must produce **no diff**, because the post-commit hook runs this on
every commit and a churning file would add a meaningless commit each time. That
rules out any generation timestamp in the output, and requires gzip with
`mtime=0` — gzip stamps the current time into its header by default, which would
make `raw/*.jsonl.gz` differ on every run even when the bytes inside are
identical.

Stdlib only, so it runs anywhere the repo does.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from newsrec import config

# =============================================================================
# Prompt filtering
#
# The single most important part of this file. Q7.4 asks for "all prompts" and
# the prompt log is graded on quality, so anything the human did not type must
# not appear in it. Claude Code injects a lot into the `user` record stream:
# IDE state, slash-command echoes, command stdout, system reminders. Logging
# those as prompts would inflate the artefact with text Piyush never wrote.
#
# Filtering is PER CONTENT BLOCK, not per record. Verified against the live
# files: the first prompt of session bae60650 arrives as a two-block message
# where block[0] is <ide_opened_file> injected context and block[1] is the
# actual 2,806-character prompt. Dropping the whole record on a wrapper match
# would have silently deleted the most substantial prompt of the session.
# =============================================================================

#: A block consisting solely of one of these wrappers is machine-injected.
INJECTED_WRAPPERS = (
    "ide_opened_file",
    "ide_selection",
    "local-command-caveat",
    "local-command-stdout",
    "local-command-stderr",
    "command-name",
    "command-message",
    "command-args",
    "system-reminder",
)

#: Harness markers that appear as bare user text but were never typed.
INJECTED_LITERALS = (
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
    "No response requested.",
)

_WRAPPER_RE = re.compile(
    r"<(" + "|".join(INJECTED_WRAPPERS) + r")>.*?</\1>", re.DOTALL | re.IGNORECASE
)


def clean_prompt_block(text: str) -> str:
    """Strip injected spans; return what the human actually typed ('' if nothing).

    Wrappers are also stripped when they are *embedded* in an otherwise real
    prompt — a system-reminder appended to a genuine message is still not part
    of the message.
    """
    text = _WRAPPER_RE.sub("", text)
    # An unclosed wrapper (truncated injection) — drop from the tag onward.
    for tag in INJECTED_WRAPPERS:
        text = re.sub(rf"<{tag}>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    for literal in INJECTED_LITERALS:
        text = text.replace(literal, "")
    return text.strip()


# =============================================================================
# Redaction
# =============================================================================

_TOKEN_PATTERNS = (
    re.compile(r"\b(sk-[A-Za-z0-9_-]{16,})"),
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,})"),
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})"),
    re.compile(r"((?:api[_-]?key|token|secret|password)\s*[=:]\s*)\S{8,}", re.IGNORECASE),
)


def redact(text: str, home: str) -> str:
    """Rewrite the home directory to ~ and scrub token-shaped strings.

    ai_usage/ is committed and read by a grader, so it is published output.
    """
    if home:
        text = text.replace(home, "~")
    for pat in _TOKEN_PATTERNS:
        text = pat.sub(lambda m: m.group(1) + "[REDACTED]", text)
    return text


# =============================================================================
# Session model
# =============================================================================


@dataclass
class Prompt:
    timestamp: str
    text: str


@dataclass
class Session:
    session_id: str
    path: Path
    records: list[dict]
    prompts: list[Prompt] = field(default_factory=list)
    title: str = ""
    started: str = ""
    ended: str = ""
    models: list[str] = field(default_factory=list)
    tool_counts: Counter = field(default_factory=Counter)
    files_touched: list[str] = field(default_factory=list)

    @property
    def date(self) -> str:
        return (self.started or "0000-00-00")[:10]

    @property
    def slug(self) -> str:
        return slugify(self.title) or f"session-{self.session_id[:8]}"

    @property
    def stem(self) -> str:
        return f"{self.date}_{self.slug}"


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return "-".join(words[:max_words])


def blocks_of(record: dict) -> list[dict]:
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def load_session(path: Path) -> Session:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            records.append(json.loads(line))

    s = Session(session_id=path.stem, path=path, records=records)
    stamps = []

    for r in records:
        if ts := r.get("timestamp"):
            stamps.append(ts)
        if r.get("type") == "ai-title" and r.get("aiTitle"):
            s.title = r["aiTitle"]

        rtype = r.get("type")
        blocks = blocks_of(r)

        if rtype == "assistant":
            model = (r.get("message") or {}).get("model")
            if model and model not in s.models:
                s.models.append(model)
            for b in blocks:
                if b.get("type") != "tool_use":
                    continue
                s.tool_counts[b.get("name", "?")] += 1
                if b.get("name") in {"Write", "Edit", "NotebookEdit"}:
                    fp = (b.get("input") or {}).get("file_path")
                    if fp and fp not in s.files_touched:
                        s.files_touched.append(fp)

        elif rtype == "user" and is_human_turn(r):
            text = "\n\n".join(
                cleaned
                for b in blocks
                if b.get("type") == "text" and (cleaned := clean_prompt_block(b.get("text", "")))
            )
            if text:
                s.prompts.append(Prompt(r.get("timestamp", ""), text))

    if stamps:
        s.started, s.ended = min(stamps), max(stamps)
    return s


def is_human_turn(record: dict) -> bool:
    """A `user` record that represents a turn the human took.

    isMeta records are harness bookkeeping (including the image attachments from
    a PDF read). isSidechain records belong to a subagent, not to the human.
    A record carrying a tool_result is the harness returning output, and is by
    far the largest category of `user` records — 97 of 104 in this session.
    """
    if record.get("isMeta") or record.get("isSidechain"):
        return False
    return all(b.get("type") != "tool_result" for b in blocks_of(record))


# =============================================================================
# Rendering
# =============================================================================


def render_sessions_md(s: Session, home: str) -> str:
    """The graded artefact: prompts only, verbatim, in order."""
    out = [
        f"# {s.title or s.stem}",
        "",
        f"- **Session:** `{s.session_id}`",
        f"- **Date:** {s.date}",
        f"- **Prompts:** {len(s.prompts)}",
        "",
        "> Human prompts only, verbatim and in order. Injected context "
        "(IDE state, slash-command echoes, system reminders) and tool results "
        "are excluded. The full exchange, "
        "including responses and tool calls, is in "
        f"`ai_usage/transcripts/{s.stem}.md`.",
        "",
        "---",
        "",
    ]
    for i, p in enumerate(s.prompts, 1):
        out += [f"## Prompt {i}", "", f"*{p.timestamp}*", "", redact(p.text, home), "", "---", ""]
    return canonical("\n".join(out))


def render_transcript_md(s: Session, home: str, include_thinking: bool, max_lines: int) -> str:
    out = [
        f"# {s.title or s.stem} — full transcript",
        "",
        f"- **Session:** `{s.session_id}`",
        f"- **Date:** {s.date}",
        f"- **Models:** {', '.join(s.models) or '—'}",
        f"- **Tool calls:** {sum(s.tool_counts.values())}",
        "",
        f"> Audit record. Tool results are truncated to {max_lines} lines; the "
        f"untruncated source is `ai_usage/raw/{s.session_id}.jsonl.gz`.",
        "",
        "---",
        "",
    ]
    for r in s.records:
        rtype = r.get("type")
        if rtype not in {"user", "assistant"}:
            continue
        side = " *(subagent)*" if r.get("isSidechain") else ""

        for b in blocks_of(r):
            btype = b.get("type")
            if rtype == "user" and btype == "text":
                if not is_human_turn(r):
                    continue
                if text := clean_prompt_block(b.get("text", "")):
                    out += [f"### 👤 Human{side}", "", redact(text, home), ""]
            elif rtype == "user" and btype == "tool_result":
                out += [
                    "<details><summary>tool result</summary>",
                    "",
                    "```",
                    truncate(stringify(b.get("content")), max_lines, home),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            elif rtype == "assistant" and btype == "text":
                if text := b.get("text", "").strip():
                    out += [f"### 🤖 Claude{side}", "", redact(text, home), ""]
            elif rtype == "assistant" and btype == "thinking" and include_thinking:
                out += [
                    "<details><summary>thinking</summary>",
                    "",
                    redact(b.get("thinking", ""), home),
                    "",
                    "</details>",
                    "",
                ]
            elif rtype == "assistant" and btype == "tool_use":
                args = json.dumps(b.get("input") or {}, ensure_ascii=False, sort_keys=True)
                out += [
                    f"**🔧 {b.get('name')}**",
                    "",
                    "```json",
                    truncate(args, max_lines, home),
                    "```",
                    "",
                ]
    return canonical("\n".join(out))


def canonical(text: str) -> str:
    """Return markdown with no trailing whitespace and a single final newline.

    Generated output must already be in the form the repo's hooks enforce.
    Otherwise trailing-whitespace/mixed-line-ending rewrite these files on every
    commit, `make ai-log` regenerates them the original way, and the pair churn
    forever -- which would defeat the Q7.7 idempotency guarantee that the whole
    post-commit design depends on.
    """
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def stringify(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return json.dumps(content, ensure_ascii=False, default=str)


def truncate(text: str, max_lines: int, home: str) -> str:
    lines = redact(text, home).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    hidden = len(lines) - max_lines
    return "\n".join([*lines[:max_lines], f"... [{hidden} more lines truncated]"])


# =============================================================================


def project_dir_for(cwd: Path) -> Path:
    """Claude Code mangles the cwd into a project directory name by replacing
    '/' and '_' with '-'. Derived rather than hard-coded so this keeps working
    if the repo is moved or cloned elsewhere."""
    mangled = str(cwd).replace("/", "-").replace("_", "-")
    return Path.home() / ".claude" / "projects" / mangled


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--projects-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=config.AI_USAGE_DIR)
    p.add_argument("--include-thinking", action="store_true")
    p.add_argument("--max-tool-result-lines", type=int, default=15)
    args = p.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    proj = args.projects_dir or project_dir_for(REPO_ROOT)
    if not proj.is_dir():
        print(f"no session directory at {proj} — nothing to extract", file=sys.stderr)
        return 0

    home = str(Path.home())
    for sub in ("sessions", "transcripts", "raw"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    sessions = [load_session(f) for f in sorted(proj.glob("*.jsonl"))]
    sessions = [s for s in sessions if s.prompts]
    sessions.sort(key=lambda s: s.started)

    for s in sessions:
        (out_dir / "sessions" / f"{s.stem}.md").write_text(
            render_sessions_md(s, home), encoding="utf-8"
        )
        (out_dir / "transcripts" / f"{s.stem}.md").write_text(
            render_transcript_md(s, home, args.include_thinking, args.max_tool_result_lines),
            encoding="utf-8",
        )
        # mtime=0: gzip stamps the current time into its header by default,
        # which would make this file differ on every run and defeat Q7.7.
        raw = out_dir / "raw" / f"{s.session_id}.jsonl.gz"
        with gzip.GzipFile(raw, "wb", compresslevel=9, mtime=0) as fh:
            fh.write(s.path.read_bytes())

    index = out_dir / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            [
                "session_id",
                "date",
                "topic",
                "started",
                "ended",
                "n_prompts",
                "n_tool_calls",
                "models",
                "files_touched",
            ]
        )
        for s in sessions:
            w.writerow(
                [
                    s.session_id,
                    s.date,
                    s.slug,
                    s.started,
                    s.ended,
                    len(s.prompts),
                    sum(s.tool_counts.values()),
                    " ".join(s.models),
                    " ".join(sorted(redact(f, home) for f in s.files_touched)),
                ]
            )

    print(f"{len(sessions)} session(s) -> {config.repo_relative(out_dir)}/")
    for s in sessions:
        print(
            f"  {s.stem:52s} {len(s.prompts):3d} prompts  "
            f"{sum(s.tool_counts.values()):3d} tool calls"
        )
    if not (out_dir / "AI_USAGE.md").exists():
        print("note: ai_usage/AI_USAGE.md is missing — it is hand-written, not generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
