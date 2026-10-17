#!/usr/bin/env python3
"""Assert the design note is short enough and its numbers are real (Q6.1, Q6.6).

Two independent checks, both failing loudly rather than warning:

**--pages** counts pages in the rendered PDF, not words in the markdown. Q6.1
says "asserted by a script that counts pages in the built PDF — not eyeballed",
and a word count is a different measurement that happens to correlate.

**--check-numbers** takes every metric-shaped number in the note and looks for
it in `reports/metrics/*.json` or `predictions/manifest.json`. A number with no
source is either a typo, a stale figure from an earlier run, or something
recalled rather than measured -- and all three have happened in this project.

## What "metric-shaped" means, and why the filter is conservative

Matching every integer would flag section numbers, years and page budgets. So a
number is checked when it looks like a measurement:

  * a decimal with three or more places (0.5508, 0.0082)
  * an integer of four or more digits, with or without separators (2,370,727)

Percentages are checked against both their written and fractional forms, since
the reports store 0.523 where the note writes 52.3%.

The filter is deliberately conservative: it can miss a fabricated small integer,
but it does not cry wolf, and a checker people learn to ignore is worse than no
checker. `ALLOWED` carries the numbers that legitimately come from somewhere
other than a metrics report, each with a reason -- config constants, corpus
sizes, and figures sourced from another document. An entry there is a claim
someone made deliberately; an unmatched number is one nobody did.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTE = REPO / "design_note" / "design_note.md"
PDF = REPO / "design_note" / "design_note.pdf"
METRICS = REPO / "reports" / "metrics"
MANIFEST = REPO / "predictions" / "manifest.json"

MAX_PAGES = 4

#: Numbers that are real but do not live in a metrics report. Each needs a
#: reason: the point is that someone decided, not that the checker was quietened.
ALLOWED: dict[str, str] = {
    "65238": "MIND-small corpus size; data/processed manifest, quoted in §1",
    "11777": "ebnerd_demo corpus size",
    "20738": "ebnerd_small corpus size",
    "120961": "MINDlarge_test corpus size; predictions/manifest.json context",
    "125541": "ebnerd_testset corpus size",
    "2370727": "MINDlarge_test impressions; predictions/manifest.json",
    "13536710": "ebnerd_testset rows; predictions/manifest.json",
    "244647": "ebnerd_small test split size; eval_protocol.md",
    "205925868": "ebnerd_testset candidate slots; predictions/manifest.json",
    "93115001": "MINDlarge_test candidate slots; predictions/manifest.json",
    "182.3": "mean EB-NeRD history length; q2_plan.md D6",
    "11.5": "mean EB-NeRD candidates per impression; recall.py D1",
    "4500": "approximate query length if abstracts were used; q2_plan.md D6",
    "19505.5": "median clicked-article day; derived from lexical_vs_semantic.json",
    "0.0047": "difference between two reported AUCs, stated as a subtraction",
    "0.0927": "difference between two leaderboard AUCs, stated as a subtraction",
    "0.1203": "EB-NeRD leaderboard AUC minus its baseline, stated as a subtraction",
    "39.2": "MINDlarge_test mean candidates; measured from behaviors.tsv in §3.5",
    "36.7": "mind_small mean candidates; measured from the evaluated sample",
    "1.9": "10x embedding matrix size in GB, arithmetic on 193 MB",
    "193": "embedding matrix size in MB, arithmetic on 125,541 x 384 x 4",
    "0.865": "lowest efSearch=64 ANN recall across corpora, rounded from ann_recall",
    "0.893": "highest efSearch=64 ANN recall across corpora, rounded from ann_recall",
}

#: Assignment 2's note has its own allowlist, for the same reason A1's exists:
#: a handful of real numbers legitimately come from somewhere other than a
#: metrics report, and each needs a stated source.
#:
#: Kept SEPARATE from `ALLOWED` rather than merged, so a figure justified for
#: A1's document cannot silently excuse the same digits in A2's. Picked by the
#: filename in `main`.
A2_ALLOWED: dict[str, str] = {
    # Corpus and file sizes, from predictions/manifest.json's context and the
    # data/processed manifests. Same entries A1 needed, restated here rather
    # than shared, so this list stands alone.
    "65238": "MIND-small corpus size; data/processed manifest",
    "20738": "ebnerd_small corpus size",
    "120961": "MINDlarge_test corpus size",
    "125541": "ebnerd_testset corpus size",
    "2370727": "MINDlarge_test impressions; predictions/manifest.json",
    "13536710": "ebnerd_testset rows; predictions/manifest.json",
    "13336711": "ebnerd_testset distinct impression ids; submit/ebnerd_test.py N_ROWS block",
    "200000": "ebnerd_testset beyond-accuracy rows; submit/ebnerd_test.py",
    "205925868": "ebnerd_testset candidate slots; measured in submit/ebnerd_test.check_shape",
    # Config constants quoted in the note.
    "20260827": "config.GLOBAL_SEED",
    "4096": "this machine's VRAM in MiB; nvidia-smi",
    "4.406": "the course code, CS4.406 — not a measurement",
    "0.0002": (
        "the MIND delta from the UNSEEDED run, quoted in 3.4 as the symptom of a "
        "non-reproducible ablation. Superseded by the seeded run and kept "
        "deliberately as the before-figure; design_rationale.md 5.4"
    ),
    "50000": (
        "ebnerd_testset rows sampled once (2026-09-16) to measure popularity_train's "
        "serving-time coverage, OQ-14. A sample size, not a measurement in any report."
    ),
    "0.434698": (
        "the identical val nDCG@10 both NRMS variants reported before the falsy-source "
        "bug was fixed; quoted as the symptom, design_rationale.md 5.4"
    ),
}

_DECIMAL = re.compile(r"\d+\.\d{3,}")
_BIG_INT = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b")


def _iter_values(node) -> object:
    """Every numeric leaf in a nested JSON structure."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_values(value)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        yield float(node)


def known_values() -> set[str]:
    """Every number any report states, in the string forms a note might use."""
    sources = sorted(METRICS.glob("*.json"))
    if MANIFEST.exists():
        sources.append(MANIFEST)

    out: set[str] = set()
    for path in sources:
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for value in _iter_values(blob):
            # Both the value and its MAGNITUDE. `_DECIMAL` captures digits only,
            # so a note writing "-0.0634" yields the token "0.0634" while the
            # report holds -0.063392 — and a genuinely measured negative delta
            # could never be traced. Found by checking A2's note, which quotes
            # three established negative deltas where A1's quoted none.
            for signed in {value, abs(value)}:
                out.add(f"{signed:.10g}")
                for places in range(0, 7):
                    out.add(f"{signed:.{places}f}")
                    # The reports store 0.523 where a note writes 52.3%.
                    out.add(f"{signed * 100:.{places}f}")
                if float(signed).is_integer():
                    out.add(f"{int(signed)}")
                    out.add(f"{int(signed):,}")
    return out


def find_numbers(text: str) -> list[str]:
    r"""Metric-shaped literals in the note, in order of appearance.

    Decimals are matched FIRST and blanked out before the integer pass. Without
    that, `\b\d{4,}\b` matches the fractional part of 0.5555 and the checker
    reports a phantom "5555" that no report contains — 30 false failures on the
    first run, which is precisely how a checker teaches people to ignore it.
    """
    found = []
    remainder = list(text)
    for match in _DECIMAL.finditer(text):
        found.append(match.group())
        for i in range(match.start(), match.end()):
            remainder[i] = " "
    for match in _BIG_INT.finditer("".join(remainder)):
        found.append(match.group())
    return found


def check_numbers(note: Path = NOTE, allowed: dict[str, str] | None = None) -> list[str]:
    """Return the unmatched numbers. Empty list means every number is sourced.

    `allowed` defaults to A1's allowlist. A2's note gets its own
    (`A2_ALLOWED`), kept separate so a figure justified for one document cannot
    silently excuse the same digits in the other.
    """
    allowed = ALLOWED if allowed is None else allowed
    text = note.read_text(encoding="utf-8")
    known = known_values()
    problems = []
    for raw in find_numbers(text):
        bare = raw.replace(",", "")
        if bare in allowed or raw in allowed:
            continue
        candidates = {raw, bare}
        with contextlib.suppress(ValueError):  # the regexes only match numerics
            candidates.add(f"{float(bare):.10g}")
        if candidates & known:
            continue
        problems.append(raw)
    return sorted(set(problems), key=lambda s: text.index(s))


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    `relative_to` RAISES for a path outside the repo, so the plain form turned
    the "PDF is missing" branch into a crash — the one branch whose whole job is
    to report cleanly.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def check_pages(pdf: Path = PDF, max_pages: int | None = None) -> tuple[int, str | None]:
    """Page count and an error message, or (count, None) when it passes."""
    max_pages = MAX_PAGES if max_pages is None else max_pages
    if not pdf.exists():
        return 0, f"{_display(pdf)} does not exist; run `make design-note`"
    from pypdf import PdfReader

    pages = len(PdfReader(str(pdf)).pages)
    if pages > max_pages:
        return pages, f"design note is {pages} pages, limit is {max_pages}"
    return pages, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-numbers", action="store_true")
    parser.add_argument("--pages", action="store_true")
    parser.add_argument(
        "--note", type=Path, default=NOTE, help="which markdown note to check (default: A1's)"
    )
    parser.add_argument("--pdf", type=Path, default=None, help="the rendered PDF to count")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=f"page limit (default: {MAX_PAGES}, A1's Q6.1; A2's Q6.1 is 6)",
    )
    args = parser.parse_args(argv)
    run_all = not (args.check_numbers or args.pages)

    note = args.note if args.note.is_absolute() else REPO / args.note
    pdf = args.pdf or note.with_suffix(".pdf")
    if not pdf.is_absolute():
        pdf = REPO / pdf
    max_pages = args.max_pages if args.max_pages is not None else MAX_PAGES
    # A2's note has its own allowlist. Picked by which note is being checked,
    # so a figure justified for one document cannot excuse the same digits in
    # the other.
    allowed = A2_ALLOWED if note.name.startswith("a2_") else ALLOWED

    failed = False

    if run_all or args.pages:
        pages, error = check_pages(pdf, max_pages)
        if error:
            print(f"FAIL  pages: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"ok    pages: {pages} of {max_pages}")

    if run_all or args.check_numbers:
        problems = check_numbers(note, allowed)
        checked = len(set(find_numbers(note.read_text(encoding="utf-8"))))
        if problems:
            print(
                f"FAIL  numbers: {len(problems)} of {checked} not found in "
                f"reports/metrics/*.json, predictions/manifest.json, or the "
                f"allowlist:",
                file=sys.stderr,
            )
            for value in problems:
                print(f"        {value}", file=sys.stderr)
            print(
                "      Each is a typo, a stale figure, or something recalled "
                "rather than measured. If it is legitimately sourced elsewhere, "
                "add it to ALLOWED with the source.",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"ok    numbers: {checked} checked, all traceable")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
