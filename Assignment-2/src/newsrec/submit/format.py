"""The prediction-file format both competitions share (Q5.1, Q5.2, Q5.3).

Design: agent_docs/q5_plan.md

## One writer, two competitions

MIND (13967) and EB-NeRD (2469) want the same line:

```
<impression_id> [r1,r2,...]
```

confirmed independently — MIND from the scorer's own `evaluate.py` (OQ-1),
EB-NeRD from the organisers' `ebnerd-benchmark` package (OQ-10). They differ
only in the filename and whether the file is zipped, so those are the only
things the per-competition modules hold.

## The ranks are an inverse permutation, and this is the thing to get wrong

Position *i* of the array holds **the rank assigned to candidate *i***, with
candidates in the order the competition lists them. It is NOT the candidates in
ranked order, and NOT the candidate indices sorted best-first.

> Four candidates `[A, B, C, D]`, and we think `B` is best, then `C`, then `A`,
> then `D`.
>
> - **Correct:** `[3,1,2,4]` — A got rank 3, B rank 1, C rank 2, D rank 4.
> - **Wrong:** `[2,3,1,4]` — the order we would list them in.
>
> Both are well-formed arrays of the right length containing each rank exactly
> once. The wrong one is accepted, scored, and lands near random. Nothing
> errors anywhere.

That is why `ranks_from_scores` builds the inverse explicitly and why
`tests/test_submit_format.py` pins it against a hand-worked example rather than
against generated data, where a symmetric permutation can make the two forms
coincide.

## Ties must be broken before writing

Both scorers convert a rank to a score of `1/rank`, which needs distinct integer
ranks. Ranking is therefore done on a strict order supplied by the caller --
the same `(score, popularity, article_id)` order the retrieval stage serves
(q2_plan D8) -- and `ranks_from_scores` never resolves a tie itself. A tie
reaching this module is a bug upstream, so it raises.

## Blank lines are a hazard, not a convenience

The MIND scorer tolerates a blank line and fills it with all-1 ranks. That
sounds forgiving; it means a bug that silently dropped an impression yields a
*scored* submission rather than a rejected one, and the only symptom is a
disappointing number with no error anywhere -- the same failure shape as Q2's
empty-query bug (design_rationale 1.16).

So this writer refuses to emit a blank line, and the validator asserts exact
row coverage. Being stricter than the scorer is the point.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


class PredictionFormatError(RuntimeError):
    """The file we were about to write, or just wrote, is not submittable."""


def ranks_from_scores(order: np.ndarray, n: int) -> np.ndarray:
    """Ranks 1..n for candidates 0..n-1, given `order` best-first.

    `order` is a permutation of `range(n)` listing candidate indices from best
    to worst -- exactly what `np.lexsort` or `np.argsort` returns. The result is
    its **inverse**: `out[i]` is the rank given to candidate `i`.

    Raises rather than repairing a malformed order. A duplicated index would
    produce duplicate ranks, and `1/rank` needs them distinct.
    """
    order = np.asarray(order, dtype=np.int64)
    if order.shape != (n,):
        raise PredictionFormatError(f"order has {order.shape} entries, expected ({n},)")

    # Check the INPUT, not the output. Scattering a non-permutation leaves some
    # entries of `out` never written, and with np.empty those hold whatever was
    # in that memory -- so a uniqueness check on the result passes or fails by
    # luck. Found by the duplicate-index test, which passed on one run.
    if np.unique(order).size != n or order.min() < 0 or order.max() >= n:
        raise PredictionFormatError(
            f"order is not a permutation of 0..{n - 1}; ranks would repeat or be missing"
        )

    out = np.empty(n, dtype=np.int64)
    out[order] = np.arange(1, n + 1, dtype=np.int64)
    return out


def format_line(impression_id: str | int, ranks: Iterable[int]) -> str:
    """`<impression_id> [r1,r2,...]` — single space, no spaces inside the array.

    Both competitions' examples show no whitespace after the commas. Neither
    parser is known to care, and neither is known not to; matching the published
    example exactly costs nothing and removes the question.
    """
    body = ",".join(str(int(r)) for r in ranks)
    return f"{impression_id} [{body}]"


def write_lines(lines: Iterator[str], path: Path) -> int:
    """Write `lines` to `path`, refusing any blank one. Returns the count.

    Streams rather than materialising: the EB-NeRD submission is 13,536,710
    lines and roughly 0.96 GB, so building the list first would cost several
    gigabytes of memory for no benefit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            if not line.strip():
                raise PredictionFormatError(
                    f"blank line at position {written + 1}. The MIND scorer would "
                    f"accept this and silently score it as all-1 ranks."
                )
            handle.write(line)
            handle.write("\n")
            written += 1
    if written == 0:
        raise PredictionFormatError(f"wrote no lines to {path}")
    return written


def zip_prediction(path: Path, archive: Path | None = None, member_dir: str = "") -> Path:
    """Zip `path`, placing it at `member_dir/<filename>` inside the archive.

    The arcname is the whole contract for a zipped submission, and **the two
    competitions disagree about it**:

    - **EB-NeRD (2469)** wants the bare filename at the archive root. The
      organisers' own `zip_submission_file` passes `arcname=path.name`, so
      `member_dir` stays empty.
    - **MIND (13967)** wants it under `res/`. Learned from the scorer's own
      error message on our first upload, not from documentation -- the archive
      must be `submission.zip` containing `res/prediction.txt`.

    An archive that accidentally carries the local directory (`predictions/…`)
    matches neither, so the arcname is always constructed here rather than left
    to `ZipFile.write`'s default, which stores the path as given.
    """
    archive = path.with_suffix(".zip") if archive is None else archive
    if archive.suffix != ".zip":
        raise PredictionFormatError(f"{archive.name} does not end in .zip")
    arcname = f"{member_dir.strip('/')}/{path.name}" if member_dir else path.name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, arcname=arcname)
    return archive


def validate_archive(
    archive: Path,
    member: str,
    expected_lines: int,
) -> dict:
    """Check the uploaded artefact is the archive the competition asked for.

    Validating the **zip**, not just the text file that went into it: the zip is
    what gets uploaded, and our first MIND submission failed on the archive
    layout while the prediction inside it was perfectly correct. A test that
    only checked the text would have passed throughout.

    Counts lines by streaming the member rather than reading it whole -- MIND's
    is 291 MB uncompressed.
    """
    if not zipfile.is_zipfile(archive):
        raise PredictionFormatError(f"{archive.name} is not a zip file")

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if member not in names:
            raise PredictionFormatError(
                f"{archive.name} does not contain {member!r}; it holds {names}. "
                f"The scorer looks for that exact path and rejects anything else."
            )
        bad = zf.testzip()
        if bad is not None:
            raise PredictionFormatError(f"{archive.name} is corrupt at {bad}")

        lines = 0
        with zf.open(member) as handle:
            for _ in handle:
                lines += 1

    if lines != expected_lines:
        raise PredictionFormatError(
            f"{archive.name}:{member} has {lines} lines, expected {expected_lines}"
        )
    return {
        "archive": archive.name,
        "member": member,
        "lines": lines,
        "members": names,
        "bytes": archive.stat().st_size,
    }


# =============================================================================
# Validation (Q5.3)
# =============================================================================


def parse_line(line: str) -> tuple[str, list[int]]:
    """Inverse of `format_line`, for validating a file we just wrote.

    Reading the file back rather than trusting the writer: Q5.3 is a claim about
    the artefact, and the artefact is what gets uploaded.
    """
    stripped = line.rstrip("\n")
    if " " not in stripped:
        raise PredictionFormatError(f"no space separating id from ranks: {stripped[:60]!r}")
    impression_id, body = stripped.split(" ", 1)
    if not (body.startswith("[") and body.endswith("]")):
        raise PredictionFormatError(f"ranks are not bracketed: {body[:60]!r}")
    inner = body[1:-1]
    if not inner:
        raise PredictionFormatError(f"empty rank array for impression {impression_id}")
    return impression_id, [int(x) for x in inner.split(",")]


def validate_file(
    path: Path,
    expected_ids: Iterable[str],
    expected_lengths: Iterable[int],
) -> dict:
    """Assert the written file matches the competition's rows exactly (Q5.3).

    `expected_ids` and `expected_lengths` are **positional**, one entry per row
    of the competition's behaviours file in its own order. Positional and not
    keyed by id, because EB-NeRD's 200,000 beyond-accuracy rows all carry
    `impression_id = 0` -- anything that groups or de-duplicates by id collapses
    them into one line and loses the entire beyond-accuracy subset without
    raising (q5_plan, hazard 1).

    Both expectations are consumed as **iterables, in lockstep with the file**,
    rather than indexed. EB-NeRD has 13,536,710 rows, and materialising that many
    Python strings purely to index into them costs about a gigabyte for nothing.
    The price of streaming is that a count mismatch surfaces at the end instead
    of up front, so both exhaustion directions are checked explicitly -- a file
    that is too short and one that is too long fail differently and must both
    fail.

    Returns a summary for the report; raises on any mismatch.
    """
    ids = iter(expected_ids)
    lengths = iter(expected_lengths)
    sentinel = object()

    seen = 0
    rank_integers = 0
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            want_id = next(ids, sentinel)
            want_len = next(lengths, sentinel)
            if want_id is sentinel or want_len is sentinel:
                raise PredictionFormatError(
                    f"{path.name} has more lines than the {lineno - 1} rows expected"
                )
            impression_id, ranks = parse_line(line)
            if impression_id != want_id:
                raise PredictionFormatError(
                    f"line {lineno}: impression_id {impression_id!r}, expected {want_id!r}. "
                    f"Order must match the competition's rows exactly."
                )
            if len(ranks) != want_len:
                raise PredictionFormatError(
                    f"line {lineno} (impression {impression_id}): {len(ranks)} ranks "
                    f"for {want_len} candidates"
                )
            if sorted(ranks) != list(range(1, want_len + 1)):
                raise PredictionFormatError(
                    f"line {lineno} (impression {impression_id}): ranks are not a "
                    f"permutation of 1..{want_len}; 1/rank requires distinct ranks"
                )
            seen = lineno
            rank_integers += want_len

    if next(ids, sentinel) is not sentinel:
        raise PredictionFormatError(
            f"{path.name} has {seen} lines but more rows were expected. A short file is "
            f"the dangerous case: the scorer fills the gap rather than rejecting it."
        )
    if seen == 0:
        raise PredictionFormatError(f"{path.name} is empty")
    return {"lines": seen, "rank_integers": rank_integers}
