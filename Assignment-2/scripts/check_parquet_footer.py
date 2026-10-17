"""Refuse a Parquet file that is not complete.

Used by `make record-checksum` so a partial download can never have its checksum
recorded. See OQ-5 (a 143 MB prefix of a 361 MB archive) and OQ-2 (a 156 MB
prefix of a 1.5 GB one) for why this gate exists rather than trusting the file.

## Why the footer is the right thing to read

Parquet is laid out with its metadata at the *end*: the four magic bytes `PAR1`
open the file, the column data follows, and then a Thrift-encoded footer records
the schema, the row-group offsets and the row counts, followed by a 4-byte
footer length and a closing `PAR1`.

That layout is what makes the check strong. To read the footer you must seek to
the end, read the length, seek backwards by it and parse. None of that can
succeed on a prefix of the file -- a truncated download has no closing magic, and
even a file that somehow ended on the right bytes would have a footer length
pointing into data that is not a footer. It is the same property that makes
`unzip -t` trustworthy (a zip's central directory is also at the end), and it is
much stronger than the last-line heuristic we have to use for TSV.

Reading the footer does NOT read the data, so this is fast even on a 1.8 GB file.

Exit 0 and print a one-line summary if the file is whole; exit non-zero with the
reason otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

MAGIC = b"PAR1"


def check(path: Path) -> str:
    """Return a summary line, or raise ValueError describing why it is not whole."""
    size = path.stat().st_size
    # A valid file is at least: opening magic + footer length + closing magic.
    if size < 12:
        raise ValueError(f"{size} bytes is too small to be a Parquet file")

    with path.open("rb") as fh:
        if fh.read(4) != MAGIC:
            raise ValueError("does not start with the PAR1 magic -- not a Parquet file")
        fh.seek(-4, 2)
        if fh.read(4) != MAGIC:
            raise ValueError("does not END with the PAR1 magic -- truncated download?")

    # The cheap magic-byte test above is not sufficient on its own: it would pass
    # a file whose tail bytes happened to be right while its footer was garbage.
    # Parsing the footer is the actual claim.
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).metadata
    return (
        f"ok {path}: {meta.num_rows:,} rows, {meta.num_columns} columns, "
        f"{meta.num_row_groups} row group(s), {size:,} bytes"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_parquet_footer.py <file.parquet>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"refusing: no such file: {path}", file=sys.stderr)
        return 2
    try:
        print(check(path))
    except Exception as exc:
        print(f"refusing: {path} footer does not parse ({exc})", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
