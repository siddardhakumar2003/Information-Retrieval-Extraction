"""Prediction file for Codabench competition 2469 (Q5.2, Q5.3).

Design: agent_docs/q5_plan.md — format confirmed from the organisers'
`ebnerd-benchmark` package (OQ-10 RESOLVED 2026-08-23).

The line format lives in `submit/format.py`, shared with MIND. This module holds
only what is specific to 2469: the filename, and that the upload is a **zip**
whose single member is the bare `predictions.txt`.
"""

from __future__ import annotations

from pathlib import Path

#: Note the plural, unlike MIND's `prediction.txt`. Both names are taken from
#: their own competition's tooling; the near-collision is theirs, not ours.
FILENAME = "predictions.txt"

ZIPPED = True
ARCHIVE_NAME = "predictions.zip"

#: The prediction sits at the archive ROOT, with no directory component --
#: `zip_submission_file` passes `arcname=path.name`. MIND wants its file under
#: `res/` instead (confirmed from 13967's error message), so the two layouts are
#: genuinely different and neither is inferred from the other.
MEMBER_DIR = ""
MEMBER = FILENAME

COMPETITION = "2469"
COMPETITION_URL = "https://www.codabench.org/competitions/2469/"

#: Every beyond-accuracy row in the test set carries this impression_id. All
#: 200,000 of them, measured. Exposed so the validator can assert the count
#: rather than trusting the writer not to have de-duplicated them.
BEYOND_ACCURACY_IMPRESSION_ID = "0"
BEYOND_ACCURACY_ROWS = 200_000


def output_path(root: Path) -> Path:
    return root / FILENAME


def archive_path(root: Path) -> Path:
    return root / ARCHIVE_NAME
