"""Prediction file for Codabench competition 13967 (Q5.1, Q5.3).

Design: agent_docs/q5_plan.md

The line format lives in `submit/format.py`, shared with EB-NeRD. This module
holds only what is specific to 13967 -- and after our first upload, that turned
out to include the archive layout.

## The upload is a zip holding a bare `prediction.txt`

```
mind_submission.zip
└── prediction.txt
```

**Settled 2026-08-23 by two rejections, and the second one is the instructive
one.** The line format from `evaluate.py` (OQ-1) was right throughout; what took
three attempts was the packaging, which `evaluate.py` does not describe.

1. Uploaded a bare `.txt`. Rejected -- it must be a zip.
2. Read the rejection `FileNotFoundError: '/app/input/res/prediction.txt'` as
   the *archive* layout and shipped a zip containing `res/prediction.txt`.
   Rejected with the identical message.
3. `/app/input/res/` is Codabench's **extraction destination**, not a path
   inside the archive. It unpacks a submission into `res/`, so our zip landed at
   `/app/input/res/res/prediction.txt`. The zip must hold the bare filename.

The trap is that the failing path and the required path look the same. A server
error names where the scorer *looked*, which is the platform's own directory
convention combined with the archive contents -- and only the second half is
ours to control. Reading such a path as a spec for our file is a plausible
inference, and it was wrong.

So a submission has two independent ways to fail and only one is about
recommendation quality: the ranks can be wrong, or the archive can be shaped
wrong. Ours was shaped wrong twice while all 2,370,727 prediction lines inside
were correct -- which is why `format.validate_archive` exists, and why it now
pins the layout that a live scorer accepted rather than one we reasoned to.

EB-NeRD's `zip_submission_file` uses the same bare-filename layout, so the two
competitions agree after all. They are still kept as separate constants: they
agree as a measured fact, not by derivation, and only one of them has been
confirmed against a live scorer.
"""

from __future__ import annotations

from pathlib import Path

#: The scorer expects this exact name for the prediction itself.
FILENAME = "prediction.txt"

#: At the archive ROOT. Codabench extracts the submission into its own `res/`
#: directory, so a `res/` inside the zip is nested one level too deep -- that is
#: what the second rejection was.
MEMBER_DIR = ""
MEMBER = FILENAME

ZIPPED = True
ARCHIVE_NAME = "mind_submission.zip"

COMPETITION = "13967"
COMPETITION_URL = "https://www.codabench.org/competitions/13967/"


def output_path(root: Path) -> Path:
    """Where the raw prediction text is written before archiving."""
    return root / FILENAME


def archive_path(root: Path) -> Path:
    """The file that actually gets uploaded."""
    return root / ARCHIVE_NAME
