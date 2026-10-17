"""Build a stand-in Codabench competition bundle for a LOCAL EB-NeRD run.

Why this exists. The real EB-NeRD scorer (competition 2469) and its hidden labels
are not on this machine, so a local Codabench server cannot reproduce its AUC. What
it CAN do is run our submission through a real compute worker and show us real
logs -- the thing the online failure never gave us (its status reported null
exit codes, i.e. the job never started, so there was nothing to read).

The scoring program here therefore checks *structure*, not quality. For every row
of ebnerd_testset it confirms the submission line has the right impression id,
the right number of ranks, and that the ranks are a permutation of 1..n. It
streams both files, so memory stays flat on 13.5M rows.

Reference data is `impression_id n_candidates` per row, taken from
ebnerd_testset's `article_ids_inview` in file order (`ebnerd_test.iter_chunks`:
positional, never sorted or de-duplicated -- the 200,000 beyond-accuracy rows all
carry impression_id 0).

    make codabench-standin-bundle
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import zipfile
import zlib
from pathlib import Path

from newsrec import config
from newsrec.submit import ebnerd_test

BUNDLE_NAME = "ebnerd_standin_bundle"


# The unpacker refuses a bundle without an image, so build the smallest valid PNG
# (1x1, transparent) at run time rather than committing a binary.
def png_1x1() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixel = zlib.compress(b"\x00\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixel) + chunk(b"IEND", b"")
    )


COMPETITION_YAML = """\
version: 2
title: EB-NeRD stand-in (local format check)
description: Local stand-in for competition 2469. Checks structure only; the hidden labels are not on this machine.
registration_auto_approve: True
docker_image: codalab/codalab-legacy:py39

image: logo.png
terms: pages/terms.md
pages:
  - title: About
    file: pages/about.md

tasks:
  - index: 0
    name: Format check
    description: Every row has the right id, the right number of ranks, and the ranks are a permutation.
    reference_data: reference_data/
    scoring_program: scoring_program/

phases:
  - index: 0
    name: Format check
    description: Upload predictions.zip
    start: 1-1-2026
    tasks:
      - 0

leaderboards:
  - title: Results
    key: main
    submission_rule: "Force_Last"
    columns:
      - title: Valid rows
        key: valid_rows
        index: 0
        sorting: desc
      - title: Bad rows
        key: bad_rows
        index: 1
        sorting: asc
"""

ABOUT_MD = "Local stand-in used to obtain worker logs for the EB-NeRD submission.\n"

METADATA_YAML = "command: python3 /app/program/score.py /app/input/ref /app/input/res /app/output\n"

# Runs inside codalab-legacy:py39: standard library only.
SCORE_PY = '''\
"""Structure check for an EB-NeRD predictions.txt. Standard library only."""
import json
import os
import sys
import time

ref_dir, res_dir, out_dir = sys.argv[1:4]
MAX_REPORTED = 20


def find(directory, name):
    for root, _dirs, files in os.walk(directory):
        if name in files:
            return os.path.join(root, name)
    print("MISSING %s under %s; found: %s" % (name, directory, [
        os.path.join(r, f) for r, _d, fs in os.walk(directory) for f in fs]))
    sys.exit(1)


started = time.time()
truth_path = find(ref_dir, "truth.txt")
pred_path = find(res_dir, "predictions.txt")
print("truth:", truth_path, os.path.getsize(truth_path), "bytes")
print("pred :", pred_path, os.path.getsize(pred_path), "bytes")

valid = bad = 0
reasons = []


def fail(row, why):
    global bad
    bad += 1
    if len(reasons) < MAX_REPORTED:
        reasons.append("row %d: %s" % (row, why))


with open(truth_path) as truth, open(pred_path) as pred:
    row = 0
    for t_line in truth:
        row += 1
        p_line = pred.readline()
        if not p_line:
            fail(row, "prediction file ended early")
            break
        t_id, t_n = t_line.split()
        t_n = int(t_n)
        try:
            p_id, rest = p_line.rstrip("\\n").split(" ", 1)
            ranks = [int(x) for x in rest.strip("[]").split(",")]
        except ValueError:
            fail(row, "unparseable line %r" % p_line[:60])
            continue
        if p_id != t_id:
            fail(row, "impression_id %s != %s" % (p_id, t_id))
        elif len(ranks) != t_n:
            fail(row, "%d ranks for %d candidates" % (len(ranks), t_n))
        elif sorted(ranks) != list(range(1, t_n + 1)):
            fail(row, "ranks are not a permutation of 1..%d" % t_n)
        else:
            valid += 1
        if row % 1000000 == 0:
            print("checked %d rows, %d bad, %.0fs" % (row, bad, time.time() - started))
    extra = sum(1 for _ in pred)
    if extra:
        fail(row, "%d unexpected extra lines" % extra)

print("rows checked: %d  valid: %d  bad: %d" % (row, valid, bad))
for r in reasons:
    print("  ", r)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "scores.json"), "w") as f:
    json.dump({"valid_rows": valid, "bad_rows": bad}, f)
print("elapsed %.1fs" % (time.time() - started))
'''


def write_truth(path: Path) -> int:
    """One `impression_id n_candidates` line per row, in file order."""
    rows = 0
    with path.open("w") as out:
        for frame in ebnerd_test.iter_chunks():
            ids = frame["impression_id"].to_list()
            lens = frame["article_ids_inview"].list.len().to_list()
            out.writelines(f"{i} {n}\n" for i, n in zip(ids, lens))
            rows += len(ids)
    return rows


def build(out_root: Path) -> Path:
    src = out_root / BUNDLE_NAME
    if src.exists():
        shutil.rmtree(src)
    (src / "reference_data").mkdir(parents=True)
    (src / "scoring_program").mkdir()
    (src / "pages").mkdir()

    (src / "competition.yaml").write_text(COMPETITION_YAML)
    (src / "logo.png").write_bytes(png_1x1())
    (src / "pages" / "about.md").write_text(ABOUT_MD)
    (src / "pages" / "terms.md").write_text("Local stand-in. No terms.\n")
    (src / "scoring_program" / "metadata.yaml").write_text(METADATA_YAML)
    (src / "scoring_program" / "score.py").write_text(SCORE_PY)
    rows = write_truth(src / "reference_data" / "truth.txt")

    archive = out_root / f"{BUNDLE_NAME}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src))
    print(f"truth rows: {rows:,}")
    print(f"bundle: {config.repo_relative(archive)} ({archive.stat().st_size / 1e6:.1f} MB)")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "codabench_standin")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
