"""Build a LABELLED stand-in Codabench competition and a submission for it.

`build_standin_bundle.py` checks the 13.5M-row EB-NeRD test file's structure but
cannot score it: ebnerd_testset has no labels. This one scores quality. It uses
the same 2,000 `ebnerd_small` test-split impressions `rerank/evaluate.py` reports
on, so the worker's AUC/MRR/nDCG can be compared with `rerank_eval.json`.

Writes to `artifacts/codabench_standin/`:
  * ebnerd_small_bundle.zip       -- competition: labels + scripts/codabench_ranking_score.py
  * ebnerd_small_nrms.zip         -- a submission: predictions.txt from NRMS+category

    make codabench-labelled-standin
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from build_standin_bundle import png_1x1
from newsrec import config
from newsrec.data import feature_store
from newsrec.rerank import features, nrms
from newsrec.rerank.evaluate import _variant_scores, regroup
from newsrec.retrieval import recall
from newsrec.submit import format as fmt

DATASET = "ebnerd_small"
VARIANT = "nrms_category_aware"
BUNDLE_NAME = "ebnerd_small_bundle"
SCORER = Path(__file__).with_name("codabench_ranking_score.py")

COMPETITION_YAML = """\
version: 2
title: EB-NeRD small, labelled stand-in
description: Scores a rank submission against ebnerd_small labels with AUC, MRR and nDCG.
registration_auto_approve: True
docker_image: codalab/codalab-legacy:py39
image: logo.png
terms: pages/terms.md
pages:
  - title: About
    file: pages/about.md

tasks:
  - index: 0
    name: Ranking metrics
    description: AUC, MRR, nDCG@5 and nDCG@10 over 2,000 ebnerd_small test impressions.
    reference_data: reference_data/
    scoring_program: scoring_program/

phases:
  - index: 0
    name: Ranking metrics
    description: Upload predictions.zip
    start: 1-1-2026
    tasks:
      - 0

leaderboards:
  - title: Results
    key: main
    submission_rule: "Force_Last"
    columns:
      - title: AUC
        key: AUC
        index: 0
        sorting: desc
      - title: MRR
        key: MRR
        index: 1
        sorting: desc
      - title: nDCG@5
        key: nDCG@5
        index: 2
        sorting: desc
      - title: nDCG@10
        key: nDCG@10
        index: 3
        sorting: desc
      - title: Impressions scored
        key: impressions_scored
        index: 4
        sorting: desc
      - title: Bad rows
        key: bad_rows
        index: 5
        sorting: asc
"""


def score_sample() -> tuple[list[str], list[list[int]], list[np.ndarray]]:
    """Impression ids, per-candidate labels, and NRMS scores, in sample order."""
    processed = config.DATA_DIR / "processed" / DATASET
    impressions = feature_store.read_table(processed / "impressions.parquet")
    # The IDENTICAL call rerank/evaluate.py and eval/harness.py make.
    sample = recall.sample_impressions(
        impressions, "test", config.RECALL_SAMPLE_SIZE, config.GLOBAL_SEED
    )
    table = features.build_candidate_table(DATASET, "test", size=config.RECALL_SAMPLE_SIZE)
    resources = nrms.load_resources(DATASET)
    scores = regroup(table, _variant_scores(DATASET, table, resources)[VARIANT], sample)

    ids, labels = [], []
    for row in sample.iter_rows(named=True):
        clicked = set(row["clicked_ids"])
        ids.append(str(row["impression_id"]))
        labels.append([1 if a in clicked else 0 for a in row["candidate_ids"]])
    return ids, labels, scores


def build(out_root: Path) -> None:
    ids, labels, scores = score_sample()

    src = out_root / BUNDLE_NAME
    if src.exists():
        shutil.rmtree(src)
    for sub in ("reference_data", "scoring_program", "pages"):
        (src / sub).mkdir(parents=True)
    (src / "competition.yaml").write_text(COMPETITION_YAML)
    (src / "logo.png").write_bytes(png_1x1())
    (src / "pages" / "about.md").write_text("Local labelled stand-in.\n")
    (src / "pages" / "terms.md").write_text("Local stand-in. No terms.\n")
    shutil.copy(SCORER, src / "scoring_program" / "score.py")
    (src / "scoring_program" / "metadata.yaml").write_text(
        "command: python3 /app/program/score.py /app/input/ref /app/input/res /app/output\n"
    )
    (src / "reference_data" / "truth.txt").write_text(
        "".join(f"{i}\t{','.join(map(str, y))}\n" for i, y in zip(ids, labels, strict=True))
    )

    # Ties fall back to the publisher's candidate order, as nrms_predict does.
    lines = [
        fmt.format_line(i, fmt.ranks_from_scores(np.argsort(-np.asarray(s), kind="stable"), len(s)))
        for i, s in zip(ids, scores, strict=True)
    ]
    text = out_root / "predictions.txt"
    fmt.write_lines(iter(lines), text)

    bundle = out_root / f"{BUNDLE_NAME}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src))
    submission = fmt.zip_prediction(text, out_root / "ebnerd_small_nrms.zip")
    print(f"impressions: {len(ids):,}")
    print(f"bundle:     {config.repo_relative(bundle)}")
    print(f"submission: {config.repo_relative(submission)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=config.ARTIFACTS_DIR / "codabench_standin")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
