"""The stand-in Codabench scorer must agree with the offline metric definitions.

`scripts/codabench_ranking_score.py` re-implements AUC / MRR / nDCG in standard
library Python because it runs in a worker image without numpy. A second
implementation of a metric is only safe if it is pinned to the first, so these
tests feed both the same random impressions.
"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from newsrec import config
from newsrec.eval import metrics

_SPEC = importlib.util.spec_from_file_location(
    "codabench_ranking_score", config.REPO_ROOT / "scripts" / "codabench_ranking_score.py"
)
scorer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scorer)


def _random_impression(rng: random.Random):
    n = rng.randint(2, 40)
    labels = [1 if rng.random() < 0.2 else 0 for _ in range(n)]
    ranks = list(range(1, n + 1))
    rng.shuffle(ranks)
    return labels, ranks


def test_agrees_with_offline_metrics_on_random_impressions():
    rng = random.Random(0)
    checked = 0
    for _ in range(500):
        labels, ranks = _random_impression(rng)
        if scorer.is_degenerate(labels):
            assert metrics.degenerate_reason(labels) is not None
            continue
        # rank 1 = best  ->  higher score = better
        expected = metrics.ranking_metrics(labels, -np.asarray(ranks, dtype=float))
        got = scorer.ranking_metrics(labels, ranks)
        assert got["AUC"] == pytest.approx(expected["auc"])
        assert got["MRR"] == pytest.approx(expected["mrr"])
        assert got["nDCG@5"] == pytest.approx(expected["ndcg@5"])
        assert got["nDCG@10"] == pytest.approx(expected["ndcg@10"])
        checked += 1
    assert checked > 300


def test_perfect_and_worst_orderings():
    labels = [0, 1, 0, 0]
    best = scorer.ranking_metrics(labels, [2, 1, 3, 4])
    worst = scorer.ranking_metrics(labels, [1, 4, 2, 3])
    assert best["AUC"] == 1.0 and best["MRR"] == 1.0 and best["nDCG@10"] == 1.0
    assert worst["AUC"] == 0.0 and worst["MRR"] == 0.25


def _write(tmp_path: Path, truth: list[str], pred: list[str]):
    (tmp_path / "truth.txt").write_text("".join(t + "\n" for t in truth))
    (tmp_path / "predictions.txt").write_text("".join(p + "\n" for p in pred))
    return scorer.score(tmp_path / "truth.txt", tmp_path / "predictions.txt")


def test_degenerate_rows_are_excluded_not_scored_as_zero(tmp_path):
    result, _, _ = _write(
        tmp_path,
        ["a\t0,1,0", "b\t0,0,0", "c\t1"],
        ["a [2,1,3]", "b [1,2,3]", "c [1]"],
    )
    assert result["impressions_scored"] == 1
    assert result["degenerate_excluded"] == 2
    assert result["AUC"] == 1.0  # not diluted by the two undefined rows


def test_a_wrong_id_or_non_permutation_is_counted_bad(tmp_path):
    result, reasons, _ = _write(
        tmp_path,
        ["a\t0,1,0", "b\t0,1,0", "c\t0,1,0"],
        ["a [2,1,3]", "x [2,1,3]", "c [1,1,3]"],
    )
    assert result["bad_rows"] == 2
    assert result["impressions_scored"] == 1
    assert len(reasons) == 2
