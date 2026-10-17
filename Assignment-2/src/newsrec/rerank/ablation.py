"""The A2 Q2 ablation: GBDT vs MLP vs the retrieval-only baseline
(a2_q2_plan.md).

## The three-way split discipline this module exists to enforce

- **train** table: fits every model. Never scored, never read for selection.
- **val** table: scores every hyperparameter combination; the best combo per
  option is the one with the highest mean nDCG@10 here. Never used to fit a
  weight.
- **test** table: scored exactly once per option (the already-selected best
  combo) and against the retrieval-only baseline, for the final reported
  comparison. Never touched before this point.

Three separate tables, built once each by `features.build_candidate_table`,
which itself only ever contains rows from the split it was asked for
(`recall.sample_impressions` filters on `split` before anything else runs).
`tests/test_rerank_ablation.py` checks this discipline holds, not just that
the numbers come out plausible.

## The paired comparison

Baseline, GBDT-best and MLP-best are all scored on the SAME test table, so
`bootstrap_ci(values - base[metric])` (identical to `eval/ablations.py`'s
pattern) measures a genuine paired delta: every row contributes to both sides
of the subtraction, so sampling noise in *which* impressions were drawn cancels
out and the interval is about the delta specifically, not about the two
methods separately.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.eval import metrics
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.eval.harness import _ranked_scores, _serving_order
from newsrec.rerank import features, gbdt, mlp

REPORT_NAME = "reranker_ablation.json"

#: "before re-ranking" = the retrieval stage's own score, ordered the same way
#: `eval/harness.py` orders BM25/semantic results (score, then training
#: popularity, then article_id). Semantic rather than BM25: on every dataset
#: `eval.json` scores it higher, so it is the stronger single retriever to
#: measure a re-ranker against.
BASELINE_SCORE_COLUMN = "semantic_score"


def _grid(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(param_grid[k] for k in keys))
    ]


def _ranked_metrics(table: pl.DataFrame, scores: np.ndarray) -> dict[str, list[float]]:
    """Per-impression AUC/MRR/nDCG@5/nDCG@10 for one scored table, using the
    same tie-break `eval/harness.py` uses so a re-ranker's "before" numbers
    match `eval.json`'s semantic row on the same impressions."""
    out: dict[str, list[float]] = {m: [] for m in metrics.RANKING_METRICS}
    scored = table.with_columns(pl.Series("_score", scores))
    for _, group in scored.group_by("impression_id", maintain_order=True):
        labels = group["label"].to_numpy()
        if metrics.degenerate_reason(labels) is not None:
            continue
        pop = group["popularity_train"].to_numpy().astype(np.float64)
        article_ids = group["article_id"].to_list()
        order = _serving_order(group["_score"].to_numpy(), pop, article_ids)
        got = metrics.ranking_metrics(labels, _ranked_scores(order))
        for m in metrics.RANKING_METRICS:
            out[m].append(got[m])
    return out


def _mean_ndcg10(per_impression: dict[str, list[float]]) -> float:
    values = per_impression["ndcg@10"]
    return float(np.mean(values)) if values else float("-inf")


def _select_best_gbdt(train: pl.DataFrame, val: pl.DataFrame) -> tuple[dict, object, list[dict]]:
    """Each combo trains with EARLY STOPPING against `val` (gbdt.py), so `val`
    is read twice: once inside training to decide the tree count, once here to
    score the result for the grid comparison. Neither use fits a weight from
    `val`'s labels directly — see gbdt.py's docstring for why that is not a
    leak."""
    grid_results = []
    best = None
    for combo in _grid(config.GBDT_PARAM_GRID):
        model = gbdt.train(train, val, params=combo)
        val_scores = gbdt.predict(model, val)
        val_metrics = _ranked_metrics(val, val_scores)
        score = _mean_ndcg10(val_metrics)
        grid_results.append(
            {
                "params": combo,
                "val_ndcg@10": round(score, 6),
                "best_iteration": model.best_iteration,
            }
        )
        if best is None or score > best[0]:
            best = (score, combo, model)
    return best[1], best[2], grid_results


def _select_best_mlp(train: pl.DataFrame, val: pl.DataFrame) -> tuple[dict, object, list[dict]]:
    """Same early-stopping-on-val discipline as `_select_best_gbdt`; see
    mlp.py's docstring."""
    grid_results = []
    best = None
    for combo in _grid(config.MLP_PARAM_GRID):
        model = mlp.train(train, val, params=combo)
        val_scores = mlp.predict(model, val)
        val_metrics = _ranked_metrics(val, val_scores)
        score = _mean_ndcg10(val_metrics)
        grid_results.append({"params": combo, "val_ndcg@10": round(score, 6)})
        if best is None or score > best[0]:
            best = (score, combo, model)
    return best[1], best[2], grid_results


def _summary(values: np.ndarray) -> dict:
    ci = bootstrap_ci(values)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ci.items()}


def evaluate_dataset(dataset: str, store_root: Path | None = None) -> dict:
    """Build the three tables, sweep both grids, compare on test."""
    train = features.build_candidate_table(
        dataset, "train", size=config.RERANK_TRAIN_SAMPLE_SIZE, store_root=store_root
    )
    val = features.build_candidate_table(
        dataset, "val", size=config.RERANK_VAL_SAMPLE_SIZE, store_root=store_root
    )
    test = features.build_candidate_table(
        dataset, "test", size=config.RECALL_SAMPLE_SIZE, store_root=store_root
    )

    gbdt_params, gbdt_model, gbdt_grid = _select_best_gbdt(train, val)
    mlp_params, mlp_model, mlp_grid = _select_best_mlp(train, val)

    baseline_scores = test[BASELINE_SCORE_COLUMN].to_numpy()
    gbdt_scores = gbdt.predict(gbdt_model, test)
    mlp_scores = mlp.predict(mlp_model, test)

    per_variant = {
        "baseline_retrieval_only": _ranked_metrics(test, baseline_scores),
        "gbdt_best": _ranked_metrics(test, gbdt_scores),
        "mlp_best": _ranked_metrics(test, mlp_scores),
    }

    base = {
        m: np.asarray(per_variant["baseline_retrieval_only"][m]) for m in metrics.RANKING_METRICS
    }
    variants_out = {}
    for name, per_impression in per_variant.items():
        block = {}
        for m in metrics.RANKING_METRICS:
            values = np.asarray(per_impression[m])
            entry = _summary(values)
            if name != "baseline_retrieval_only" and len(values) == len(base[m]):
                delta = bootstrap_ci(values - base[m])
                entry["delta_vs_baseline"] = {
                    "mean": round(delta["mean"], 6),
                    "ci_low": round(delta["ci_low"], 6),
                    "ci_high": round(delta["ci_high"], 6),
                    "established": bool(delta["ci_low"] > 0.0 or delta["ci_high"] < 0.0),
                }
            block[m] = entry
        variants_out[name] = block

    return {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "train_impressions": int(train["impression_id"].n_unique()),
        "val_impressions": int(val["impression_id"].n_unique()),
        "test_impressions": int(test["impression_id"].n_unique()),
        "gbdt": {"selected_params": gbdt_params, "grid": gbdt_grid},
        "mlp": {"selected_params": mlp_params, "grid": mlp_grid},
        "baseline_score_column": BASELINE_SCORE_COLUMN,
        "variants": variants_out,
    }


def run(datasets: tuple[str, ...] | None = None, out_dir: Path | None = None) -> Path:
    """Write `reranker_ablation.json` (A2 Q2)."""
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "reranker_ablation",
        "requirement": "A2.Q2",
        "feature_columns": list(features.FEATURE_COLUMNS),
        "datasets": {d: evaluate_dataset(d) for d in datasets},
    }
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        lines.append(
            f"\n{dataset} ({block['language']})  "
            f"train={block['train_impressions']:,} val={block['val_impressions']:,} "
            f"test={block['test_impressions']:,}"
        )
        lines.append(f"  gbdt best params: {block['gbdt']['selected_params']}")
        lines.append(f"  mlp  best params: {block['mlp']['selected_params']}")
        for name, metric_block in block["variants"].items():
            ndcg10 = metric_block["ndcg@10"]
            tag = ""
            delta = ndcg10.get("delta_vs_baseline")
            if delta:
                mark = "established" if delta["established"] else "not established"
                tag = f"   delta {delta['mean']:+.4f} [{delta['ci_low']:+.4f},{delta['ci_high']:+.4f}] {mark}"
            lines.append(f"  {name:26s} nDCG@10 {ndcg10['mean']:.4f}{tag}")
    return "\n".join(lines)
