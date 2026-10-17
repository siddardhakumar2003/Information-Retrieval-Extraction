"""A2 Q9.1 — the two-stage pipeline with and without serving-unavailable features.

Design: `agent_docs/a2_q5_plan.md`. A1's retrieval-stage answer is
`eval/ablations.py` → `reports/metrics/ablations.json`; this is the re-ranker's.

## What "unavailable at serving time" means, and the two kinds

`config.SERVING_UNAVAILABLE_FEATURES` holds seven columns in two groups, and
they are unusable for different reasons:

- **Post-click** (`read_time`, `scroll_percentage`, `next_read_time`,
  `next_scroll_percentage`): recorded *after* a click happens. Knowing them at
  ranking time is knowing the outcome being predicted.
- **Dataset-wide aggregates** (`total_inviews`, `total_pageviews`,
  `total_read_time`): totals over the whole collection period, including the
  future relative to any given impression.

A2 Q9.1 asks for metrics **with and without** them, so the ablation is a
labelled partner to the headline number, never a headline of its own.

## Only three of the seven can change a re-ranker's ranking at all

This is the finding that shapes the module, and it is a property of *what is
being ranked* rather than of the data.

A re-ranker orders the candidates **within one impression**. `read_time` and
`scroll_percentage` are recorded per *impression*, not per candidate — so
adding them to a candidate-level feature table gives every candidate in an
impression the identical value. A feature that is constant across the things
being ordered cannot reorder them. It can shift a model's output for a whole
impression, which changes nothing about that impression's ranking, and AUC,
MRR and nDCG are all computed within an impression.

So the four post-click columns are **structurally incapable** of inflating
these metrics, and the three article-level aggregates are the whole risk. The
ablation adds those three. The other four are recorded here as
`excluded_because_constant_within_an_impression` rather than silently dropped —
a reader is entitled to know that "we tested three of seven" was a reasoned
choice and not an oversight.

This also sharpens A1's Q9.1 result rather than contradicting it: there, the
leak ranked candidates by `total_pageviews` *alone* and beat the entire
pipeline. Same three columns, same mechanism, and it is per-article popularity
that does the damage.

## MIND ships none of them

`mind_small` carries no column in the set, so the ablation is impossible there
and the report says so explicitly with `ablation_possible: false` — the same
handling `eval/ablations.py` uses, for the same reason: an absent dataset reads
as an oversight, an explicit not-applicable does not.

## Why the MLP and not NRMS

The MLP consumes `features.FEATURE_COLUMNS`, so adding three columns to that
list is the entire change. NRMS consumes article embeddings and a category
label and has no scalar feature vector to extend — asking "what if NRMS could
see `total_pageviews`" would mean designing a different architecture, which is
a different question. The MLP is the model this ablation can actually be run on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import metrics
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.rerank import features, mlp
from newsrec.rerank.nrms_ablation import _ranked_metrics

REPORT_NAME = "two_stage_ablation.json"

#: The serving-unavailable columns this ablation adds. Article-level, so they
#: vary between the candidates of one impression and can therefore change a
#: within-impression ranking. Every member is asserted to be in
#: `config.SERVING_UNAVAILABLE_FEATURES` by
#: `tests/test_two_stage_ablation.py`.
ABLATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "total_inviews",
    "total_pageviews",
    "total_read_time",
)

#: Recorded rather than dropped. Per-IMPRESSION, so identical for every
#: candidate in an impression, so incapable of reordering one — see the module
#: docstring. Excluded because they cannot affect the metric, not because they
#: are safe.
CONSTANT_WITHIN_IMPRESSION: tuple[str, ...] = (
    "read_time",
    "scroll_percentage",
    "next_read_time",
    "next_scroll_percentage",
)

HEADLINE_VARIANT = "two_stage_headline"
ABLATION_VARIANT = "two_stage_with_unavailable_features"


def available_columns(dataset: str, store_root: Path | None = None) -> tuple[str, ...]:
    """Which of `ABLATION_FEATURE_COLUMNS` this dataset actually ships."""
    root = store_root or config.DATA_DIR
    path = root / "processed" / dataset / "articles_extra.parquet"
    if not path.exists():
        return ()
    present = set(feature_store.read_table(path).columns)
    return tuple(c for c in ABLATION_FEATURE_COLUMNS if c in present)


def with_unavailable(
    table: pl.DataFrame, dataset: str, store_root: Path | None = None
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """`table` plus the available serving-unavailable columns, joined per candidate.

    A candidate with no value scores 0, the same null handling
    `eval/ablations.py` documents — an article absent from the aggregate table
    is not evidence of low popularity, but 0 is the only value that does not
    invent one.
    """
    columns = available_columns(dataset, store_root)
    if not columns:
        return table, ()

    root = store_root or config.DATA_DIR
    extra = feature_store.read_table(
        root / "processed" / dataset / "articles_extra.parquet"
    ).select("article_id", *columns)
    joined = table.join(extra, on="article_id", how="left").with_columns(
        [pl.col(c).fill_null(0).cast(pl.Float64) for c in columns]
    )
    return joined, columns


def _train_and_score(
    train_table: pl.DataFrame,
    val_table: pl.DataFrame,
    test_table: pl.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, list[float]]:
    """Fit an MLP on `feature_columns` and return its per-impression metrics.

    `features.FEATURE_COLUMNS` is module-level and read by
    `features.feature_matrix`, so it is swapped for the duration of this call
    and restored afterwards. Monkeypatching a module constant is not pretty;
    the alternative is threading a column list through four functions to serve
    one ablation, and this way the headline path stays exactly as it is.
    """
    original = features.FEATURE_COLUMNS
    features.FEATURE_COLUMNS = feature_columns
    try:
        model = mlp.train(train_table, val_table)
        scores = mlp.predict(model, test_table)
    finally:
        features.FEATURE_COLUMNS = original
    return _ranked_metrics(test_table, scores)


def evaluate_dataset(dataset: str, store_root: Path | None = None) -> dict:
    """The paired with/without comparison for one dataset."""
    columns = available_columns(dataset, store_root)
    if not columns:
        return {
            "dataset": dataset,
            "language": config.DATASETS[dataset].language,
            "ablation_possible": False,
            "note": (
                f"{dataset} ships none of {sorted(ABLATION_FEATURE_COLUMNS)}, so there "
                f"is nothing to add and nothing to remove. Reported explicitly rather "
                f"than omitted: an absent dataset reads as an oversight, an explicit "
                f"not-applicable does not."
            ),
        }

    tables = {
        split: features.build_candidate_table(
            dataset,
            split,
            size=size,
            store_root=store_root,
        )
        for split, size in (
            ("train", config.RERANK_TRAIN_SAMPLE_SIZE),
            ("val", config.RERANK_VAL_SAMPLE_SIZE),
            ("test", config.RECALL_SAMPLE_SIZE),
        )
    }
    extended = {
        split: with_unavailable(table, dataset, store_root)[0] for split, table in tables.items()
    }

    headline = _train_and_score(
        tables["train"], tables["val"], tables["test"], features.FEATURE_COLUMNS
    )
    ablation = _train_and_score(
        extended["train"],
        extended["val"],
        extended["test"],
        (*features.FEATURE_COLUMNS, *columns),
    )

    variants = {}
    for name, per_impression in ((HEADLINE_VARIANT, headline), (ABLATION_VARIANT, ablation)):
        block = {}
        for metric in metrics.RANKING_METRICS:
            values = np.asarray(per_impression[metric])
            summary = bootstrap_ci(values)
            entry = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()}
            if name == ABLATION_VARIANT:
                base = np.asarray(headline[metric])
                if len(values) == len(base):
                    ci = bootstrap_ci(values - base)
                    entry["delta_vs_headline"] = {
                        "mean": round(ci["mean"], 6),
                        "ci_low": round(ci["ci_low"], 6),
                        "ci_high": round(ci["ci_high"], 6),
                        "established": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                    }
            block[metric] = entry
        variants[name] = block

    return {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "ablation_possible": True,
        "added_columns": list(columns),
        "test_impressions": int(tables["test"]["impression_id"].n_unique()),
        "variants": variants,
    }


def run(datasets: tuple[str, ...] | None = None, out_dir: Path | None = None) -> Path:
    """Write `two_stage_ablation.json` (A2 Q9.1)."""
    datasets = tuple(config.HEADLINE_DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "two_stage_ablation",
        "requirement": "A2.Q9.1",
        "question": (
            "How much would the serving-unavailable features be worth to the "
            "RE-RANKER if we were willing to use them? A1's ablations.json answers "
            "the same question for the retrieval stage."
        ),
        "serving_unavailable_features": sorted(config.SERVING_UNAVAILABLE_FEATURES),
        "added_columns": list(ABLATION_FEATURE_COLUMNS),
        "excluded_because_constant_within_an_impression": {
            "columns": list(CONSTANT_WITHIN_IMPRESSION),
            "why": (
                "These are recorded per IMPRESSION, not per candidate, so adding them "
                "to a candidate-level table gives every candidate in an impression the "
                "identical value. A feature constant across the things being ordered "
                "cannot reorder them, and AUC/MRR/nDCG are all computed within an "
                "impression. They are structurally incapable of inflating these "
                "metrics, which is a stronger statement than 'we chose not to test "
                "them' — and it is why the three article-level aggregates are the "
                "whole risk."
            ),
        },
        "model": (
            "the MLP, because it consumes features.FEATURE_COLUMNS and adding three "
            "columns to that list is the entire change. NRMS consumes embeddings and a "
            "category label and has no scalar feature vector to extend."
        ),
        "null_handling": (
            "A candidate with no value for an added column scores 0. An article absent "
            "from the aggregate table is not evidence of low popularity, but 0 is the "
            "only value that does not invent one."
        ),
        "bootstrap": {
            "ci": config.BOOTSTRAP_CI,
            "n_resamples": config.BOOTSTRAP_N,
            "paired": True,
            "paired_against": HEADLINE_VARIANT,
        },
        "datasets": {d: evaluate_dataset(d) for d in datasets},
    }
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        if not block["ablation_possible"]:
            lines.append(f"\n{dataset}: not applicable — {block['note'].split('.')[0]}.")
            continue
        lines.append(
            f"\n{dataset} ({block['language']})  n={block['test_impressions']:,}  "
            f"added {block['added_columns']}"
        )
        for name, metric_block in block["variants"].items():
            ndcg10 = metric_block["ndcg@10"]
            line = f"  {name:38s} nDCG@10 {ndcg10['mean']:.4f}"
            delta = ndcg10.get("delta_vs_headline")
            if delta:
                mark = "established" if delta["established"] else "not established"
                line += (
                    f"   delta {delta['mean']:+.4f} "
                    f"[{delta['ci_low']:+.4f},{delta['ci_high']:+.4f}] {mark}"
                )
            lines.append(line)
    return "\n".join(lines)
