"""A2 Q3 — the ablation that isolates the one principled change.

Design: `agent_docs/a2_q3_plan.md`.

Mirrors `rerank/ablation.py` deliberately: the same three-table split
discipline, the same `bootstrap_ci(values - base[metric])` pairing, the same
`established` contract. Q2 and Q3 should be comparable at a glance, and two
ablation modules that measured "the same thing" slightly differently would make
that impossible.

## What the three variants are for

- `baseline_retrieval_only` — the `semantic_score` column, ordered with the
  codebase's score-then-popularity-then-id tie-break. **The same column and the
  same tie-break `reranker_ablation.json` uses**, so Q2's "before re-ranking"
  and Q3's are the identical number rather than two similar ones.
- `nrms_reproduction` — the reproduction (A2.Q3.1).
- `nrms_category_aware` — the reproduction plus the one change (A2.Q3.2).

## The comparison that is actually graded

A2.Q3.4 asks for a paired 95% CI excluding zero for a claimed gain. There are
two deltas here and only one of them answers that:

1. each NRMS variant against retrieval-only — "did a re-ranker help at all";
2. **`nrms_category_aware` minus `nrms_reproduction`** — the only comparison
   that isolates the change, because the two models differ by exactly one
   constructor argument (asserted by
   `tests/test_nrms.py::test_the_variant_differs_from_the_baseline_in_one_declared_way`).

Both are paired: every variant is scored on the *same* test table, so
subtracting per-impression values cancels the noise in which impressions were
drawn and the interval describes the delta rather than the two methods
separately.

A delta whose interval contains zero is written `established: false` and **no
improvement is claimed for it** in the design note. A2 Q2 already set that
precedent — `a2_q2_insights.md` reports GBDT losing everywhere — and it is worth
more than a flattering number.

## No hyperparameter grid

One declared configuration per variant. A2.Q3.3 asks for an ablation isolating
the improvement, which is baseline-vs-variant; a grid answers a different
question. Q2's 8+8 sweep already carries the hyperparameter-sensitivity lesson,
so it is not missing from the submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.eval import metrics
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.eval.harness import _ranked_scores, _serving_order
from newsrec.rerank import features, nrms, store

REPORT_NAME = "nrms_ablation.json"

#: Same choice and same reason as `rerank/ablation.py`: on every dataset
#: `eval.json` scores the semantic retriever above BM25, so it is the stronger
#: single retriever to measure a re-ranker against.
BASELINE_SCORE_COLUMN = "semantic_score"

BASELINE_VARIANT = "baseline_retrieval_only"


def _ranked_metrics(table: pl.DataFrame, scores: np.ndarray) -> dict[str, list[float]]:
    """Per-impression AUC/MRR/nDCG@5/nDCG@10 for one scored table.

    Identical to `rerank/ablation.py::_ranked_metrics`, including the
    score-then-popularity-then-id tie-break, so a number here and a number
    there mean the same thing.
    """
    out: dict[str, list[float]] = {m: [] for m in metrics.RANKING_METRICS}
    scored = table.with_columns(pl.Series("_score", scores))
    for _, group in scored.group_by("impression_id", maintain_order=True):
        labels = group["label"].to_numpy()
        if metrics.degenerate_reason(labels) is not None:
            continue
        popularity = group["popularity_train"].to_numpy().astype(np.float64)
        order = _serving_order(
            group["_score"].to_numpy(), popularity, group["article_id"].to_list()
        )
        got = metrics.ranking_metrics(labels, _ranked_scores(order))
        for m in metrics.RANKING_METRICS:
            out[m].append(got[m])
    return out


def _summary(values: np.ndarray) -> dict:
    ci = bootstrap_ci(values)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ci.items()}


def _delta(values: np.ndarray, base: np.ndarray) -> dict:
    """A paired bootstrap interval for `values - base`, with the verdict.

    `established` is the whole point: it says the interval excludes zero, not
    that the difference is large.
    """
    ci = bootstrap_ci(values - base)
    return {
        "mean": round(ci["mean"], 6),
        "ci_low": round(ci["ci_low"], 6),
        "ci_high": round(ci["ci_high"], 6),
        "established": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
    }


def evaluate_dataset(dataset: str, store_root: Path | None = None) -> dict:
    """Train both variants on one dataset and compare them on the test table."""
    train_table = features.build_candidate_table(
        dataset, "train", size=config.RERANK_TRAIN_SAMPLE_SIZE, store_root=store_root
    )
    val_table = features.build_candidate_table(
        dataset, "val", size=config.RERANK_VAL_SAMPLE_SIZE, store_root=store_root
    )
    test_table = features.build_candidate_table(
        dataset, "test", size=config.RECALL_SAMPLE_SIZE, store_root=store_root
    )

    resources = nrms.load_resources(dataset, store_root=store_root)
    train_source = nrms.build_source(resources, train_table, seed=config.GLOBAL_SEED)
    val_source = nrms.build_source(resources, val_table, with_groups=False)
    test_source = nrms.build_source(resources, test_table, with_groups=False)

    per_variant: dict[str, dict[str, list[float]]] = {
        BASELINE_VARIANT: _ranked_metrics(test_table, test_table[BASELINE_SCORE_COLUMN].to_numpy())
    }
    trained: dict[str, dict] = {}

    for variant in config.NRMS_VARIANTS:
        category_aware = variant == "nrms_category_aware"
        model = nrms.train(
            train_source,
            val_table,
            val_rows=val_source,
            category_aware=category_aware,
        )
        store.save(model, store.model_dir(dataset, variant))
        per_variant[variant] = _ranked_metrics(
            test_table, nrms.predict(model, test_source, test_table)
        )
        trained[variant] = {
            "category_aware": category_aware,
            "epochs_trained": model.epochs_trained,
            "best_val_ndcg@10": round(model.best_val_ndcg10, 6),
            "saved_to": str(store.model_dir(dataset, variant).relative_to(config.REPO_ROOT)),
        }

    base = {m: np.asarray(per_variant[BASELINE_VARIANT][m]) for m in metrics.RANKING_METRICS}
    reproduction = {
        m: np.asarray(per_variant["nrms_reproduction"][m]) for m in metrics.RANKING_METRICS
    }

    variants_out: dict[str, dict] = {}
    for name, per_impression in per_variant.items():
        block: dict[str, dict] = {}
        for m in metrics.RANKING_METRICS:
            values = np.asarray(per_impression[m])
            entry = _summary(values)
            if name != BASELINE_VARIANT and len(values) == len(base[m]):
                entry["delta_vs_baseline"] = _delta(values, base[m])
            # THE GRADED COMPARISON (A2.Q3.4): the one change, isolated.
            if name == "nrms_category_aware" and len(values) == len(reproduction[m]):
                entry["delta_vs_reproduction"] = _delta(values, reproduction[m])
            block[m] = entry
        variants_out[name] = block

    candidates = test_table.group_by("impression_id").len()["len"].to_numpy()
    return {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "train_impressions": int(train_table["impression_id"].n_unique()),
        "val_impressions": int(val_table["impression_id"].n_unique()),
        "test_impressions": int(test_table["impression_id"].n_unique()),
        "baseline_score_column": BASELINE_SCORE_COLUMN,
        "candidates_per_impression": {
            "median": int(np.median(candidates)),
            "min": int(candidates.min()),
            "max": int(candidates.max()),
        },
        # A2 Q1's hypothesis, now a number: how often an impression had fewer
        # than NRMS_NEG_SAMPLES non-clicked candidates and had to be sampled
        # with replacement. a2_q2_insights.md named short EB-NeRD candidate
        # lists as a suspected cause of GBDT's failure; this measures it.
        "negative_sampling": {
            "n_negatives": config.NRMS_NEG_SAMPLES,
            "impressions_sampled_with_replacement": train_source.padded_negatives,
            "share": round(train_source.padded_negatives / max(len(train_source.groups), 1), 6),
        },
        "models": trained,
        "variants": variants_out,
    }


def run(datasets: tuple[str, ...] | None = None, out_dir: Path | None = None) -> Path:
    """Write `nrms_ablation.json` (A2 Q3).

    Defaults to `config.HEADLINE_DATASETS` rather than every registered dataset:
    one English corpus and one Danish one satisfies "both datasets", and
    `ebnerd_demo` is what A1 made it — a fast-iteration corpus for fixtures, not
    a headline number. `rerank/ablation.py` covers all three because Q2's grids
    were cheap enough; training two neural variants is not.
    """
    datasets = tuple(config.HEADLINE_DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "nrms_ablation",
        "requirement": "A2.Q3",
        "graded_comparison": (
            "delta_vs_reproduction on nrms_category_aware — the only comparison that "
            "isolates the one principled change (A2.Q3.3, A2.Q3.4). A delta whose "
            "interval contains zero is established: false, and no improvement is claimed."
        ),
        "declared_deviations_from_published_nrms": [
            "The news encoder is a linear projection of A1's pinned sentence embeddings "
            "(multilingual-e5-small, 384-d) rather than learned word embeddings with "
            "word-level self-attention. See a2_q3_plan.md for the three reasons.",
            "Hyperparameters are our declared choices, not the paper's or "
            "ebnerd-benchmark's — neither has been read (OQ-12).",
            "The user encoder is permutation-invariant over the click history, as "
            "published NRMS is: attention without positional encoding is a set "
            "operation. A1's 'recency' pooling used order; this architecture cannot "
            "express it.",
        ],
        "bootstrap": {
            "ci": config.BOOTSTRAP_CI,
            "n_resamples": config.BOOTSTRAP_N,
            "paired": True,
            "resampled_over": "impressions",
        },
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
        sampling = block["negative_sampling"]
        lines.append(
            f"  negatives={sampling['n_negatives']}  "
            f"sampled with replacement in {sampling['share']:.1%} of impressions "
            f"(median {block['candidates_per_impression']['median']} candidates)"
        )
        for name, metric_block in block["variants"].items():
            ndcg10 = metric_block["ndcg@10"]
            line = f"  {name:24s} nDCG@10 {ndcg10['mean']:.4f}"
            for key, label in (
                ("delta_vs_baseline", "vs retrieval"),
                ("delta_vs_reproduction", "vs reproduction"),
            ):
                delta = ndcg10.get(key)
                if delta:
                    mark = "established" if delta["established"] else "not established"
                    line += (
                        f"   {label} {delta['mean']:+.4f} "
                        f"[{delta['ci_low']:+.4f},{delta['ci_high']:+.4f}] {mark}"
                    )
            lines.append(line)
    return "\n".join(lines)
