"""What the serving-unavailable features would buy us (Q9.1).

Design: agent_docs/eval_protocol.md

## The question this answers

`eval_protocol.md` splits every number into two kinds. A **headline** metric is
computed with serving-available features only and is what the design note
quotes. An **ablation** metric is its pair, computed with the
serving-unavailable features *included*, always labelled and never quoted alone.

The point is not to improve a score. It is to measure how much the forbidden
features are worth, so that "we excluded them" is a quantified decision rather
than a claim of virtue. If they turn out to be worth a great deal, that is the
strongest possible argument for the discipline; if they are worth nothing, the
discipline was cheap and we should say so.

## Which features, and why they are forbidden

`config.SERVING_UNAVAILABLE_FEATURES` holds seven columns in two groups. The
**post-click four** -- `read_time`, `scroll_percentage`, `next_read_time`,
`next_scroll_percentage` -- are obvious once named: you cannot know how long
someone read an article you have not shown them yet.

The **dataset-wide aggregates** are the dangerous ones, and they are what this
module uses: `total_inviews`, `total_pageviews`, `total_read_time` are computed
over the whole corpus window, so every row carries popularity drawn from its own
future. They sit in `articles_extra.parquet` next to `title` and `category`,
they look like ordinary article metadata, and a well-meaning pipeline picks them
up by accident. Ranking by one is the cleanest demonstration of the leak.

## MIND ships none of them

`config.DATASETS["mind_small"].unavailable_columns` is empty -- MIND's news.tsv
carries no post-click fields and no aggregates. So no ablation is possible
there, and the report says so explicitly rather than omitting the dataset and
leaving a reader to wonder whether it was forgotten.

## Nulls are unknown, not zero

`total_pageviews` is present for only 9,856 of 20,738 `ebnerd_small` articles
(48%) and 7,506 of 11,777 on `ebnerd_demo` (64%). Missing ones score 0 and rank
last rather than being dropped: dropping them would change the candidate count
and make the variants score different lists, which is the one thing a paired
comparison cannot survive.

That the leak wins decisively *despite* being absent for half the corpus is part
of the finding, not a caveat against it.
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
from newsrec.eval.harness import _ranked_scores, _semantic_scores, _serving_order
from newsrec.eval.hybrid import blend
from newsrec.retrieval import embed, query, recall

REPORT_NAME = "ablations.json"

#: The leaked feature this ablation ranks by. One of the three dataset-wide
#: aggregates; `total_pageviews` because it is the most directly a popularity
#: count, and because using all three would confound "how much does the leak buy"
#: with "how well can three features be combined".
LEAK_COLUMN = "total_pageviews"

#: Blend weight for the "honest signal plus leak" variant. Deliberately 0.5 --
#: an equal vote, not a tuned optimum. Tuning it would make this an attempt to
#: win rather than a measurement of what the leak is worth.
LEAK_ALPHA = 0.5

VARIANTS = ("headline_semantic", "ablation_leak_only", "ablation_semantic_plus_leak")


def leak_values(dataset: str, store_root: Path | None = None) -> tuple[dict[str, float], dict]:
    """article_id -> the leaked aggregate, plus its coverage.

    Returns an empty mapping for a dataset that ships none of these columns,
    which is how MIND is handled without a special case at the call site.
    """
    root = store_root or config.DATA_DIR
    path = root / "processed" / dataset / "articles_extra.parquet"
    if not path.exists():
        return {}, {"available": False, "reason": "no articles_extra table"}

    frame = feature_store.read_table(path)
    if LEAK_COLUMN not in frame.columns:
        return {}, {"available": False, "reason": f"{LEAK_COLUMN} not present"}

    pairs = frame.select("article_id", LEAK_COLUMN).drop_nulls(LEAK_COLUMN)
    values = dict(pairs.iter_rows())
    return values, {
        "available": True,
        "column": LEAK_COLUMN,
        "articles_with_value": len(values),
        "articles_total": len(frame),
        "null_share": round(1.0 - len(values) / len(frame), 4) if len(frame) else float("nan"),
    }


def evaluate_dataset(dataset: str, store_root: Path | None = None) -> dict:
    """Headline and ablation metrics over one dataset's evaluated sample."""
    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset

    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")
    sample = recall.sample_impressions(impressions, "test")
    users = sample["user_id"].unique().to_list()
    user_history = history.filter(pl.col("user_id").is_in(users))

    embeddings = embed.load(embed.embeddings_dir(dataset))
    popularity = query.training_popularity(impressions)
    leaked, coverage = leak_values(dataset, store_root)

    if not coverage["available"]:
        return {
            "dataset": dataset,
            "leak_feature": coverage,
            "ablation_possible": False,
            "note": (
                f"{dataset} ships none of config.SERVING_UNAVAILABLE_FEATURES, so there "
                f"is nothing to ablate. Reported rather than omitted: an absent dataset "
                f"reads as an oversight, an explicit 'not applicable' does not."
            ),
        }

    sem_scores, _ = _semantic_scores(sample, user_history, embeddings)

    per_variant: dict[str, dict[str, list[float]]] = {
        v: {m: [] for m in metrics.RANKING_METRICS} for v in VARIANTS
    }
    for i, row in enumerate(sample.iter_rows(named=True)):
        candidates = list(row["candidate_ids"])
        clicked = set(row["clicked_ids"])
        labels = np.array([1 if a in clicked else 0 for a in candidates], dtype=np.int64)
        if metrics.degenerate_reason(labels) is not None:
            continue

        pop = np.array([popularity.get(a, 0) for a in candidates], dtype=np.float64)
        # Unknown is not popular, and not dropped: a missing aggregate scores 0
        # so the article ranks last among scored ones while the candidate count
        # stays identical across variants. A paired comparison needs that.
        leak = np.array([float(leaked.get(a, 0.0)) for a in candidates], dtype=np.float64)

        for name, scores in (
            ("headline_semantic", sem_scores[i]),
            ("ablation_leak_only", leak),
            ("ablation_semantic_plus_leak", blend(leak, sem_scores[i], LEAK_ALPHA)),
        ):
            order = _serving_order(scores, pop, candidates)
            got = metrics.ranking_metrics(labels, _ranked_scores(order))
            for metric, value in got.items():
                per_variant[name][metric].append(value)

    out = {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "impressions_evaluated": len(sample),
        "impression_set_fingerprint": recall.impression_set_fingerprint(sample),
        "leak_feature": coverage,
        "ablation_possible": True,
        "variants": {},
    }
    base = {m: np.asarray(per_variant["headline_semantic"][m]) for m in metrics.RANKING_METRICS}
    for name in VARIANTS:
        block = {}
        for metric in metrics.RANKING_METRICS:
            values = np.asarray(per_variant[name][metric])
            ci = bootstrap_ci(values)
            entry = {
                "mean": round(ci["mean"], 6),
                "ci_low": round(ci["ci_low"], 6),
                "ci_high": round(ci["ci_high"], 6),
                "n": ci["n"],
            }
            if name != "headline_semantic":
                d = bootstrap_ci(values - base[metric])
                entry["delta_vs_headline"] = {
                    "mean": round(d["mean"], 6),
                    "ci_low": round(d["ci_low"], 6),
                    "ci_high": round(d["ci_high"], 6),
                    "established": bool(d["ci_low"] > 0.0 or d["ci_high"] < 0.0),
                }
            block[metric] = entry
        out["variants"][name] = block
    return out


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> Path:
    """Write `ablations.json` (Q9.1)."""
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, **kwargs) for d in datasets}
    report = {
        "report": "ablations",
        "requirement": "Q9.1",
        "question": (
            "How much would the serving-unavailable features be worth if we used them? "
            "Not an attempt to score better -- a measurement of what the discipline costs."
        ),
        "serving_unavailable_features": sorted(config.SERVING_UNAVAILABLE_FEATURES),
        "leak_column": LEAK_COLUMN,
        "leak_alpha": LEAK_ALPHA,
        "why_this_column": (
            "One of the three dataset-wide aggregates, which are the dangerous group: "
            "they are computed over the whole corpus window, so every row carries "
            "popularity from its own future, and they sit in articles_extra.parquet "
            "next to title and category where a pipeline picks them up by accident. "
            "One column rather than three, so the result measures what the leak is "
            "worth rather than how well three features combine."
        ),
        "null_handling": (
            "An article with no value for the leaked column scores 0 and ranks last "
            "among scored candidates. Not dropped: dropping would change the candidate "
            "count and make the variants score different lists, which is the one thing "
            "a paired comparison cannot survive."
        ),
        "datasets_without_the_features": [
            d for d, b in measured.items() if not b["ablation_possible"]
        ],
        "answer": (
            "See delta_vs_headline on each ablation variant. Every headline number in "
            "the design note is the serving-available one; these are its labelled pairs "
            "and are never quoted alone."
        ),
        "bootstrap": {
            "n_resamples": config.BOOTSTRAP_N,
            "ci": config.BOOTSTRAP_CI,
            "paired": True,
            "paired_against": "headline_semantic",
        },
        "datasets": measured,
    }
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        if not block["ablation_possible"]:
            lines.append(
                f"\n{dataset}: no serving-unavailable features shipped — nothing to ablate"
            )
            continue
        cov = block["leak_feature"]
        lines.append(
            f"\n{dataset} ({block['language']})  n={block['impressions_evaluated']}  "
            f"{cov['column']} present for {cov['articles_with_value']:,}/{cov['articles_total']:,} "
            f"articles ({1 - cov['null_share']:.0%})"
        )
        for name in VARIANTS:
            auc = block["variants"][name]["auc"]
            delta = auc.get("delta_vs_headline")
            tag = ""
            if delta:
                mark = "established" if delta["established"] else "not established"
                tag = (
                    f"   delta {delta['mean']:+.4f} "
                    f"[{delta['ci_low']:+.4f},{delta['ci_high']:+.4f}] {mark}"
                )
            lines.append(f"  {name:30s} AUC {auc['mean']:.4f}{tag}")
    return "\n".join(lines)
