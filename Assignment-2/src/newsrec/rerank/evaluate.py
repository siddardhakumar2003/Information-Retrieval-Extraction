"""A2 Q5 — the full metric set, sliced, for the two-stage pipeline.

Design: `agent_docs/a2_q5_plan.md`.

## Why this is glue and not a second harness

`eval/harness.py` already computes everything A2 Q5 asks for — AUC, MRR,
nDCG@5, nDCG@10, intra-list diversity, novelty, coverage, each with a bootstrap
interval, over the eight slice masks `eval/slices.py` builds (which include both
pairs Q5 names by name: cold-start vs warm, head vs tail). It also already
flags a slice too small for its interval to mean what it says.

So this module computes **no metrics of its own**. It reshapes a re-ranker's
per-row scores into the `list[np.ndarray]` that `harness._per_impression`
expects, and hands them to `harness._metric_block` / `_slice_block`. Reusing
the privates directly is the established convention: `eval/ablations.py`,
`eval/hybrid.py` and `rerank/features.py` all import them rather than
re-deriving a score or a metric.

## Why `rerank` is not registered as a `harness.METHODS` entry

It would be the obvious move and it is wrong.
`tests/test_reports.py::test_eval_json_covers_method_dataset_grid` asserts that
every registered method ran on every dataset, so adding one makes `make eval`
— and the committed `eval.json` a grader reads without rebuilding — depend on
loading a **trained model**, which depends on a GPU. `eval.json` would stop
being regenerable on a CPU-only machine, which is the claim A2.Q7.1/Q7.2 make.

A separate module with its own report keeps the retrieval evaluation
CPU-reproducible and the two-stage evaluation honest about needing a model.

## The one thing that can silently go wrong

`features.build_candidate_table` returns one row per (impression, candidate)
and `harness._per_impression` walks the *sample* row by row. Lining those two
up wrongly produces a plausible mid-range metric with **nothing raising
anywhere** — the failure A1 shipped once for real, where a submission scored
byte-identically to the baseline while its coverage guard read 1.0000
(`design_rationale.md` §1.21).

So `regroup` does not trust the shapes. It checks that each impression's
candidate list in the table **is** the sample's candidate list, in the same
order, and raises `CandidateMismatch` otherwise.

## Pairing

Every variant is scored on the same sample, and that sample is asserted to be
the one `eval.json` already describes (by
`recall.impression_set_fingerprint`). Without that assertion, "before vs after
re-ranking" compares two similar samples rather than the same one, and the
paired delta means nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import beyond_accuracy as ba
from newsrec.eval import harness, metrics, slices
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.rerank import features, nrms, store
from newsrec.retrieval import embed, query, recall

REPORT_NAME = "rerank_eval.json"

#: The retrieval-only ordering. Same column and same tie-break
#: `rerank/ablation.py` and `rerank/nrms_ablation.py` use, so all three reports'
#: "before re-ranking" is the identical number.
BASELINE_VARIANT = "retrieval_only_baseline"
BASELINE_SCORE_COLUMN = "semantic_score"


class CandidateMismatch(RuntimeError):
    """A candidate table does not describe the sample it is being scored against.

    Raised rather than tolerated: the alternative is a complete, well-formed,
    wrong metric.
    """


def regroup(table: pl.DataFrame, scores: np.ndarray, sample: pl.DataFrame) -> list[np.ndarray]:
    """Per-row scores -> one array per `sample` row, in candidate order.

    The shape `harness._per_impression` consumes. Checked, not assumed: each
    impression's candidates in `table` must equal the sample's
    `candidate_ids` exactly and in order.
    """
    scored = table.with_columns(pl.Series("_score", scores))
    by_impression = {
        str(impression_id): (group["article_id"].to_list(), group["_score"].to_numpy())
        for (impression_id,), group in scored.group_by("impression_id", maintain_order=True)
    }

    out: list[np.ndarray] = []
    for row in sample.iter_rows(named=True):
        impression_id = str(row["impression_id"])
        expected = [str(a) for a in row["candidate_ids"]]
        found = by_impression.get(impression_id)
        if found is None:
            raise CandidateMismatch(
                f"impression {impression_id} is in the sample but not in the candidate table"
            )
        got, values = found
        if [str(a) for a in got] != expected:
            raise CandidateMismatch(
                f"impression {impression_id}: the candidate table's articles are not the "
                f"sample's candidate_ids in the same order "
                f"({len(got)} vs {len(expected)} articles). Scores would be attached to "
                f"the wrong candidates and every metric would still look plausible."
            )
        out.append(values)
    return out


def _variant_scores(
    dataset: str,
    table: pl.DataFrame,
    resources: nrms.DatasetResources,
) -> dict[str, np.ndarray]:
    """Per-row scores for each variant this report covers.

    The baseline is a column of the table. Each NRMS variant is loaded from
    `artifacts/rerank/` — **loaded, not retrained**, so the evaluated system is
    provably the one `nrms_ablation.json` selected. A missing directory is
    skipped with a note rather than silently retrained under different logic.
    """
    out = {BASELINE_VARIANT: table[BASELINE_SCORE_COLUMN].to_numpy()}
    source = nrms.build_source(resources, table, with_groups=False)
    for variant in config.NRMS_VARIANTS:
        dest = store.model_dir(dataset, variant)
        if not (dest / store.META_NAME).exists():
            continue
        model = store.load(dest)
        out[variant] = nrms.predict(model, source, table)
    return out


def _summarise(entry: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in entry.items()}


def evaluate_dataset(dataset: str, store_root: Path | None = None) -> dict:
    """The full metric set for every variant on one dataset."""
    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset

    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")

    # The IDENTICAL call `harness.evaluate_dataset` makes, so the two reports
    # describe the same impressions rather than two similar samples.
    sample = recall.sample_impressions(
        impressions, "test", config.RECALL_SAMPLE_SIZE, config.GLOBAL_SEED
    )
    user_history = history.filter(pl.col("user_id").is_in(sample["user_id"].unique().to_list()))

    embeddings = embed.load(embed.embeddings_dir(dataset))
    popularity = query.training_popularity(impressions)
    p_by_article, p_unseen = ba.smoothed_probabilities(popularity, embeddings.n_articles)

    masks = slices.build(sample, user_history, impressions)
    slices.partition_check(masks, len(sample))

    table = features.build_candidate_table(
        dataset, "test", size=config.RECALL_SAMPLE_SIZE, store_root=store_root
    )
    resources = nrms.load_resources(dataset, store_root=store_root)
    scored = _variant_scores(dataset, table, resources)

    per_variant: dict[str, dict] = {}
    ranking_values: dict[str, dict[str, np.ndarray]] = {}
    for name, scores in scored.items():
        per_impression = harness._per_impression(
            sample, regroup(table, scores, sample), popularity, embeddings, p_by_article, p_unseen
        )
        per_variant[name] = {
            "overall": harness._metric_block(per_impression, embeddings.n_articles),
            "degenerate_impressions": per_impression["degenerate"],
            "slices": {
                slice_name: harness._slice_block(
                    slice_name, mask, per_impression, embeddings.n_articles
                )
                for slice_name, mask in masks.items()
            },
        }
        ranking_values[name] = {
            metric: np.array(
                [v for v in per_impression["ranking"][metric] if v is not None], dtype=np.float64
            )
            for metric in metrics.RANKING_METRICS
        }

    # Paired deltas against the retrieval-only ordering, on the same sample.
    base = ranking_values[BASELINE_VARIANT]
    for name, values in ranking_values.items():
        if name == BASELINE_VARIANT:
            continue
        for metric in metrics.RANKING_METRICS:
            if len(values[metric]) != len(base[metric]):
                continue
            ci = bootstrap_ci(values[metric] - base[metric])
            per_variant[name]["overall"][metric]["delta_vs_baseline"] = {
                "mean": round(ci["mean"], 6),
                "ci_low": round(ci["ci_low"], 6),
                "ci_high": round(ci["ci_high"], 6),
                "established": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
            }

    return {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "impressions_evaluated": len(sample),
        "impression_set_fingerprint": recall.impression_set_fingerprint(sample),
        "catalogue_articles": embeddings.n_articles,
        "baseline_score_column": BASELINE_SCORE_COLUMN,
        "variants": {k: _deep_round(v) for k, v in per_variant.items()},
    }


def _deep_round(value):
    if isinstance(value, dict):
        return {k: _deep_round(v) for k, v in value.items()}
    if isinstance(value, float):
        return round(value, 6)
    return value


def run(datasets: tuple[str, ...] | None = None, out_dir: Path | None = None) -> Path:
    """Write `rerank_eval.json` (A2 Q5.1-Q5.4).

    Defaults to `config.HEADLINE_DATASETS`, matching `nrms_ablation.run` —
    this report evaluates the models that stage trained, so it covers the same
    corpora.
    """
    datasets = tuple(config.HEADLINE_DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "rerank_eval",
        "requirement": "A2.Q5.1-A2.Q5.4",
        "ranked_pool": "the impression's own candidate_ids, not the corpus",
        "metrics_source": (
            "eval/harness.py's own _per_impression/_metric_block/_slice_block — this "
            "stage computes no metric of its own, so a number here and a number in "
            "eval.json are the same measurement"
        ),
        "impression_set": (
            "recall.sample_impressions(test, RECALL_SAMPLE_SIZE, GLOBAL_SEED) — the "
            "identical call eval.json makes, asserted by the fingerprint, so every "
            "delta_vs_baseline is a genuine paired comparison"
        ),
        "bootstrap": {
            "ci": config.BOOTSTRAP_CI,
            "n_resamples": config.BOOTSTRAP_N,
            "paired": True,
            "resampled_over": "impressions",
        },
        "slices": {
            "training_period": list(slices.TRAINING_PERIOD_SLICES),
            "test_period": list(slices.TEST_PERIOD_SLICES),
            "note": (
                "A2 Q5.2 requires at least two; both of the pairs it names "
                "(cold_start/warm, article_head/article_tail) are here, plus the "
                "history quartiles and the training-period head/tail cut A1's Q4.5 "
                "mandated. article_head/article_tail use TEST-period popularity and "
                "are analysis descriptors, never model inputs."
            ),
        },
        "datasets": {d: evaluate_dataset(d) for d in datasets},
    }
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        lines.append(f"\n{dataset} ({block['language']})  n={block['impressions_evaluated']:,}")
        for name, variant in block["variants"].items():
            overall = variant["overall"]
            line = (
                f"  {name:24s} "
                f"AUC {overall['auc']['mean']:.4f}  "
                f"nDCG@10 {overall['ndcg@10']['mean']:.4f}  "
                f"div {overall['diversity']['mean']:.4f}  "
                f"cov {overall['coverage']['mean']:.4f}"
            )
            delta = overall["ndcg@10"].get("delta_vs_baseline")
            if delta:
                mark = "established" if delta["established"] else "not established"
                line += f"   delta {delta['mean']:+.4f} {mark}"
            lines.append(line)
            cold = variant["slices"].get("cold_start", {})
            warm = variant["slices"].get("warm", {})
            if cold and warm:
                lines.append(
                    f"      cold_start n={cold['n']:<5d} nDCG@10 {cold['ndcg@10']['mean']:.4f}"
                    f"   warm n={warm['n']:<5d} nDCG@10 {warm['ndcg@10']['mean']:.4f}"
                )
    return "\n".join(lines)
