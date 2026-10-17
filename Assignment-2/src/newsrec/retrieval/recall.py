"""recall@K for BM25 candidate generation (q2_plan.md D1, D9).

Requirements covered: Q2.7, Q2.8.

## What recall@K means here, and what it deliberately does not mean

For one impression: of the articles the user actually clicked, what fraction
appear in the top K we retrieved? Averaged over impressions, **not pooled** --
pooling would let one impression with many clicks dominate the average, which is
the same reasoning `eval_protocol.md` applies to the ranking metrics.

**Retrieval is from the WHOLE corpus** -- all 65,238 MIND articles, all 20,738
`ebnerd_small` articles -- not from the impression's own candidate list. This is
q2_plan D1, and it was settled by a measurement rather than a preference:
EB-NeRD shows a mean of 11.5 and a maximum of 100 candidates per impression, so
recall@200 over that list would be **exactly 1.0 for every impression**. The
metric would be arithmetic, not measurement. `test_recall.py` asserts the pool
size equals the corpus size so this cannot quietly change.

Expect the absolute numbers to be low. Retrieving 200 of 65,238 and hoping to
include the ~1.5 that were clicked is genuinely hard. That is the difficulty of
the task, not a verdict on BM25, and the temptation it creates -- shrinking the
pool to make the number look better -- is D2's rejected leak.

## The cold-start path is the tie-break, not a separate branch

An impression with no usable history produces an empty query, so every article
scores exactly 0 and the ranking is decided entirely by `top_k`'s tie-break --
which is training-period popularity (D7, D8). That *is* the popularity fallback
Q2.6 requires; it falls out of the ordering rather than needing a second code
path that could drift from it. `test_recall.py` asserts the two agree.

## Sampling, and why the interval is the point

D9 evaluates on a seeded sample rather than every impression: 244,647 test
impressions x a corpus scan each is slow enough to discourage re-running, which
is how stale numbers happen. The sample is a deterministic function of
(dataset, split, size, seed), all recorded in the output, and a fingerprint of
the selected impression ids is stored so Q3.5 can *prove* it evaluated the same
set rather than asserting it.

Because the number is a sample estimate, it is reported with a bootstrap
confidence interval. If BM25 and embeddings differ by less than their intervals
overlap, the design note must say the difference is not established.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.retrieval import bm25, query, tokenise
from newsrec.retrieval import index as idx

REPORT_NAME = "bm25_recall.json"


def sample_impressions(
    impressions: pl.DataFrame,
    split: str = "test",
    size: int | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """The evaluated impression set — deterministic, and shared with Q3.

    Impressions with no clicked article are dropped: recall is undefined when
    the denominator is zero. They are counted by the caller and reported rather
    than silently disappearing (design_rationale 2.12) — an undefined value
    quietly imputed as 0 would drag the average down for a reason that is not
    about retrieval at all.

    Exposed as a function so Q3's semantic recall imports it instead of
    re-deriving the sample, which is what makes the two comparable (Q3.5).
    """
    size = config.RECALL_SAMPLE_SIZE if size is None else size
    seed = config.GLOBAL_SEED if seed is None else seed

    evaluable = impressions.filter(
        (pl.col("split") == split) & (pl.col("clicked_ids").list.len() > 0)
    ).sort("impression_id")  # total order before sampling, so the draw is stable

    if len(evaluable) <= size:
        return evaluable
    return evaluable.sample(n=size, seed=seed, shuffle=False).sort("impression_id")


def impression_set_fingerprint(impressions: pl.DataFrame) -> str:
    """A hash of the evaluated impression ids.

    Q3.5 requires semantic recall to be computed over the SAME impressions as
    Q2.8. Recording the seed and size makes that reproducible; recording this
    makes it *checkable*, which is the difference between a claim and evidence.
    """
    blob = "\n".join(impressions["impression_id"].to_list()).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def evaluate_dataset(
    dataset: str,
    k_values: tuple[int, ...] | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    store_root: Path | None = None,
    spec: config.TokeniserSpec | None = None,
) -> dict:
    """Measure recall@K for one dataset. Returns the report fragment.

    `spec` overrides the dataset's headline tokenisation variant. It exists for
    Q2.9's ablation, which must compare variants under *one* definition of
    recall — a second implementation would let the two drift, and then a
    difference between variants would be indistinguishable from a difference
    between the two measurement paths. Nothing in the headline pipeline passes it.
    """
    k_values = config.RECALL_K_VALUES if k_values is None else k_values
    sample_size = config.RECALL_SAMPLE_SIZE if sample_size is None else sample_size
    seed = config.GLOBAL_SEED if seed is None else seed
    spec = config.tokeniser_for(dataset) if spec is None else spec

    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset
    store = root / "feature_store" / dataset

    articles = feature_store.read_table(store / "article_features.parquet").select(
        "article_id", "text"
    )
    titles = feature_store.read_table(processed / "articles.parquet").select("article_id", "title")
    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")

    # THE POOL. Every article in the corpus (D2 option A), which is what makes
    # the K in recall@K a meaningful cut rather than a formality.
    t0 = time.perf_counter()
    index = idx.build(articles, spec)
    build_seconds = time.perf_counter() - t0

    # Popularity from TRAINING rows only. Serves double duty as the cold-start
    # ranking and as top_k's tie-break, from one table (D7, D8).
    popularity = query.training_popularity(impressions)
    tie_break = np.asarray(query.popularity_array(index.article_ids, popularity), dtype=np.int64)

    sample = sample_impressions(impressions, "test", sample_size, seed)
    all_test = impressions.filter(pl.col("split") == "test")
    undefined = len(all_test) - len(all_test.filter(pl.col("clicked_ids").list.len() > 0))

    users = sample["user_id"].unique().to_list()
    queries = query.build_queries(
        sample, history.filter(pl.col("user_id").is_in(users)), titles, spec
    )

    idf = bm25.idf_array(index)
    doc_to_article = np.asarray(index.article_ids, dtype=object)
    max_k = max(k_values)

    per_k: dict[int, list[float]] = {k: [] for k in k_values}
    cold_start = 0
    t1 = time.perf_counter()

    for row in sample.iter_rows(named=True):
        terms = queries.get(row["impression_id"], [])
        if not terms:
            cold_start += 1
        clicked = set(row["clicked_ids"])
        scores = bm25.score(index, terms, idf=idf)
        top = bm25.top_k(scores, max_k, tie_break=tie_break)
        retrieved = doc_to_article[top]
        for k in k_values:
            hits = len(clicked.intersection(retrieved[:k]))
            per_k[k].append(hits / len(clicked))

    retrieve_seconds = time.perf_counter() - t1

    return {
        "dataset": dataset,
        "tokeniser": tokenise.fingerprint(spec),
        "tokeniser_variant": spec.name,
        "corpus_articles": index.n_docs,
        "pool": "full_corpus",
        "impressions_evaluated": len(sample),
        "impressions_in_test_split": len(all_test),
        "impressions_undefined_excluded": undefined,
        "cold_start_impressions": cold_start,
        "sample_size": sample_size,
        "seed": seed,
        "impression_set_fingerprint": impression_set_fingerprint(sample),
        "recall": {
            str(k): {
                **bootstrap_ci(per_k[k]),
                "k": k,
            }
            for k in k_values
        },
        # Wall-clock, and therefore NOT part of the recall report: two runs of
        # unchanged code would differ in the third decimal and the determinism
        # check would fail every time. `run()` splits this out into scale.json,
        # where varying timings are the point (Q6.5). This is the same rule the
        # Q1 manifest follows by carrying no "built at" field
        # (design_rationale 2.9) -- and it was re-learned here the hard way,
        # because the first version of this report embedded it and
        # test_recall_is_measured_consistently caught it immediately.
        "timing_seconds": {
            "index_build": round(build_seconds, 3),
            "retrieval_total": round(retrieve_seconds, 3),
            "per_impression_ms": round(1000 * retrieve_seconds / max(1, len(sample)), 3),
        },
        # One recall value per evaluated impression, in `sample` order — the raw
        # material for a PAIRED comparison. Q2.9 needs to say what the Danish
        # stemmer was worth, and two independent intervals cannot answer that:
        # they overlap for a variant that improves every single impression by a
        # little. Bootstrapping the per-impression DIFFERENCE can. Popped by
        # `run()` for the same reason `timing_seconds` is — 2,000 floats per K
        # do not belong in a summary report.
        "per_impression_recall": {str(k): per_k[k] for k in k_values},
    }


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> Path:
    """Measure every dataset and write the report (Q2.8)."""
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, **kwargs) for d in datasets}

    # Split the wall-clock measurements out. Everything left is a deterministic
    # function of the code and the data, so two runs produce identical bytes.
    timings = {d: m.pop("timing_seconds") for d, m in measured.items()}
    for m in measured.values():
        m.pop("per_impression_recall")

    report = {
        "metric": "recall_at_k",
        "retriever": "bm25",
        # Recorded so a reader can reproduce the number without reading code.
        "k_values": list(config.RECALL_K_VALUES),
        "bm25": {"k1": config.BM25_K1, "b": config.BM25_B, "epsilon": config.BM25_EPSILON},
        "query_history_limit": config.QUERY_HISTORY_LIMIT,
        "bootstrap": {"n_resamples": config.BOOTSTRAP_N, "ci": config.BOOTSTRAP_CI},
        "datasets": measured,
    }
    dest = out_dir / REPORT_NAME
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Feeds scale_analysis.md / Q6.5. Timings here are EXPECTED to vary between
    # runs -- that is what they measure -- which is exactly why they cannot live
    # in the report above.
    scale_path = out_dir / "scale.json"
    scale = json.loads(scale_path.read_text()) if scale_path.exists() else {}
    scale.setdefault("bm25_recall", {}).update(timings)
    scale_path.write_text(json.dumps(scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def format_table(report: dict) -> str:
    """The dataset x K table, for the design note and the terminal."""
    lines = [
        f"{'dataset':<14}{'articles':>10}{'impr':>8}{'cold':>7}{'K':>6}{'recall':>9}{'95% CI':>20}",
        "-" * 74,
    ]
    for name, d in report["datasets"].items():
        for k in report["k_values"]:
            r = d["recall"][str(k)]
            ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            lines.append(
                f"{name:<14}{d['corpus_articles']:>10,}{d['impressions_evaluated']:>8,}"
                f"{d['cold_start_impressions']:>7,}{k:>6}{r['mean']:>9.4f}{ci:>20}"
            )
    return "\n".join(lines)
