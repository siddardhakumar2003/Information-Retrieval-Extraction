"""Semantic recall@K (q3_plan D7).

Requirements covered: Q3.5 — recall@K for K in {50, 100, 200}, both datasets,
computed over the **same impression set** as Q2.8 so the two are comparable.

## "The same impressions" is asserted, not intended

`recall.sample_impressions` is imported rather than reimplemented, and the
resulting `impression_set_fingerprint` is compared against the one in
`bm25_recall.json`. A mismatch **raises**.

That check is the difference between a claim and evidence. Two retrievers scored
on different impressions produce two perfectly plausible numbers whose difference
means nothing, and nothing about the output would look wrong. Q2.8 wrote the
fingerprint for exactly this moment.

## Everything else is Q2's protocol, deliberately

Same seeded 2,000-impression sample. Same whole-corpus pool. Same exclusion of
impressions with no clicked article (recall is undefined with a zero
denominator, design_rationale 2.12). Same per-impression averaging rather than
pooling. Same bootstrap confidence intervals. Same training-period popularity
fallback for cold-start users.

The point of holding all of that fixed is that the *only* thing differing
between `bm25_recall.json` and `semantic_recall.json` is how a candidate is
scored. Any other difference would be a confound in Q3.6.

## Cold start goes to the popularity list, exactly as BM25 does

An impression with no legitimate prior click has no user vector -- `user_vector`
reports it through a mask. Those impressions are served the training-period
popularity ranking, the same table and the same order `query.cold_start_candidates`
gives BM25. So both retrievers behave identically on those rows, and a
lexical-vs-semantic difference cannot come from the fallback.

MIND has 62 such impressions in the sample and EB-NeRD has 0 (q3_plan D8), so
this path matters for one dataset only -- which is itself worth reporting rather
than averaging away.

## Why the ANN index is used rather than exact search

Q3.3 measured the approximation at 0.9830-0.9944 agreement with exact search, so
the ANN index is what the *system* would serve and therefore what should be
measured. The residual disagreement is a real, quantified property of the system
and is reported alongside, not hidden.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.retrieval import ann, embed, query, recall, user_vector

REPORT_NAME = "semantic_recall.json"


class ImpressionSetMismatch(RuntimeError):
    """Semantic recall was computed over different impressions than BM25.

    Raised, never warned. Q3.5's entire requirement is that the two are
    comparable; a mismatch makes every comparison in Q3.6 meaningless while
    leaving both numbers looking entirely reasonable.
    """


def _bm25_fingerprints(metrics_dir: Path) -> dict[str, str]:
    """dataset -> impression_set_fingerprint, from the committed Q2.8 report."""
    path = metrics_dir / recall.REPORT_NAME
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    return {
        name: d.get("impression_set_fingerprint") for name, d in report.get("datasets", {}).items()
    }


def evaluate_dataset(
    dataset: str,
    k_values: tuple[int, ...] | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    store_root: Path | None = None,
    pooling: str | None = None,
    embeddings: embed.ArticleEmbeddings | None = None,
    index: ann.AnnIndex | None = None,
) -> dict:
    """Measure semantic recall@K for one dataset."""
    k_values = config.RECALL_K_VALUES if k_values is None else k_values
    sample_size = config.RECALL_SAMPLE_SIZE if sample_size is None else sample_size
    seed = config.GLOBAL_SEED if seed is None else seed
    pooling = config.USER_POOLING if pooling is None else pooling

    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset

    if embeddings is None:
        embeddings = embed.load(embed.embeddings_dir(dataset))
    if index is None:
        index = ann.load(ann.ann_dir(dataset), embeddings=embeddings)

    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")

    # THE SAME SAMPLE as Q2.8 -- imported, not reimplemented (Q3.5).
    sample = recall.sample_impressions(impressions, "test", sample_size, seed)
    all_test = impressions.filter(pl.col("split") == "test")
    undefined = len(all_test) - len(all_test.filter(pl.col("clicked_ids").list.len() > 0))

    users = sample["user_id"].unique().to_list()
    t0 = time.perf_counter()
    vectors, has_vector = user_vector.user_vectors(
        sample.select("impression_id", "user_id", "impression_time"),
        history.filter(pl.col("user_id").is_in(users)),
        embeddings,
        rule=pooling,
    )
    pool_seconds = time.perf_counter() - t0

    # The identical cold-start fallback BM25 uses (q2_plan D7).
    #
    # popularity_ranking, NOT cold_start_candidates: the latter ranks only
    # articles clicked during training and so returns fewer than K on a corpus
    # where few articles were ever clicked, while BM25's zero-score tie-break
    # keeps going through the rest of the corpus. The two retrievers would then
    # be compared over different numbers of candidates. See that function's
    # docstring -- the bug was invisible on the real corpora.
    popularity = query.training_popularity(impressions)
    fallback = query.popularity_ranking(embeddings.article_ids, popularity, k=max(k_values))

    article_of = np.asarray(embeddings.article_ids, dtype=object)
    max_k = max(k_values)

    warm = np.flatnonzero(has_vector)
    t1 = time.perf_counter()
    retrieved: dict[int, np.ndarray] = {}
    if warm.size:
        # One batched search rather than a loop: the whole reason the ANN index
        # exists is that it answers many queries at once.
        hits = index.search(vectors[warm], max_k)
        for row, i in enumerate(warm):
            retrieved[int(i)] = article_of[hits.indices[row]]
    search_seconds = time.perf_counter() - t1

    per_k: dict[int, list[float]] = {k: [] for k in k_values}
    for i, row in enumerate(sample.iter_rows(named=True)):
        clicked = set(row["clicked_ids"])
        candidates = retrieved.get(i) if has_vector[i] else np.asarray(fallback, dtype=object)
        for k in k_values:
            hits_k = len(clicked.intersection(candidates[:k]))
            per_k[k].append(hits_k / len(clicked))

    return {
        "dataset": dataset,
        "retriever": "semantic",
        "encoder": embeddings.spec.model_id,
        "encoder_revision": embeddings.spec.revision,
        "embedding_fingerprint": embed.fingerprint(embeddings.spec),
        "pooling": pooling,
        "ann_ef_search": index.ef_search,
        "corpus_articles": embeddings.n_articles,
        "pool": "full_corpus",
        "impressions_evaluated": len(sample),
        "impressions_in_test_split": len(all_test),
        "impressions_undefined_excluded": undefined,
        "cold_start_impressions": int((~has_vector).sum()),
        "sample_size": sample_size,
        "seed": seed,
        "impression_set_fingerprint": recall.impression_set_fingerprint(sample),
        "recall": {str(k): {**bootstrap_ci(per_k[k]), "k": k} for k in k_values},
        "timing_seconds": {
            "pool_user_vectors": round(pool_seconds, 3),
            "ann_search": round(search_seconds, 3),
            "per_impression_ms": round(
                1000 * (pool_seconds + search_seconds) / max(1, len(sample)), 3
            ),
        },
        "_per_impression_recall": {str(k): per_k[k] for k in k_values},
    }


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> Path:
    """Measure every dataset and write `semantic_recall.json` (Q3.5).

    **Raises** if the evaluated impression set differs from Q2.8's. Q3.5 exists
    to make the two comparable; proceeding past a mismatch would produce a
    report whose every downstream comparison is meaningless and whose numbers
    look fine.
    """
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, **kwargs) for d in datasets}

    lexical = _bm25_fingerprints(out_dir)
    mismatched = {
        d: {"semantic": m["impression_set_fingerprint"], "bm25": lexical[d]}
        for d, m in measured.items()
        if d in lexical and lexical[d] != m["impression_set_fingerprint"]
    }
    if mismatched:
        raise ImpressionSetMismatch(
            f"Q3.5 requires the SAME impressions as Q2.8, and they differ: {mismatched}. "
            f"Every lexical-vs-semantic comparison built on this would be meaningless "
            f"while both numbers looked entirely reasonable. Refusing to write the report."
        )

    timings = {d: m.pop("timing_seconds") for d, m in measured.items()}
    for m in measured.values():
        m.pop("_per_impression_recall")

    report = {
        "metric": "recall_at_k",
        "retriever": "semantic",
        "k_values": list(config.RECALL_K_VALUES),
        "encoder": config.ENCODER.model_id,
        "encoder_revision": config.ENCODER.revision,
        "pooling": kwargs.get("pooling") or config.USER_POOLING,
        "ann": {
            "index": "hnsw_flat_ip",
            "m": config.ANN_HNSW_M,
            "ef_search": config.ANN_HNSW_EF_SEARCH,
        },
        "query_history_limit": config.QUERY_HISTORY_LIMIT,
        "bootstrap": {"n_resamples": config.BOOTSTRAP_N, "ci": config.BOOTSTRAP_CI},
        "impression_set_matches_bm25": bool(lexical) and not mismatched,
        "datasets": measured,
    }
    dest = out_dir / REPORT_NAME
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if out_dir == config.METRICS_DIR:
        scale_path = out_dir / "scale.json"
        scale = json.loads(scale_path.read_text()) if scale_path.exists() else {}
        scale.setdefault("semantic_recall", {}).update(timings)
        scale_path.write_text(json.dumps(scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def format_table(report: dict) -> str:
    lines = [
        f"{'dataset':<14}{'articles':>10}{'impr':>7}{'cold':>6}{'K':>6}{'recall':>9}{'95% CI':>20}",
        "-" * 72,
    ]
    for name, d in report["datasets"].items():
        for k in report["k_values"]:
            r = d["recall"][str(k)]
            ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            lines.append(
                f"{name:<14}{d['corpus_articles']:>10,}{d['impressions_evaluated']:>7,}"
                f"{d['cold_start_impressions']:>6,}{k:>6}{r['mean']:>9.4f}{ci:>20}"
            )
    return "\n".join(lines)
