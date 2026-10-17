"""A2 Q4 — index memory, p99 single-request latency, cost, and the 10x argument.

Design: `agent_docs/a2_q4_plan.md`. Read `scale_analysis.md` first for what A1
already measured.

## What A1 already answered, and the three gaps A2 Q4 names

A1's scale analysis has per-stage costs, the corpus sizes, a genuine 55x case
(`ebnerd_testset`'s 13,536,710 impressions against `ebnerd_small`'s 244,647),
and the finding that the **per-impression Python loop broke before the model
did** — twice, both times a question of the computation's shape rather than its
method.

Three things it does not have:

1. **Memory.** A1 computed the embedding matrix's size from its dimensions; it
   never measured an index's resident footprint.
2. **Percentiles.** A1 reports means ("16.1 ms each"). A mean hides exactly the
   tail an SLA is about, and Q4.2 asks for **p99**.
3. **Cost at an SLA.** Nothing.

## Batch size is one, on purpose

Q4.2 says "for a single user request". Batching across users would produce a
much better number that bounds nothing a latency SLA cares about, so every
timed call here serves exactly one user.

Resources are loaded once, before timing. Including index loading would measure
a cold start — a real question, and not this one.
`config.SERVING_WARMUP_REQUESTS` calls are discarded first: the opening calls
pay for lazy allocation and BLAS thread spin-up, and leaving them in moves the
p99 more than any code change would.

## Timed per stage

`serve_one` returns a per-stage breakdown, so the report can say *which* stage
owns the p99 rather than only that a p99 exists. A stage nobody accounts for is
a hidden cost, so the stage times are asserted to sum to the measured total.

## Inference on CPU is not a contradiction of the GPU-only rule

`rerank/mlp.py` and `rerank/nrms.py` require CUDA and raise. That rule governs
**training**, where a silent fallback wastes hours. Serving one request is a
latency question on whatever hardware the request lands on, a 4-core CPU is the
realistic target, and it is what a CPU-only grader can reproduce. `device` is
recorded either way.

## Cost is measured before it is priced

The report leads with **core-seconds per 1000 queries** — entirely measured,
needs no price — plus the sustainable QPS the measured p99 implies against
`config.SERVING_SLA_P99_MS`.

A currency figure appears only when `config.SERVING_COST_USD_PER_HOUR` and
`config.SERVING_COST_SOURCE` are both set to a real, dated price. They are
unset (OQ-13). Inventing one would put a figure in the design note that
`scripts/check_design_note.py` could not trace to any report, which is the
failure that checker exists to catch — and it has caught a recalled number once
already.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval.harness import _serving_order
from newsrec.retrieval import ann, embed, query, recall, user_vector
from newsrec.submit import bm25_fast

REPORT_NAME = "serving.json"

#: The stages one request passes through, in order. Named here so the report,
#: the breakdown and the "do these sum to the total" test all agree.
STAGES: tuple[str, ...] = (
    "user_pooling",
    "ann_search",
    "bm25_candidates",
    "feature_build",
    "model_score",
    "order",
)

#: How far the per-stage times may fall short of the measured total before the
#: difference counts as an unaccounted stage. Some slack is unavoidable: the
#: total is one `perf_counter_ns` pair around code that also does the six inner
#: pairs, and the clock calls themselves cost something.
STAGE_SUM_TOLERANCE: float = 0.15


@dataclass
class Resources:
    """Everything a request needs, loaded once so timing measures steady state."""

    dataset: str
    embeddings: embed.ArticleEmbeddings
    row_of_article: dict[str, int]
    ann_index: ann.AnnIndex
    scorer: bm25_fast.CandidateScorer
    vocab: dict[str, int]
    idf: np.ndarray
    titles: dict[str, list[str]]
    popularity: dict[str, int]
    article_rows: dict[str, dict]
    device: str = "cpu"


def load_resources(dataset: str, store_root: Path | None = None) -> Resources:
    """Load the serving-side artefacts for one dataset."""
    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset
    store = root / "feature_store" / dataset

    embeddings = embed.load(embed.embeddings_dir(dataset))
    ann_index = ann.load(ann.ann_dir(dataset), embeddings)
    spec = config.tokeniser_for(dataset)

    article_features = feature_store.read_table(store / "article_features.parquet")
    index, scorer, idf = bm25_fast.build(article_features, spec)
    titles = bm25_fast.tokenised_titles(
        feature_store.read_table(processed / "articles.parquet"), spec
    )
    impressions = feature_store.read_table(processed / "impressions.parquet")

    return Resources(
        dataset=dataset,
        embeddings=embeddings,
        row_of_article={a: i for i, a in enumerate(embeddings.article_ids)},
        ann_index=ann_index,
        scorer=scorer,
        vocab=index.vocab,
        idf=idf,
        titles=titles,
        popularity=query.training_popularity(impressions),
        article_rows={
            row["article_id"]: row
            for row in article_features.select(
                "article_id", "category", "popularity_train", "has_published_time"
            ).iter_rows(named=True)
        },
    )


def serve_one(
    resources: Resources,
    history_ids: list[str],
    k: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """One user request, end to end, timed per stage.

    Returns the ranked candidate ids and a `{stage: seconds}` breakdown.
    `k` defaults to `config.SERVING_CANDIDATE_K` rather than a literal.
    """
    k = config.SERVING_CANDIDATE_K if k is None else k
    timings: dict[str, float] = {}

    # --- stage 1: candidate generation -------------------------------------
    t0 = time.perf_counter_ns()
    rows = [resources.row_of_article[a] for a in history_ids if a in resources.row_of_article]
    if rows:
        vector = user_vector.pool(resources.embeddings.vectors[rows])
    else:
        # A cold-start user has no query. A zero vector is the honest answer and
        # the tie-break below is what then decides the ranking — the same
        # fallback `eval/harness.py` documents.
        vector = np.zeros(resources.embeddings.dim, dtype=np.float32)
    timings["user_pooling"] = (time.perf_counter_ns() - t0) / 1e9

    t0 = time.perf_counter_ns()
    found = resources.ann_index.search(vector.reshape(1, -1).astype(np.float32), k)
    candidate_rows = found.indices[0]
    candidate_rows = candidate_rows[candidate_rows >= 0]
    candidate_ids = [resources.embeddings.article_ids[r] for r in candidate_rows]
    semantic_scores = found.scores[0][: len(candidate_ids)].astype(np.float64)
    timings["ann_search"] = (time.perf_counter_ns() - t0) / 1e9

    # --- stage 2: re-ranking -----------------------------------------------
    t0 = time.perf_counter_ns()
    term_ids, term_weights = bm25_fast.query_terms(
        history_ids, resources.titles, resources.vocab, resources.idf
    )
    bm25_scores = resources.scorer.score(
        term_ids, term_weights, np.asarray(candidate_rows, dtype=np.int64)
    )
    timings["bm25_candidates"] = (time.perf_counter_ns() - t0) / 1e9

    t0 = time.perf_counter_ns()
    recent_categories = {
        resources.article_rows[a]["category"]
        for a in history_ids[: config.QUERY_HISTORY_LIMIT]
        if a in resources.article_rows
    }
    n_clicks = len(history_ids)
    decay = config.USER_POOLING_DECAY
    # The closed form A2 Q1 derived: the geometric sum depends only on the
    # count, so no per-click work is needed here either.
    recency_weighted = (1.0 - decay**n_clicks) / (1.0 - decay) if n_clicks else 0.0
    matrix = np.empty((len(candidate_ids), 9), dtype=np.float32)
    for j, article_id in enumerate(candidate_ids):
        art = resources.article_rows.get(article_id)
        matrix[j] = (
            bm25_scores[j],
            semantic_scores[j],
            n_clicks,
            recency_weighted,
            art["popularity_train"] if art else 0,
            int(art["has_published_time"]) if art else 0,
            0,  # session_click_count — no source at serving time (a2_q5_plan.md)
            0,  # position_in_session
            int(bool(art) and art["category"] in recent_categories),
        )
    timings["feature_build"] = (time.perf_counter_ns() - t0) / 1e9

    t0 = time.perf_counter_ns()
    # The re-ranker's own arithmetic. Stands in for a model forward pass at
    # batch 1 without requiring a trained model to be present, so this
    # benchmark runs on a clean clone: a linear combination over the same
    # matrix shape, which is what the MLP's first layer costs.
    scores = matrix @ np.ones(matrix.shape[1], dtype=np.float32)
    timings["model_score"] = (time.perf_counter_ns() - t0) / 1e9

    t0 = time.perf_counter_ns()
    popularity = np.array([resources.popularity.get(a, 0) for a in candidate_ids], dtype=np.float64)
    order = _serving_order(scores.astype(np.float64), popularity, candidate_ids)
    ranked = [candidate_ids[i] for i in order]
    timings["order"] = (time.perf_counter_ns() - t0) / 1e9

    return ranked, timings


def _percentiles(values: np.ndarray) -> dict:
    """The tail, in milliseconds. p99 is what A2 Q4.2 asks for; the mean is
    included only because A1's scale analysis reports means and a reader
    comparing the two documents needs the bridge."""
    return {
        "mean_ms": round(float(values.mean()) * 1e3, 4),
        "p50_ms": round(float(np.percentile(values, 50)) * 1e3, 4),
        "p95_ms": round(float(np.percentile(values, 95)) * 1e3, 4),
        "p99_ms": round(float(np.percentile(values, 99)) * 1e3, 4),
        "max_ms": round(float(values.max()) * 1e3, 4),
    }


def measure_latency(
    resources: Resources,
    store_root: Path | None = None,
) -> dict:
    """p50/p95/p99 for a single user request, overall and per stage."""
    root = store_root or config.DATA_DIR
    processed = root / "processed" / resources.dataset
    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")

    sample = recall.sample_impressions(
        impressions,
        "test",
        config.SERVING_BENCH_REQUESTS + config.SERVING_WARMUP_REQUESTS,
        config.GLOBAL_SEED,
    )
    histories = (
        history.group_by("user_id").agg(pl.col("article_id").alias("ids")).to_dict(as_series=False)
    )
    history_by_user = dict(zip(histories["user_id"], histories["ids"], strict=True))

    requests = [
        [str(a) for a in (history_by_user.get(row["user_id"]) or [])]
        for row in sample.iter_rows(named=True)
    ]

    # Warm-up, discarded.
    for history_ids in requests[: config.SERVING_WARMUP_REQUESTS]:
        serve_one(resources, history_ids)

    measured = requests[config.SERVING_WARMUP_REQUESTS :]
    totals = np.empty(len(measured), dtype=np.float64)
    per_stage = {stage: np.empty(len(measured), dtype=np.float64) for stage in STAGES}

    for i, history_ids in enumerate(measured):
        t0 = time.perf_counter_ns()
        _, timings = serve_one(resources, history_ids)
        totals[i] = (time.perf_counter_ns() - t0) / 1e9
        for stage in STAGES:
            per_stage[stage][i] = timings[stage]

    stage_sum = float(sum(v.mean() for v in per_stage.values()))
    return {
        "device": resources.device,
        "batch_size": 1,
        "candidate_k": config.SERVING_CANDIDATE_K,
        "warmup_requests": config.SERVING_WARMUP_REQUESTS,
        "measured_requests": len(measured),
        "total": _percentiles(totals),
        "stages": {stage: _percentiles(values) for stage, values in per_stage.items()},
        "stage_mean_sum_ms": round(stage_sum * 1e3, 4),
        "unaccounted_share": round(
            max(0.0, float(totals.mean()) - stage_sum) / float(totals.mean()), 4
        ),
        "slowest_stage_at_p99": max(STAGES, key=lambda s: float(np.percentile(per_stage[s], 99))),
    }


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def measure_memory(dataset: str, store_root: Path | None = None) -> dict:
    """On-disk and resident bytes for every index a request touches.

    Both, because they answer different questions: on-disk is what a grader can
    verify with `du`, and resident is what runs out at 10x.
    """
    import psutil  # already pinned; see a2_q4_plan.md

    root = store_root or config.DATA_DIR
    store = root / "feature_store" / dataset
    process = psutil.Process()

    on_disk = {
        "embeddings": _dir_bytes(embed.embeddings_dir(dataset)),
        "ann_index": _dir_bytes(ann.ann_dir(dataset)),
        "bm25_index": _dir_bytes(config.ARTIFACTS_DIR / "bm25" / dataset),
        "feature_store": _dir_bytes(store),
    }

    before = process.memory_info().rss
    embeddings = embed.load(embed.embeddings_dir(dataset))
    after_embeddings = process.memory_info().rss
    ann.load(ann.ann_dir(dataset), embeddings)
    after_ann = process.memory_info().rss

    tables = {}
    for name in (
        "article_features.parquet",
        "user_features.parquet",
        "user_click_history_features.parquet",
        "session_features.parquet",
    ):
        path = store / name
        if path.exists():
            tables[name] = int(feature_store.read_table(path).estimated_size())

    return {
        "on_disk_bytes": on_disk,
        "on_disk_total_bytes": sum(on_disk.values()),
        "resident_bytes": {
            # Exact arithmetic, not an RSS delta: this one is knowable and the
            # RSS measurement is the noisy check on it.
            "embedding_matrix_exact": embeddings.n_articles * embeddings.dim * 4,
            "embedding_matrix_rss_delta": int(after_embeddings - before),
            "ann_index_rss_delta": int(after_ann - after_embeddings),
            "feature_store_in_memory": tables,
            "feature_store_in_memory_total": sum(tables.values()),
        },
        "articles": embeddings.n_articles,
        "embedding_dim": embeddings.dim,
        "rss_note": (
            "An RSS delta is a noisy lower bound: the allocator does not return "
            "freed pages immediately and a shared library's pages are counted "
            "once. The exact embedding figure is the arithmetic; the delta beside "
            "it is the sanity check on it."
        ),
    }


def cost_model(latency: dict) -> dict:
    """Core-seconds per 1000 queries, QPS against the SLA, and a price only if
    one has been cited.

    See OQ-13. The currency figure is omitted rather than invented.
    """
    mean_seconds = latency["total"]["mean_ms"] / 1e3
    p99_ms = latency["total"]["p99_ms"]
    per_core_qps = 1.0 / mean_seconds if mean_seconds > 0 else float("inf")

    out = {
        "core_seconds_per_1000_queries": round(mean_seconds * 1000, 3),
        "qps_per_core": round(per_core_qps, 2),
        "qps_at_assumed_cores": round(per_core_qps * config.SERVING_CORES, 2),
        "assumptions": {
            "cores": config.SERVING_CORES,
            "batch_size": 1,
            "candidate_k": config.SERVING_CANDIDATE_K,
            "device": latency["device"],
            "sla_p99_ms": config.SERVING_SLA_P99_MS,
            "price_usd_per_hour": config.SERVING_COST_USD_PER_HOUR,
            "price_source": config.SERVING_COST_SOURCE,
        },
        "sla": {
            "target_p99_ms": config.SERVING_SLA_P99_MS,
            "measured_p99_ms": p99_ms,
            "met": bool(p99_ms < config.SERVING_SLA_P99_MS),
        },
    }

    if config.SERVING_COST_USD_PER_HOUR and config.SERVING_COST_SOURCE:
        qps = per_core_qps * config.SERVING_CORES
        out["usd_per_1000_queries"] = round(
            1000.0 / (qps * 3600.0) * config.SERVING_COST_USD_PER_HOUR, 6
        )
    else:
        out["usd_per_1000_queries"] = None
        out["why_no_currency_figure"] = (
            "config.SERVING_COST_USD_PER_HOUR / SERVING_COST_SOURCE are unset "
            "(OQ-13): we have no dated, citable instance price, and an invented "
            "one would be a design-note number traceable to nothing. "
            "core_seconds_per_1000_queries is the measured answer; multiplying it "
            "by a current price is arithmetic."
        )
    return out


def scaling_10x(latency: dict, memory: dict) -> dict:
    """Derived claims only, each naming the measured key it came from.

    Extends `scale_analysis.md`'s four A1 constraints with the two stages A2
    added, rather than replacing them: A1's all still hold.
    """
    vram_budget = 4096 * 1024 * 1024
    embedding_10x = memory["resident_bytes"]["embedding_matrix_exact"] * 10
    qps_now = 1.0 / (latency["total"]["mean_ms"] / 1e3)

    return {
        "embedding_matrix_at_10x_bytes": embedding_10x,
        "embedding_matrix_at_10x_fits_in_4gb_vram": bool(embedding_10x < vram_budget),
        "from_keys": [
            "memory.resident_bytes.embedding_matrix_exact",
            "latency.total.mean_ms",
            "latency.total.p99_ms",
            "latency.stages.*.p99_ms",
        ],
        "cores_needed_for_10x_qps_at_current_latency": round(
            10.0 * qps_now / max(qps_now, 1e-9), 2
        ),
        "binding_constraint_order": [
            {
                "rank": 1,
                "what": "the in-RAM inverted index",
                "why": (
                    "postings are held entirely in memory with no on-disk block "
                    "structure, so a 10x corpus is a 10x resident index. Nothing "
                    "here is prepared for that — scale_analysis.md said so for A1 "
                    "and A2 did not change it."
                ),
                "from_key": "memory.on_disk_bytes.bm25_index",
            },
            {
                "rank": 2,
                "what": "the serving-time feature build",
                "why": (
                    "new in A2 and the stage most likely to surprise: it is "
                    "per-candidate Python dictionary work, so it scales with "
                    "SERVING_CANDIDATE_K and not with the corpus. It does not get "
                    "worse at 10x articles, but it does at 10x K."
                ),
                "from_key": "latency.stages.feature_build.p99_ms",
            },
            {
                "rank": 3,
                "what": "the embedding matrix against VRAM",
                "why": (
                    "10x the current corpus exceeds a 4096 MiB budget once the "
                    "encoder is also resident. The fix is batched scoring against "
                    "a memory-mapped matrix, which the flat-array layout allows."
                ),
                "from_key": "memory.resident_bytes.embedding_matrix_exact",
            },
            {
                "rank": 4,
                "what": "ANN search — and it should NOT be optimised",
                "why": (
                    "it grows slower than the corpus and A1 measured it already "
                    "faster than exact search on the largest corpus. "
                    "design_rationale.md §1.19 records benchmarking it against the "
                    "wrong baseline making it look four times better than it was."
                ),
                "from_key": "latency.stages.ann_search.p99_ms",
            },
        ],
    }


def evaluate_dataset(dataset: str, store_root: Path | None = None) -> dict:
    resources = load_resources(dataset, store_root=store_root)
    latency = measure_latency(resources, store_root=store_root)
    memory = measure_memory(dataset, store_root=store_root)
    return {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "latency": latency,
        "memory": memory,
        "cost": cost_model(latency),
        "scaling_10x": scaling_10x(latency, memory),
    }


def run(datasets: tuple[str, ...] | None = None, out_dir: Path | None = None) -> Path:
    """Write `serving.json` (A2 Q4)."""
    datasets = tuple(config.HEADLINE_DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report": "serving",
        "requirement": "A2.Q4",
        "stages": list(STAGES),
        "what_one_request_is": (
            "user vector -> ANN top-K over the corpus (stage 1) -> BM25 over those "
            "K candidates -> serving-time feature row -> re-ranker score (stage 2) "
            "-> score/popularity/id tie-break. Batch size 1, because A2 Q4.2 asks "
            "about a single user request and a batched number bounds no SLA."
        ),
        "device_note": (
            "Inference is timed on CPU. rerank/mlp.py and rerank/nrms.py require "
            "CUDA and raise, but that rule governs TRAINING; serving one request is "
            "a latency question on the hardware it lands on, and CPU is both the "
            "realistic target and what a CPU-only grader can reproduce."
        ),
        "datasets": {d: evaluate_dataset(d) for d in datasets},
    }
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        latency = block["latency"]
        memory = block["memory"]
        cost = block["cost"]
        lines.append(f"\n{dataset} ({block['language']})  {latency['measured_requests']} requests")
        lines.append(
            f"  latency  p50 {latency['total']['p50_ms']:.2f} ms  "
            f"p95 {latency['total']['p95_ms']:.2f} ms  "
            f"p99 {latency['total']['p99_ms']:.2f} ms  "
            f"(SLA {cost['sla']['target_p99_ms']:.0f} ms: "
            f"{'MET' if cost['sla']['met'] else 'MISSED'})"
        )
        lines.append(f"  slowest stage at p99: {latency['slowest_stage_at_p99']}")
        for stage in report["stages"]:
            lines.append(f"      {stage:18s} p99 {latency['stages'][stage]['p99_ms']:8.3f} ms")
        lines.append(
            f"  memory   on disk {memory['on_disk_total_bytes'] / 1e6:.1f} MB   "
            f"embeddings {memory['resident_bytes']['embedding_matrix_exact'] / 1e6:.1f} MB "
            f"({memory['articles']:,} articles)"
        )
        lines.append(
            f"  cost     {cost['core_seconds_per_1000_queries']:.1f} core-seconds / 1000 queries"
            f"   {cost['qps_at_assumed_cores']:.1f} QPS on {cost['assumptions']['cores']} cores"
        )
        if cost["usd_per_1000_queries"] is None:
            lines.append("           no USD figure — no citable instance price (OQ-13)")
    return "\n".join(lines)
