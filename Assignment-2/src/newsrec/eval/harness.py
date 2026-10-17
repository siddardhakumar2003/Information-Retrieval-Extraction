"""Run the full metric set over every (method, dataset) pair (Q4.8).

Design: agent_docs/q4_plan.md D1, D2, D6

## What is ranked, and why it is not the corpus

Q2 and Q3 rank the whole corpus and ask whether the clicked article is in the
top K. Q4 ranks **the impression's own candidate list** -- the articles the
publisher actually showed -- and asks how well the clicked ones are ordered
within it.

That is forced rather than chosen. AUC needs positives and negatives inside one
impression; recall at K=200 over the corpus is about 0.03, so a corpus shortlist
contains no positive at all for roughly 97% of impressions and AUC would be
undefined for almost every row. See metrics.py for the consequence that belongs
in the design note.

## Two scorers, one interface, and the tie-break is the point

Each scorer returns one score per candidate. Both then go through
`_serving_order`, which converts scores to the ranking a served system would
actually produce: BM25 score first, then training-period popularity, then
`article_id`. That is exactly `bm25.top_k`'s tie-break (q2_plan D8), so an
impression scored here ranks the way the retriever would rank it.

The tie-break is load-bearing rather than cosmetic. A cold-start user has no
query, so **every candidate scores exactly 0** and the entire ranking is decided
by the tie-break. Without it those impressions would be ordered by whatever
sequence the publisher happened to write the candidates in, and their AUC would
measure the publisher's list order rather than our system. With it, both
retrievers fall back to the same popularity prior and are directly comparable --
`compare.py` established that the two cold paths must agree, and this keeps that
true here.

Metrics therefore see a strict ranking with no ties. `metrics.auc` still
implements average-rank tie handling because its sklearn oracle requires it, but
the harness does not exercise that path.

## Missing articles

A candidate absent from the embedding table cannot be scored semantically. It
gets a sentinel below any achievable cosine, so it sorts last among genuinely
scored candidates rather than being dropped -- dropping it would change the
denominator of every metric and quietly make the two methods score different
lists.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import beyond_accuracy as ba
from newsrec.eval import bootstrap, metrics, slices
from newsrec.eval.bootstrap import MIN_TRUSTWORTHY_N, bootstrap_ci
from newsrec.retrieval import bm25, embed, query, recall, semantic, user_vector
from newsrec.retrieval import index as idx

REPORT_NAME = "eval.json"

#: Below any achievable cosine similarity (which is bounded by -1), so an
#: unembeddable candidate ranks last among scored ones without colliding with a
#: real score.
UNSCORABLE = -2.0

METHODS: tuple[str, ...] = ("bm25", "semantic")


def _serving_order(
    scores: np.ndarray,
    popularity: np.ndarray,
    article_ids: list[str],
) -> np.ndarray:
    """Candidate indices best-first under the served tie-break (q2_plan D8).

    `np.lexsort` orders by the LAST key first, so the keys are listed in reverse
    priority. Negations because lexsort is ascending and we want descending
    score and descending popularity, with `article_id` ascending last.
    """
    return np.lexsort((np.asarray(article_ids, dtype=object), -popularity, -scores))


def _ranked_scores(order: np.ndarray) -> np.ndarray:
    """A strictly decreasing score reproducing `order`.

    The metrics take scores, not orders. Handing them `-rank` gives a ranking
    with no ties, which is what a served system produces: the tie-break has
    already decided every comparison.
    """
    out = np.empty(order.size, dtype=np.float64)
    out[order] = -np.arange(order.size, dtype=np.float64)
    return out


def _bm25_scores(
    sample: pl.DataFrame,
    history: pl.DataFrame,
    titles: pl.DataFrame,
    index: idx.InvertedIndex,
    spec: config.TokeniserSpec,
) -> list[np.ndarray]:
    """One score array per impression, aligned to its `candidate_ids`."""
    idf = bm25.idf_array(index)
    doc_of = {a: i for i, a in enumerate(index.article_ids)}
    queries = query.build_queries(sample, history, titles, spec)

    out: list[np.ndarray] = []
    for row in sample.iter_rows(named=True):
        terms = queries.get(row["impression_id"], [])
        candidates = row["candidate_ids"]
        if not terms:
            # Cold start: every candidate scores 0 and the popularity tie-break
            # decides the whole ranking. Not a special case in the code -- this
            # is what `bm25.score` returns for an empty query -- but it is the
            # case the tie-break exists for.
            out.append(np.zeros(len(candidates), dtype=np.float64))
            continue
        corpus_scores = bm25.score(index, terms, idf=idf)
        # A candidate the corpus does not carry scores 0, the same as a
        # candidate that matched no query term. Both are "no evidence".
        out.append(
            np.array(
                [corpus_scores[doc_of[a]] if a in doc_of else 0.0 for a in candidates],
                dtype=np.float64,
            )
        )
    return out


def _semantic_scores(
    sample: pl.DataFrame,
    history: pl.DataFrame,
    embeddings: embed.ArticleEmbeddings,
) -> tuple[list[np.ndarray], np.ndarray]:
    """One cosine-score array per impression, plus the warm mask.

    Cold users get all-zero scores so the popularity tie-break decides their
    ranking, identically to BM25's cold path.
    """
    vectors, has_vector = user_vector.user_vectors(
        sample.select("impression_id", "user_id", "impression_time"), history, embeddings
    )
    row_of = ba.row_index(embeddings)

    out: list[np.ndarray] = []
    for i, row in enumerate(sample.iter_rows(named=True)):
        candidates = row["candidate_ids"]
        if not has_vector[i]:
            out.append(np.zeros(len(candidates), dtype=np.float64))
            continue
        rows = np.array([row_of.get(a, -1) for a in candidates], dtype=np.int64)
        known = rows >= 0
        scores = np.full(len(candidates), UNSCORABLE, dtype=np.float64)
        if known.any():
            # Both sides unit-norm, so the dot product is the cosine.
            scores[known] = embeddings.vectors[rows[known]] @ vectors[i]
        out.append(scores)
    return out, has_vector


def _per_impression(
    sample: pl.DataFrame,
    scores: list[np.ndarray],
    popularity: dict[str, int],
    embeddings: embed.ArticleEmbeddings,
    p_by_article: dict[str, float],
    p_unseen: float,
) -> dict:
    """Every per-impression value the report needs, in one pass over the sample.

    Returns arrays aligned to `sample` rows, with `None` where a metric is
    undefined for that impression, plus the degenerate counts and the top-N
    lists coverage is computed from.
    """
    row_of = ba.row_index(embeddings)
    ranking: dict[str, list[float | None]] = {m: [] for m in metrics.RANKING_METRICS}
    diversity: list[float | None] = []
    novelty: list[float | None] = []
    novelty_unseen_fraction: list[float | None] = []
    top_lists: list[list[str]] = []
    degenerate: dict[str, int] = {}

    for i, row in enumerate(sample.iter_rows(named=True)):
        candidates = list(row["candidate_ids"])
        clicked = set(row["clicked_ids"])
        labels = np.array([1 if a in clicked else 0 for a in candidates], dtype=np.int64)

        pop = np.array([popularity.get(a, 0) for a in candidates], dtype=np.float64)
        order = _serving_order(scores[i], pop, candidates)
        ranked_ids = [candidates[j] for j in order]
        top = ranked_ids[: ba.TOP_N]
        top_lists.append(top)

        got = metrics.ranking_metrics(labels, _ranked_scores(order))
        if got is None:
            reason = metrics.degenerate_reason(labels)
            degenerate[reason] = degenerate.get(reason, 0) + 1
            for name in ranking:
                ranking[name].append(None)
        else:
            for name in ranking:
                ranking[name].append(got[name])

        diversity.append(ba.diversity(top, embeddings, row_of=row_of))
        novelty.append(ba.novelty(top, p_by_article, p_unseen))
        novelty_unseen_fraction.append(ba.novelty_unseen_fraction(top, popularity))

    return {
        "ranking": ranking,
        "diversity": diversity,
        "novelty": novelty,
        "novelty_unseen_fraction": novelty_unseen_fraction,
        "top_lists": top_lists,
        "degenerate": degenerate,
    }


def _summarise(values: list[float | None], mask: np.ndarray | None = None) -> dict:
    """`bootstrap_ci` over the defined values, with the undefined ones counted.

    `n_excluded` is reported rather than folded away: a metric averaged over 900
    of 2,000 impressions is a different claim from one averaged over all of
    them, and the reader cannot tell from the mean alone.
    """
    rows = values if mask is None else [v for v, keep in zip(values, mask) if keep]
    defined = [v for v in rows if v is not None]
    out = bootstrap_ci(defined)
    out = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}
    out["n_excluded"] = len(rows) - len(defined)
    return out


def auc_verdict(auc: dict) -> str:
    """Whether this AUC is distinguishable from random guessing.

    0.5 is the AUC of a coin flip, so an interval containing 0.5 is a system
    that has not been shown to rank better than chance. Recorded as a field
    rather than left to the reader, because a mean of 0.4956 printed beside an
    interval of [0.4822, 0.5105] reads as "nearly 0.5, close enough" unless the
    report says outright that it is not distinguishable from random.

    Note what this does NOT say. "Better than random" is a statement about the
    sign at this sample size, not about the effect being large or useful.
    """
    if not math.isfinite(auc["mean"]):
        return "undefined"
    if auc["ci_low"] > 0.5:
        return "better_than_random"
    if auc["ci_high"] < 0.5:
        return "worse_than_random"
    return "indistinguishable_from_random"


def _metric_block(
    per_impression: dict,
    catalogue_size: int,
    mask: np.ndarray | None = None,
) -> dict:
    """Every metric for one (method, dataset, slice) cell."""
    block = {name: _summarise(values, mask) for name, values in per_impression["ranking"].items()}
    block["auc"]["verdict"] = auc_verdict(block["auc"])
    block["diversity"] = _summarise(per_impression["diversity"], mask)
    block["novelty"] = _summarise(per_impression["novelty"], mask)
    block["novelty_unseen_fraction"] = _summarise(per_impression["novelty_unseen_fraction"], mask)

    lists = per_impression["top_lists"]
    if mask is not None:
        lists = [row for row, keep in zip(lists, mask) if keep]
    block["coverage"] = ba.coverage_ci(lists, catalogue_size)
    block["coverage"] = {
        k: (round(v, 6) if isinstance(v, float) else v) for k, v in block["coverage"].items()
    }
    block["coverage"]["distinct_articles"] = len({a for row in lists for a in row})

    n = len(lists)
    if 0 < n < MIN_TRUSTWORTHY_N:
        # Measured: a 95% interval over a sparse metric covers ~0.85 at n=100.
        # The numbers stay in the report; the confidence claim does not.
        block["ci_warning"] = (
            f"n={n} is below MIN_TRUSTWORTHY_N={MIN_TRUSTWORTHY_N}; "
            "the bootstrap interval here is narrower than its nominal 95%"
        )
    return block


#: A slice holding less than this share of the sample, or more than its
#: complement, is flagged in the report. Measured, the Q4.5-mandated
#: training-period head/tail cut lands at 0.004 -- 8 impressions of 2,000 -- so
#: this fires on exactly the slice the assignment requires us to report.
DEGENERATE_SLICE_SHARE: float = 0.05


def _slice_block(
    name: str,
    mask: np.ndarray,
    per_impression: dict,
    catalogue_size: int,
) -> dict:
    """One slice's metrics, with its share and any warning attached.

    The warning sits inside the slice next to its own numbers rather than in a
    note at the top of the report, because a reader who scrolls to
    `article_head_train` and finds an AUC of 0.85 needs to see, in that same
    object, that it was computed over 8 impressions.
    """
    share = float(mask.mean())
    block = {
        "n": int(mask.sum()),
        "share": round(share, 4),
        **_metric_block(per_impression, catalogue_size, mask),
    }
    if 0 < share < DEGENERATE_SLICE_SHARE or 1.0 - DEGENERATE_SLICE_SHARE < share < 1.0:
        block["degenerate_slice"] = (
            f"this slice holds {share:.1%} of the sample; its complement holds "
            f"{1 - share:.1%}. A comparison between the two is not a comparison."
        )
    if not mask.any():
        block["empty_slice"] = "no impressions fall in this slice on this dataset"
    return block


def evaluate_dataset(
    dataset: str,
    methods: tuple[str, ...] = METHODS,
    sample_size: int | None = None,
    seed: int | None = None,
    store_root: Path | None = None,
) -> dict:
    """Ranking and beyond-accuracy metrics for one dataset (Q4.1-Q4.6)."""
    sample_size = config.RECALL_SAMPLE_SIZE if sample_size is None else sample_size
    seed = config.GLOBAL_SEED if seed is None else seed
    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset

    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")
    sample = recall.sample_impressions(impressions, "test", sample_size, seed)
    users = sample["user_id"].unique().to_list()
    user_history = history.filter(pl.col("user_id").is_in(users))

    embeddings = embed.load(embed.embeddings_dir(dataset))
    popularity = query.training_popularity(impressions)
    p_by_article, p_unseen = ba.smoothed_probabilities(popularity, embeddings.n_articles)

    masks = slices.build(sample, user_history, impressions)
    slices.partition_check(masks, len(sample))

    spec = config.tokeniser_for(dataset)
    scored: dict[str, list[np.ndarray]] = {}
    timings: dict[str, float] = {}
    if "bm25" in methods:
        articles = feature_store.read_table(
            root / "feature_store" / dataset / "article_features.parquet"
        ).select("article_id", "text")
        titles = feature_store.read_table(processed / "articles.parquet").select(
            "article_id", "title"
        )
        index = idx.build(articles, spec)
        t0 = time.perf_counter()
        scored["bm25"] = _bm25_scores(sample, user_history, titles, index, spec)
        timings["bm25"] = time.perf_counter() - t0
    if "semantic" in methods:
        t0 = time.perf_counter()
        scored["semantic"], _ = _semantic_scores(sample, user_history, embeddings)
        timings["semantic"] = time.perf_counter() - t0

    out_methods = {}
    for method, scores in scored.items():
        per_impression = _per_impression(
            sample, scores, popularity, embeddings, p_by_article, p_unseen
        )
        out_methods[method] = {
            "overall": _metric_block(per_impression, embeddings.n_articles),
            "degenerate_impressions": per_impression["degenerate"],
            "slices": {
                name: _slice_block(name, mask, per_impression, embeddings.n_articles)
                for name, mask in masks.items()
            },
        }

    candidates = sample["candidate_ids"].list.len().to_numpy()
    return {
        "dataset": dataset,
        "impressions_evaluated": len(sample),
        "impression_set_fingerprint": recall.impression_set_fingerprint(sample),
        "catalogue_articles": embeddings.n_articles,
        "candidates_per_impression": {
            "median": int(np.median(candidates)),
            "p95": int(np.percentile(candidates, 95)),
            "max": int(candidates.max()),
        },
        "novelty_smoothing": {
            "scheme": "add_one",
            "vocabulary": embeddings.n_articles,
            "training_clicks": int(sum(popularity.values())),
        },
        "methods": out_methods,
        "timing_seconds": {k: round(v, 3) for k, v in timings.items()},
    }


def run(
    datasets: tuple[str, ...] | None = None,
    methods: tuple[str, ...] = METHODS,
    out_dir: Path | None = None,
    **kwargs,
) -> Path:
    """Write `eval.json` (Q4.8).

    Refuses to write unless every dataset's impression set matches the one Q2.8
    committed. Q4 does not pair per impression the way Q3.6 does, but a reader
    comparing `eval.json`'s AUC against `bm25_recall.json`'s recall is entitled
    to assume both describe the same impressions.
    """
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, methods=methods, **kwargs) for d in datasets}

    committed = semantic._bm25_fingerprints(out_dir)
    problems = {
        d: {"here": m["impression_set_fingerprint"], "bm25_recall": committed[d]}
        for d, m in measured.items()
        if d in committed and committed[d] != m["impression_set_fingerprint"]
    }
    if problems:
        raise semantic.ImpressionSetMismatch(
            f"Q4 must describe the same impressions as Q2.8: {problems}."
        )

    timings = {d: m.pop("timing_seconds") for d, m in measured.items()}

    report = {
        "report": "eval",
        "requirement": "Q4.1-Q4.8",
        "ranked_pool": "the impression's own candidate_ids, not the corpus",
        "pool_note": (
            "recall@K in bm25_recall.json / semantic_recall.json measures candidate "
            "generation over the whole corpus; AUC/MRR/nDCG here measure ranking "
            "within the supplied candidate list. Different questions."
        ),
        "tie_break": "bm25 or cosine score, then training-period popularity, then article_id",
        "bootstrap": {
            "n_resamples": config.BOOTSTRAP_N,
            "ci": config.BOOTSTRAP_CI,
            "resampled_over": "impressions",
            "coverage_validation": "see bootstrap_coverage below (Q4.7)",
            "min_trustworthy_n": MIN_TRUSTWORTHY_N,
        },
        "slice_note": (
            "article_head/article_tail use TEST-period popularity (q3_plan D8) and are "
            "analysis descriptors, never model inputs -- every number was computed "
            "before the slice existed, and no score, ranking or feature sees them."
        ),
        "training_period_head_tail_warning": (
            "article_head_train/article_tail_train are the Q4.5-compliant cut, computed "
            "from training-period clicks only. MEASURED, THEY ARE DEGENERATE: the head "
            "bucket holds 8-9 of 2,000 impressions (0.4%) against 99.6% in the tail, "
            "because news is fresh and test-period articles were mostly never clicked "
            "during training. Every figure in those two slices carries a ci_warning and "
            "must not be read as a head-versus-tail comparison -- MIND's "
            "article_head_train AUC of 0.85 is computed over 8 impressions. They are "
            "reported because Q4.5 requires a training-period cut, and labelled "
            "unreliable because that is what the data shows."
        ),
        "training_period_slices": list(slices.TRAINING_PERIOD_SLICES),
        "test_period_slices": list(slices.TEST_PERIOD_SLICES),
        "coverage_note": (
            "coverage is a property of the whole run, not a per-impression value: its "
            "mean is the sample's actual coverage and its interval comes from "
            "resampling impressions and recomputing the union, which is biased low."
        ),
        # Q4.7. Written here rather than left in a test docstring: the design
        # note quotes these, and a number quoted from an assertion is not
        # regenerable. The header above promised this key before it existed.
        "bootstrap_coverage": bootstrap.validate_coverage(trials=300, n_resamples=200),
        "datasets": measured,
    }

    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    scale = out_dir / "scale.json"
    if scale.exists():
        blob = json.loads(scale.read_text(encoding="utf-8"))
        blob.setdefault("eval", {}).update({d: t for d, t in timings.items()})
        scale.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    """A readable summary of `eval.json` for the terminal."""
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        lines.append(
            f"\n{dataset}  n={block['impressions_evaluated']}  "
            f"candidates median={block['candidates_per_impression']['median']}"
        )
        header = f"  {'method':10s} " + " ".join(
            f"{m:>18s}" for m in (*metrics.RANKING_METRICS, "diversity", "novelty", "coverage")
        )
        lines.append(header)
        for method, m in sorted(block["methods"].items()):
            cells = []
            for name in (*metrics.RANKING_METRICS, "diversity", "novelty", "coverage"):
                got = m["overall"][name]
                cells.append(f"{got['mean']:.4f} +-{(got['ci_high'] - got['ci_low']) / 2:.4f}")
            lines.append(f"  {method:10s} " + " ".join(f"{c:>18s}" for c in cells))
    return "\n".join(lines)
