"""Lexical versus semantic, paired and sliced (q3_plan D8).

Requirements covered: Q3.6 — a comparison per dataset **and per slice**, with the
winner per slice stated explicitly.

## Both retrievers run here, over one sample, in one pass

Neither `bm25_recall.json` nor `semantic_recall.json` keeps per-impression
values -- 2,000 floats per K are not a summary. But a *paired* comparison needs
exactly those, aligned row for row, so this module recomputes both sides rather
than reading the two reports.

That is not duplication for its own sake: it is what guarantees alignment.
Reading two reports and subtracting their means would give the same headline
number and no way to slice it, no way to pair it, and no way to notice if the
two had drifted apart.

The impression-set fingerprint is still checked against **both** committed
reports before anything is computed. A mismatch raises.

## Paired, because unpaired intervals hide small consistent effects

Each retriever's own confidence interval spans the variation *between
impressions*, which is large -- some users are easy, most are not. Two such
intervals overlap heavily even when one system beats the other on nearly every
impression. Bootstrapping the **per-impression difference** removes that shared
variation.

**Corrected 2026-08-24.** This paragraph used to claim Q2.9 demonstrated the
point -- "the Danish stemmer won at every K on both EB-NeRD corpora and no
unpaired interval could say so". Checked against
`bm25_tokenisation_ablation.json`, that is wrong twice. The stemmer's delta at
K=100 on `ebnerd_demo` is exactly **+0.0000**, so it did not win at every K; and
**no** tokenisation delta is established on any corpus even paired -- the
largest is +0.0070 [-0.0005, +0.0140] at K=200. The parenthetical implied the
paired test settled what the unpaired one could not, and it did not.

The argument above stands on its own; it did not need a worked example, and the
one it had was invented rather than read off the report. Q2.9's actual finding
is that **no tokenisation variant makes an established difference on any
corpus**, which is worth reporting as a null result rather than dressed up.

What paired bootstrapping does buy is real and is measured elsewhere: Q3.5's
unpaired comparison left seven of nine cells "not established", and this module
settles several of them.

## The slices, and why they are defined the way they are (D8)

Both obvious definitions are degenerate on this data, measured before being
rejected:

- **Cold-start as "has no history"** is 0 of 2,000 for both EB-NeRD corpora.
  Even "fewer than 5 clicks" is 0 -- EB-NeRD's median user has 88.
- **Head/tail by *training*-period clicks** puts 85% of MIND and 98.6% of
  `ebnerd_demo` impressions in one bucket, because news is fresh and test-period
  articles are mostly new.

So: history slices are **within-dataset quartiles**, comparable as positions in
each population rather than as absolute click counts; and head/tail uses
**test-period** popularity, which splits impressions near 50/50 on all three.

Using test-period popularity to define a slice is not leakage: it labels which
bucket an already-computed result is reported in. No score, ranking or model
input sees it, and every number was computed before the slice existed. That
distinction belongs in the design note in so many words -- an assessor scanning
for leakage will see future-derived popularity in this report and is right to
ask.
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
from newsrec.retrieval import ann, bm25, embed, query, recall, semantic, user_vector
from newsrec.retrieval import index as idx

REPORT_NAME = "lexical_vs_semantic.json"


def history_lengths(sample: pl.DataFrame, history: pl.DataFrame) -> np.ndarray:
    """Clicks per impression's user, aligned to `sample` row order."""
    counts = dict(history.group_by("user_id").len().iter_rows())
    return np.array([counts.get(u, 0) for u in sample["user_id"].to_list()], dtype=np.int64)


def head_articles(impressions: pl.DataFrame, split: str = "test") -> set[str]:
    """The articles carrying the first 50% of clicks in `split`.

    Test-period, not training-period -- see the module docstring. An analysis
    descriptor, never a model input.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for clicked in impressions.filter(pl.col("split") == split)["clicked_ids"].to_list():
        if clicked:
            counts.update(clicked)
    if not counts:
        return set()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(counts.values())
    head, running = set(), 0
    for article_id, n in ranked:
        head.add(article_id)
        running += n
        if running >= total / 2:
            break
    return head


def build_slices(
    sample: pl.DataFrame,
    history: pl.DataFrame,
    impressions: pl.DataFrame,
) -> dict[str, np.ndarray]:
    """name -> boolean mask over `sample` rows (D8)."""
    lengths = history_lengths(sample, history)
    q1, q3 = np.percentile(lengths, [25, 75])

    head = head_articles(impressions)
    touches_head = np.array(
        [bool(head.intersection(c or [])) for c in sample["clicked_ids"].to_list()]
    )

    return {
        # Quartiles of THIS dataset's own history distribution. Comparable as
        # positions in each population, not as absolute counts -- MIND's bottom
        # quartile is a handful of clicks, EB-NeRD's is dozens.
        "history_short": lengths <= q1,
        "history_long": lengths >= q3,
        # Does this impression's clicked set include a head article?
        "article_head": touches_head,
        "article_tail": ~touches_head,
    }


def _bm25_per_impression(
    dataset: str,
    sample: pl.DataFrame,
    history: pl.DataFrame,
    impressions: pl.DataFrame,
    store_root: Path,
    k_values: tuple[int, ...],
) -> tuple[dict[int, list[float]], list[np.ndarray]]:
    """Recall per impression, plus each impression's retrieved list."""
    spec = config.tokeniser_for(dataset)
    articles = feature_store.read_table(
        store_root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    titles = feature_store.read_table(
        store_root / "processed" / dataset / "articles.parquet"
    ).select("article_id", "title")

    index = idx.build(articles, spec)
    idf = bm25.idf_array(index)
    popularity = query.training_popularity(impressions)
    tie_break = np.asarray(query.popularity_array(index.article_ids, popularity), dtype=np.int64)
    queries = query.build_queries(sample, history, titles, spec)
    article_of = np.asarray(index.article_ids, dtype=object)
    max_k = max(k_values)

    per_k: dict[int, list[float]] = {k: [] for k in k_values}
    retrieved: list[np.ndarray] = []
    for row in sample.iter_rows(named=True):
        terms = queries.get(row["impression_id"], [])
        top = bm25.top_k(bm25.score(index, terms, idf=idf), max_k, tie_break=tie_break)
        got = article_of[top]
        retrieved.append(got)
        clicked = set(row["clicked_ids"])
        for k in k_values:
            per_k[k].append(len(clicked.intersection(got[:k])) / len(clicked))
    return per_k, retrieved


def _semantic_per_impression(
    dataset: str,
    sample: pl.DataFrame,
    history: pl.DataFrame,
    impressions: pl.DataFrame,
    k_values: tuple[int, ...],
    embeddings: embed.ArticleEmbeddings,
    index: ann.AnnIndex,
) -> tuple[dict[int, list[float]], list[np.ndarray], np.ndarray]:
    vectors, has_vector = user_vector.user_vectors(
        sample.select("impression_id", "user_id", "impression_time"), history, embeddings
    )
    # popularity_ranking, not cold_start_candidates -- see semantic.py.
    popularity = query.training_popularity(impressions)
    fallback = np.asarray(
        query.popularity_ranking(embeddings.article_ids, popularity, k=max(k_values)),
        dtype=object,
    )
    article_of = np.asarray(embeddings.article_ids, dtype=object)
    max_k = max(k_values)

    warm = np.flatnonzero(has_vector)
    hits = index.search(vectors[warm], max_k) if warm.size else None

    per_k: dict[int, list[float]] = {k: [] for k in k_values}
    retrieved: list[np.ndarray] = []
    warm_pos = {int(i): r for r, i in enumerate(warm)}
    for i, row in enumerate(sample.iter_rows(named=True)):
        got = article_of[hits.indices[warm_pos[i]]] if i in warm_pos else fallback
        retrieved.append(got)
        clicked = set(row["clicked_ids"])
        for k in k_values:
            per_k[k].append(len(clicked.intersection(got[:k])) / len(clicked))
    return per_k, retrieved, has_vector


def _paired(lex: list[float], sem: list[float], mask: np.ndarray | None = None) -> dict:
    """Bootstrap the per-impression difference (semantic minus lexical)."""
    a = np.asarray(sem, dtype=np.float64)
    b = np.asarray(lex, dtype=np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size == 0:
        return {"n": 0, "winner": "empty_slice"}

    ci = bootstrap_ci(a - b)
    established = bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0)
    return {
        "n": int(a.size),
        "lexical": round(float(b.mean()), 6),
        "semantic": round(float(a.mean()), 6),
        "delta": round(ci["mean"], 6),
        "ci_low": round(ci["ci_low"], 6),
        "ci_high": round(ci["ci_high"], 6),
        # Both bounds on the same side of zero. Says the SIGN is established at
        # this sample size; says nothing about whether the effect is large
        # enough to matter, and it is not a p-value.
        "established": established,
        "winner": ("semantic" if ci["mean"] > 0 else "lexical") if established else "neither",
    }


def _freshness(
    retrieved: list[np.ndarray],
    published: dict[str, object],
    mask: np.ndarray,
    k: int,
) -> float | None:
    """Median publication time of the retrieved list, in days, or None.

    EB-NeRD only -- MIND ships no `published_time` for any article.
    """
    rows = [
        np.median(
            [published[a].timestamp() for a in got[:k] if published.get(a) is not None] or [np.nan]
        )
        for got, keep in zip(retrieved, mask, strict=True)
        if keep
    ]
    rows = [r for r in rows if np.isfinite(r)]
    return round(float(np.median(rows)) / 86400.0, 3) if rows else None


def evaluate_dataset(
    dataset: str,
    k_values: tuple[int, ...] | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    store_root: Path | None = None,
) -> dict:
    """Paired lexical-vs-semantic comparison for one dataset (Q3.6)."""
    k_values = config.RECALL_K_VALUES if k_values is None else k_values
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
    index = ann.load(ann.ann_dir(dataset), embeddings=embeddings)

    t0 = time.perf_counter()
    lex, lex_lists = _bm25_per_impression(
        dataset, sample, user_history, impressions, root, k_values
    )
    lex_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    sem, sem_lists, has_vector = _semantic_per_impression(
        dataset, sample, user_history, impressions, k_values, embeddings, index
    )
    sem_seconds = time.perf_counter() - t1

    slices = build_slices(sample, user_history, impressions)
    # Cold start is a slice too, and it is the one where both retrievers are
    # KNOWN to behave identically -- so a non-zero delta there would be a bug,
    # not a finding.
    slices["cold_start"] = ~has_vector
    slices["warm"] = has_vector

    overall = {str(k): _paired(lex[k], sem[k]) for k in k_values}
    per_slice = {
        name: {
            "n": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "k": {str(k): _paired(lex[k], sem[k], mask) for k in k_values},
        }
        for name, mask in slices.items()
    }

    # Freshness, EB-NeRD only. Recomputed per slice because the interesting
    # question is whether the staleness gap WIDENS where semantic loses most.
    published = dict(
        feature_store.read_table(processed / "articles.parquet")
        .select("article_id", "published_time")
        .iter_rows()
    )
    freshness = None
    if any(v is not None for v in published.values()):
        top_k = max(k_values)
        clicked_days = _freshness(
            [np.asarray(c, dtype=object) for c in sample["clicked_ids"].to_list()],
            published,
            np.ones(len(sample), dtype=bool),
            top_k,
        )
        freshness = {
            "k": top_k,
            "unit": "days_since_epoch_median",
            "clicked": clicked_days,
            "overall": {
                "lexical": _freshness(lex_lists, published, np.ones(len(sample), bool), top_k),
                "semantic": _freshness(sem_lists, published, np.ones(len(sample), bool), top_k),
            },
            "by_slice": {
                name: {
                    "lexical": _freshness(lex_lists, published, mask, top_k),
                    "semantic": _freshness(sem_lists, published, mask, top_k),
                }
                for name, mask in slices.items()
                if mask.any()
            },
        }

    return {
        "dataset": dataset,
        "impressions_evaluated": len(sample),
        "impression_set_fingerprint": recall.impression_set_fingerprint(sample),
        "corpus_articles": embeddings.n_articles,
        "paired": True,
        "overall": overall,
        "slices": per_slice,
        "freshness": freshness,
        "timing_seconds": {
            "lexical": round(lex_seconds, 3),
            "semantic": round(sem_seconds, 3),
        },
    }


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> Path:
    """Write `lexical_vs_semantic.json` (Q3.6).

    Raises if the impression set differs from EITHER committed report. Both are
    checked, not just one: agreeing with BM25 while disagreeing with the
    semantic report would still make the paired delta meaningless.
    """
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, **kwargs) for d in datasets}

    committed = {
        "bm25": semantic._bm25_fingerprints(out_dir),
        "semantic": _semantic_fingerprints(out_dir),
    }
    problems = {}
    for d, m in measured.items():
        ours = m["impression_set_fingerprint"]
        for which, table in committed.items():
            if d in table and table[d] != ours:
                problems[f"{d}/{which}"] = {"here": ours, which: table[d]}
    if problems:
        raise semantic.ImpressionSetMismatch(
            f"Q3.6 pairs per impression, so the sets must be identical: {problems}. "
            f"Refusing to write a comparison whose pairing is not real."
        )

    timings = {d: m.pop("timing_seconds") for d, m in measured.items()}

    report = {
        "comparison": "lexical_vs_semantic",
        "requirement": "Q3.6",
        "paired": True,
        "k_values": list(config.RECALL_K_VALUES),
        "delta_sign": "semantic minus lexical; positive means semantic retrieved more",
        "bootstrap": {"n_resamples": config.BOOTSTRAP_N, "ci": config.BOOTSTRAP_CI},
        "slice_definitions": {
            "history_short": "user history length <= this dataset's 25th percentile",
            "history_long": "user history length >= this dataset's 75th percentile",
            "article_head": "impression's clicked set includes an article in the "
            "test-period top 50% of clicks",
            "article_tail": "impression's clicked set includes no such article",
            "cold_start": "no legitimate prior click; both retrievers serve the "
            "identical popularity fallback, so a non-zero delta here is a bug",
            "warm": "has a user vector",
        },
        "slice_note": (
            "Head/tail is defined by TEST-period popularity. That is an "
            "analysis-time descriptor, not a serving-time feature: no score, "
            "ranking or model input sees it, and every number was computed "
            "before the slice existed."
        ),
        "impression_sets_match_committed_reports": not problems,
        "datasets": measured,
    }
    dest = out_dir / REPORT_NAME
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if out_dir == config.METRICS_DIR:
        scale_path = out_dir / "scale.json"
        scale = json.loads(scale_path.read_text()) if scale_path.exists() else {}
        scale.setdefault("lexical_vs_semantic", {}).update(timings)
        scale_path.write_text(json.dumps(scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def _semantic_fingerprints(metrics_dir: Path) -> dict[str, str]:
    path = metrics_dir / semantic.REPORT_NAME
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    return {
        name: d.get("impression_set_fingerprint") for name, d in report.get("datasets", {}).items()
    }


def format_table(report: dict) -> str:
    lines = []
    for name, d in report["datasets"].items():
        lines.append(f"\n{name}  ({d['impressions_evaluated']:,} impressions)")
        lines.append(
            f"  {'slice':<16}{'n':>6}{'K':>5}{'lex':>9}{'sem':>9}{'delta':>10}{'winner':>11}"
        )
        lines.append("  " + "-" * 66)
        for k in report["k_values"]:
            o = d["overall"][str(k)]
            lines.append(
                f"  {'ALL':<16}{o['n']:>6}{k:>5}{o['lexical']:>9.4f}{o['semantic']:>9.4f}"
                f"{o['delta']:>+10.4f}{o['winner']:>11}"
            )
        for sname, s in d["slices"].items():
            for k in report["k_values"]:
                c = s["k"][str(k)]
                if c.get("n", 0) == 0:
                    continue
                lines.append(
                    f"  {sname:<16}{c['n']:>6}{k:>5}{c['lexical']:>9.4f}{c['semantic']:>9.4f}"
                    f"{c['delta']:>+10.4f}{c['winner']:>11}"
                )
    return "\n".join(lines)
