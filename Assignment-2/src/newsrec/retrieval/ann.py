"""Approximate nearest-neighbour search over article embeddings (q3_plan D4).

Requirements covered: Q3.3 — an ANN index whose top-K is *measured* against
exact search rather than trusted.

> **What "approximate" means here.** Exact search compares a query against every
> article vector: 65,238 dot products for MIND, and the answer is by definition
> correct. **HNSW** (Hierarchical Navigable Small World) instead pre-builds a
> graph in which each article links to a few of its near neighbours, and answers
> a query by walking that graph greedily towards the query. It touches a few
> hundred articles rather than all of them -- but a greedy walk can settle in a
> local pocket and miss a true neighbour. That miss rate is the "approximation",
> and Q3.3 exists because the only honest way to state it is to measure it.

## Inner product, because the vectors are already unit length

`embed.py` normalises every vector on write, so cosine similarity and the dot
product are the same number and the index can use `METRIC_INNER_PRODUCT`
directly. `build` re-checks the norms rather than assuming: an un-normalised
vector makes inner product rank by *magnitude* as well as direction, which
silently turns "most similar" into "most similar, but longer articles first".

## efSearch is a declared parameter, not a default

Measured on **our own vectors**, 2,000 queries at K=200, against exact search:

| efSearch | mind_small (65,238) | ebnerd_small (20,738) | ebnerd_demo (11,777) |
|---|---|---|---|
| 64 (library default) | 0.8646 **FAIL** | 0.8668 **FAIL** | 0.8934 **FAIL** |
| 128 | 0.9482 **FAIL** | 0.9553 ok | 0.9708 ok |
| **256 (pinned)** | **0.9830 ok** | **0.9893 ok** | **0.9944 ok** |
| 512 | 0.9954 ok | 0.9980 ok | 0.9992 ok |

Two things that only measurement could have told us. The library default fails
on all three. And **128 passes on both EB-NeRD corpora and fails on MIND** --
the most dangerous kind of value, since testing on the small corpus alone would
have blessed it. Agreement falls as the corpus grows, so MIND is the binding
constraint and 256 is chosen for it.

## The exact index is kept, and it is not slow

`ExactIndex` is the ground truth Q3.3 scores against, and it is deliberately a
numpy/BLAS matmul rather than `faiss.IndexFlatIP` -- both because an independent
implementation makes a disagreement unambiguously the ANN index's fault, and
because it is the *faster* exact baseline: faiss's flat index measured 0.54-0.58x
of numpy's throughput on this data. Benchmarking HNSW against the slower
baseline would have flattered it.

Against the honest baseline, search-only, 2,000 queries at K=200:

| | build | ANN search | exact (numpy) | speedup |
|---|---|---|---|---|
| ebnerd_demo | 8.3 s | 0.35 s | 0.34 s | **0.96x** |
| ebnerd_small | 20.4 s | 0.37 s | 0.58 s | **1.54x** |
| mind_small | 103.4 s | 0.74 s | 1.87 s | **2.55x** |

So on the smallest corpus the ANN index is **break-even**, and its advantage
grows with corpus size -- which is the honest shape of the result and the one
the design note should carry. It is not the ~4x the plan projected from a sweep
against faiss's flat index on 768-dimensional vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from newsrec import config
from newsrec.retrieval import embed

INDEX_FILE = "hnsw.faiss"
META_FILE = "meta.json"
REPORT_NAME = "ann_recall.json"


class AnnMismatch(RuntimeError):
    """An ANN index on disk does not match the embeddings it is used with.

    Raised rather than warned, for the reason `index.IndexMismatch` and
    `embed.EmbeddingMismatch` exist: a graph built over one set of vectors and
    queried with another returns row numbers that are valid integers pointing at
    the wrong articles. Every downstream shape check passes.
    """


@dataclass(frozen=True)
class SearchResult:
    """Top-K neighbours for a batch of queries."""

    #: (n_queries, k) row indices into the embedding matrix.
    indices: np.ndarray
    #: (n_queries, k) similarity scores, descending within each row.
    scores: np.ndarray


class ExactIndex:
    """Brute-force inner product — the ground truth Q3.3 measures against.

    Not a faiss index deliberately: a plain matrix product has no parameters, no
    graph and no approximation, so when it disagrees with HNSW the disagreement
    is unambiguously the ANN index's. Using `faiss.IndexFlatIP` as the reference
    would make both sides depend on the same library.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, len(self.vectors))
        sims = q @ self.vectors.T
        # argpartition then sort within the slice: the full sort of 65,238
        # columns per query is the dominant cost otherwise, and only the top-k
        # ordering is ever read.
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        idx = part[rows, order]
        return SearchResult(indices=idx, scores=sims[rows, idx])


class AnnIndex:
    """HNSW over unit-norm article vectors."""

    def __init__(self, index, article_ids: list[str], fingerprint: str, ef_search: int):
        self._index = index
        self.article_ids = article_ids
        self.fingerprint = fingerprint
        self.ef_search = ef_search
        self._index.hnsw.efSearch = ef_search

    @property
    def n_articles(self) -> int:
        return self._index.ntotal

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, indices = self._index.search(q, min(k, self.n_articles))
        return SearchResult(indices=indices, scores=scores)


def _check_normalised(vectors: np.ndarray) -> None:
    """Inner product is cosine ONLY for unit vectors. Verified, not assumed."""
    if len(vectors) == 0:
        return
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise AnnMismatch(
            f"vectors are not unit length (norms {norms.min():.4f}..{norms.max():.4f}). "
            f"Inner product would rank by magnitude as well as direction, so a longer "
            f"article would outrank a more similar one. embed.py normalises on write; "
            f"these did not come from there."
        )


def build(
    embeddings: embed.ArticleEmbeddings,
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
) -> AnnIndex:
    """Build the HNSW graph (q3_plan D4).

    Single-threaded on purpose. faiss parallelises graph construction across
    OpenMP threads, and the order in which concurrent inserts link a node is not
    fixed -- so a multi-threaded build produces a *different graph* on each run,
    and with it different approximate neighbours and a recall figure that moves
    when nothing changed. Q1.13 applies to every stage; `run()` reports the build
    cost so the price of this choice is visible rather than hidden.
    """
    import faiss

    m = config.ANN_HNSW_M if m is None else m
    ef_construction = (
        config.ANN_HNSW_EF_CONSTRUCTION if ef_construction is None else ef_construction
    )
    ef_search = config.ANN_HNSW_EF_SEARCH if ef_search is None else ef_search

    vectors = np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
    _check_normalised(vectors)

    previous = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        index = faiss.IndexHNSWFlat(embeddings.dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(vectors)
    finally:
        faiss.omp_set_num_threads(previous)

    return AnnIndex(
        index=index,
        article_ids=list(embeddings.article_ids),
        fingerprint=embed.fingerprint(embeddings.spec),
        ef_search=ef_search,
    )


def agreement_with_exact(
    ann: AnnIndex,
    exact: ExactIndex,
    queries: np.ndarray,
    k: int,
) -> float:
    """Fraction of the exact top-K that the ANN index also returned (Q3.3).

    Set overlap, not rank correlation. For candidate *generation* what matters
    is whether an article reaches the shortlist at all; where it sits inside the
    shortlist is the ranking stage's problem.
    """
    want = exact.search(queries, k).indices
    got = ann.search(queries, k).indices
    return float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(got, want, strict=True)]))


def _meta(ann: AnnIndex) -> dict:
    return {
        "ann_kind": "hnsw_flat_ip",
        "embedding_fingerprint": ann.fingerprint,
        "n_articles": ann.n_articles,
        "m": config.ANN_HNSW_M,
        "ef_construction": config.ANN_HNSW_EF_CONSTRUCTION,
        "ef_search": ann.ef_search,
    }


def save(ann: AnnIndex, dest: Path) -> Path:
    import faiss

    dest.mkdir(parents=True, exist_ok=True)
    faiss.write_index(ann._index, str(dest / INDEX_FILE))
    (dest / META_FILE).write_text(
        json.dumps(_meta(ann), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dest


def load(src: Path, embeddings: embed.ArticleEmbeddings | None = None) -> AnnIndex:
    """Read the graph back, refusing one built over different vectors.

    `embeddings` is the artefact this index is about to be used with. Checking
    the fingerprint here is what stops a graph built over MIND's vectors being
    queried alongside EB-NeRD's article ids -- the row numbers would all be in
    range, and every article returned would be the wrong one.
    """
    import faiss

    try:
        meta = json.loads((src / META_FILE).read_text(encoding="utf-8"))
        index = faiss.read_index(str(src / INDEX_FILE))
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise AnnMismatch(f"cannot read the ANN index at {src}: {exc}") from exc

    if embeddings is not None:
        want = embed.fingerprint(embeddings.spec)
        if meta.get("embedding_fingerprint") != want:
            raise AnnMismatch(
                f"ANN index at {src} was built over {meta.get('embedding_fingerprint')} "
                f"but {want} was supplied. Row numbers would point at the wrong articles."
            )
        if index.ntotal != len(embeddings.article_ids):
            raise AnnMismatch(
                f"ANN index at {src} holds {index.ntotal} vectors but {len(embeddings.article_ids)} "
                f"article ids were supplied."
            )
        article_ids = list(embeddings.article_ids)
    else:
        article_ids = []

    return AnnIndex(
        index=index,
        article_ids=article_ids,
        fingerprint=meta.get("embedding_fingerprint", ""),
        ef_search=meta.get("ef_search", config.ANN_HNSW_EF_SEARCH),
    )


def ann_dir(dataset: str) -> Path:
    """Where this dataset's ANN index lives."""
    return config.ARTIFACTS_DIR / "ann" / dataset


def sample_queries(
    embeddings: embed.ArticleEmbeddings,
    size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Query vectors for the Q3.3 measurement, drawn from the corpus itself.

    **A stated approximation.** The system's real queries are *user* vectors --
    means of up to 20 article vectors -- and those do not exist until step 5. A
    mean sits nearer the centre of the space than any single article, and central
    queries are the *harder* case for a greedy graph walk, because many
    candidates are near-equidistant. So this measurement is likely to
    **over**-state ANN agreement, and step 6 re-measures it with the real user
    vectors rather than carrying this number forward unchallenged.
    """
    size = config.RECALL_SAMPLE_SIZE if size is None else size
    seed = config.GLOBAL_SEED if seed is None else seed
    n = len(embeddings.vectors)
    if n <= size:
        return np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(n, size=size, replace=False))
    return np.ascontiguousarray(embeddings.vectors[picks], dtype=np.float32)


def evaluate_dataset(
    dataset: str,
    ef_values: tuple[int, ...] = (64, 128, 256, 512),
    sample_size: int | None = None,
    seed: int | None = None,
    embeddings: embed.ArticleEmbeddings | None = None,
) -> dict:
    """Build the index and measure its agreement with exact search (Q3.3)."""
    import time

    if embeddings is None:
        embeddings = embed.load(embed.embeddings_dir(dataset))

    # Reuse the persisted graph when one exists and matches these vectors.
    # Rebuilding costs 103 s for MIND, and `make` invokes the build stage and
    # the measurement stage separately -- doing it twice per `make semantic`
    # would double the stage cost to prove nothing. `load` enforces the
    # embedding fingerprint, so a stale graph is refused rather than reused.
    t0 = time.perf_counter()
    saved = ann_dir(dataset)
    try:
        ann = load(saved, embeddings=embeddings) if (saved / INDEX_FILE).exists() else None
    except AnnMismatch:
        ann = None  # stale or foreign; rebuild rather than refuse
    if ann is None:
        ann = build(embeddings)
    build_seconds = time.perf_counter() - t0

    exact = ExactIndex(embeddings.vectors)
    queries = sample_queries(embeddings, sample_size, seed)
    max_k = max(config.RECALL_K_VALUES)

    t1 = time.perf_counter()
    truth = exact.search(queries, max_k).indices
    exact_seconds = time.perf_counter() - t1

    sweep: dict[str, dict] = {}
    sweep_timings: dict[str, float] = {}
    for ef in ef_values:
        ann.ef_search = ef
        ann._index.hnsw.efSearch = ef
        # Time the ANN SEARCH ALONE. The first version of this timed
        # `agreement_with_exact`, which runs an exact search and a 2,000-iteration
        # set-overlap loop as well -- so the "ANN search" figure was really
        # exact + ANN + Python, and it reported the ANN index as ~2x SLOWER than
        # brute force when it is in fact ~2x faster. A speed number that silently
        # measures three things is worse than no speed number.
        t2 = time.perf_counter()
        got = ann.search(queries, max_k).indices
        ann_seconds = time.perf_counter() - t2

        recall = float(
            np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(got, truth, strict=True)])
        )
        sweep[str(ef)] = {
            "ef_search": ef,
            "recall_vs_exact": round(recall, 4),
            "meets_floor": bool(recall >= config.ANN_RECALL_FLOOR),
        }
        # Wall-clock goes to scale.json, NOT here. This report must be
        # byte-identical across runs or "the recall moved" stops being
        # informative. A `seconds` field lived in this dict for one commit and
        # made every rerun dirty the file while every recall value stayed
        # identical -- the fourth time this project has made the same mistake
        # (design_rationale 2.9, 1.19; exam_notes 5.1).
        sweep_timings[str(ef)] = round(ann_seconds, 3)

    ann.ef_search = config.ANN_HNSW_EF_SEARCH
    ann._index.hnsw.efSearch = config.ANN_HNSW_EF_SEARCH
    chosen = sweep[str(config.ANN_HNSW_EF_SEARCH)]

    return {
        "dataset": dataset,
        "n_articles": ann.n_articles,
        "dim": embeddings.dim,
        "queries": len(queries),
        "k": max_k,
        "embedding_fingerprint": ann.fingerprint,
        "m": config.ANN_HNSW_M,
        "ef_construction": config.ANN_HNSW_EF_CONSTRUCTION,
        "ef_search": config.ANN_HNSW_EF_SEARCH,
        "recall_vs_exact": chosen["recall_vs_exact"],
        "meets_floor": chosen["meets_floor"],
        "floor": config.ANN_RECALL_FLOOR,
        "query_source": "article_vectors",
        "sweep": sweep,
        "timing_seconds": {
            "build": round(build_seconds, 3),
            # Both are search-only, over the same queries at the same K, so the
            # ratio is meaningful. `exact_search` is a numpy/BLAS matmul, which
            # is the FASTEST honest baseline -- notably faster than
            # faiss.IndexFlatIP (measured 0.54-0.58x of numpy). Benchmarking
            # against the slower library baseline would have flattered HNSW.
            "exact_search_numpy": round(exact_seconds, 3),
            "ann_search": sweep_timings[str(config.ANN_HNSW_EF_SEARCH)],
            "speedup_vs_exact": round(
                exact_seconds / max(1e-9, sweep_timings[str(config.ANN_HNSW_EF_SEARCH)]), 2
            ),
            "by_ef_search": sweep_timings,
        },
        "_index": ann,
    }


def build_and_save(dataset: str) -> Path:
    """Build one dataset's graph and persist it. No measurement, no report.

    Separate from `run` so the Makefile can express the two as what they are:
    `artifacts/ann/<ds>/` is an artefact per dataset, `ann_recall.json` is one
    report over all of them. Collapsing them would mean either rebuilding every
    graph to write the report, or writing a report that covers whichever dataset
    ran last.
    """
    embeddings = embed.load(embed.embeddings_dir(dataset))
    return save(build(embeddings), ann_dir(dataset))


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    write_index: bool = True,
    **kwargs,
) -> Path:
    """Build every ANN index and write `reports/metrics/ann_recall.json` (Q3.3).

    **Fails** if any dataset misses `config.ANN_RECALL_FLOOR`. Q3.3 asks for the
    approximation to be measured; a measurement that cannot fail is a decoration,
    and a silently-degraded index would depress semantic recall in a way that
    looks like the embeddings underperforming.
    """
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, **kwargs) for d in datasets}

    if write_index:
        for dataset, m in measured.items():
            save(m["_index"], ann_dir(dataset))

    timings = {}
    for m in measured.values():
        timings[m["dataset"]] = m.pop("timing_seconds")
        m.pop("_index")

    failed = {d: m["recall_vs_exact"] for d, m in measured.items() if not m["meets_floor"]}

    report = {
        "metric": "ann_recall_vs_exact",
        "requirement": "Q3.3",
        "floor": config.ANN_RECALL_FLOOR,
        "index": "hnsw_flat_ip",
        "m": config.ANN_HNSW_M,
        "ef_construction": config.ANN_HNSW_EF_CONSTRUCTION,
        "ef_search": config.ANN_HNSW_EF_SEARCH,
        "all_datasets_meet_floor": not failed,
        "datasets": measured,
    }
    dest = out_dir / REPORT_NAME
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if out_dir == config.METRICS_DIR:
        scale_path = out_dir / "scale.json"
        scale = json.loads(scale_path.read_text()) if scale_path.exists() else {}
        scale.setdefault("ann", {}).update(timings)
        scale_path.write_text(json.dumps(scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failed:
        raise AnnMismatch(
            f"ANN recall below the {config.ANN_RECALL_FLOOR} floor (Q3.3): {failed}. "
            f"Raise ANN_HNSW_EF_SEARCH; the sweep in {config.repo_relative(dest)} shows "
            f"what each value buys. Refusing to proceed with a silently degraded index."
        )
    return dest


def format_table(report: dict) -> str:
    lines = [
        f"{'dataset':<14}{'articles':>10}{'efSearch':>10}{'recall':>9}{'floor':>8}",
        "-" * 51,
    ]
    for name, d in report["datasets"].items():
        for ef, s in sorted(d["sweep"].items(), key=lambda kv: int(kv[0])):
            mark = " *" if int(ef) == d["ef_search"] else ""
            lines.append(
                f"{name:<14}{d['n_articles']:>10,}{ef + mark:>10}"
                f"{s['recall_vs_exact']:>9.4f}{'ok' if s['meets_floor'] else 'FAIL':>8}"
            )
    lines.append("* = the pinned config.ANN_HNSW_EF_SEARCH")
    # The exact-search baseline and the speedup are deliberately NOT in this
    # table. They are one number per dataset, not one per efSearch row, and
    # padding them into every row invited exactly the confusion that produced
    # the wrong speedup in the first place (design_rationale 1.19). They live in
    # scale.json under `ann`, named `exact_search_numpy` and `ann_search`.
    return "\n".join(lines)
