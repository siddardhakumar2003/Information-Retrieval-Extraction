"""The MIND hybrid submission: blend lexical into semantic, then rank (Q5.1).

Design: agent_docs/q5_plan.md; blend weight from `reports/metrics/hybrid_ablation.json`

## Why MIND gets a blend and EB-NeRD does not

`eval/hybrid.py` swept the blend weight on the **validation** split and applied
the winner to test, so the choice and the reported number come from different
data. MIND selected alpha=0.30 and kept an established gain on test
(0.6067 vs 0.5985, +0.0082 [+0.0027, +0.0136]). Both EB-NeRD corpora
independently selected **alpha=0.0** -- their val curves fall monotonically from
zero, so every amount of lexical signal makes them worse.

So this module exists for MIND. Running it on a Danish corpus would be
overriding a measurement with a preference.

## The tie-break differs from the offline one, deliberately

Offline, ties break on `(score, training-period popularity, article_id)`
(q2_plan D8). `MINDlarge_test` has **no labels**, so there is no training-period
popularity to compute from it and none is invented. Ties break on
`(score, article_id)` instead.

This matters for exactly one case and it is worth naming: a row with no history
scores every candidate 0, so its entire ranking is decided by the tie-break --
which makes it identical to the `article_id` baseline that scored 0.5187. That
is the honest fallback for a user we know nothing about, and `run` reports how
many rows it applies to rather than leaving it implicit.

## One pass, no caching of user vectors

`score_fast` pools one vector per USER because EB-NeRD's test set ships one
fixed history per user. MIND's test file carries a history **per row**, so there
is no such saving: two rows from the same user may legitimately differ. Pooling
inline costs ~50 microseconds against the ~334 microseconds BM25 already needs,
so caching would add a correctness risk to save nothing measurable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from newsrec import config
from newsrec.eval.hybrid import blend
from newsrec.retrieval import embed, user_vector
from newsrec.submit import bm25_fast, encode_test, mind_format, mind_test
from newsrec.submit import format as fmt

#: Chosen on the VALIDATION split by `eval/hybrid.py::select_on_val`, never on
#: the split any number is reported from. Written here as a literal rather than
#: read from the report so a submission cannot silently change when the report
#: is regenerated; `test_hybrid_predict.py` asserts the two agree.
MIND_ALPHA = 0.30


def user_vector_for(
    history_ids: list[str],
    embeddings: embed.ArticleEmbeddings,
    row_of_article: dict[str, int],
    limit: int | None = None,
) -> np.ndarray | None:
    """Pooled unit-norm vector for one row's history, or None if unusable.

    `None` rather than a zero vector: a zero vector would score every candidate
    0.0, which is indistinguishable from a real cosine of zero, and would quietly
    put a cold row on the same footing as a warm one whose interests happen to be
    orthogonal to everything.
    """
    limit = config.QUERY_HISTORY_LIMIT if limit is None else limit
    if not history_ids:
        return None
    # Most recent first, capped — the same rule as the offline path. MIND's raw
    # history is oldest-first, so the recent end is the tail.
    rows = [row_of_article[a] for a in history_ids[-limit:][::-1] if a in row_of_article]
    if not rows:
        return None
    return user_vector.pool(embeddings.vectors[rows])


def _order(scores: np.ndarray, article_ids: list[str]) -> np.ndarray:
    """Candidate indices best-first, ties broken by `article_id` ascending."""
    return np.lexsort((np.asarray(article_ids, dtype=object), -scores))


def score_row(
    history_ids: list[str],
    candidates: list[str],
    *,
    scorer: bm25_fast.CandidateScorer,
    vocab: dict[str, int],
    idf: np.ndarray,
    titles: dict[str, list[str]],
    embeddings: embed.ArticleEmbeddings,
    doc_of: dict[str, int],
    row_of_article: dict[str, int],
    alpha: float,
) -> tuple[np.ndarray, bool, bool]:
    """Blended scores for one row, plus whether each side had anything to say.

    Returns `(scores, had_lexical, had_semantic)`. The two flags are counted by
    the caller and reported: a submission where one side was silent for most
    rows is a different system from the one we measured, and the count is the
    only way to notice.
    """
    n = len(candidates)

    doc_rows = np.array([doc_of.get(a, -1) for a in candidates], dtype=np.int64)
    known_doc = doc_rows >= 0
    lex = np.zeros(n, dtype=np.float64)
    term_ids, term_weights = bm25_fast.query_terms(history_ids, titles, vocab, idf)
    if term_ids.size and known_doc.any():
        lex[known_doc] = scorer.score(term_ids, term_weights, doc_rows[known_doc])

    sem = np.zeros(n, dtype=np.float64)
    vector = user_vector_for(history_ids, embeddings, row_of_article)
    art_rows = np.array([row_of_article.get(a, -1) for a in candidates], dtype=np.int64)
    known_art = art_rows >= 0
    if vector is not None and known_art.any():
        sem[known_art] = embeddings.vectors[art_rows[known_art]] @ vector

    return blend(lex, sem, alpha), bool(term_ids.size), vector is not None


def mind_lines(
    source: Path | None = None,
    alpha: float = MIND_ALPHA,
    raw_root: Path | None = None,
) -> Iterator[tuple[str, str, int, bool, bool]]:
    """Yield `(line, impression_id, n_candidates, had_lexical, had_semantic)`."""
    spec = config.tokeniser_for("mind_small")  # English; MIND's tokeniser spec
    articles = encode_test.mind_articles(raw_root)
    index, scorer, idf = bm25_fast.build(articles, spec)

    titles_frame = _mind_titles(raw_root)
    titles = bm25_fast.tokenised_titles(titles_frame, spec)
    doc_of = {a: i for i, a in enumerate(index.article_ids)}

    embeddings = embed.load(config.ARTIFACTS_DIR / "embeddings" / encode_test.MIND_KEY)
    row_of_article = {a: i for i, a in enumerate(embeddings.article_ids)}

    for impression_id, candidates, history_ids in _mind_rows(source, raw_root):
        scores, had_lex, had_sem = score_row(
            history_ids,
            candidates,
            scorer=scorer,
            vocab=index.vocab,
            idf=idf,
            titles=titles,
            embeddings=embeddings,
            doc_of=doc_of,
            row_of_article=row_of_article,
            alpha=alpha,
        )
        ranks = fmt.ranks_from_scores(_order(scores, candidates), len(candidates))
        yield (
            fmt.format_line(impression_id, ranks),
            impression_id,
            len(candidates),
            had_lex,
            had_sem,
        )


def _mind_titles(raw_root: Path | None = None):
    """`(article_id, title)` for MINDlarge_test — queries use titles only (D6)."""
    import polars as pl

    path = mind_test.behaviors_path(raw_root).parent / "news.tsv"
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=[
            "article_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ],
        quote_char=None,
    ).select("article_id", "title")


def _mind_rows(
    source: Path | None = None, raw_root: Path | None = None
) -> Iterator[tuple[str, list[str], list[str]]]:
    """`(impression_id, candidates, history)` in file order.

    `mind_test.read_rows` deliberately reads only the two fields the baseline
    needed; the hybrid also needs the history column, so this reads all three
    rather than widening that function's contract for one caller.
    """
    path = mind_test.behaviors_path(raw_root) if source is None else source
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != mind_test.N_COLUMNS:
                raise mind_test.MindTestError(
                    f"{path.name} line {lineno}: {len(fields)} fields, "
                    f"expected {mind_test.N_COLUMNS}"
                )
            candidates = fields[4].split()
            if not candidates:
                raise mind_test.MindTestError(
                    f"{path.name} line {lineno}: no candidates for impression {fields[0]}"
                )
            yield fields[0], candidates, fields[3].split()


def write_mind_hybrid(
    out_dir: Path,
    source: Path | None = None,
    alpha: float = MIND_ALPHA,
    raw_root: Path | None = None,
) -> dict:
    """Write, archive and validate the MIND hybrid submission."""
    path = mind_format.output_path(out_dir)
    ids: list[str] = []
    lengths: list[int] = []
    counts = {"lexical_silent": 0, "semantic_silent": 0, "both_silent": 0}

    def stream() -> Iterator[str]:
        for line, impression_id, n, had_lex, had_sem in mind_lines(source, alpha, raw_root):
            ids.append(impression_id)
            lengths.append(n)
            if not had_lex:
                counts["lexical_silent"] += 1
            if not had_sem:
                counts["semantic_silent"] += 1
            if not had_lex and not had_sem:
                counts["both_silent"] += 1
            yield line

    written = fmt.write_lines(stream(), path)
    summary = fmt.validate_file(path, ids, lengths)
    if written != summary["lines"]:
        raise fmt.PredictionFormatError(f"wrote {written} lines, read back {summary['lines']}")

    archive = fmt.zip_prediction(
        path, mind_format.archive_path(out_dir), member_dir=mind_format.MEMBER_DIR
    )
    summary.update(
        path=str(path),
        bytes=path.stat().st_size,
        alpha=alpha,
        coverage=counts,
        archive=fmt.validate_archive(archive, mind_format.MEMBER, summary["lines"]),
    )
    return summary


def alpha_from_report(metrics_dir: Path | None = None) -> float | None:
    """The val-selected alpha recorded in `hybrid_ablation.json`, if present.

    Used by a test to assert `MIND_ALPHA` still matches what was measured. The
    submission does not read it at runtime: a prediction file should not change
    because a report was regenerated.
    """
    path = (metrics_dir or config.METRICS_DIR) / "hybrid_ablation.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    selected = blob["datasets"].get("mind_small", {}).get("val_selected")
    return None if selected is None else float(selected["selected_alpha"])
