"""Encoding the two competition test corpora (Q5.1, Q5.2).

Design: agent_docs/q5_plan.md

## Why these are not `make embed --dataset ...`

`MINDlarge_test` and `ebnerd_testset` are not registered in `config.DATASETS`
and deliberately never will be: they carry no labels and no split, so every
split and leakage test would have to grow an exception for a corpus that has
neither. They also never enter the feature store, whose tables all carry a
`split` column.

What they DO share with the registered corpora is the thing that matters for
embeddings: the definition of an article's text. That definition is not
re-written here -- `TEXT_EXPR` reproduces `feature_store`'s
`title + " " + abstract`, stripped, and the modules are cross-checked by a test.
If the two ever diverge, a test-corpus vector would mean something subtly
different from a training-corpus vector while comparing perfectly happily
against it.

## Same encoder, necessarily

`config.ENCODER` is pinned to a revision hash (design_rationale 1.17). A
submission encoded with a different model would produce vectors that are not
comparable with the user histories they are scored against -- and the failure
would be silent, since cosines between unrelated embedding spaces are still
perfectly well-defined numbers. `embed.build` reads the same pinned spec.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from newsrec import config
from newsrec.retrieval import embed
from newsrec.submit import ebnerd_test, mind_test

#: The canonical article text, identical to `feature_store`'s. Not re-derived:
#: see the module docstring.
TEXT_EXPR = (
    pl.concat_str([pl.col("title"), pl.col("abstract").fill_null("")], separator=" ")
    .str.strip_chars()
    .alias("text")
)

#: Where each test corpus's vectors land. Named so they cannot collide with the
#: registered datasets' directories.
MIND_KEY = "mind_large_test"
EBNERD_KEY = "ebnerd_testset"


def mind_articles(raw_root: Path | None = None) -> pl.DataFrame:
    """`(article_id, text)` for MINDlarge_test's news.tsv, sorted by id.

    Sorted because `embed.save` persists row order alongside the vectors and a
    reordered corpus must never silently reassign a row to a different article.
    """
    path = mind_test.behaviors_path(raw_root).parent / "news.tsv"
    frame = pl.read_csv(
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
    )
    return (
        frame.select("article_id", "title", "abstract")
        .with_columns(TEXT_EXPR)
        .select("article_id", "text")
        .sort("article_id")
    )


def ebnerd_articles(raw_root: Path | None = None) -> pl.DataFrame:
    """`(article_id, text)` for ebnerd_testset's articles.parquet, sorted by id.

    EB-NeRD's `subtitle` plays MIND's `abstract` role (`data/ebnerd.py`), so it
    is renamed before the shared text expression is applied rather than the
    expression being duplicated with a different column name.
    """
    path = ebnerd_test.testset_root(raw_root) / "articles.parquet"
    frame = pl.read_parquet(path, columns=["article_id", "title", "subtitle"]).rename(
        {"subtitle": "abstract"}
    )
    return frame.with_columns(TEXT_EXPR).select("article_id", "text").sort("article_id")


def encode(
    which: str,
    device: str = "auto",
    raw_root: Path | None = None,
    out_root: Path | None = None,
) -> dict:
    """Encode one test corpus and persist it. Returns a summary.

    Idempotent in the same way `embed.run` is: if vectors already exist for this
    corpus under the same encoder fingerprint, they are loaded rather than
    recomputed. A re-encode is ~30 minutes on CPU and produces byte-identical
    output, so doing it twice is pure waste.
    """
    loader = {MIND_KEY: mind_articles, EBNERD_KEY: ebnerd_articles}[which]
    articles = loader(raw_root)

    dest = (out_root or config.ARTIFACTS_DIR / "embeddings") / which
    if dest.exists():
        try:
            existing = embed.load(dest)
        except Exception:
            existing = None
        if existing is not None and existing.n_articles == len(articles):
            return {
                "corpus": which,
                "articles": existing.n_articles,
                "dim": existing.dim,
                "reused": True,
                "path": str(dest),
            }

    embeddings = embed.build(articles, device=device)
    embed.save(embeddings, dest)
    return {
        "corpus": which,
        "articles": embeddings.n_articles,
        "dim": embeddings.dim,
        "reused": False,
        "path": str(dest),
        "empty_text": int((articles["text"].str.len_chars() == 0).sum()),
    }
