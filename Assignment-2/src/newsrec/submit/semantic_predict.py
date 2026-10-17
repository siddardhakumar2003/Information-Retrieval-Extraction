"""The EB-NeRD semantic submission (Q5.2).

Design: agent_docs/q5_plan.md; blend weight decision from
`reports/metrics/hybrid_ablation.json`

## Semantic only, and that is a measurement rather than a preference

`eval/hybrid.py` swept the lexical/semantic blend weight on the **validation**
split of both EB-NeRD corpora. Both selected `alpha=0` independently, their val
curves falling monotonically from zero -- every amount of lexical signal made
them worse. `hybrid_ablation.json` records why: Danish BM25 almost always
produces a score (1.3% zeros against MIND's 52.3%) and those scores barely
separate candidates within an impression (coefficient of variation 0.50 against
MIND's 1.45). A signal that speaks constantly and says nothing displaces the
tie-break with noise.

So there is no lexical term here, and adding one would be overriding the
measurement.

## The key type is the thing that kills this silently

EB-NeRD candidate ids are `Int32` in the parquet; the embedding table keys on
`str`. Measured on the real data: string keys give coverage **0.0000**, integer
keys **1.0000**. With the wrong one every candidate is `UNSCORABLE`, every
ranking falls through to the `article_id` tie-break, and the result is a
well-formed, accepted submission that is exactly the baseline -- with nothing
anywhere reporting a problem.

`run` therefore measures coverage on a sample and refuses to write below
`MIN_COVERAGE`, rather than trusting the cast.

## Leakage, deliberately not filtered

The test history is the pre-period history the competition supplies -- by
construction the state before the test period -- so there is no boundary to
enforce and no `click_time` to enforce it against. Recorded in `score_fast` and
repeated here because "no leakage check in the submission path" should read as a
decision rather than an oversight.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from newsrec import config
from newsrec.retrieval import embed
from newsrec.submit import ebnerd_format, ebnerd_test, encode_test, score_fast
from newsrec.submit import format as fmt

#: Below this share of candidate slots resolving to a vector, refuse to write.
#: Not a tuning knob: the real value is 1.0000 and the failure mode is 0.0000,
#: so anything in between means something has gone wrong that we have not seen
#: before and should look at rather than ship.
MIN_COVERAGE = 0.99

#: ...and the same for USERS. Both halves resolve article ids independently, so
#: checking only candidates is not enough: the first real run had candidate
#: coverage 1.0000 and zero user vectors, which would have shipped the baseline.
#: Every EB-NeRD test impression has a history row (measured, 0 missing), so
#: anything below this means the id types have diverged again.
MIN_USER_COVERAGE = 0.99

#: Behaviour rows per chunk. Each chunk explodes to roughly 15 candidate rows
#: per behaviour row, so this is about 15M scored pairs at a time.
CHUNK = 1_000_000


class CoverageTooLow(RuntimeError):
    """Candidates are not resolving to embeddings; the file would be a baseline."""


def history_path(raw_root: Path | None = None) -> Path:
    return ebnerd_test.testset_root(raw_root) / "test" / "history.parquet"


def load_pieces(raw_root: Path | None = None):
    """Embeddings, the integer-keyed row index, and per-user pooled vectors."""
    embeddings = embed.load(config.ARTIFACTS_DIR / "embeddings" / encode_test.EBNERD_KEY)
    # int, not str. See the module docstring -- this single argument is the
    # difference between a real submission and the baseline.
    row_of_article = score_fast.article_row_index(embeddings, cast=int)

    history = pl.read_parquet(history_path(raw_root), columns=["user_id", "article_id_fixed"])
    # The SAME integer-keyed index. Letting this function build its own default
    # is what produced zero user vectors on the first run: history ids are Int32
    # and the embedding table keys on str.
    row_of_user, user_matrix = score_fast.user_vectors_from_fixed_history(
        history, embeddings, row_of_article=row_of_article
    )
    return embeddings, row_of_article, row_of_user, user_matrix


def measure_coverage(
    row_of_article: dict,
    source: Path | None = None,
    rows: int = 50_000,
) -> float:
    """Share of candidate slots in the first `rows` behaviours that have a vector."""
    frame = (
        pl.scan_parquet(ebnerd_test.behaviors_path() if source is None else source)
        .select("article_ids_inview")
        .head(rows)
        .collect()
    )
    return score_fast.coverage(row_of_article, frame["article_ids_inview"].explode())


def lines(
    embeddings: embed.ArticleEmbeddings,
    row_of_article: dict,
    row_of_user,
    user_matrix,
    source: Path | None = None,
    chunk: int = CHUNK,
) -> Iterator[str]:
    """Prediction lines for every behaviour row, in file order."""
    for frame in ebnerd_test.iter_chunks(
        source, chunk, columns=("impression_id", "user_id", "article_ids_inview")
    ):
        ranks = score_fast.score_frame(frame, row_of_user, user_matrix, embeddings, row_of_article)
        yield from score_fast.lines_for_frame(frame, ranks)


def write_ebnerd_semantic(
    out_dir: Path,
    source: Path | None = None,
    raw_root: Path | None = None,
    chunk: int = CHUNK,
) -> dict:
    """Write, archive and validate the EB-NeRD semantic submission."""
    shape = ebnerd_test.check_shape(source)
    embeddings, row_of_article, row_of_user, user_matrix = load_pieces(raw_root)

    coverage = measure_coverage(row_of_article, source)
    if coverage < MIN_COVERAGE:
        raise CoverageTooLow(
            f"only {coverage:.4f} of candidate slots resolve to an embedding "
            f"(need >= {MIN_COVERAGE}). At 0.0 this writes a valid, accepted file "
            f"that is really the article_id baseline. Check the id key type: "
            f"candidates are Int32 and the embedding table keys on str."
        )

    n_history = pl.scan_parquet(history_path(raw_root)).select(pl.len()).collect().item()
    user_coverage = len(row_of_user) / n_history if n_history else 0.0
    if user_coverage < MIN_USER_COVERAGE:
        raise CoverageTooLow(
            f"only {len(row_of_user):,} of {n_history:,} users ({user_coverage:.4f}) got a "
            f"vector (need >= {MIN_USER_COVERAGE}). Every candidate can resolve while "
            f"every user fails: the two halves look ids up independently. At 0.0 every "
            f"score is UNSCORABLE and the file is the article_id baseline."
        )

    path = ebnerd_format.output_path(out_dir)
    written = fmt.write_lines(
        lines(embeddings, row_of_article, row_of_user, user_matrix, source, chunk), path
    )

    def expected_ids() -> Iterator[str]:
        for frame in ebnerd_test.iter_chunks(source, chunk, columns=("impression_id",)):
            yield from (str(v) for v in frame["impression_id"].to_list())

    def expected_lengths() -> Iterator[int]:
        for frame in ebnerd_test.iter_chunks(source, chunk, columns=("article_ids_inview",)):
            yield from frame["article_ids_inview"].list.len().to_list()

    summary = fmt.validate_file(path, expected_ids(), expected_lengths())
    if written != summary["lines"]:
        raise fmt.PredictionFormatError(f"wrote {written} lines, read back {summary['lines']}")

    archive = fmt.zip_prediction(
        path, ebnerd_format.archive_path(out_dir), member_dir=ebnerd_format.MEMBER_DIR
    )
    summary.update(
        path=str(path),
        bytes=path.stat().st_size,
        method="semantic",
        candidate_coverage=round(coverage, 6),
        users_with_vectors=len(row_of_user),
        user_coverage=round(user_coverage, 6),
        source_shape=shape,
        archive=fmt.validate_archive(archive, ebnerd_format.MEMBER, summary["lines"]),
    )
    return summary
