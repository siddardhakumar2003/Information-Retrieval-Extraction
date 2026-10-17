"""A2 Q5.5 — two-stage predictions for both leaderboards, scored by NRMS.

Design: `agent_docs/a2_q5_plan.md`.

## Why the neural model is the one that ships, and it is not because it won

The rule declared before any test number was looked at was "per dataset, the
variant with the best **val** nDCG@10". Applied to the measured val scores it
picks NRMS+category on EB-NeRD (0.5152) and the **MLP** on MIND (0.3692).

The MLP is nevertheless disqualified, on a measurement rather than a
preference. Its highest-importance feature is `popularity_train` — clicks in the
*training split of the small corpus* — and the leaderboard test sets are later,
much larger corpora whose articles mostly did not exist then. Measured on 20,000
lines of `MINDlarge_test`, only **5.6%** of candidate slots carry a non-zero
value; on 50,000 rows of `ebnerd_testset`, **8.2%**. A model whose dominant
feature is 94.4% constant at serving time is not running in the distribution it
was fitted on, and its leaderboard score would not estimate the offline number
that selected it. OQ-14 records the decision and what it gives up.

NRMS has the opposite property, and this is the interesting part: **it is more
servable than the feature-based re-ranker.** Its inputs are an article's
embedding and its category label, and the test bundles supply both in full —
`submit/encode_test.py` already encodes the test corpora with the same pinned
encoder A1 used. Nothing about the model's input distribution changes between
the offline evaluation and the submission.

## What the model needs, and what could silently go wrong

Two lookups, both of which have a documented way of failing quietly:

**Article ids into embedding rows.** A1 shipped this bug for real: a submission
that was valid, accepted and byte-identical to the baseline, while its
candidate-side coverage guard read 1.0000 and *every one of 807,677 user-side
lookups* had failed, because EB-NeRD stores `article_id_fixed` as `Int32` and
the embedding table keys on `str` (`design_rationale.md` §1.21). Both sides are
therefore measured and both are refused below a floor.

**Category labels into embedding indices.** The category embedding is a learned
table with one row per *training-corpus* category, in the order
`nrms.load_resources` assigns. A test-corpus category must be looked up through
that same map or the model reads a row belonging to a different category — with
no error, because every index is in range. So the map is rebuilt from the
training corpus by the identical code path, the model's saved `n_categories` is
asserted against it, and a test-corpus category the training corpus never had
maps to index 0 ("no category on record") rather than to whatever integer a
fresh sort would give it.

## Click order does not matter here, but click *selection* does

The user encoder is permutation-invariant (`rerank/nrms.py`, measured), so
whether a history is oldest-first or newest-first changes nothing. **Which**
`config.QUERY_HISTORY_LIMIT` clicks are taken does matter, so the tail of the
chronological history is taken, exactly as `score_fast` and `hybrid_predict`
already do. Reversing it afterwards is a no-op kept only so this path reads the
same as the others.

## Shape

Chunked and batched throughout. A1's scale analysis found the per-impression
Python loop twice, and the second time it was a 37-hour job that became 14
minutes. Rows are never sorted, grouped or de-duplicated: both competitions
match the prediction file to the truth file **positionally**.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.rerank import nrms, store
from newsrec.retrieval import embed
from newsrec.submit import ebnerd_test, encode_test, hybrid_predict, mind_test, semantic_predict
from newsrec.submit import format as fmt

#: Which trained model scores which competition. The test corpora are not in
#: `config.DATASETS`, so this mapping cannot be derived and is declared.
TRAINED_ON: dict[str, str] = {
    encode_test.MIND_KEY: "mind_small",
    encode_test.EBNERD_KEY: "ebnerd_small",
}

#: The variant that ships. See the module docstring for why it is this one on
#: both competitions rather than the literal winner of the val rule.
VARIANT = "nrms_category_aware"

#: Below these shares of resolvable ids, refuse to write. A submission whose
#: lookups failed is still a well-formed file that the scorer accepts — A1
#: produced one. Both sides are checked because only one of them failing is
#: exactly what happened.
MIN_CANDIDATE_COVERAGE = 0.5
MIN_USER_COVERAGE = 0.5

#: Impressions read per pass. Kept modest because `_candidate_batch` pads every
#: row to the widest candidate list in the batch, and EB-NeRD's run from 1 to
#: 250 — at 4,096 impressions a batch would allocate
#: 4096 x 250 x 384 x 4 bytes = 1.5 GB of mostly padding, which does not fit
#: this machine's 4 GB of VRAM.
CHUNK_IMPRESSIONS = 8_192

#: Rows per forward pass, after the chunk has been bucketed by candidate count.
#: 512 x 250 x 384 x 4 = 196 MB worst case, and typically far less because a
#: bucket holds rows of similar width.
BATCH_IMPRESSIONS = 512


class SubmissionRefused(RuntimeError):
    """A guard fired. Writing anyway would produce a plausible, wrong file."""


def category_index_from_training_corpus(dataset: str) -> dict[str, int]:
    """`category label -> embedding row`, built from the TRAINING corpus.

    Rebuilt by calling `nrms.load_resources`, the same function the ablation
    used, rather than re-deriving the ordering here. Index 0 is reserved for
    "no category on record", so an unseen label is a category of its own
    instead of colliding with a real one.
    """
    from newsrec.data import feature_store

    store_dir = config.DATA_DIR / "feature_store" / dataset
    article_features = feature_store.read_table(store_dir / "article_features.parquet")
    names = sorted({c for c in article_features["category"].to_list() if c is not None})
    return {name: i + 1 for i, name in enumerate(names)}


def test_categories(which: str, raw_root: Path | None = None) -> dict[str, str]:
    """`article_id -> category label` for one test corpus.

    MIND's `news.tsv` column 2 and EB-NeRD's `category_str` are the labels the
    training corpora's `category` column was built from (`data/mind.py`,
    `data/ebnerd.py`), so the two vocabularies are comparable.
    """
    if which == encode_test.MIND_KEY:
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
        ).select("article_id", "category")
    else:
        path = ebnerd_test.testset_root(raw_root) / "articles.parquet"
        frame = pl.read_parquet(path, columns=["article_id", "category_str"]).rename(
            {"category_str": "category"}
        )
    return {
        str(article_id): category
        for article_id, category in frame.iter_rows()
        if category is not None
    }


def _inference_device() -> str:
    """CUDA when available, CPU otherwise — and unlike TRAINING, falling back
    here is correct.

    `rerank/nrms.py::train` raises without a GPU on purpose: a silent fallback
    would turn a hyperparameter sweep into an overnight job. Scoring 13.5 million
    rows is a throughput problem, and a CPU-only grader must still be able to
    regenerate the submission — slowly is fine, impossible is not.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_resources(which: str, raw_root: Path | None = None) -> dict:
    """The model, the test-corpus embeddings, and both id maps.

    Asserts the model's `n_categories` against the map rebuilt here. A mismatch
    means the training corpus changed since the model was fitted and every
    category lookup would be off by an unknown amount, silently.
    """
    dataset = TRAINED_ON[which]
    model = store.load(store.model_dir(dataset, VARIANT), device=_inference_device())

    embeddings = embed.load(config.ARTIFACTS_DIR / "embeddings" / which)
    row_of_article = {a: i for i, a in enumerate(embeddings.article_ids)}

    index_of_category = category_index_from_training_corpus(dataset)
    expected = len(index_of_category) + 1
    if model.category_aware and model.n_categories != expected:
        raise SubmissionRefused(
            f"{VARIANT} on {dataset} was trained with {model.n_categories} categories "
            f"but the training corpus now yields {expected}. Every category lookup "
            f"would read a row belonging to a different category, in range and wrong."
        )

    category_of_article = test_categories(which, raw_root)
    category_of_row = np.zeros(embeddings.n_articles, dtype=np.int64)
    known = 0
    for article_id, row in row_of_article.items():
        label = category_of_article.get(str(article_id))
        if label is not None and label in index_of_category:
            category_of_row[row] = index_of_category[label]
            known += 1

    return {
        "model": model,
        "embeddings": embeddings,
        "row_of_article": row_of_article,
        "category_of_row": category_of_row,
        "category_coverage": known / max(embeddings.n_articles, 1),
        "n_categories": expected,
        "dataset": dataset,
        "device": model.device,
    }


def _source(resources: dict) -> nrms.ListwiseBatchSource:
    """A scoring source over the TEST corpus's vectors.

    `history_rows` is filled per chunk, so this is built once and mutated —
    the embedding matrix is up to 190 MB and copying it per chunk would be the
    kind of shape mistake A1's scale analysis is about.
    """
    return nrms.ListwiseBatchSource(
        groups=[],
        history_rows={},
        row_of_article=resources["row_of_article"],
        embeddings=resources["embeddings"].vectors,
        category_of_row=resources["category_of_row"],
        n_categories=resources["n_categories"],
    )


def _score_chunk(
    resources: dict,
    source: nrms.ListwiseBatchSource,
    candidate_lists: list[list[str]],
    history_lists: list[list[str]],
) -> tuple[list[np.ndarray], int]:
    """One NRMS forward pass over a batch. Returns per-row score arrays.

    **Keyed by POSITION, not by `impression_id`.** EB-NeRD's test file has
    13,536,710 rows but only 13,336,711 distinct impression ids, because all
    200,000 beyond-accuracy rows carry `impression_id = 0`
    (`submit/ebnerd_test.py`). A history dict keyed by impression id would hand
    every one of those 200,000 rows the same user's click history — in range,
    well formed, and wrong for 199,999 of them.
    """
    limit = config.QUERY_HISTORY_LIMIT
    row_of_article = resources["row_of_article"]

    keys = [str(i) for i in range(len(candidate_lists))]
    source.history_rows = {}
    resolved_history = 0
    for key, history in zip(keys, history_lists, strict=True):
        # Tail of the chronological history, then reversed. The reversal is a
        # no-op for this model (the encoder is order-blind) and is kept so this
        # path reads identically to score_fast/hybrid_predict.
        recent = list(history)[-limit:][::-1]
        rows = [row_of_article[a] for a in recent if a in row_of_article]
        source.history_rows[key] = np.asarray(rows, dtype=np.int64)
        if rows:
            resolved_history += 1

    model = resources["model"]
    history_t, mask_t, history_categories = nrms._history_batch(source, keys, model.device)
    candidates_t, candidate_categories = nrms._candidate_batch(
        source, candidate_lists, model.device
    )
    import torch

    with torch.no_grad():
        scores = (
            model.net(
                history_t,
                mask_t,
                candidates_t,
                history_categories if model.category_aware else None,
                candidate_categories if model.category_aware else None,
            )
            .cpu()
            .numpy()
        )
    scored = [scores[i, : len(candidate_lists[i])] for i in range(len(candidate_lists))]
    return scored, resolved_history


def _lines(
    resources: dict,
    rows: Iterator[tuple[str, list[str], list[str]]],
    tally: dict,
) -> Iterator[str]:
    """One output line per input row, in file order.

    Accumulates the coverage tallies into `tally` as it goes, so the caller can
    refuse *after* writing but *before* uploading — the counts are only known
    once the whole file has been walked.
    """
    source = _source(resources)
    row_of_article = resources["row_of_article"]
    buffer: list[tuple[str, list[str], list[str]]] = []

    def flush() -> Iterator[str]:
        """Score the buffered chunk and emit its lines in the ORIGINAL order.

        Rows are bucketed by candidate count before scoring. `_candidate_batch`
        pads every row in a batch to the widest one, and EB-NeRD's lists run
        from 1 to 250, so batching them as they arrive would make most of every
        tensor padding — 13.5 million rows' worth. Bucketing costs a sort and
        cuts the wasted work by roughly an order of magnitude.

        The results are scattered back to the arrival order before being
        emitted, because both competitions match the prediction file to the
        truth file **positionally**. This is the one place in this module where
        rows are reordered, and it is undone before anything is written.
        """
        if not buffer:
            return
        scores_by_position: dict[int, np.ndarray] = {}
        by_width = sorted(range(len(buffer)), key=lambda i: len(buffer[i][1]))

        for start in range(0, len(by_width), BATCH_IMPRESSIONS):
            positions = by_width[start : start + BATCH_IMPRESSIONS]
            scored, resolved_history = _score_chunk(
                resources,
                source,
                [buffer[i][1] for i in positions],
                [buffer[i][2] for i in positions],
            )
            tally["users_with_history"] += resolved_history
            for position, scores in zip(positions, scored, strict=True):
                scores_by_position[position] = scores

        for position, (impression_id, candidates, _) in enumerate(buffer):
            tally["rows"] += 1
            tally["candidate_slots"] += len(candidates)
            tally["candidates_resolved"] += sum(1 for a in candidates if a in row_of_article)
            # Descending score; ties by the order the publisher listed them,
            # which `argsort(kind="stable")` preserves.
            order = np.argsort(-np.asarray(scores_by_position[position]), kind="stable")
            yield fmt.format_line(impression_id, fmt.ranks_from_scores(order, len(candidates)))
        buffer.clear()

    for row in rows:
        buffer.append(row)
        if len(buffer) >= CHUNK_IMPRESSIONS:
            yield from flush()
    yield from flush()


def _mind_rows(raw_root: Path | None = None) -> Iterator[tuple[str, list[str], list[str]]]:
    """Reuses `hybrid_predict._mind_rows` — the same three fields, read once."""
    yield from hybrid_predict._mind_rows(raw_root=raw_root)


def _ebnerd_rows(raw_root: Path | None = None) -> Iterator[tuple[str, list[str], list[str]]]:
    """`(impression_id, candidates, history)` in file order.

    EB-NeRD ships one fixed pre-period history per *user*, in a separate table,
    rather than one per row as MIND does. It is loaded once and joined in
    memory; `semantic_predict` measured zero test impressions missing a history
    row, so this is a lookup rather than a fallback path.
    """
    history = pl.read_parquet(
        semantic_predict.history_path(raw_root), columns=["user_id", "article_id_fixed"]
    )
    history_of_user = {
        user_id: [str(a) for a in (clicks or [])] for user_id, clicks in history.iter_rows()
    }
    for chunk in ebnerd_test.iter_chunks(
        columns=("impression_id", "article_ids_inview", "user_id")
    ):
        for impression_id, candidates, user_id in chunk.iter_rows():
            yield (
                str(impression_id),
                [str(a) for a in candidates],
                history_of_user.get(user_id, []),
            )


def write(which: str, raw_root: Path | None = None, out_dir: Path | None = None) -> dict:
    """Score one competition's test file and write its archive.

    Returns a manifest block. Refuses to zip if either coverage guard fires.
    """
    out_dir = config.PREDICTIONS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resources = build_resources(which, raw_root)
    reader = _mind_rows if which == encode_test.MIND_KEY else _ebnerd_rows
    stem = "mind" if which == encode_test.MIND_KEY else "ebnerd"
    text_path = out_dir / ("prediction.txt" if stem == "mind" else "predictions.txt")

    tally = {
        "rows": 0,
        "candidate_slots": 0,
        "candidates_resolved": 0,
        "users_with_history": 0,
    }
    written = fmt.write_lines(_lines(resources, reader(raw_root), tally), text_path)

    candidate_coverage = tally["candidates_resolved"] / max(tally["candidate_slots"], 1)
    user_coverage = tally["users_with_history"] / max(tally["rows"], 1)

    if candidate_coverage < MIN_CANDIDATE_COVERAGE:
        raise SubmissionRefused(
            f"only {candidate_coverage:.4f} of candidate ids resolved to an embedding row; "
            f"the file would be well-formed and meaningless"
        )
    if user_coverage < MIN_USER_COVERAGE:
        raise SubmissionRefused(
            f"only {user_coverage:.4f} of rows got a usable click history. This is the "
            f"exact failure of design_rationale.md 1.21, where the candidate-side guard "
            f"read 1.0000 while every user-side lookup had failed."
        )

    archive = fmt.zip_prediction(text_path, out_dir / f"{stem}_nrms_submission.zip")
    return {
        "competition": which,
        "variant": VARIANT,
        "trained_on": resources["dataset"],
        "lines": written,
        "candidate_slots": tally["candidate_slots"],
        "candidate_coverage": round(candidate_coverage, 6),
        "user_coverage": round(user_coverage, 6),
        "category_coverage": round(resources["category_coverage"], 6),
        "device": resources["device"],
        "archive": str(config.repo_relative(archive)),
        "why_this_variant": (
            "NRMS+category on both competitions. The val rule alone picks the MLP on "
            "MIND (0.3692 vs 0.3412), but the MLP's dominant feature popularity_train "
            "is non-zero for only 5.6% of MIND submission candidates, so it would run "
            "far outside its training distribution (OQ-14). NRMS consumes article "
            "embeddings and category labels, which the test bundles supply in full."
        ),
        "features_unavailable_at_serving_time": {
            "note": (
                "Recorded because it is the finding, not a caveat. These are the "
                "feature-based re-ranker's inputs; NRMS uses none of them, which is "
                "why it is the servable option."
            ),
            "popularity_train": "5.6% non-zero on MINDlarge_test, 8.2% on ebnerd_testset",
            "session_click_count": (
                "no source at all — ebnerd_testset/test/behaviors.parquet ships "
                "session_id but no article_ids_clicked, so a click-derived session "
                "feature cannot be computed from it"
            ),
        },
    }


def run(out_dir: Path | None = None, which: tuple[str, ...] | None = None) -> Path:
    """Write both submissions and a manifest block for each."""
    out_dir = config.PREDICTIONS_DIR if out_dir is None else out_dir
    targets = which or (encode_test.MIND_KEY, encode_test.EBNERD_KEY)
    blocks = {target: write(target, out_dir=out_dir) for target in targets}
    path = out_dir / "nrms_submission_manifest.json"
    path.write_text(
        json.dumps(
            {"report": "nrms_submissions", "requirement": "A2.Q5.5", "submissions": blocks},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
