"""Our own inverted index over article text (q2_plan.md D3).

Requirements covered: Q2.1 (build the index), Q2.2 (index title + abstract).

## What an inverted index is, and why we write our own

A **forward** index maps document -> its terms. An **inverted** index maps
term -> the documents containing it. That inversion is the whole trick: a query
for "police" touches one posting list instead of scanning all 65,238 articles.

D7 settled that we implement this ourselves rather than calling a library. Q2.1
says "build an inverted index", and it is the graded artefact; `rank_bm25`
appears in exactly one place, as an oracle in `tests/test_bm25_scoring.py`.

## The layout, and why this one

Chosen (D3 option A): a term -> term_id vocabulary, plus postings held as three
flat numpy arrays in CSR style:

    offsets[i] .. offsets[i+1]   the slice of postings belonging to term_id i
    doc_ids[...]                 which documents contain it
    term_freqs[...]              how often, in each

CSR ("compressed sparse row") means the per-term lists are concatenated into one
array with an index of where each begins, rather than being stored as thousands
of separate small arrays. Scoring a term is then a single contiguous numpy slice,
which is why this is fast enough to run 230,117 queries without dropping into a
Python loop per posting.

Rejected: a scipy sparse matrix (compact and less code, but it hides the postings
structure that Q2.1 asks us to build), a dict of lists of tuples (3M Python
tuples, slow and memory-hungry), and SQLite (out-of-core for free, but every
query pays database round-trips at a scale that fits comfortably in RAM).

## Document frequency is stored, not recomputed

`doc_freq(term)` is just the length of the term's posting slice, so it *could* be
derived. It is stored anyway because BM25's inverse document frequency needs it
for every query term on every query, and because a stored df that disagrees with
the postings is a bug worth being able to test for directly.

## Empty documents keep their slot

Both corpora contain articles with no text -- 19 in `ebnerd_small`. They occupy a
doc_id with length 0 and appear in no posting list. Skipping them would shift
every subsequent doc_id by one, so the index would point at the wrong articles
while still returning perfectly plausible results.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.retrieval import tokenise

VOCAB_FILE = "vocab.json"
POSTINGS_FILE = "postings.npz"
DOC_STATS_FILE = "doc_stats.npz"
META_FILE = "index_meta.json"


class IndexMismatch(RuntimeError):
    """An index on disk does not match what the caller expects to load.

    Raised rather than warned. The failure this prevents is an index built under
    one tokeniser being reused by a run configured for another: every query term
    is looked up in a vocabulary that never contained it, recall collapses to
    near zero, and nothing anywhere reports an error.
    """


@dataclass(frozen=True)
class InvertedIndex:
    """An inverted index over one corpus, under one tokeniser."""

    #: term -> term_id
    vocab: dict[str, int]
    #: CSR offsets into doc_ids / term_freqs; length len(vocab) + 1
    offsets: np.ndarray
    doc_ids: np.ndarray
    term_freqs: np.ndarray
    #: doc_id -> token count (0 for an empty article)
    doc_lengths: np.ndarray
    #: doc_id -> article_id, so retrieval can name what it found
    article_ids: list[str]
    tokeniser: config.TokeniserSpec

    @property
    def n_docs(self) -> int:
        return len(self.article_ids)

    @property
    def n_postings(self) -> int:
        return int(self.doc_ids.size)

    @property
    def avg_doc_length(self) -> float:
        """Mean tokens per document, over ALL documents including empty ones.

        BM25 divides by this, so it must be a real number even for a corpus of
        entirely empty articles -- a NaN here propagates silently into every
        score rather than raising anywhere.
        """
        if self.n_docs == 0:
            return 0.0
        return float(self.doc_lengths.sum()) / self.n_docs

    def term_slice(self, term: str) -> slice | None:
        """The CSR slice for `term`, or None if it is out of vocabulary."""
        tid = self.vocab.get(term)
        if tid is None:
            return None
        return slice(int(self.offsets[tid]), int(self.offsets[tid + 1]))

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """(doc_ids, term_freqs) for `term`; empty arrays when unknown.

        Returns array views rather than Python objects, because this is the hot
        path: bm25.py scores a whole posting list as one vectorised expression.
        """
        sl = self.term_slice(term)
        if sl is None:
            empty_d = np.empty(0, dtype=self.doc_ids.dtype)
            empty_f = np.empty(0, dtype=self.term_freqs.dtype)
            return empty_d, empty_f
        return self.doc_ids[sl], self.term_freqs[sl]

    def postings_for(self, term: str) -> list[tuple[int, int]]:
        """(doc_id, term_freq) pairs -- the readable form, for tests and debugging."""
        docs, freqs = self.postings(term)
        return [(int(d), int(f)) for d, f in zip(docs, freqs, strict=True)]

    def doc_freq(self, term: str) -> int:
        """How many documents contain `term`."""
        sl = self.term_slice(term)
        return 0 if sl is None else sl.stop - sl.start


# =============================================================================
# Build
# =============================================================================


def build(articles: pl.DataFrame, spec: config.TokeniserSpec) -> InvertedIndex:
    """Build an index from a frame with `article_id` and `text` columns.

    `text` is taken as given. The feature store builds it (title + abstract) in
    one place -- Q2.2, and design_rationale 2.8 -- so re-concatenating fields
    here would create a second definition that could drift from the first.
    """
    missing = {"article_id", "text"} - set(articles.columns)
    if missing:
        raise ValueError(
            f"index.build needs an `article_id` and a `text` column; missing {sorted(missing)}. "
            f"Pass the feature store's article_features frame, which already builds `text`."
        )

    article_ids = articles["article_id"].to_list()
    if len(set(article_ids)) != len(article_ids):
        raise ValueError(
            "duplicate article_id in the corpus: the doc_id -> article_id map would be "
            "ambiguous and retrieval would return one of the duplicates arbitrarily."
        )

    # term -> {doc_id: term_freq}. Built as a dict first because the postings
    # for a term are discovered scattered across the corpus; the flat CSR arrays
    # are packed once at the end, when every list's final length is known.
    accum: dict[str, dict[int, int]] = defaultdict(dict)
    doc_lengths = np.zeros(len(article_ids), dtype=np.int32)

    for doc_id, text in enumerate(articles["text"].to_list()):
        terms = tokenise.tokenise(text, spec)
        doc_lengths[doc_id] = len(terms)
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        for term, freq in counts.items():
            accum[term][doc_id] = freq

    # Sorted vocabulary so term_ids -- and therefore the whole on-disk layout --
    # depend only on the corpus, never on dict iteration order (Q1.13).
    vocab = {term: i for i, term in enumerate(sorted(accum))}

    n_postings = sum(len(v) for v in accum.values())
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    doc_ids = np.zeros(n_postings, dtype=np.int32)
    term_freqs = np.zeros(n_postings, dtype=np.int32)

    cursor = 0
    for term, tid in vocab.items():
        postings = accum[term]
        offsets[tid] = cursor
        # Ascending doc_id within a posting list: deterministic, and it keeps
        # the arrays in an order a human can read when debugging.
        for doc_id in sorted(postings):
            doc_ids[cursor] = doc_id
            term_freqs[cursor] = postings[doc_id]
            cursor += 1
    offsets[len(vocab)] = cursor

    return InvertedIndex(
        vocab=vocab,
        offsets=offsets,
        doc_ids=doc_ids,
        term_freqs=term_freqs,
        doc_lengths=doc_lengths,
        article_ids=article_ids,
        tokeniser=spec,
    )


# =============================================================================
# Persistence
# =============================================================================


def _meta(index: InvertedIndex) -> dict:
    return {
        "index_version": config.INDEX_VERSION,
        "n_docs": index.n_docs,
        "n_terms": len(index.vocab),
        "n_postings": index.n_postings,
        "avg_doc_length": index.avg_doc_length,
        "tokeniser": tokenise.fingerprint(index.tokeniser),
        "tokeniser_name": index.tokeniser.name,
        "k1": config.BM25_K1,
        "b": config.BM25_B,
    }


def save(index: InvertedIndex, dest: Path) -> Path:
    """Write the four files q2_plan D3 specifies, deterministically.

    No timestamps and no wall-clock anything, for the same reason the Q1 manifest
    carries none: two builds of the same corpus must be byte-identical, so that
    "the index changed" is informative rather than routine (design_rationale 2.9).
    """
    dest.mkdir(parents=True, exist_ok=True)

    (dest / VOCAB_FILE).write_text(
        json.dumps(index.vocab, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    # savez, not savez_compressed: the compressed form embeds a per-member
    # timestamp in its zip headers, which would make two identical saves differ.
    with (dest / POSTINGS_FILE).open("wb") as fh:
        np.savez(fh, offsets=index.offsets, doc_ids=index.doc_ids, term_freqs=index.term_freqs)
    with (dest / DOC_STATS_FILE).open("wb") as fh:
        np.savez(
            fh,
            doc_lengths=index.doc_lengths,
            article_ids=np.array(index.article_ids, dtype=object),
        )
    (dest / META_FILE).write_text(
        json.dumps(_meta(index), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dest


def load(src: Path, expect_tokeniser: config.TokeniserSpec | None = None) -> InvertedIndex:
    """Read an index back, refusing one that does not match what is expected.

    `expect_tokeniser` is checked rather than merely recorded, following the Q1
    precedent where the feature store's `config_fingerprint` is enforced on read
    (design_rationale 2.10). Recording a fingerprint nobody verifies documents
    the bug instead of preventing it.
    """
    try:
        meta = json.loads((src / META_FILE).read_text(encoding="utf-8"))
        vocab = json.loads((src / VOCAB_FILE).read_text(encoding="utf-8"))
        with np.load(src / POSTINGS_FILE) as p:
            offsets, doc_ids, term_freqs = p["offsets"], p["doc_ids"], p["term_freqs"]
        with np.load(src / DOC_STATS_FILE, allow_pickle=True) as d:
            doc_lengths, article_ids = d["doc_lengths"], d["article_ids"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise IndexMismatch(f"cannot read the index at {src}: {exc}") from exc

    for key in ("index_version", "tokeniser", "tokeniser_name", "n_docs"):
        if key not in meta:
            raise IndexMismatch(
                f"{src / META_FILE} is missing {key!r}; it was written by an older or "
                f"corrupt build. Rebuild the index rather than trusting it."
            )

    if meta["index_version"] != config.INDEX_VERSION:
        raise IndexMismatch(
            f"index at {src} was built at version {meta['index_version']}, "
            f"this code is version {config.INDEX_VERSION}. Rebuild it."
        )

    if expect_tokeniser is not None:
        want = tokenise.fingerprint(expect_tokeniser)
        if meta["tokeniser"] != want:
            raise IndexMismatch(
                f"index at {src} was built with tokeniser {meta['tokeniser']} "
                f"({meta['tokeniser_name']}) but {want} ({expect_tokeniser.name}) was "
                f"requested. Query terms would be looked up in a vocabulary built under "
                f"different rules, so recall would collapse silently. Rebuild the index."
            )
        spec = expect_tokeniser
    else:
        spec = config.TOKENISER_VARIANTS.get(
            meta["tokeniser_name"],
            config.TokeniserSpec(name=meta["tokeniser_name"], language="?"),
        )

    if len(article_ids) != meta["n_docs"]:
        raise IndexMismatch(
            f"index at {src} claims {meta['n_docs']} documents but stores "
            f"{len(article_ids)} article ids."
        )

    return InvertedIndex(
        vocab=vocab,
        offsets=offsets,
        doc_ids=doc_ids,
        term_freqs=term_freqs,
        doc_lengths=doc_lengths,
        article_ids=[str(a) for a in article_ids],
        tokeniser=spec,
    )


def index_dir(dataset: str) -> Path:
    """Where this dataset's index lives (q2_plan D3)."""
    return config.ARTIFACTS_DIR / "bm25" / dataset
