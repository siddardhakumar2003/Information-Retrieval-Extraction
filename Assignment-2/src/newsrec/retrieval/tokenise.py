"""Text to terms — the first step of Q2 (q2_plan.md D4).

Requirements covered: Q2.9 (language-aware tokenisation, declared and ablated).

## Why this is its own module

Tokenisation is the quietest place in the pipeline to introduce an error.
Nothing here can raise on bad input: it takes a string and returns a list of
strings. A wrong choice -- splitting Danish `ø` as a boundary, dropping numbers,
deduplicating terms -- produces slightly different tokens, then slightly
different scores, then a recall number that is simply wrong with no symptom
anywhere. Isolating it means it can be tested exhaustively against stated
expectations rather than inspected inside a larger function.

## One pipeline, configured per language

D4 rejected writing two tokenisers. Two implementations would mean a
MIND-versus-EB-NeRD difference is partly a difference between *our own two
functions*, which would undermine the cross-dataset comparison Q3.6 depends on.
So there is one code path here and the language is a parameter, supplied by
`config.TokeniserSpec`.

The baselines are deliberately asymmetric -- English removes stopwords, Danish
does not. That is not an oversight. Danish is morphologically richer than
English (compounding, definite suffixes: `bil` / `bilen` / `biler` are three
distinct tokens under a naive scheme), and D4 keeps the Danish baseline naive so
that Q2.9's stemming ablation has a real baseline to improve on. Adopting the
stemmer quietly would leave nothing to report.

## Resources are declared in config and loaded here

`config.TokeniserSpec` names a stopword resource and a Snowball algorithm; it
does not open files or import stemmers. Config declares, this module resolves.
That keeps `config.py` importable with no dependency on anything but the
standard library, which `scripts/check_no_data_files.py` relies on.
"""

from __future__ import annotations

import re
from functools import cache, lru_cache
from pathlib import Path

from newsrec import config

#: Unicode word characters. `\w` on a `str` pattern is Unicode-aware by default
#: in Python 3, so `ø`, `æ`, `å` and accented Latin are word characters and
#: Danish words are not split apart. Digits are word characters too, which is
#: what keeps numbers (D4: "40 points", "2019" carry meaning in news).
#:
#: Everything else -- punctuation, the em-dashes that fill headlines, quotes --
#: is a separator and never becomes a term.
WORD_RE = re.compile(r"\w+")

#: Vendored stopword lists. In-repo rather than pulled from a library at runtime
#: so the exact discarded vocabulary is auditable, and so English and Danish are
#: resolved the same way (there is no scikit-learn Danish list).
STOPWORD_DIR = Path(__file__).parent / "stopwords"


@cache
def load_stopwords(resource: str) -> frozenset[str]:
    """Load a vendored stopword list by resource id.

    Raises `FileNotFoundError` for an unknown resource rather than returning an
    empty set. The empty-set behaviour is the dangerous one: it degrades
    silently to "no stopwords at all", so the English baseline would quietly
    become the naive variant and every English number would move for a reason
    nobody could see.
    """
    path = STOPWORD_DIR / f"{resource}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in STOPWORD_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"no stopword resource {resource!r} at {config.repo_relative(path)}; "
            f"available: {available}"
        )
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            words.add(term)
    return frozenset(words)


@cache
def _stemmer(algorithm: str):
    """Resolve a Snowball stemmer, or fail loudly.

    Imported lazily and deliberately NOT wrapped in a try/except that falls back
    to no stemming. A silent fallback would report the baseline's numbers as the
    ablation's, which is a false experimental result -- much worse than a crash.
    """
    import snowballstemmer

    return snowballstemmer.stemmer(algorithm)


@lru_cache(maxsize=1_000_000)
def _stem_one(algorithm: str, term: str) -> str:
    """Stem a single term, memoised.

    News vocabulary is small and extremely repetitive -- the same few thousand
    terms recur across 65,238 articles -- so caching per term turns stemming
    from a per-token cost into a per-vocabulary one.
    """
    return _stemmer(algorithm).stemWord(term)


def tokenise(text: str | None, spec: config.TokeniserSpec) -> list[str]:
    """Split `text` into index terms under `spec`.

    Returns terms **in order and with repetition preserved**. BM25 needs term
    frequency, so collapsing duplicates here would silently discard the signal
    that a word appearing five times in a document matters more than one
    appearing once (D6 option D, rejected).

    A null or empty `text` yields `[]` rather than raising: articles with no
    abstract exist in both datasets and must not stop an index build.
    """
    if not text:
        return []

    if spec.lowercase:
        text = text.lower()

    terms = WORD_RE.findall(text)

    if spec.stopwords is not None:
        stops = load_stopwords(spec.stopwords)
        terms = [t for t in terms if t not in stops]

    if spec.stemmer is not None:
        terms = [_stem_one(spec.stemmer, t) for t in terms]

    return terms


def fingerprint(spec: config.TokeniserSpec) -> str:
    """A stable identifier for a tokenisation variant.

    Goes into `index_meta.json` so an index built under one variant cannot be
    silently reused under another -- the same discipline as the Q1 feature-store
    fingerprint, which is enforced on read rather than merely recorded
    (design_rationale.md 2.10).

    Built from the *behaviour-bearing* fields, not from `name` alone: two
    variants that were renamed but configured identically should collide, and
    one variant edited in place must produce a different fingerprint even if its
    name did not change.
    """
    return (
        f"{spec.language}"
        f"|lower={int(spec.lowercase)}"
        f"|stop={spec.stopwords or '-'}"
        f"|stem={spec.stemmer or '-'}"
    )
