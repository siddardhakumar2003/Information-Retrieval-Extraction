"""Q2 tokenisation (q2_plan.md D4).

Each test names the silent failure it prevents. The theme running through them:
a tokeniser fails *quietly*. It does not raise; it returns a slightly different
list of strings, and every number downstream shifts by an amount nobody can
attribute. So the assertions here are about specific, nameable behaviours rather
than "it returns some tokens".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from newsrec import config
from newsrec.retrieval import tokenise

# snowballstemmer is only needed for the Q2.9 ablation variants, not for the
# headline baselines. Skip rather than pretend, and name what is missing.
try:  # pragma: no cover - import probe
    import snowballstemmer  # noqa: F401

    HAS_SNOWBALL = True
except ImportError:  # pragma: no cover
    HAS_SNOWBALL = False

needs_snowball = pytest.mark.skipif(
    not HAS_SNOWBALL,
    reason="snowballstemmer not installed; required only for the Q2.9 stemming ablation",
)


# =============================================================================
# The registry wiring (D4: language comes from the dataset, not from a literal)
# =============================================================================


def test_language_comes_from_the_dataset_registry():
    """A hard-coded "en" on the EB-NeRD path yields plausible-looking nonsense.

    This is the failure that would never raise: Danish text tokenised under an
    English configuration produces tokens, produces scores, and produces a
    recall number. Nothing looks wrong.
    """
    assert config.tokeniser_for("mind_small").language == "en"
    assert config.tokeniser_for("ebnerd_demo").language == "da"
    assert config.tokeniser_for("ebnerd_small").language == "da"


def test_every_dataset_resolves_to_a_variant_that_exists():
    for key in config.DATASETS:
        spec = config.tokeniser_for(key)
        assert spec.name in config.TOKENISER_VARIANTS


def test_baseline_variants_are_the_defaults_not_the_stemmed_ones():
    """Q2.9: the stemmed variant must be measured, never silently adopted."""
    for key in config.DATASETS:
        assert config.tokeniser_for(key).stemmer is None


def test_danish_baseline_is_deliberately_naive():
    """D4 states this asymmetry explicitly; assert it so it cannot drift."""
    da = config.TOKENISER_VARIANTS["da_baseline"]
    assert da.stopwords is None
    assert da.stemmer is None
    en = config.TOKENISER_VARIANTS["en_baseline"]
    assert en.stopwords == "en"


# =============================================================================
# Core splitting behaviour
# =============================================================================


def test_lowercases():
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("Politiet MELDER Om", spec) == ["politiet", "melder", "om"]


def test_punctuation_is_split_not_kept():
    """Headlines are full of em-dashes and quotes; these must not become terms."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("Se her — 'chokerende' billeder!", spec) == [
        "se",
        "her",
        "chokerende",
        "billeder",
    ]


def test_numbers_survive():
    """D4 keeps numbers: '40 points' and '2019' carry real meaning in news."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("Scored 40 points in 2019", spec) == [
        "scored",
        "40",
        "points",
        "in",
        "2019",
    ]


def test_no_minimum_term_length():
    """An English-tuned minimum of 3 would silently damage Danish short words."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("er du på vej", spec) == ["er", "du", "på", "vej"]


def test_danish_letters_are_word_characters():
    """If æ/ø/å split words, every Danish token is wrong and nothing raises."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("blå gæster på øen", spec) == ["blå", "gæster", "på", "øen"]


def test_repetition_is_preserved():
    """BM25 needs term frequency. Deduplicating here would flatten it (D6 option D)."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("mand mand mand", spec) == ["mand", "mand", "mand"]


@pytest.mark.parametrize("text", ["", "   ", "!!! --- ???", None])
def test_degenerate_input_returns_empty_list_without_raising(text):
    """Articles with a null or empty abstract exist; the index build must survive."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise(text, spec) == []


# =============================================================================
# Stopwords
# =============================================================================


def test_english_stopwords_are_removed():
    spec = config.TOKENISER_VARIANTS["en_baseline"]
    assert tokenise.tokenise("the president of the united states", spec) == [
        "president",
        "united",
        "states",
    ]


def test_danish_baseline_keeps_its_stopwords():
    """The counterpart of the test above: the asymmetry is real and intended."""
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert "og" in tokenise.tokenise("mand og kvinde", spec)


def test_stopword_resource_is_vendored_and_nonempty():
    words = tokenise.load_stopwords("en")
    assert len(words) > 100
    assert "the" in words
    # Comment and blank lines in the vendored file must not become stopwords.
    assert not any(w.startswith("#") for w in words)
    assert "" not in words


def test_danish_stopword_resource_is_vendored():
    """OQ-8: the Snowball Danish list, fetched from source rather than recited."""
    words = tokenise.load_stopwords("da")
    assert len(words) == 94
    for term in ("og", "ikke", "være", "hendes"):
        assert term in words
    assert not any(w.startswith("#") for w in words)


def test_danish_stopword_file_carries_its_provenance():
    """OQ-8's actual requirement, which is not "a Danish list exists".

    The entry's whole argument is that a list *resembling* the Snowball list is
    not the Snowball list. What makes this file citable is the source URL, the
    retrieval date and the upstream checksum recorded in its header — and a
    header can be lost to a careless rewrite far more easily than 94 words can.

    Deliberately offline: asserting the recorded provenance, not re-fetching it.
    A test that hits snowballstem.org would fail in a clean clone behind a
    firewall, which is the environment `make test` is required to pass in.
    Re-verification is a manual step, and the recorded checksum is what makes it
    possible at all.
    """
    header = (Path(tokenise.__file__).parent / "stopwords" / "da.txt").read_text(encoding="utf-8")
    assert "https://snowballstem.org/algorithms/danish/stop.txt" in header
    assert "retrieved 2026-08-21" in header or "2026-08-21" in header
    assert "ef3ac6c540db45e8e292e526d6da3a11383f90435ba26000f9202dcecf68b8fd" in header


def test_apostrophe_fragments_are_stopwords():
    """'s' and 't' are debris of OUR splitting rule, not words.

    Measured on mind_small: 's' appeared in 43.4% of documents and was the most
    frequent English term overall. Left in, it inflates document length — the
    exact quantity BM25's length normalisation reacts to — so documents were
    penalised in proportion to how many possessives their headlines carried.
    """
    words = tokenise.load_stopwords("en")
    assert "s" in words
    assert "t" in words


def test_apostrophe_fragments_do_not_survive_tokenisation():
    """The end-to-end form of the test above: what actually reaches the index."""
    spec = config.TOKENISER_VARIANTS["en_baseline"]
    assert tokenise.tokenise("Trump's lawyer said don't", spec) == [
        "trump",
        "lawyer",
        "said",
        "don",
    ]


def test_other_single_characters_are_still_kept():
    """D4 rejected a minimum term length; only the two artefacts are removed.

    Guards against the over-broad fix — dropping every one-character token —
    which would silently damage Danish, where short words carry meaning.
    """
    spec = config.TOKENISER_VARIANTS["da_baseline"]
    assert tokenise.tokenise("a b c 5 ø", spec) == ["a", "b", "c", "5", "ø"]


def test_unknown_stopword_resource_raises_rather_than_silently_keeping_everything():
    """The dangerous failure is a missing file treated as an empty stopword set.

    That degrades to "no stopwords at all" while still producing results, so the
    English baseline would silently become the naive one.
    """
    with pytest.raises(FileNotFoundError):
        tokenise.load_stopwords("klingon")


# =============================================================================
# Determinism (Q1.13 applies to every stage, not only the data build)
# =============================================================================


def test_output_is_stable_across_calls():
    spec = config.TOKENISER_VARIANTS["en_baseline"]
    text = "Apple unveils the new iPhone 15 Pro — priced at $999"
    first = tokenise.tokenise(text, spec)
    assert all(tokenise.tokenise(text, spec) == first for _ in range(5))


def test_fingerprint_changes_when_the_variant_changes():
    """An index built under one variant must not be silently reused under another."""
    seen = {tokenise.fingerprint(spec): name for name, spec in config.TOKENISER_VARIANTS.items()}
    assert len(seen) == len(config.TOKENISER_VARIANTS)


def test_fingerprint_is_stable_for_the_same_variant():
    spec = config.TOKENISER_VARIANTS["en_baseline"]
    assert tokenise.fingerprint(spec) == tokenise.fingerprint(spec)


# =============================================================================
# The stemming ablation (Q2.9) — measured, never silently swapped in
# =============================================================================


@needs_snowball
def test_danish_stemmer_conflates_a_known_compound_family():
    """The point of the Danish ablation, in one assertion.

    Danish inflects with suffixes: 'bil' (car), 'bilen' (the car), 'biler'
    (cars). Under the naive baseline these are three unrelated index terms, so a
    query about 'bilen' cannot match an article about 'biler'. The stemmer is
    supposed to collapse them. If it does not, the ablation is measuring nothing.
    """
    naive = config.TOKENISER_VARIANTS["da_baseline"]
    stemmed = config.TOKENISER_VARIANTS["da_stem"]

    assert len(set(tokenise.tokenise("bil bilen biler", naive))) == 3
    assert len(set(tokenise.tokenise("bil bilen biler", stemmed))) == 1


@needs_snowball
def test_english_stemmer_conflates_a_known_family():
    stemmed = config.TOKENISER_VARIANTS["en_stem"]
    assert len(set(tokenise.tokenise("connect connected connecting", stemmed))) == 1


@needs_snowball
def test_stemmed_variant_differs_from_its_baseline():
    """If the two variants agree everywhere, the ablation cannot show anything."""
    text = "politiet melder om flere biler og cyklister på vejene"
    naive = tokenise.tokenise(text, config.TOKENISER_VARIANTS["da_baseline"])
    stemmed = tokenise.tokenise(text, config.TOKENISER_VARIANTS["da_stem"])
    assert naive != stemmed


def test_stemming_requested_without_the_dependency_fails_loudly():
    """Never degrade to "no stemming" — that would report the baseline as the ablation.

    Uses a spec with a stemmer but no stopword resource, so this exercises the
    stemmer path alone. `da_stem` would also trip the missing Danish stopword
    list (OQ-8) and the test would pass for the wrong reason.
    """
    if HAS_SNOWBALL:
        pytest.skip("snowballstemmer is installed; the degradation path cannot be exercised")
    spec = config.TokeniserSpec(name="probe", language="da", stemmer="danish")
    with pytest.raises(ImportError):
        tokenise.tokenise("biler", spec)
