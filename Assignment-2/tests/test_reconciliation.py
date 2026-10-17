"""Q1.10-Q1.11 — nothing is dropped silently.

A parser allowed to discard a malformed row quietly turns a 30,000-row loss into
a number that merely looks a bit low. These assertions are the reason
_manifest.json exists.
"""

from __future__ import annotations

import json

import pytest


def test_no_silent_drops(processed_store, dataset):
    """Q1.10 — raw == parsed + rejected, with rejects counted BY REASON."""
    m = json.loads((processed_store / "_manifest.json").read_text())
    counts = m["counts"]
    assert counts["raw"] == counts["parsed"] + counts["rejected"]
    assert sum(m["rejections"].values()) == counts["rejected"], "rejects not attributed to reasons"


def test_split_sizes_sum_to_total(processed_store, dataset):
    m = json.loads((processed_store / "_manifest.json").read_text())
    assert sum(m["splits"].values()) == m["counts"]["parsed"]


def test_article_dedupe_accounted(processed_store, dataset):
    m = json.loads((processed_store / "_manifest.json").read_text())
    a = m["articles"]
    assert a["raw"] - a["duplicates"] == a["unique"]


@pytest.mark.realdata
def test_mind_known_counts(processed_store, dataset):
    """Q1.11 — a canary against a misparse.

    Counted from behaviors.tsv on 2026-08-19. If a parser change moves these,
    it is a bug until proven otherwise.
    """
    if dataset != "mind_small":
        pytest.skip("MIND-specific canary")
    m = json.loads((processed_store / "_manifest.json").read_text())
    assert m["counts"]["raw"] == 156965 + 73152
    assert m["splits"] == {"train": 126695, "val": 30270, "test": 73152}
    assert m["articles"]["raw"] >= 51282
