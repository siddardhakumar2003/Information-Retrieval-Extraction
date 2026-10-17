"""A2 Q5.5 — the two-stage submission writer's positional contract.

Both competitions match the prediction file to a truth file we do not have, **by
line position**. So the only property that really matters here is that line *i*
of the output describes row *i* of the input, with a rank for each of that row's
candidates and no others.

Everything that can break that is invisible in the output: a reordered batch, a
dict keyed by something that is not unique per row, a rank vector that is not a
permutation. A1 shipped exactly this class of bug once — a submission that was
valid, accepted and byte-identical to the baseline while every user-side id
lookup had silently failed (`design_rationale.md` §1.21).

These tests do not need the 13.5 million-row files. They exercise the writer on
hand-built rows, including the two shapes the real files contain that a naive
implementation gets wrong: duplicate impression ids, and candidate lists of
wildly different lengths in one batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from newsrec import config
from newsrec.rerank import nrms
from newsrec.submit import encode_test, nrms_predict
from newsrec.submit import format as fmt

EMBED_DIM = 8
N_ARTICLES = 40
N_CATEGORIES = 4


def _resources(device: str = "cpu") -> dict:
    """A tiny stand-in for `build_resources`, with the same keys."""
    torch.manual_seed(0)
    article_ids = [f"a{j}" for j in range(N_ARTICLES)]
    rng = np.random.default_rng(5)
    net = nrms.NRMSNet(
        embedding_dim=EMBED_DIM,
        dim=8,
        heads=2,
        attention_dim=4,
        n_categories=N_CATEGORIES,
        category_dim=2,
        dropout=0.0,
    )
    model = nrms.NRMSModel(
        net=net,
        device=device,
        category_aware=True,
        n_categories=N_CATEGORIES,
        embedding_dim=EMBED_DIM,
        dim=8,
        heads=2,
        attention_dim=4,
        category_dim=2,
    )
    return {
        "model": model,
        "embeddings": type(
            "E",
            (),
            {
                "vectors": rng.standard_normal((N_ARTICLES, EMBED_DIM)).astype(np.float32),
                "article_ids": article_ids,
                "n_articles": N_ARTICLES,
                "dim": EMBED_DIM,
            },
        )(),
        "row_of_article": {a: i for i, a in enumerate(article_ids)},
        "category_of_row": rng.integers(0, N_CATEGORIES, size=N_ARTICLES),
        "category_coverage": 1.0,
        "n_categories": N_CATEGORIES,
        "dataset": "mind_small",
        "device": device,
    }


def _rows(n: int = 30, seed: int = 1):
    """`(impression_id, candidates, history)` with deliberately awkward shapes:
    varying candidate counts, some empty histories, and DUPLICATE impression
    ids — all three occur in the real files."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        width = int(rng.integers(1, 12))
        candidates = [f"a{j}" for j in rng.choice(N_ARTICLES, size=width, replace=False)]
        history = [f"a{j}" for j in rng.choice(N_ARTICLES, size=int(rng.integers(0, 6)))]
        # Every third row reuses id "0", mimicking EB-NeRD's 200,000
        # beyond-accuracy rows.
        impression_id = "0" if i % 3 == 0 else str(1000 + i)
        out.append((impression_id, candidates, history))
    return out


def _write(rows, resources=None, tally=None) -> list[str]:
    resources = resources or _resources()
    tally = (
        tally
        if tally is not None
        else {
            "rows": 0,
            "candidate_slots": 0,
            "candidates_resolved": 0,
            "users_with_history": 0,
        }
    )
    return list(nrms_predict._lines(resources, iter(rows), tally))


# =============================================================================
# The positional contract
# =============================================================================


def test_one_line_is_written_per_input_row():
    rows = _rows()
    assert len(_write(rows)) == len(rows)


def test_each_line_carries_its_own_rows_impression_id_in_order():
    """Line i must describe row i. A reordered batch would still produce the
    right number of well-formed lines."""
    rows = _rows()
    lines = _write(rows)
    for (impression_id, _, _), line in zip(rows, lines, strict=True):
        got_id, _ = fmt.parse_line(line)
        assert got_id == impression_id


def test_each_line_has_exactly_one_rank_per_candidate():
    rows = _rows()
    for (_, candidates, _), line in zip(rows, _write(rows), strict=True):
        _, ranks = fmt.parse_line(line)
        assert len(ranks) == len(candidates)


def test_every_line_is_a_permutation_of_one_to_n():
    """A rank vector that repeats or skips a rank is not a ranking. The MIND
    scorer would still read it and return a number."""
    rows = _rows()
    for (_, candidates, _), line in zip(rows, _write(rows), strict=True):
        _, ranks = fmt.parse_line(line)
        assert sorted(ranks) == list(range(1, len(candidates) + 1))


def test_duplicate_impression_ids_get_their_own_histories():
    """EB-NeRD's test file has 13,536,710 rows and 13,336,711 distinct
    impression ids: all 200,000 beyond-accuracy rows carry id 0.

    A history dict keyed by impression id would hand all of them the same
    user's clicks. Here two rows share an id and have deliberately different
    histories over disjoint articles, so if they were conflated both rows would
    score identically.
    """
    resources = _resources()
    source = nrms_predict._source(resources)
    candidates = [[f"a{j}" for j in range(4)]] * 2
    histories = [["a5", "a6", "a7"], ["a30", "a31", "a32"]]

    scored, _ = nrms_predict._score_chunk(resources, source, candidates, histories)

    # Asserted on the SCORES, not on the emitted ranking. With only four
    # candidates two different user vectors can easily produce the same
    # permutation by coincidence — which is exactly what happened when this
    # test first checked the output lines, and it made a passing implementation
    # look broken.
    assert set(source.history_rows) == {"0", "1"}, (
        "the history dict is keyed by something that is not unique per row"
    )
    assert not np.allclose(scored[0], scored[1]), (
        "two rows sharing impression_id 0 scored identically despite disjoint "
        "click histories — the history is being keyed by impression id"
    )


def test_bucketing_by_candidate_width_does_not_change_any_line():
    """The chunk is sorted by candidate count before scoring, to stop short
    lists being padded out to the longest one, and unsorted afterwards. This
    asserts the round trip: scoring 30 mixed-width rows in one bucketed chunk
    must give the same lines as scoring them one at a time.
    """
    rows = _rows()
    resources = _resources()
    together = _write(rows, resources)
    separately = [_write([row], _resources())[0] for row in rows]
    assert together == separately


def test_chunk_boundaries_do_not_change_any_line(monkeypatch):
    """The reader is chunked, so a row must not be affected by which chunk it
    landed in."""
    rows = _rows(n=25)
    baseline = _write(rows, _resources())
    for chunk, batch in ((3, 2), (7, 3), (100, 100)):
        monkeypatch.setattr(nrms_predict, "CHUNK_IMPRESSIONS", chunk)
        monkeypatch.setattr(nrms_predict, "BATCH_IMPRESSIONS", batch)
        assert _write(rows, _resources()) == baseline, f"chunk={chunk} batch={batch}"


def test_a_row_with_no_usable_history_still_gets_a_full_ranking():
    """A cold-start user must not drop a line — the scorer fills a missing line
    with all-1 ranks and scores it, so the submission would come back with a
    real number and a silent hole in it (`submit/mind_test.py`)."""
    rows = [("77", ["a1", "a2", "a3"], [])]
    line = _write(rows)[0]
    got_id, ranks = fmt.parse_line(line)
    assert got_id == "77"
    assert sorted(ranks) == [1, 2, 3]


def test_a_candidate_with_no_embedding_row_is_ranked_not_dropped():
    """Dropping it would change that row's candidate count and break the
    positional contract with the truth file."""
    rows = [("88", ["a1", "never_embedded", "a2"], ["a3"])]
    _, ranks = fmt.parse_line(_write(rows)[0])
    assert sorted(ranks) == [1, 2, 3]


# =============================================================================
# The coverage tallies — the guards that would have caught A1's §1.21 bug
# =============================================================================


def test_the_tally_counts_resolvable_candidates_and_histories():
    rows = _rows()
    tally = {
        "rows": 0,
        "candidate_slots": 0,
        "candidates_resolved": 0,
        "users_with_history": 0,
    }
    _write(rows, _resources(), tally)
    assert tally["rows"] == len(rows)
    assert tally["candidate_slots"] == sum(len(r[1]) for r in rows)
    # Every candidate in the fixture is embeddable, so these must agree.
    assert tally["candidates_resolved"] == tally["candidate_slots"]
    assert tally["users_with_history"] == sum(1 for r in rows if r[2])


def test_unresolvable_candidates_are_counted_as_such():
    rows = [("1", ["a1", "ghost", "phantom"], ["a2"])]
    tally = {
        "rows": 0,
        "candidate_slots": 0,
        "candidates_resolved": 0,
        "users_with_history": 0,
    }
    _write(rows, _resources(), tally)
    assert tally["candidate_slots"] == 3
    assert tally["candidates_resolved"] == 1


def test_the_coverage_floors_are_set_below_one_but_well_above_zero():
    """A1's failure read 1.0000 on one side and 0.0 on the other, so a floor of
    0 would not have caught it and a floor of 1 would refuse legitimate runs
    (a test corpus does legitimately contain unembeddable articles)."""
    assert 0.0 < nrms_predict.MIN_CANDIDATE_COVERAGE < 1.0
    assert 0.0 < nrms_predict.MIN_USER_COVERAGE < 1.0


# =============================================================================
# Declared configuration
# =============================================================================


def test_each_competition_is_mapped_to_the_corpus_its_model_was_trained_on():
    assert nrms_predict.TRAINED_ON[encode_test.MIND_KEY] == "mind_small"
    assert nrms_predict.TRAINED_ON[encode_test.EBNERD_KEY] == "ebnerd_small"


def test_the_shipped_variant_is_one_the_ablation_actually_trained():
    assert nrms_predict.VARIANT in config.NRMS_VARIANTS


def test_the_history_limit_comes_from_config():
    from pathlib import Path

    text = Path(nrms_predict.__file__).read_text(encoding="utf-8")
    assert "config.QUERY_HISTORY_LIMIT" in text


def test_inference_falls_back_to_cpu_unlike_training(monkeypatch):
    """Training raises without a GPU on purpose; scoring must not. A CPU-only
    grader has to be able to regenerate the submission — slowly is acceptable,
    impossible is not."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert nrms_predict._inference_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert nrms_predict._inference_device() == "cuda"


@pytest.mark.realdata
def test_the_category_map_is_rebuilt_from_the_training_corpus():
    """The learned category table has one row per training-corpus category. A
    test-corpus label must be looked up through that same map or the model
    reads a row belonging to a different category, in range and wrong."""
    index = nrms_predict.category_index_from_training_corpus("mind_small")
    assert index, "no categories found in the training corpus"
    assert min(index.values()) == 1, "index 0 must stay reserved for 'no category'"
    assert len(set(index.values())) == len(index), "two categories share an index"
