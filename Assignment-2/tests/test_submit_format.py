"""Q5.1-Q5.3 — the prediction file formats.

The failure guarded against is a submission that is **accepted and scored** while
being wrong. Both competitions convert ranks to scores with `1/rank`, and both
tolerate more than they should: MIND fills a blank line with all-1 ranks rather
than rejecting the file. So a dropped impression, or an inverse permutation
written the right way round, produces a well-formed file, a real leaderboard
number, and no error anywhere.

Every test here therefore checks the artefact rather than the intent, and the
inverse-permutation tests use a hand-worked asymmetric example — on a symmetric
one the permutation and its inverse coincide and the test proves nothing.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from newsrec.submit import ebnerd_format, mind_format
from newsrec.submit import format as fmt

# -----------------------------------------------------------------------------
# The worked example from q5_plan.md
#
# Candidates [A, B, C, D]; we rank B best, then C, then A, then D.
#   order (best-first indices) = [1, 2, 0, 3]
#   ranks (inverse)            = [3, 1, 2, 4]
# The wrong answer — writing the permutation itself — is [2, 3, 1, 4].
# -----------------------------------------------------------------------------
WORKED_ORDER = np.array([1, 2, 0, 3])
WORKED_RANKS = [3, 1, 2, 4]
WORKED_WRONG = [2, 3, 1, 4]


def test_ranks_are_the_inverse_permutation_not_the_permutation():
    got = fmt.ranks_from_scores(WORKED_ORDER, 4)
    assert list(got) == WORKED_RANKS
    assert list(got) != WORKED_WRONG, "wrote the ranked order instead of the ranks"


def test_the_worked_example_is_asymmetric_so_the_test_can_fail():
    """A canary for the test above.

    On a symmetric permutation the inverse equals the original and the
    assertion would hold for a wrong implementation too.
    """
    assert WORKED_RANKS != WORKED_WRONG
    assert list(np.asarray(WORKED_ORDER) + 1) != WORKED_RANKS


def test_the_best_candidate_gets_rank_one():
    ranks = fmt.ranks_from_scores(np.array([2, 0, 1]), 3)
    assert ranks[2] == 1, "candidate 2 was ranked best, so it must carry rank 1"


def test_ranks_round_trip_back_to_the_order():
    for n in (1, 2, 5, 50):
        rng = np.random.default_rng(n)
        order = rng.permutation(n)
        ranks = fmt.ranks_from_scores(order, n)
        assert list(np.argsort(ranks)) == list(order)


def test_ranks_are_a_permutation_of_one_to_n():
    ranks = fmt.ranks_from_scores(np.array([3, 0, 2, 1]), 4)
    assert sorted(ranks) == [1, 2, 3, 4]


@pytest.mark.parametrize("order", [[0, 0, 1], [0, 1, 1], [0, 1, 3], [-1, 0, 1]])
def test_a_malformed_order_raises_rather_than_repeating_a_rank(order):
    """`1/rank` needs distinct ranks, so a duplicate is not recoverable.

    The check has to read the *input*. Scattering a non-permutation leaves some
    output entries never written, and `np.empty` fills those with whatever was
    in that memory — so validating the result passes or fails by luck. This test
    caught exactly that.
    """
    with pytest.raises(fmt.PredictionFormatError, match="permutation"):
        fmt.ranks_from_scores(np.array(order), 3)


def test_a_wrong_length_order_raises():
    with pytest.raises(fmt.PredictionFormatError, match="expected"):
        fmt.ranks_from_scores(np.array([0, 1]), 3)


# -----------------------------------------------------------------------------
# The line
# -----------------------------------------------------------------------------


def test_format_line_matches_the_published_example_byte_for_byte():
    """Both competitions publish this shape; no spaces inside the array."""
    assert fmt.format_line(1, [3, 1, 2, 4]) == "1 [3,1,2,4]"
    assert fmt.format_line(237, [2, 3, 1]) == "237 [2,3,1]"


def test_format_line_accepts_a_string_id():
    """EB-NeRD ids are integers; MIND's are strings in our schema. One writer."""
    assert fmt.format_line("42", [1]) == "42 [1]"


def test_a_line_round_trips_through_the_parser():
    line = fmt.format_line(7, [2, 1, 3])
    assert fmt.parse_line(line) == ("7", [2, 1, 3])


# -----------------------------------------------------------------------------
# Writing
# -----------------------------------------------------------------------------


def test_write_lines_refuses_a_blank_line(tmp_path):
    """The hazard, not a convenience.

    MIND's scorer fills a blank line with all-1 ranks and scores it, so a bug
    that dropped an impression would produce a disappointing number rather than
    a rejection. Being stricter than the scorer is the point.
    """
    path = tmp_path / "prediction.txt"
    with pytest.raises(fmt.PredictionFormatError, match="blank line"):
        fmt.write_lines(iter(["1 [1]", "", "3 [1]"]), path)


def test_write_lines_refuses_to_write_nothing(tmp_path):
    with pytest.raises(fmt.PredictionFormatError, match="no lines"):
        fmt.write_lines(iter([]), tmp_path / "prediction.txt")


def test_written_file_ends_with_a_newline_and_has_no_header(tmp_path):
    path = tmp_path / "prediction.txt"
    fmt.write_lines(iter(["1 [1,2]", "2 [2,1]"]), path)
    text = path.read_text(encoding="utf-8")
    assert text == "1 [1,2]\n2 [2,1]\n"
    assert not text.startswith("impression")


# -----------------------------------------------------------------------------
# Q5.3 — exact coverage
# -----------------------------------------------------------------------------


def _write(tmp_path, lines):
    path = tmp_path / "prediction.txt"
    fmt.write_lines(iter(lines), path)
    return path


def test_impression_coverage_exact(tmp_path):
    """The named Q5.3 condition: every required row exactly once, in order."""
    path = _write(tmp_path, ["a [1,2]", "b [2,1,3]", "c [1]"])
    got = fmt.validate_file(path, ["a", "b", "c"], [2, 3, 1])
    assert got == {"lines": 3, "rank_integers": 6}


def test_a_missing_row_is_caught(tmp_path):
    """The dangerous direction: the scorer fills the gap rather than rejecting."""
    path = _write(tmp_path, ["a [1,2]", "b [2,1,3]"])
    with pytest.raises(fmt.PredictionFormatError, match="has 2 lines but more rows"):
        fmt.validate_file(path, ["a", "b", "c"], [2, 3, 1])


def test_an_extra_row_is_caught(tmp_path):
    path = _write(tmp_path, ["a [1,2]", "b [2,1,3]", "c [1]"])
    with pytest.raises(fmt.PredictionFormatError, match="more lines"):
        fmt.validate_file(path, ["a", "b"], [2, 3])


def test_rows_out_of_order_are_caught(tmp_path):
    """Both competitions match positionally, so order is part of correctness."""
    path = _write(tmp_path, ["b [2,1,3]", "a [1,2]"])
    with pytest.raises(fmt.PredictionFormatError, match="impression_id"):
        fmt.validate_file(path, ["a", "b"], [2, 3])


def test_a_wrong_rank_count_is_caught(tmp_path):
    path = _write(tmp_path, ["a [1,2]"])
    with pytest.raises(fmt.PredictionFormatError, match="2 ranks for 3 candidates"):
        fmt.validate_file(path, ["a"], [3])


def test_duplicate_ranks_within_a_line_are_caught(tmp_path):
    """A tie that survived to the file. `1/rank` cannot express it."""
    path = _write(tmp_path, ["a [1,1,2]"])
    with pytest.raises(fmt.PredictionFormatError, match=r"permutation of 1\.\.3"):
        fmt.validate_file(path, ["a"], [3])


def test_ranks_starting_at_zero_are_caught(tmp_path):
    """Off-by-one from a 0-indexed argsort — a well-formed, wrong file."""
    path = _write(tmp_path, ["a [0,1,2]"])
    with pytest.raises(fmt.PredictionFormatError, match=r"permutation of 1\.\.3"):
        fmt.validate_file(path, ["a"], [3])


def test_repeated_impression_ids_validate_positionally(tmp_path):
    """EB-NeRD's 200,000 beyond-accuracy rows ALL carry impression_id 0.

    Anything keyed by id collapses them to one line and loses the entire
    beyond-accuracy subset silently. The validator must accept a file with
    repeated ids and must still count them.
    """
    path = _write(tmp_path, ["0 [1,2]", "0 [2,1]", "0 [1,2]"])
    got = fmt.validate_file(path, ["0", "0", "0"], [2, 2, 2])
    assert got["lines"] == 3


def test_a_deduplicated_beyond_accuracy_file_is_caught(tmp_path):
    """The concrete bug the hazard describes: three rows written as one."""
    path = _write(tmp_path, ["0 [1,2]"])
    with pytest.raises(fmt.PredictionFormatError, match="has 1 lines but more rows"):
        fmt.validate_file(path, ["0", "0", "0"], [2, 2, 2])


# -----------------------------------------------------------------------------
# Per-competition specifics
# -----------------------------------------------------------------------------


def test_mind_format(tmp_path):
    """The named Q5.1 condition: line format, and the archive layout.

    The archive half was learned the hard way — our first upload was rejected
    while every one of its 2,370,727 prediction lines was correct.
    """
    assert mind_format.FILENAME == "prediction.txt"
    assert mind_format.ZIPPED is True
    assert mind_format.MEMBER == "prediction.txt"
    assert mind_format.COMPETITION == "13967"

    path = mind_format.output_path(tmp_path)
    fmt.write_lines(iter([fmt.format_line(1, [3, 1, 2, 4]), fmt.format_line(2, [1, 2])]), path)
    assert path.read_text(encoding="utf-8") == "1 [3,1,2,4]\n2 [1,2]\n"
    fmt.validate_file(path, ["1", "2"], [4, 2])

    archive = fmt.zip_prediction(
        path, mind_format.archive_path(tmp_path), member_dir=mind_format.MEMBER_DIR
    )
    assert archive.name == "mind_submission.zip"
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["prediction.txt"], (
            "13967 wants the bare filename: Codabench extracts the archive into "
            "its own res/, so a res/ inside the zip nests one level too deep"
        )
        assert zf.read("prediction.txt").decode() == "1 [3,1,2,4]\n2 [1,2]\n"
    fmt.validate_archive(archive, mind_format.MEMBER, 2)


def test_a_mind_archive_with_a_res_prefix_is_rejected(tmp_path):
    """The exact shape of our SECOND rejected submission.

    `res/prediction.txt` inside the zip is a perfectly valid archive holding a
    perfectly valid prediction, and the scorer cannot find it — Codabench has
    already put the archive's contents in `res/`, so this lands at
    `/app/input/res/res/prediction.txt`.

    Kept as a test because the failing path and the required path look almost
    identical in the error message, and the wrong reading is the plausible one.
    """
    path = mind_format.output_path(tmp_path)
    fmt.write_lines(iter([fmt.format_line(1, [1])]), path)
    archive = fmt.zip_prediction(path, mind_format.archive_path(tmp_path), member_dir="res")

    with pytest.raises(fmt.PredictionFormatError, match=r"does not contain"):
        fmt.validate_archive(archive, mind_format.MEMBER, 1)


def test_ebnerd_format(tmp_path):
    """The named Q5.2 condition: filename, zipped, and the arcname inside it."""
    assert ebnerd_format.FILENAME == "predictions.txt"
    assert ebnerd_format.ZIPPED is True
    assert ebnerd_format.MEMBER == "predictions.txt"
    assert ebnerd_format.COMPETITION == "2469"

    path = ebnerd_format.output_path(tmp_path)
    fmt.write_lines(iter([fmt.format_line(237, [2, 3, 1]), fmt.format_line(291, [2, 1])]), path)
    archive = fmt.zip_prediction(
        path, ebnerd_format.archive_path(tmp_path), member_dir=ebnerd_format.MEMBER_DIR
    )

    assert archive.name == "predictions.zip"
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["predictions.txt"], "2469 wants the file at the archive root"
        assert zf.read("predictions.txt").decode() == "237 [2,3,1]\n291 [2,1]\n"
    fmt.validate_archive(archive, ebnerd_format.MEMBER, 2)


def test_both_competitions_want_the_prediction_at_the_archive_root():
    """They agree on the layout — but only as a measured fact.

    MIND's was established by a live scorer accepting it; EB-NeRD's comes from
    the organisers' `zip_submission_file`. Same answer, different and
    independent evidence, so the constants stay separate rather than one being
    derived from the other. The filenames still differ by one letter.
    """
    assert mind_format.MEMBER_DIR == ""
    assert ebnerd_format.MEMBER_DIR == ""
    assert mind_format.MEMBER == "prediction.txt"
    assert ebnerd_format.MEMBER == "predictions.txt"
    assert mind_format.MEMBER != ebnerd_format.MEMBER


def test_validate_archive_catches_a_short_member(tmp_path):
    """A truncated prediction inside a well-formed zip."""
    path = tmp_path / "predictions.txt"
    fmt.write_lines(iter(["1 [1]", "2 [1]"]), path)
    archive = fmt.zip_prediction(path, tmp_path / "predictions.zip")
    with pytest.raises(fmt.PredictionFormatError, match="2 lines, expected 3"):
        fmt.validate_archive(archive, "predictions.txt", 3)


def test_validate_archive_rejects_a_file_that_is_not_a_zip(tmp_path):
    path = tmp_path / "predictions.txt"
    fmt.write_lines(iter(["1 [1]"]), path)
    with pytest.raises(fmt.PredictionFormatError, match="not a zip"):
        fmt.validate_archive(path, "predictions.txt", 1)


def test_the_archive_never_carries_the_local_directory(tmp_path):
    """`ZipFile.write` stores the path as given, so `predictions/prediction.txt`
    would end up inside the archive if the arcname were left to the default.
    That matches neither competition."""
    nested = tmp_path / "predictions"
    nested.mkdir()
    path = nested / "prediction.txt"
    fmt.write_lines(iter(["1 [1]"]), path)

    archive = fmt.zip_prediction(path, tmp_path / "a.zip", member_dir="res")
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["res/prediction.txt"]


def test_the_two_competitions_do_not_share_a_filename():
    """`prediction.txt` and `predictions.txt` differ by one letter.

    Uploading the wrong one is a rejection at best. Pinned so a well-meaning
    tidy-up cannot unify them.
    """
    assert mind_format.FILENAME != ebnerd_format.FILENAME


def test_zip_refuses_a_non_zip_archive_name(tmp_path):
    path = _write(tmp_path, ["a [1]"])
    with pytest.raises(fmt.PredictionFormatError, match=r"does not end in \.zip"):
        fmt.zip_prediction(path, tmp_path / "predictions.tar")


def test_the_beyond_accuracy_constants_match_what_was_measured():
    """13,536,710 test rows, of which 200,000 are beyond-accuracy and all carry
    impression_id 0. If the test set is ever re-downloaded and these change, the
    writer's positional assumption needs re-checking."""
    assert ebnerd_format.BEYOND_ACCURACY_IMPRESSION_ID == "0"
    assert ebnerd_format.BEYOND_ACCURACY_ROWS == 200_000


# -----------------------------------------------------------------------------
# Q5.4 — what is committed, and what deliberately is not
# -----------------------------------------------------------------------------


def test_predictions_committed_and_checksummed():
    """The named Q5.4 condition, as amended by OQ-11.

    The artefacts are NOT committed and that is deliberate: measured, MIND's
    prediction is 291 MB (107 MB gzipped) and EB-NeRD's is ~0.96 GB. D6 assumed
    both would fit under 1 MB gzipped; neither is within two orders of magnitude
    of it, and committing either would break invariant 3.

    So what must be present is the evidence: a checksum for every prediction the
    manifest claims, and a regeneration command for each. This test asserts the
    manifest and the checksum file agree, which is the property that would break
    if a prediction were regenerated and only one of the two updated.
    """
    import json

    from newsrec import config

    sums = config.PREDICTIONS_DIR / "SHA256SUMS"
    manifest_path = config.PREDICTIONS_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("no predictions generated yet; run `make submit-baseline`")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["predictions"], "manifest lists no predictions"

    assert sums.exists(), "manifest lists predictions but SHA256SUMS is absent"
    checksums = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        checksums[name] = digest

    for name, entry in manifest["predictions"].items():
        assert name in checksums, f"{name} is in the manifest but not in SHA256SUMS"
        assert checksums[name] == entry["sha256"], (
            f"{name}: manifest and SHA256SUMS disagree, so one was regenerated without the other"
        )
        assert entry["regenerate"], f"{name} records no regeneration command"
        assert entry["lines"] > 0

    for name in checksums:
        assert name in manifest["predictions"], f"{name} is checksummed but undocumented"


def test_the_prediction_artefacts_are_ignored_by_git(repo_root):
    """`.gitignore` must actively ignore them, not merely happen not to track them.

    This is the gap that bit: the first version of the ignore rules re-included
    `predictions/` wholesale and relied on a comment claiming the raw forms were
    "ignored above by extension". Nothing ignores `.txt`, so `git add -A` staged
    a 291 MB file. The tracked-files test below passed throughout, because the
    file was staged and not yet committed.

    So this asks git the question directly, for a path that need not exist.
    """
    import subprocess

    for name in ("prediction.txt", "predictions.txt", "predictions.zip"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", f"predictions/{name}"],
            cwd=repo_root,
            check=False,
        )
        assert result.returncode == 0, (
            f"predictions/{name} is NOT ignored; `git add -A` would stage a "
            f"multi-hundred-megabyte artefact into history"
        )


def test_the_prediction_artefacts_are_not_committed_to_git(repo_root):
    """Invariant 3, checked directly rather than trusted to .gitignore.

    A 291 MB text file added by accident is the kind of thing that is very hard
    to remove from history later.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "predictions/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for path in tracked:
        assert not path.endswith(".txt"), f"{path} is a prediction artefact and is tracked"
        assert not path.endswith(".zip"), f"{path} is a prediction artefact and is tracked"


# -----------------------------------------------------------------------------
# The EB-NeRD baseline's vectorised ranking
# -----------------------------------------------------------------------------


def test_the_polars_ranking_agrees_with_the_numpy_inverse_permutation():
    """Two independent implementations of the same contract must agree.

    MIND's path builds the inverse permutation explicitly with numpy;
    EB-NeRD's uses `rank("ordinal")` in polars, which produces it directly
    because a rank IS "this element's position in the order". They are written
    differently enough that agreeing on the asymmetric cases is real evidence
    rather than a tautology.
    """
    import numpy as np
    import polars as pl

    from newsrec.submit import baseline

    rows = [[30, 10, 20], [5, 7], [9], [4, 3, 2, 1], [100, 1, 50, 2]]
    got = baseline.ebnerd_baseline_ranks(pl.Series("c", rows, dtype=pl.List(pl.Int32))).to_list()
    want = [[int(x) for x in fmt.ranks_from_scores(np.argsort(np.array(r)), len(r))] for r in rows]
    assert got == want


def test_the_vectorised_ranks_are_a_permutation_of_one_to_n():
    import polars as pl

    from newsrec.submit import baseline

    rows = [[30, 10, 20], [5, 7], [9], [8, 8 + 1, 8 + 2]]
    got = baseline.ebnerd_baseline_ranks(pl.Series("c", rows, dtype=pl.List(pl.Int32))).to_list()
    for row, ranks in zip(rows, got, strict=True):
        assert sorted(ranks) == list(range(1, len(row) + 1))


def test_the_ebnerd_shape_guard_fires_when_the_test_set_changes(tmp_path, monkeypatch):
    """The positional writer rests on measured facts about this exact file.

    If a re-download shipped a different row count, or stopped collapsing the
    beyond-accuracy rows onto id 0, the writer would still be correct but every
    count recorded in q5_plan and the manifest would be wrong. Cheap to check,
    invisible if unchecked.
    """
    import polars as pl

    from newsrec.submit import ebnerd_test

    path = tmp_path / "behaviors.parquet"
    pl.DataFrame(
        {
            "impression_id": [1, 2],
            "article_ids_inview": [[10, 11], [12]],
            "is_beyond_accuracy": [False, False],
        }
    ).write_parquet(path)

    with pytest.raises(ebnerd_test.EbnerdTestError, match="rows, expected"):
        ebnerd_test.check_shape(path)


def test_ebnerd_chunks_come_back_in_file_order_and_cover_every_row(tmp_path):
    """Row order is the contract; a chunked reader that reordered would be a
    silent, total corruption of the submission."""
    import polars as pl

    from newsrec.submit import ebnerd_test

    path = tmp_path / "behaviors.parquet"
    ids = list(range(250))
    pl.DataFrame(
        {"impression_id": ids, "article_ids_inview": [[i, i + 1] for i in ids]}
    ).write_parquet(path)

    seen = []
    for frame in ebnerd_test.iter_chunks(path, chunk=37):
        seen.extend(frame["impression_id"].to_list())
    assert seen == ids
