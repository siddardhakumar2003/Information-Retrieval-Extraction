"""Single source of truth for paths, the dataset registry, seeds, split boundaries,
and the two policy sets that Q8 and Q9 are graded on.

This module is *declarative configuration*, not pipeline code — every value here
is a constant or a small dataclass. It is written during setup, before any
implementation exists, because three separate consumers must agree on these
values and would otherwise each hard-code their own copy:

  * the pre-commit hook and ``scripts/check_no_data_files.py`` read
    ``FORBIDDEN_PATH_PATTERNS`` (Q8.2);
  * the leakage tests read ``SERVING_UNAVAILABLE_FEATURES`` (Q9.2);
  * the split tests read ``DATASETS[...].split`` and assert the produced
    boundaries equal it rather than a literal (Q1.8).

Changing a value here must change behaviour. If a boundary or a seed is repeated
anywhere else in the repo, that is a bug.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

#: Repository root, resolved from this file rather than from the cwd, so the CLI
#: and the tests agree no matter where they are invoked from.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    """Allow every root to be relocated, so a grader with a small disk can point
    the heavy directories elsewhere without editing tracked files."""
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


#: Immutable raw zone. `make download` writes here; NOTHING else ever does.
#: Keeping raw inputs read-only is what makes a full rebuild always possible.
RAW_DIR: Path = _env_path("NEWSREC_RAW_DIR", REPO_ROOT / "Data")

#: Everything generated. Safe to delete: `make clean-derived` does exactly that.
DATA_DIR: Path = _env_path("NEWSREC_DATA_DIR", REPO_ROOT / "data")
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURE_STORE_DIR: Path = DATA_DIR / "feature_store"

#: Large binary build products: BM25 index, embedding matrices, ANN indexes.
ARTIFACTS_DIR: Path = _env_path("NEWSREC_ARTIFACTS_DIR", REPO_ROOT / "artifacts")

#: Committed outputs — small JSON/CSV/PNG only.
REPORTS_DIR: Path = REPO_ROOT / "reports"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
PREDICTIONS_DIR: Path = REPO_ROOT / "predictions"
DESIGN_NOTE_DIR: Path = REPO_ROOT / "design_note"
AI_USAGE_DIR: Path = REPO_ROOT / "ai_usage"


def repo_relative(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    `Path.relative_to` RAISES rather than falling back, which is wrong for every
    caller here: paths legitimately point outside the repo (a temp directory in
    a test, a relocated NEWSREC_RAW_DIR). This was written twice locally before
    being hoisted -- once in extract_ai_log.py and once in mind.py -- which is
    exactly the duplication CLAUDE.md forbids.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# Determinism (Q1.13)
# =============================================================================

#: Every stochastic step derives its seed from this one. No bare literal seeds.
GLOBAL_SEED: int = 20260827

#: Bootstrap resampling (Q4.6): resamples are over impressions, seeded.
BOOTSTRAP_N: int = 1000
BOOTSTRAP_CI: float = 0.95

# =============================================================================
# Q9 — features unavailable at serving time
#
# Declared before the dataset registry because the registry references it.
# =============================================================================

#: Fields recorded only AFTER a click happens. Knowing them at ranking time is
#: knowing the outcome you are trying to predict.
POST_CLICK_FEATURES: frozenset[str] = frozenset(
    {
        "read_time",
        "scroll_percentage",
        "next_read_time",
        "next_scroll_percentage",
    }
)

#: Dataset-period aggregates. These are computed over the WHOLE corpus window,
#: so every row carries popularity from its own future. Subtler than the
#: post-click set and easier to use by accident, which is why it is named here.
FUTURE_AGGREGATE_FEATURES: frozenset[str] = frozenset(
    {
        "total_inviews",
        "total_pageviews",
        "total_read_time",
    }
)

#: The Q9.1 set. Usable ONLY inside a frame explicitly flagged as an ablation;
#: never in a headline number. tests/test_leakage.py asserts this list matches
#: what the pipeline actually excludes.
SERVING_UNAVAILABLE_FEATURES: frozenset[str] = POST_CLICK_FEATURES | FUTURE_AGGREGATE_FEATURES

# =============================================================================
# Dataset registry
# =============================================================================


@dataclass(frozen=True)
class SplitSpec:
    """Temporal split boundaries, as tz-aware UTC **datetimes**.

    The rule is identical for both datasets: split the *source train* period by
    timestamp into train (earlier) and val (its final day), and use the *source
    validation/dev* period as test. Interaction data is NEVER split randomly
    (Q1.6), so there is deliberately no fraction or shuffle parameter here — the
    type itself makes a random split unrepresentable.

    Datetimes rather than dates, and this is not fussiness. Verified from the
    data on 2026-08-19:

      * MIND periods are calendar-aligned — train runs 2019-11-09 00:00:19 to
        2019-11-14 23:59:13, dev is 2019-11-15.
      * **EB-NeRD periods break at 07:00 UTC, not midnight.** train ends
        2023-05-25 06:59:52 and validation begins 2023-05-25 07:00:15.

    A date-typed boundary is therefore correct for MIND and silently wrong for
    EB-NeRD: it would cut 25 May in half, putting the first seven hours of the
    *validation* period into our train split. That is future data in the
    training set — a leak that no test looking only at dates would ever catch,
    which is precisely why config.SplitSpec held a placeholder until the real
    timestamps could be read.
    """

    #: Impressions with time < ``val_start`` are train.
    val_start: datetime
    #: Impressions with ``val_start`` <= time < ``test_start`` are val.
    test_start: datetime
    #: Human-readable justification, quoted in the design note.
    rationale: str

    def __post_init__(self) -> None:
        for name in ("val_start", "test_start"):
            value = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(f"SplitSpec.{name} must be tz-aware UTC, got naive {value!r}")
        if self.val_start >= self.test_start:
            raise ValueError("SplitSpec: val_start must precede test_start")


@dataclass(frozen=True)
class DatasetSpec:
    """One entry in the dataset registry."""

    key: str
    language: str
    #: Directory under RAW_DIR holding this dataset's extracted files.
    raw_subdir: str
    split: SplitSpec
    #: True for the variant whose numbers are quoted as headline results.
    #: The other variants exist for fast iteration and for fixtures.
    headline: bool = False
    #: Columns present in this dataset that are unavailable at serving time.
    #: Subset of SERVING_UNAVAILABLE_FEATURES; empty when the dataset has none.
    unavailable_columns: frozenset[str] = field(default_factory=frozenset)


# MIND-small: train covers Nov 9-14 2019, dev is Nov 15 2019.
#   train = Nov 9-13, val = Nov 14, test = the dev period (Nov 15).
_MIND_SPLIT = SplitSpec(
    val_start=datetime(2019, 11, 14, tzinfo=UTC),
    test_start=datetime(2019, 11, 15, tzinfo=UTC),
    rationale=(
        "MINDsmall_train spans 2019-11-09 00:00:19 to 2019-11-14 23:59:13 and "
        "MINDsmall_dev is 2019-11-15 -- calendar-aligned, verified from "
        "behaviors.tsv. The final train day becomes val and dev becomes test, so "
        "our test set is the same period the MIND leaderboard evaluates. "
        "Sizes: train 126,695 / val 30,270 / test 73,152."
    ),
)

# EB-NeRD demo and small share one collection window, verified 2026-08-19 by
# reading impression_time from both behaviors.parquet files (OQ-4, now resolved):
#
#   train/       2023-05-18 07:00:01  ->  2023-05-25 06:59:58
#   validation/  2023-05-25 07:00:02  ->  2023-06-01 06:59:59
#
# The periods break at 07:00 UTC, NOT at midnight. val_start is therefore
# 05-24 07:00 (the final 24h of the train period) and test_start is 05-25 07:00
# (where the shipped validation period actually begins). Using midnight would
# put the first seven hours of the validation period into train.
_EBNERD_SPLIT = SplitSpec(
    val_start=datetime(2023, 5, 24, 7, 0, tzinfo=UTC),
    test_start=datetime(2023, 5, 25, 7, 0, tzinfo=UTC),
    rationale=(
        "EB-NeRD collection periods run 07:00-to-07:00 UTC, not midnight to "
        "midnight. train/ spans 2023-05-18 07:00 to 2023-05-25 07:00 and "
        "validation/ spans 2023-05-25 07:00 to 2023-06-01 07:00. val is the "
        "final 24h of the train period; test is the shipped validation period, "
        "which is the window the RecSys 2024 leaderboard evaluates. "
        "Sizes -- demo: 21,318 / 3,406 / 25,356; small: 200,328 / 32,559 / 244,647."
    ),
)

DATASETS: dict[str, DatasetSpec] = {
    "mind_small": DatasetSpec(
        key="mind_small",
        language="en",
        raw_subdir="MIND",
        split=_MIND_SPLIT,
        headline=True,
        unavailable_columns=frozenset(),  # MIND ships no post-click fields
    ),
    "ebnerd_demo": DatasetSpec(
        key="ebnerd_demo",
        language="da",
        raw_subdir="EB-NeRD/ebnerd_demo",
        split=_EBNERD_SPLIT,
        headline=False,
        unavailable_columns=SERVING_UNAVAILABLE_FEATURES,
    ),
    "ebnerd_small": DatasetSpec(
        key="ebnerd_small",
        language="da",
        raw_subdir="EB-NeRD/ebnerd_small",
        split=_EBNERD_SPLIT,
        headline=True,
        unavailable_columns=SERVING_UNAVAILABLE_FEATURES,
    ),
}

#: Datasets whose numbers are quoted as headline results (see OQ-3).
HEADLINE_DATASETS: tuple[str, ...] = tuple(k for k, d in DATASETS.items() if d.headline)


# =============================================================================
# Q2 — tokenisation (q2_plan.md D4)
# =============================================================================


@dataclass(frozen=True)
class TokeniserSpec:
    """One named tokenisation variant.

    D4 chose a single shared pipeline whose behaviour is *configured* per
    language rather than two separate tokenisers, so that a MIND-vs-EB-NeRD
    difference is a difference in the data and not in our own two
    implementations. This dataclass is that configuration.

    `name` is part of the index fingerprint: an index built under one variant
    must never be silently reused under another, exactly as the Q1 feature-store
    fingerprint works (see design_rationale.md 2.10).
    """

    name: str
    language: str
    lowercase: bool = True
    #: Stopword resource id, or None to keep every term. Resolved by
    #: `newsrec.retrieval.tokenise`, not here -- config declares, it does not load.
    stopwords: str | None = None
    #: Snowball algorithm name (e.g. "porter", "danish"), or None for no stemming.
    stemmer: str | None = None


#: Every tokenisation variant Q2 knows about.
#:
#: The two `_baseline` entries are the headline configuration; the two `_stem`
#: entries exist to be *measured* against them (Q2.9), never to be swapped in
#: quietly. Note the deliberate asymmetry in the baselines: English removes
#: stopwords, Danish does not. That is D4's choice, not an oversight -- the
#: Danish baseline is deliberately naive so the ablation has something to show.
TOKENISER_VARIANTS: dict[str, TokeniserSpec] = {
    "en_baseline": TokeniserSpec(
        name="en_baseline",
        language="en",
        stopwords="en",
    ),
    "en_stem": TokeniserSpec(
        name="en_stem",
        language="en",
        stopwords="en",
        stemmer="porter",
    ),
    "da_baseline": TokeniserSpec(
        name="da_baseline",
        language="da",
    ),
    "da_stem": TokeniserSpec(
        name="da_stem",
        language="da",
        stopwords="da",
        stemmer="danish",
    ),
}

#: Which variant each language uses for headline numbers. The ablation reads
#: `TOKENISER_VARIANTS` directly; nothing else may hard-code a variant name.
DEFAULT_TOKENISER: dict[str, str] = {"en": "en_baseline", "da": "da_baseline"}


def tokeniser_for(dataset_key: str) -> TokeniserSpec:
    """The headline tokenisation variant for a dataset.

    Goes through `DATASETS[...].language` on purpose. D4's test asserts that the
    language comes from the dataset registry rather than being written as a
    literal at the call site, because a hard-coded "en" in the EB-NeRD path is
    the kind of mistake that produces plausible-looking Danish results.
    """
    language = DATASETS[dataset_key].language
    return TOKENISER_VARIANTS[DEFAULT_TOKENISER[language]]


def tokeniser_variants_for(dataset_key: str) -> tuple[TokeniserSpec, ...]:
    """Every variant Q2.9's ablation should score for this dataset, headline first.

    Derived from `TOKENISER_VARIANTS` by language rather than written as a second
    hard-coded list. A new variant added above is therefore ablated automatically;
    a second list would have to be remembered, and the failure mode of forgetting
    is a report that silently covers less than it claims to.

    Headline first because the ablation reports deltas *against* the headline, and
    the baseline has to be identifiable without matching on the name.
    """
    language = DATASETS[dataset_key].language
    headline = DEFAULT_TOKENISER[language]
    others = [
        name
        for name, spec in TOKENISER_VARIANTS.items()
        if spec.language == language and name != headline
    ]
    return tuple(TOKENISER_VARIANTS[name] for name in [headline, *others])


#: Q2 D6 — how many of a user's most recent clicks become the query. Declared
#: here rather than written at the call site so the ablation sweeps a config
#: value and the design note can report which number produced which result.
QUERY_HISTORY_LIMIT: int = 20

#: Q2 D5 — BM25-Okapi parameters. The standard defaults, and deliberately the
#: same ones `rank_bm25.BM25Okapi` uses, so the oracle test in
#: tests/test_bm25_scoring.py is an exact comparison rather than an approximate
#: one. `k1` controls how fast term frequency saturates; `b` how strongly
#: document length is normalised (0 = not at all, 1 = fully).
BM25_K1: float = 1.2
BM25_B: float = 0.75

#: Okapi's classic idf goes NEGATIVE for a term appearing in more than half the
#: documents, which would mean penalising a document for containing a query
#: term. `rank_bm25.BM25Okapi` handles this by flooring every negative idf at
#: `epsilon * average_idf`, and we inherit that rule verbatim so the D7
#: correctness test is an exact comparison rather than an approximate one.
#: 0.25 is rank_bm25's own default. See design_rationale.md 1.15.
BM25_EPSILON: float = 0.25

#: Bumped when a change to the index format or the tokeniser pipeline makes
#: previously written indexes unreadable or wrong. Part of the index
#: fingerprint, so a stale index on disk is detected rather than silently reused
#: (the same discipline as STORE_VERSION in the Q1 feature store).
INDEX_VERSION: int = 1


# =============================================================================
# Q3 — semantic retrieval (q3_plan.md D1-D5)
# =============================================================================


@dataclass(frozen=True)
class EncoderSpec:
    """The article/user encoder, pinned hard enough to be reproducible.

    Q3 D1 chose ONE multilingual model for both datasets, so that a
    MIND-vs-EB-NeRD difference in Q3.6 is a difference in the data and not in
    two different encoders. This dataclass is that choice, and `revision` is the
    part that makes it a fact rather than a name: a bare model id resolves to
    whatever `main` points at today, so two runs months apart can silently use
    different weights and produce different numbers with no diff anywhere.
    """

    #: Hugging Face repository id.
    model_id: str
    #: Immutable commit sha on the Hub. NEVER a branch name -- see above.
    revision: str
    #: Output dimension. Recorded, and asserted against the loaded model, so a
    #: swapped model is caught at build time rather than at index time.
    dim: int
    #: Tokens per text. The model's own default is 512; D2 sets 128 to meet the
    #: Q3.2 budget, accepting a measured 9.1% truncation on MIND and 0.0% on
    #: EB-NeRD.
    max_seq_length: int
    #: The e5 family REQUIRES these prefixes -- confirmed from the model card,
    #: not from memory: "Each input text should start with 'query: ' or
    #: 'passage: ', even for non-English texts." Omitting them degrades the
    #: space quietly and the numbers still look plausible, which is why
    #: tests/test_embeddings.py asserts which side gets which.
    passage_prefix: str
    query_prefix: str


#: Q3 D1. `intfloat/multilingual-e5-small`: 384 dimensions, mean pooling,
#: 118M parameters, English and Danish in one space. Small enough that all
#: three corpora encode on CPU in ~31 minutes (measured), which is what made
#: "one model for both datasets" affordable at all.
ENCODER = EncoderSpec(
    model_id="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dim=384,
    max_seq_length=128,
    passage_prefix="passage: ",
    query_prefix="query: ",
)

#: Batch size for the encoding pass. 64 at seq 128 is what the CPU throughput
#: in q3_plan's "Measured scale" was measured at; it is a speed knob and
#: changes no number, but it is declared so the timing is reproducible.
ENCODE_BATCH_SIZE: int = 64

#: Bumped when a change makes previously written embeddings wrong or
#: unreadable. Part of the embedding fingerprint, enforced ON READ -- the same
#: discipline as INDEX_VERSION above and STORE_VERSION in Q1.
EMBEDDING_VERSION: int = 1

#: Q3 D5 -- how clicked-article vectors become one user vector.
#:
#: "mean" is the headline; "recency" is the scored ablation, weighting more
#: recent clicks higher. A config value rather than a constant because Q3.4
#: requires it to be one, and because both arms are actually measured.
#:
#: For EB-NeRD "recent" is click_time; MIND ships no click timestamps, so it is
#: history_rank -- position, which is defined for both. That asymmetry is real
#: and is stated in the design note rather than papered over.
USER_POOLING: str = "mean"
USER_POOLING_VARIANTS: tuple[str, ...] = ("mean", "recency")

#: Geometric decay for USER_POOLING="recency": weight r^i for the i-th most
#: recent click. 0.9 over the QUERY_HISTORY_LIMIT=20 window gives the oldest
#: click ~12% of the newest one's weight -- a real gradient, not a rounding
#: difference, which is the point of running the ablation at all.
USER_POOLING_DECAY: float = 0.9

#: Q3 D4 -- faiss HNSW parameters. Measured on real 768-d vectors: efSearch=64
#: (the library default) scores 0.9188 against exact search, BELOW Q3.3's 0.95
#: bar; 256 scores 0.9962 at 0.43s. Shipping the default would have failed the
#: requirement, which is why these are pinned here and reported.
ANN_HNSW_M: int = 32
ANN_HNSW_EF_CONSTRUCTION: int = 200
ANN_HNSW_EF_SEARCH: int = 256

#: Q3.3's bar: ANN top-K must agree with exact search at least this well.
ANN_RECALL_FLOOR: float = 0.95


# =============================================================================
# Q8 — paths that must never enter git history
# =============================================================================

#: fnmatch-style patterns, matched against repo-relative POSIX paths. Read by
#: the `no-data-artefacts` pre-commit hook AND by scripts/check_no_data_files.py.
#: Hooks do not survive a clone, so the script is the guarantee that holds in
#: the grader's copy; both must consult this one list.
#:
#: A LEADING SLASH means "only at the repo root"; without it a pattern matches at
#: any depth. That distinction is load-bearing, not cosmetic: an unanchored
#: `data/*` also matches `src/newsrec/data/mind.py`, and would forbid the entire
#: Q1 pipeline package from ever being committed. `__pycache__/*` is deliberately
#: left unanchored, because it must be caught at every depth.
FORBIDDEN_PATH_PATTERNS: tuple[str, ...] = (
    # Root-anchored directories — the generated and raw zones only.
    "/data/*",
    "/Data/*",
    "/artifacts/*",
    # Any depth.
    "__pycache__/*",
    "*.zip",
    "*.pt",
    "*.ckpt",
    "*.parquet",
    "*.npy",
    "*.npz",
    "*.faiss",
    "*.bin",
    "*.tsv",
    "*.vec",
    "*.safetensors",
    "*.h5",
    "*.pkl",
)

#: Paths exempt from the patterns above, because they are required deliverables
#: that happen to match. Each exemption is deliberate and justified:
#:   predictions/*.gz  -- D6: Q7.1 lists prediction files as a deliverable while
#:                        Q8 forbids large files; gzipped keeps them under the
#:                        size threshold and present in the repo.
#:   ai_usage/raw/*.gz -- Q7.4 requires the verbatim chat history export.
#:   tests/fixtures/*  -- `make test` must pass in a clean clone with no data.
#: Anchored, for the same reason as the patterns above: an exemption that
#: matched at any depth would silently excuse a stray `predictions/` directory
#: created anywhere in the tree.
FORBIDDEN_PATH_EXEMPTIONS: tuple[str, ...] = (
    "/predictions/*.gz",
    "/predictions/SHA256SUMS",
    "/ai_usage/raw/*.jsonl.gz",
    # The rendered transcript is the "chat history export" Q7.4 asks for. It
    # crossed 1 MiB during the Q1 session (295 tool calls), so the general limit
    # was about to make a required deliverable uncommittable -- the same failure
    # the raw capture hit one session earlier. Truncating it further would mean
    # shrinking evidence to fit a rule, so it gets the exempt ceiling instead.
    "/ai_usage/transcripts/*.md",
    "/tests/fixtures/*",
)

#: Anything staged must be smaller than this. Matches the pre-commit
#: `check-added-large-files --maxkb` setting; kept here so both agree.
MAX_COMMITTED_FILE_BYTES: int = 1024 * 1024

#: Ceiling for the exempted deliverables above. They are required by Q7 and
#: their size is not a choice: `ai_usage/raw/<session>.jsonl.gz` is the verbatim
#: chat-history export Q7.4 asks for, and a long working session legitimately
#: exceeds 1 MiB even gzipped. A separate, higher, still-finite limit keeps the
#: Q8 spirit (nothing unbounded enters git) without making a required deliverable
#: uncommittable. If a file exceeds THIS, something is wrong -- raw session logs
#: do not reach 8 MiB compressed.
MAX_EXEMPT_FILE_BYTES: int = 8 * 1024 * 1024

# =============================================================================
# Assignment constants
# =============================================================================

# =============================================================================
# A2 Q1 — behavioural features (a2_q1_plan.md)
# =============================================================================

#: Bumped when behaviour_features.py's logic changes in a way that invalidates
#: previously written click-history/session/popularity tables. Folded into
#: feature_store.config_fingerprint, the same discipline as STORE_VERSION.
BEHAVIOURAL_FEATURE_VERSION: int = 1

# =============================================================================
# A2 Q2 — re-ranking (a2_q2_plan.md)
# =============================================================================

#: Impressions used to FIT a re-ranker. Bigger than RECALL_SAMPLE_SIZE: that
#: constant sizes a sample for MEASURING one metric precisely; this one sizes a
#: sample for training a model, which needs more rows to learn from, not just
#: enough to bound a confidence interval.
RERANK_TRAIN_SAMPLE_SIZE: int = 20_000

#: Impressions used only to CHOOSE a hyperparameter combination (never to fit a
#: weight). Same size as RECALL_SAMPLE_SIZE so the selection metric has a
#: comparably precise interval.
RERANK_VAL_SAMPLE_SIZE: int = 2_000

#: A small, declared grid over REGULARISATION strength -- a demonstration of
#: hyperparameter sensitivity for the required ablation study (A2 Q3.1-Q3.3),
#: not a production tuning search. `learning_rate` and the tree-count ceiling
#: are fixed (early stopping on the val set decides the actual tree count per
#: combination, see GBDT_MAX_ROUNDS/GBDT_EARLY_STOPPING_ROUNDS); what is swept
#: is how strongly the model is regularised, which measured (a2_q2_plan.md)
#: to be what actually separated an overfit model from a useful one at this
#: data scale.
#: `device_type="cpu"`: LightGBM's OpenCL GPU backend is not repeatable (two
#: identical runs gave different trees on both EB-NeRD datasets; LightGBM warns
#: of this), and CPU with `deterministic=True` is exact. At 20,000 impressions
#: by 9 features CPU training is cheap, so repeatability costs nothing here.
GBDT_PARAM_GRID: dict[str, list] = {
    "num_leaves": [15, 31],
    "min_data_in_leaf": [100, 200],
    "lambda_l2": [1.0, 5.0],
}
GBDT_FIXED_PARAMS: dict = {
    "objective": "lambdarank",
    "device_type": "cpu",
    "verbosity": -1,
    "learning_rate": 0.05,
    "ndcg_eval_at": [10],
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": GLOBAL_SEED,
    "deterministic": True,
    "force_row_wise": True,
}
#: Upper ceiling on boosting rounds; early stopping on the val set's nDCG@10
#: almost always stops well before this (measured: ~30-100 rounds).
GBDT_MAX_ROUNDS: int = 500
GBDT_EARLY_STOPPING_ROUNDS: int = 30

#: Same reasoning as GBDT_PARAM_GRID, over the MLP's own regularisers:
#: capacity (`hidden_dim`), step size (`lr`), and L2 weight decay.
MLP_PARAM_GRID: dict[str, list] = {
    "hidden_dim": [32, 64],
    "lr": [1e-3, 1e-2],
    "weight_decay": [1e-4, 1e-3],
}
#: Upper ceiling on epochs; early stopping on val nDCG@10 decides the actual
#: count per combination, the same discipline as GBDT_MAX_ROUNDS.
MLP_MAX_EPOCHS: int = 100
MLP_EARLY_STOPPING_PATIENCE: int = 10
#: How often (in epochs) early stopping checks val nDCG@10 -- checking every
#: single epoch on a GPU is cheap, but batching the check keeps the training
#: loop simple to read.
MLP_EVAL_EVERY: int = 1

#: Bumped when gbdt.py/mlp.py's feature set or training logic changes in a way
#: that makes a previously trained/cached model stale.
RERANK_VERSION: int = 1

# =============================================================================
# A2 Q3 — NRMS reproduction and the one principled change (a2_q3_plan.md)
#
# THESE ARE OUR DECLARED CHOICES, NOT THE PAPER'S. We have read neither
# `ebnerd-benchmark`'s NRMS configuration nor Wu et al. 2019's stated settings
# (OQ-12), so quoting either as the reference would be a guess dressed as a
# citation. The architecture is implemented from the described mechanism —
# self-attention over past clicks, additive-attention pooling, a listwise
# softmax loss — and A2.Q3.1 is claimed as "reproduced, with the deviations
# declared" rather than as bit-equality with the starter code.
# =============================================================================

#: Working width of the news and user vectors. The 384-d sentence embeddings are
#: projected to this; NRMS_HEADS must divide it.
NRMS_DIM: int = 128

#: Heads in the user encoder's multi-head self-attention -- the layer that lets
#: past clicks weigh each other, replacing A1's fixed USER_POOLING_DECAY.
NRMS_HEADS: int = 8

#: Hidden width of the additive attention that pools the attended click vectors
#: into one user vector.
NRMS_ATTENTION_DIM: int = 64

#: Width of the learned category embedding -- THE ONE PRINCIPLED CHANGE
#: (A2.Q3.2). Used only when a model is built with `category_aware=True`, which
#: is the single switch separating the two ablation variants.
NRMS_CATEGORY_DIM: int = 32

NRMS_DROPOUT: float = 0.2

#: Non-clicked candidates sampled from the SAME impression to sit beside the
#: clicked one in the listwise softmax. Four is the largest ratio both corpora
#: can usually supply: eval.json measures a median of 9 candidates per
#: impression on EB-NeRD against 23 on MIND. Impressions with fewer are sampled
#: with replacement and COUNTED -- a2_q2_insights.md names short EB-NeRD
#: candidate lists as an unverified hypothesis for GBDT's failure there, and
#: reporting the padded share turns that hypothesis into a measured number.
NRMS_NEG_SAMPLES: int = 4

#: Impressions per training batch. Each carries 1 + NRMS_NEG_SAMPLES candidates
#: and up to QUERY_HISTORY_LIMIT history vectors, so this is the knob to halve
#: first if 4 GB of VRAM runs short.
NRMS_BATCH_IMPRESSIONS: int = 64

NRMS_LR: float = 1e-3

#: Upper ceiling on epochs; early stopping on val nDCG@10 decides the actual
#: count, the same discipline as GBDT_MAX_ROUNDS and MLP_MAX_EPOCHS and for the
#: same measured reason (A2 Q2 found GBDT at train nDCG@10 0.66 vs val 0.30).
NRMS_MAX_EPOCHS: int = 50
NRMS_EARLY_STOPPING_PATIENCE: int = 5

#: Bumped when nrms.py's architecture or training logic changes in a way that
#: makes a previously saved model stale. `rerank/store.load` refuses a mismatch
#: rather than scoring with the wrong model.
NRMS_VERSION: int = 1

#: Where trained re-rankers are saved. Inside ARTIFACTS_DIR, so it is already
#: gitignored and already matched by FORBIDDEN_PATH_PATTERNS -- a `*.pt` cannot
#: reach git from here even by accident.
RERANK_MODEL_DIR: Path = ARTIFACTS_DIR / "rerank"

# =============================================================================
# A2 Q4 — serving and scale analysis (a2_q4_plan.md)
# =============================================================================

#: Candidates stage 1 retrieves per request, before re-ranking. The top of the
#: 100-200 band A2 Q2 states. Read from here rather than written at the call
#: site so the benchmark and any future serving path cannot disagree about what
#: was measured.
SERVING_CANDIDATE_K: int = 200

#: Single-user requests timed for the latency percentiles. Batch size is ONE by
#: design: A2 Q4.2 asks about "a single user request", and a batched number
#: would look better while bounding nothing an SLA cares about.
SERVING_BENCH_REQUESTS: int = 1_000

#: Requests discarded before timing starts. The first few calls pay for lazy
#: allocation and BLAS thread spin-up; leaving them in moves the p99 by more
#: than anything in the code does.
SERVING_WARMUP_REQUESTS: int = 50

#: The latency target the cost estimate is quoted against (A2 Q4.3's own
#: example). Reported as met or not met, never assumed.
SERVING_SLA_P99_MS: float = 100.0

#: USD per hour for the machine the cost estimate assumes, and the dated source
#: it came from. BOTH are deliberately unset: we have no citable price (OQ-13),
#: and a plausible invented one would be a number the design-note checker could
#: not trace to any report -- the exact failure that checker exists to catch.
#:
#: While these are unset, `eval/serving.py` reports CORE-SECONDS per 1000
#: queries, which is fully measured and needs no price at all. Set both to emit
#: a currency figure; `tests/test_serving.py` fails on one without the other.
SERVING_COST_USD_PER_HOUR: float | None = None
SERVING_COST_SOURCE: str = ""

#: Cores the QPS estimate assumes are available to serve. The development
#: machine has 4; recorded rather than detected so the arithmetic in the report
#: is reproducible on a different machine.
SERVING_CORES: int = 4

#: The two ablation variants. `nrms_reproduction` is the baseline being
#: reproduced; `nrms_category_aware` is it plus the one change. Named here so
#: the ablation, the model store and the report cannot disagree about what the
#: variants are called.
NRMS_VARIANTS: tuple[str, ...] = ("nrms_reproduction", "nrms_category_aware")

#: Assignment 1's deadline (at Quiz-1). Kept rather than overwritten: A1 is
#: submitted, and its checklist and design note both date from before it.
DUE_DATE: date = date(2026, 8, 27)

#: Assignment 2's deadline. A2 builds on A1 rather than replacing it, so the two
#: dates coexist — see agent_docs/a2_requirements.md.
A2_DUE_DATE: date = date(2026, 9, 20)

RECALL_K_VALUES: tuple[int, ...] = (50, 100, 200)

#: How many test impressions recall@K is measured over (q2_plan D9).
#:
#: Not every impression: `ebnerd_small` has 244,647 in its test split and each
#: one costs a full corpus scan plus a sort, so an exhaustive run is slow enough
#: to discourage re-running -- which is how a stale number ends up in a report.
#: The sample is seeded from GLOBAL_SEED, so it is the same set on every run and
#: Q3 can evaluate on exactly it (Q3.5). The sampling error is not hidden: every
#: recall value is reported with a bootstrap confidence interval.
RECALL_SAMPLE_SIZE: int = 2_000
NDCG_K_VALUES: tuple[int, ...] = (5, 10)
CODABENCH_MIND_URL = "https://www.codabench.org/competitions/13967/"
CODABENCH_EBNERD_URL = "https://www.codabench.org/competitions/2469/"
