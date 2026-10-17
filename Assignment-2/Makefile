# =============================================================================
# CS4.406 Assignment 1 — build graph
#
# The one-command-rebuild claim (Q1.5, Q7.2) is only honest if Make actually
# knows the dependency graph. So every stage declares its real output file (or a
# .stamp sentinel where the output is a directory of files) as the target, and
# its real inputs -- raw data AND the source files that transform them -- as
# prerequisites. Consequences:
#
#   * `make -n all` prints an honest, acyclic graph of what would run;
#   * editing src/newsrec/retrieval/bm25.py rebuilds only from bm25 onward;
#   * `make clean-derived && make all` is a genuine from-raw rebuild, not a
#     wall of .PHONY targets that shell out unconditionally.
#
# Stage bodies that are not implemented yet raise NotImplementedError, so an
# unbuilt stage fails loudly rather than writing an empty artefact that the next
# stage would consume and report a plausible zero for.
# =============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --warn-undefined-variables

# No .pyc files. During setup a stale __pycache__/config.cpython-312.pyc was
# observed being loaded in preference to an edited config.py, silently masking
# the change -- a restored file kept reporting the pre-edit split boundary. The
# usual mtime-granularity explanation was checked and ruled out (this NTFS mount
# resolves to ~0.3 s), so the cause is unexplained rather than understood, which
# is reason enough not to leave it in the loop. Nothing here benefits from
# bytecode caching, and a stale .pyc that hides a source edit is expensive to
# diagnose twice.
export PYTHONDONTWRITEBYTECODE := 1

VENV      := .venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
RUFF      := $(VENV)/bin/ruff
PRECOMMIT := $(VENV)/bin/pre-commit
NEWSREC   := $(PY) -m newsrec.cli

RAW        := Data
PROC       := data/processed
STORE      := data/feature_store
ART        := artifacts
METRICS    := reports/metrics
PREDS      := predictions

DATASETS   := mind_small ebnerd_demo ebnerd_small
# Datasets whose numbers are quoted as headline results (config.HEADLINE_DATASETS).
HEADLINE   := mind_small ebnerd_small

# Source files that invalidate a stage when edited. Listing them as real
# prerequisites is what makes incremental rebuilds correct rather than lucky.
SRC_CONFIG    := src/newsrec/config.py
SRC_DATA      := $(SRC_CONFIG) $(wildcard src/newsrec/data/*.py)
SRC_RETRIEVAL := $(SRC_CONFIG) $(wildcard src/newsrec/retrieval/*.py)
# Per-stage source sets. SRC_RETRIEVAL is the whole directory, which is correct
# but far too coarse for the Q3 stages: editing ann.py invalidated the
# embeddings stamp and Make re-ran a 24-minute encode to produce byte-identical
# vectors. Measured, not hypothetical -- it happened once. A prerequisite list
# should name what a stage actually reads.
SRC_TOKENISE  := src/newsrec/retrieval/tokenise.py $(wildcard src/newsrec/retrieval/stopwords/*.txt)
SRC_BM25      := $(SRC_CONFIG) $(SRC_TOKENISE) src/newsrec/retrieval/index.py \
                 src/newsrec/retrieval/bm25.py src/newsrec/retrieval/query.py
SRC_EMBED     := $(SRC_CONFIG) src/newsrec/retrieval/embed.py
SRC_ANN       := $(SRC_EMBED) src/newsrec/retrieval/ann.py
SRC_SEMANTIC  := $(SRC_ANN) src/newsrec/retrieval/user_vector.py \
                 src/newsrec/retrieval/recall.py src/newsrec/retrieval/query.py \
                 src/newsrec/retrieval/semantic.py src/newsrec/retrieval/compare.py \
                 $(SRC_TOKENISE) src/newsrec/retrieval/index.py src/newsrec/retrieval/bm25.py
SRC_EVAL      := $(SRC_CONFIG) $(wildcard src/newsrec/eval/*.py)
SRC_SUBMIT    := $(SRC_CONFIG) $(wildcard src/newsrec/submit/*.py)

# --- Raw inputs, per dataset -------------------------------------------------
RAW_mind_small   := $(RAW)/MIND/MINDsmall_train/news.tsv \
                    $(RAW)/MIND/MINDsmall_train/behaviors.tsv \
                    $(RAW)/MIND/MINDsmall_dev/news.tsv \
                    $(RAW)/MIND/MINDsmall_dev/behaviors.tsv
RAW_ebnerd_demo  := $(RAW)/EB-NeRD/ebnerd_demo/articles.parquet \
                    $(RAW)/EB-NeRD/ebnerd_demo/train/behaviors.parquet \
                    $(RAW)/EB-NeRD/ebnerd_demo/train/history.parquet \
                    $(RAW)/EB-NeRD/ebnerd_demo/validation/behaviors.parquet \
                    $(RAW)/EB-NeRD/ebnerd_demo/validation/history.parquet
RAW_ebnerd_small := $(RAW)/EB-NeRD/ebnerd_small/articles.parquet \
                    $(RAW)/EB-NeRD/ebnerd_small/train/behaviors.parquet \
                    $(RAW)/EB-NeRD/ebnerd_small/train/history.parquet \
                    $(RAW)/EB-NeRD/ebnerd_small/validation/behaviors.parquet \
                    $(RAW)/EB-NeRD/ebnerd_small/validation/history.parquet

# --- Per-dataset stage outputs -----------------------------------------------
MANIFESTS  := $(foreach d,$(DATASETS),$(PROC)/$(d)/_manifest.json)
STORES     := $(foreach d,$(DATASETS),$(STORE)/$(d)/.stamp)
BM25_IDX   := $(foreach d,$(DATASETS),$(ART)/bm25/$(d)/.stamp)
EMBED      := $(foreach d,$(DATASETS),$(ART)/embeddings/$(d)/.stamp)
ANN_IDX    := $(foreach d,$(DATASETS),$(ART)/ann/$(d)/.stamp)

# =============================================================================
# help
# =============================================================================
.PHONY: help
help:
	@echo "CS4.406 news recommendation — targets (Assignment 1 + Assignment 2)"
	@echo
	@echo "  setup             create $(VENV), install pinned deps + package, install git hooks"
	@echo "  download          fetch missing raw archives into $(RAW)/, verify SHA256, extract"
	@echo "  download-extra    EB-NeRD embedding artifacts and the test set (large, optional)"
	@echo
	@echo "Assignment 1 — retrieval (no GPU required; CPU fallback everywhere)"
	@echo "  data              raw -> unified schema -> temporal split -> feature store   [Q1]"
	@echo "  bm25              inverted index + retrieval + recall@K                      [Q2]"
	@echo "  embeddings        encode articles to unit-norm vectors                      [Q3.1]"
	@echo "  ann               HNSW index + agreement with exact search                 [Q3.3]"
	@echo "  semantic          user vectors + semantic recall@K                            [Q3]"
	@echo "  compare           paired lexical-vs-semantic, per slice                     [Q3.6]"
	@echo "  hybrid            does blending lexical into semantic help? (ablation)      [Q3.6]"
	@echo "  eval              metrics, beyond-accuracy, slices, bootstrap CIs            [Q4]"
	@echo "  ablations         paired metrics without serving-unavailable features        [Q9.1]"
	@echo "  submit            prediction files for both Codabench competitions           [Q5]"
	@echo "  all               data -> bm25 -> semantic -> eval -> ablations -> submit"
	@echo
	@echo "Assignment 2 — behavioural signals and re-ranking"
	@echo "  behaviour_features  click-history/session/article feature summary      [A2.Q1]"
	@echo "  rerank            GBDT/MLP re-rankers, ablated vs retrieval-only (GPU) [A2.Q2]"
	@echo "  nrms              reproduce NRMS, then beat it, with paired CIs  (GPU) [A2.Q3]"
	@echo "  rerank-eval       full metrics + slices + CIs for the two-stage pipeline [A2.Q5]"
	@echo "  serving           index memory, p99 latency, cost/QPS, 10x argument    [A2.Q4]"
	@echo "  two-stage-ablation  re-ranker with/without forbidden features (GPU) [A2.Q9]"
	@echo "  submit-nrms       two-stage predictions for both leaderboards          [A2.Q5]"
	@echo "  a2-design-note    render + check the 6-page A2 design note            [A2.Q6]"
	@echo "  a2-all            every A2 stage. Separate from 'all' because these"
	@echo "                    TRAIN, and training raises rather than falling back"
	@echo "                    to CPU — so 'all' stays runnable without a GPU."
	@echo
	@echo "  test              fast tests only: no dataset, no GPU, target < 30 s"
	@echo "  test-full         adds the realdata marker (needs a built data/ store)"
	@echo "  lint / format     ruff check / ruff format"
	@echo "  lock              regenerate requirements*.txt from requirements*.in"
	@echo
	@echo "  ai-log            rebuild ai_usage/ from Claude Code session JSONL   [Q7.4]"
	@echo "  check-no-data-files  no data or large binaries in git (also in CI)   [Q8]"
	@echo "  requirements      re-run every verifier in the A1 requirements checklist"
	@echo "  a2-requirements   the same, for agent_docs/a2_requirements.md     [A2.Q7]"
	@echo
	@echo "  clean-derived     rm -rf data/ artifacts/ reports/metrics/ (KEEPS $(RAW)/)"
	@echo
	@echo "Reproduce everything from raw files:  make clean-derived && make all"
	@echo "                     ...including A2:  && make a2-all   (needs a GPU)"

# =============================================================================
# setup
# =============================================================================
$(VENV)/bin/python:
	@# Debian/Ubuntu split ensurepip out into the python3-venv package, so a
	@# stock python3 can create a venv but cannot put pip inside it. Rather than
	@# dead-ending on a machine we do not have root on, fall back to a pip-less
	@# venv plus the official get-pip bootstrap. The apt package is still the
	@# clean fix, so say so rather than hiding the workaround.
	python3 -m venv $(VENV) 2>/dev/null || { \
	  echo "note: ensurepip is unavailable — on Debian/Ubuntu the clean fix is:"; \
	  echo "        sudo apt install python3.12-venv"; \
	  echo "      falling back to a pip-less venv + get-pip bootstrap (needs network)"; \
	  rm -rf $(VENV); \
	  python3 -m venv --without-pip $(VENV); \
	  curl -fsSL https://bootstrap.pypa.io/get-pip.py | $(VENV)/bin/python -; }
	$(PIP) install --quiet --upgrade pip setuptools wheel

.PHONY: setup
setup: $(VENV)/bin/python
	@# requirements.txt is a SINGLE lockfile pinning the whole environment,
	@# runtime and dev together. pip freeze reports one environment and cannot
	@# attribute a transitive dependency back to the .in file that pulled it in,
	@# so splitting the lock would mean guessing -- and a lockfile that guesses
	@# is worse than one that is honestly complete. The .in/.dev.in split still
	@# documents intent; the lock records exactly what was installed.
	@# Fall back to the loose .in files on a fresh clone where lock has not run.
	if [ -f requirements.txt ]; then \
	  $(PIP) install --quiet -r requirements.txt; \
	else \
	  echo "note: no requirements.txt yet, installing from requirements.in (run 'make lock')"; \
	  $(PIP) install --quiet -r requirements.in -r requirements-dev.in; \
	fi
	$(PIP) install --quiet -e .
	$(MAKE) hooks-install
	@echo "setup complete — $$($(PY) --version)"

.PHONY: hooks-install
hooks-install:
	@# Only the pre-commit stage now. The commit-msg hook enforced the removed
	@# Provenance trailer, and the post-commit hook auto-committed ai_usage/ --
	@# both were removed on 2026-08-21. `make ai-log` is run manually.
	if [ -x $(PRECOMMIT) ]; then $(PRECOMMIT) install --install-hooks -t pre-commit; fi
	@echo "git hooks installed (pre-commit)"

.PHONY: lock
lock: $(VENV)/bin/python
	$(PIP) install --quiet -r requirements.in -r requirements-dev.in
	@# Two things `pip freeze` alone gets wrong here:
	@#
	@# 1. It omits index directives. The CPU torch wheels are versioned `2.x.y+cpu`
	@#    and do NOT exist on PyPI, so a lockfile without the --extra-index-url
	@#    line is simply uninstallable. Carry those lines across verbatim.
	@# 2. It would record the editable install of this package itself, leaking an
	@#    absolute path that is wrong in every other checkout.
	{ echo "# GENERATED by 'make lock' — do not hand-edit. Intent lives in requirements.in."; \
	  grep -h -E '^--(extra-)?index-url' requirements.in requirements-dev.in || true; \
	  $(PIP) freeze --exclude-editable | grep -v '^newsrec=='; } > requirements.txt
	@echo "locked -> requirements.txt"

# =============================================================================
# download  (Q1.1) — idempotent, checksum-verified
# =============================================================================
MIND_URL   := https://huggingface.co/datasets/yjw1029/MIND/resolve/main
EBNERD_URL := https://ebnerd-dataset.s3.eu-west-1.amazonaws.com

$(RAW)/MIND/%.zip:
	mkdir -p $(@D)
	wget --continue --no-verbose -O $@ $(MIND_URL)/$*.zip

$(RAW)/EB-NeRD/%.zip:
	mkdir -p $(@D)
	wget --continue --no-verbose -O $@ $(EBNERD_URL)/$*.zip

# Extraction: one .extracted stamp per archive, so an archive is unzipped once
# rather than once per file it contains.
#
# The raw files depend on their stamp ORDER-ONLY (the `|` prefix). That is the
# right semantics for an immutable raw zone: if a raw file is already on disk it
# is never rebuilt, no matter how mtimes compare. Correctness of the raw zone is
# guaranteed by `verify-downloads` checking SHA256, not by timestamps -- and a
# normal prerequisite here would re-extract ~460 MB on every clone whose zips
# happen to be newer than their contents, and would rewrite files that the
# pipeline is supposed to treat as read-only.

# MIND archives have no top-level directory, so each extracts into its own.
$(RAW)/MIND/MINDsmall_train/.extracted: $(RAW)/MIND/MINDsmall_train.zip
	unzip -q -o -d $(@D) $<
	touch $@
$(RAW)/MIND/MINDsmall_dev/.extracted: $(RAW)/MIND/MINDsmall_dev.zip
	unzip -q -o -d $(@D) $<
	touch $@

# EB-NeRD archives already contain their top-level directory.
$(RAW)/EB-NeRD/ebnerd_demo/.extracted: $(RAW)/EB-NeRD/ebnerd_demo.zip
	unzip -q -o -d $(RAW)/EB-NeRD $<
	touch $@
$(RAW)/EB-NeRD/ebnerd_small/.extracted: $(RAW)/EB-NeRD/ebnerd_small.zip
	unzip -q -o -d $(RAW)/EB-NeRD $<
	touch $@

$(filter $(RAW)/MIND/MINDsmall_train/%,$(RAW_mind_small)): | $(RAW)/MIND/MINDsmall_train/.extracted
$(filter $(RAW)/MIND/MINDsmall_dev/%,$(RAW_mind_small)):   | $(RAW)/MIND/MINDsmall_dev/.extracted
$(RAW_ebnerd_demo):  | $(RAW)/EB-NeRD/ebnerd_demo/.extracted
$(RAW_ebnerd_small): | $(RAW)/EB-NeRD/ebnerd_small/.extracted

.PHONY: download
download: $(RAW_mind_small) $(RAW_ebnerd_demo) $(RAW_ebnerd_small) verify-downloads

.PHONY: verify-downloads
verify-downloads:
	@# A checksum mismatch is a HARD failure, never a silent re-download: a
	@# truncated or upstream-replaced archive would otherwise quietly change
	@# every number downstream.
	cd $(RAW) && grep -v '^#' $(CURDIR)/scripts/raw_archives.sha256 \
	  | grep -v '^[[:space:]]*$$' | sha256sum --check --quiet --ignore-missing
	@echo "raw archives verified against scripts/raw_archives.sha256"

.PHONY: record-checksum
record-checksum:
	@# Append a hash for a newly downloaded raw file. Refuses anything that
	@# cannot be shown to be COMPLETE, so a partial download can never have its
	@# checksum baked into the repo -- which would silently bless a truncated
	@# file forever (see OQ-5, and OQ-2 where the check bit for real).
	@#
	@# The completeness test depends on the format, because "is this whole?" has
	@# a different answer per format:
	@#   .zip -> `unzip -t`, which needs the end-of-central-directory record and
	@#           therefore cannot pass on a prefix of the file.
	@#   .tsv -> the last line must carry the same number of tab-separated
	@#           fields as the first AND the file must end in a newline. A cut
	@#           download almost always ends mid-record, so this catches it;
	@#           it is weaker than `unzip -t` and that is stated, not hidden.
	@#   .parquet -> the footer must parse. Parquet, like zip, stores its
	@#           metadata at the END of the file (schema, row-group offsets,
	@#           row counts, closing PAR1 magic), so reading it back is
	@#           impossible on a prefix. This is a STRONG check, on a par with
	@#           `unzip -t` rather than with the TSV heuristic.
	@# Either way the file must not have been written in the last 60 seconds,
	@# because a hash of a file a wget still holds open is a hash of a moment.
	@# Usage: make record-checksum FILE=EB-NeRD/foo.zip
	test -n "$(FILE)" || { echo "usage: make record-checksum FILE=<path relative to $(RAW)/>"; exit 2; }
	@f='$(RAW)/$(FILE)'; \
	test -f "$$f" || { echo "no such file: $$f"; exit 2; }; \
	age=$$(( $$(date +%s) - $$(stat -c %Y "$$f") )); \
	test "$$age" -ge 60 || { echo "refusing: $$f was modified $${age}s ago -- still downloading?"; exit 3; }; \
	case "$$f" in \
	  *.zip) unzip -t "$$f" > /dev/null || { echo "refusing: $$f fails unzip -t (incomplete archive)"; exit 4; };; \
	  *.tsv) test "$$(tail -c 1 "$$f" | xxd -p)" = "0a" \
	           || { echo "refusing: $$f does not end in a newline (truncated?)"; exit 4; }; \
	         test "$$(head -1 "$$f" | awk -F'\t' '{print NF}')" = "$$(tail -1 "$$f" | awk -F'\t' '{print NF}')" \
	           || { echo "refusing: $$f last line has a different field count than its first (truncated?)"; exit 4; };; \
	  *.parquet) $(PY) scripts/check_parquet_footer.py "$$f" || exit 4;; \
	  *) echo "refusing: no completeness check defined for $$f -- add one rather than skipping it"; exit 5;; \
	esac
	cd $(RAW) && sha256sum $(FILE) >> $(CURDIR)/scripts/raw_archives.sha256
	@echo "recorded $(FILE)"

#: Set to anything to make `download-models` verify the cache without network.
OFFLINE ?=

.PHONY: download-models
download-models:
	@# Q3's encoder, pinned to an immutable commit sha in config.ENCODER and
	@# VERIFIED after download -- a bare model id resolves to whatever `main`
	@# points at today, so two runs months apart can use different weights with
	@# nothing in the diff to show it. Same reasoning as raw_archives.sha256.
	@#
	@# KNOWN LIMITATION (OQ-9): this needs the network. The weights cannot be
	@# committed -- invariant 3 forbids large binaries in git -- so there is no
	@# offline fallback. `make download-models OFFLINE=1` verifies an existing
	@# cache without reaching out.
	$(PY) scripts/download_models.py $(if $(OFFLINE),--offline,)

.PHONY: download-extra
download-extra: $(RAW)/EB-NeRD/artifacts/Ekstra_Bladet_word2vec.zip \
                $(RAW)/EB-NeRD/artifacts/google_bert_base_multilingual_cased.zip
	@echo "note: ebnerd_testset.zip is still required for Q5.2 — see OQ-2."

$(RAW)/EB-NeRD/artifacts/%.zip:
	mkdir -p $(@D)
	wget --continue --no-verbose -O $@ $(EBNERD_URL)/artifacts/$*.zip

# =============================================================================
# Q1 — data pipeline
#
# One rule per dataset, generated so the graph stays explicit rather than hidden
# behind a pattern rule that Make cannot show you.
# =============================================================================
define DATASET_RULES
$(PROC)/$(1)/_manifest.json: $$(RAW_$(1)) $$(SRC_DATA)
	mkdir -p $$(@D)
	$$(NEWSREC) data --dataset $(1)

$(STORE)/$(1)/.stamp: $(PROC)/$(1)/_manifest.json
	@# The manifest rule above already ran `newsrec data`, which writes BOTH the
	@# processed store and the feature store. Re-invoking it here rebuilt every
	@# dataset twice -- 26s of work done in 52s. The stamp only records that the
	@# feature store is current with its manifest.
	test -f $$@ || $$(NEWSREC) data --dataset $(1)
	touch $$@

$(ART)/bm25/$(1)/.stamp: $(STORE)/$(1)/.stamp $$(SRC_BM25)
	mkdir -p $$(@D)
	$$(NEWSREC) bm25 --dataset $(1)
	touch $$@

$(ART)/embeddings/$(1)/.stamp: $(STORE)/$(1)/.stamp $$(SRC_EMBED)
	@# Its own stage, not a phase of `semantic`: an ANN parameter change must
	@# not re-trigger a ~31-minute encode that touches not one vector.
	mkdir -p $$(@D)
	$$(NEWSREC) embed --dataset $(1)
	touch $$@

$(ART)/ann/$(1)/.stamp: $(ART)/embeddings/$(1)/.stamp $$(SRC_ANN)
	mkdir -p $$(@D)
	$$(NEWSREC) ann --dataset $(1)
	touch $$@
endef
$(foreach d,$(DATASETS),$(eval $(call DATASET_RULES,$(d))))

.PHONY: data
data: $(STORES)

# A2 Q1 — click-history/session/article behavioural features. Built as part of
# `newsrec data` itself (feature_store.write_feature_store), so $(STORES) is
# already a complete build; this is a REPORT over the already-built tables, the
# same division `compare`/`ablations` use between "build" and "report".
$(METRICS)/behavioural_features.json: $(STORES) $(SRC_DATA)
	mkdir -p $(@D)
	$(NEWSREC) behaviour_features

.PHONY: behaviour_features
behaviour_features: $(METRICS)/behavioural_features.json

# A2 Q2 — GBDT/MLP re-rankers ablated against the retrieval-only baseline.
# Needs the full A1 retrieval artefacts (BM25 index + embeddings), not just
# $(STORES), since rerank/features.py reuses eval/harness.py's own scorers.
SRC_RERANK := $(SRC_CONFIG) $(wildcard src/newsrec/rerank/*.py)
$(METRICS)/reranker_ablation.json: $(METRICS)/eval.json $(SRC_RERANK)
	mkdir -p $(@D)
	$(NEWSREC) rerank

.PHONY: rerank
rerank: $(METRICS)/reranker_ablation.json

# A2 Q3 — reproduce an NRMS-style ranker, then beat it with one principled
# change, isolated by an ablation with paired bootstrap CIs. $(SRC_RERANK)
# already globs src/newsrec/rerank/*.py, so nrms.py, nrms_ablation.py and
# store.py are covered without listing them.
#
# Defaults to $(HEADLINE_DATASETS) rather than all three — one English corpus
# and one Danish satisfies "both datasets", and a third would be a third more
# GPU time for a corpus A1 designated a fixture. Pass --dataset to override.
$(METRICS)/nrms_ablation.json: $(METRICS)/eval.json $(SRC_RERANK)
	mkdir -p $(@D)
	$(NEWSREC) nrms

.PHONY: nrms
nrms: $(METRICS)/nrms_ablation.json

# A2 Q5 — the full metric set (adding diversity, novelty and coverage to the
# four ranking metrics), sliced, with bootstrap CIs, for the two-stage
# pipeline. Depends on nrms_ablation.json because it LOADS the models that
# stage trained and saved rather than retraining them.
$(METRICS)/rerank_eval.json: $(METRICS)/nrms_ablation.json $(SRC_RERANK) $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) rerank_eval

.PHONY: rerank-eval
rerank-eval: $(METRICS)/rerank_eval.json

# A2 Q4 — serving characteristics. Needs the retrieval artefacts a request
# actually touches (ANN graph, BM25 index, embeddings) but NOT a trained
# model: it times the re-ranker's arithmetic at batch 1, so the benchmark runs
# on a clean clone. CPU by design, see a2_q4_plan.md.
$(METRICS)/serving.json: $(ANN_IDX) $(BM25_IDX) $(STORES) $(SRC_EVAL) $(SRC_SUBMIT)
	mkdir -p $(@D)
	$(NEWSREC) serving

.PHONY: serving
serving: $(METRICS)/serving.json

# A2 Q9.1 — the re-ranker with and without the serving-unavailable features.
# Fits an MLP twice per dataset, so GPU-only. EB-NeRD only in practice: MIND
# ships none of the columns and the report says so explicitly.
$(METRICS)/two_stage_ablation.json: $(METRICS)/eval.json $(SRC_RERANK)
	mkdir -p $(@D)
	$(NEWSREC) two_stage_ablation

.PHONY: two-stage-ablation
two-stage-ablation: $(METRICS)/two_stage_ablation.json

# =============================================================================
# Q2 / Q3 — candidate generation
# =============================================================================
$(METRICS)/bm25_recall.json: $(BM25_IDX) $(SRC_BM25) src/newsrec/retrieval/recall.py $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) bm25

# Q2.9. Scores every tokenisation variant, so it costs one full recall pass per
# variant -- roughly three times `bm25_recall.json`. A prerequisite of `bm25`
# anyway: make only re-runs it when the retrieval sources change, and a stale
# ablation is worse than a slow one, because the design note quotes it.
$(METRICS)/bm25_tokenisation_ablation.json: $(BM25_IDX) $(SRC_BM25) src/newsrec/retrieval/ablation.py src/newsrec/retrieval/recall.py $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) bm25 --ablation

.PHONY: bm25
bm25: $(METRICS)/bm25_recall.json $(METRICS)/bm25_tokenisation_ablation.json

# Q3.3. Scores the persisted graphs against exact search. FAILS if any dataset
# misses ANN_RECALL_FLOOR -- a measurement that cannot fail is a decoration.
$(METRICS)/ann_recall.json: $(ANN_IDX) $(SRC_ANN)
	mkdir -p $(@D)
	$(NEWSREC) ann --measure

.PHONY: ann
ann: $(METRICS)/ann_recall.json

$(METRICS)/semantic_recall.json: $(METRICS)/ann_recall.json $(SRC_SEMANTIC) $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) semantic

.PHONY: embeddings
embeddings: $(EMBED)

# Not a graded metric -- an interpretation aid for design_rationale 1.18. Kept a
# Make target rather than an ad-hoc script because a number quoted anywhere must
# be regenerable (CLAUDE.md), which is the rule that turned Q2's ad-hoc stemmer
# comparison into bm25_tokenisation_ablation.json.
$(METRICS)/embedding_diagnostics.json: $(EMBED) $(SRC_EMBED)
	mkdir -p $(@D)
	$(NEWSREC) embed --diagnostics

.PHONY: embedding-diagnostics
embedding-diagnostics: $(METRICS)/embedding_diagnostics.json

.PHONY: semantic
semantic: $(METRICS)/semantic_recall.json

# Q3.6. Recomputes BOTH retrievers over one sample so the deltas are paired
# row for row; refuses to write if the impression set differs from either
# committed report.
$(METRICS)/lexical_vs_semantic.json: $(METRICS)/semantic_recall.json $(METRICS)/bm25_recall.json $(SRC_SEMANTIC) $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) compare

.PHONY: compare
compare: $(METRICS)/lexical_vs_semantic.json

# =============================================================================
# Q4 / Q9 — evaluation and ablations
# =============================================================================
$(METRICS)/eval.json: $(METRICS)/bm25_recall.json $(METRICS)/semantic_recall.json $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) eval

.PHONY: eval
eval: $(METRICS)/eval.json

# Q3.6 follow-up: does blending the lexical score into the semantic one help?
# Measured, not assumed — and the answer is language-dependent, which is why it
# is a reported ablation rather than a tuning knob. Depends on eval.json because
# it asserts its endpoints reproduce that report's pure-method AUCs.
$(METRICS)/hybrid_ablation.json: $(METRICS)/eval.json $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) hybrid

.PHONY: hybrid
hybrid: $(METRICS)/hybrid_ablation.json

$(METRICS)/ablations.json: $(METRICS)/eval.json $(SRC_EVAL)
	mkdir -p $(@D)
	$(NEWSREC) ablations

.PHONY: ablations
ablations: $(METRICS)/ablations.json

# =============================================================================
# Q6 — design note
# =============================================================================
# The PDF is the graded artefact and the page limit is asserted on it, not on a
# word count of the markdown. check_design_note.py --check-numbers additionally
# traces every metric-shaped figure back to a reports/metrics key, which is the
# only mechanical defence against a stale or recalled number.
design_note/design_note.pdf: design_note/design_note.md scripts/render_design_note.py
	$(PY) scripts/render_design_note.py

.PHONY: design-note
design-note: design_note/design_note.pdf
	$(PY) scripts/check_design_note.py

# A2 Q6 — its own 6-page note, its own allowlist. A2.pdf calls 6 pages "a
# guideline, not a hard cap", so the check is on the body and the script says
# what it counted rather than only pass/fail.
design_note/a2_design_note.pdf: design_note/a2_design_note.md scripts/render_design_note.py
	$(PY) scripts/render_design_note.py --source design_note/a2_design_note.md

.PHONY: a2-design-note
a2-design-note: design_note/a2_design_note.pdf
	$(PY) scripts/check_design_note.py --note design_note/a2_design_note.md --max-pages 6

# =============================================================================
# Q5 — submission
# =============================================================================
$(PREDS)/SHA256SUMS: $(METRICS)/eval.json $(SRC_SUBMIT)
	mkdir -p $(@D)
	$(NEWSREC) submit
	@# The artefacts themselves are NOT committed. D6 assumed they would fit
	@# gzipped under 1 MB; measured, MIND's is 291 MB (107 MB gzipped) and
	@# EB-NeRD's is ~0.96 GB, so committing either would break invariant 3.
	@# OQ-11: the checksum and predictions/manifest.json are what make the
	@# claim checkable, and `make submit` is what regenerates the bytes.
	@# Checksum the ARCHIVES, not the text: the zip is what gets uploaded, and
	@# our first MIND submission was rejected on archive layout while the
	@# prediction inside it was correct.
	cd $(PREDS) && sha256sum *.zip > SHA256SUMS

.PHONY: submit
submit: $(PREDS)/SHA256SUMS

# The trivial baseline (OQ-6): ranks MIND candidates by article_id, which
# carries no signal and is expected to score near random. It exists to prove the
# competition accepts our file format BEFORE the 90-minute encoding run, not to
# compete. Deliberately depends on nothing but the raw file and the writer.
# The MIND hybrid (Q5.1). Blend weight comes from the VALIDATION split via
# `make hybrid`, not from the split any number is reported on. EB-NeRD has no
# equivalent target on purpose: both its corpora selected alpha=0 on val, so a
# Danish hybrid would be overriding a measurement with a preference.
# EB-NeRD's real submission (Q5.2): semantic only, because both its corpora
# selected alpha=0 on the validation split. Refuses to write unless BOTH
# candidate and user coverage clear 0.99 — the first run had candidates at
# 1.0000 and users at 0.0000, which silently produced the baseline.
.PHONY: submit-semantic
submit-semantic:
	mkdir -p $(PREDS)
	$(NEWSREC) submit --semantic
	cd $(PREDS) && sha256sum *.zip > SHA256SUMS
	unzip -l $(PREDS)/predictions.zip

.PHONY: submit-hybrid
submit-hybrid:
	mkdir -p $(PREDS)
	$(NEWSREC) submit --hybrid
	cd $(PREDS) && sha256sum *.zip > SHA256SUMS
	unzip -l $(PREDS)/mind_submission.zip

# A2 Q5.5 — two-stage predictions, scored by the NRMS re-ranker on both
# competitions. Depends on nrms_ablation.json because it LOADS the model that
# stage trained; the test-corpus embeddings are built on demand by
# submit/encode_test.py, which reuses them when the fingerprint matches.
.PHONY: submit-nrms
submit-nrms:
	mkdir -p $(PREDS)
	$(PY) -c "from newsrec.submit import nrms_predict; print(nrms_predict.run())"
	cd $(PREDS) && sha256sum *.zip > SHA256SUMS

# Local stand-in for competition 2469, so a compute worker on this machine can
# ingest ebnerd_nrms_submission.zip and produce real logs. Structure check only:
# the hidden labels are not here. See scripts/build_standin_bundle.py.
.PHONY: codabench-standin-bundle
codabench-standin-bundle:
	PYTHONPATH=src $(PY) scripts/build_standin_bundle.py

# Labelled counterpart: scores NRMS's ebnerd_small predictions with AUC/MRR/nDCG
# on the local worker, to be compared with rerank_eval.json. Needs the trained
# NRMS model (artifacts/rerank/), so it is not part of `all`.
.PHONY: codabench-labelled-standin
codabench-labelled-standin:
	PYTHONPATH=src:scripts $(PY) scripts/build_labelled_standin.py

.PHONY: submit-baseline
submit-baseline:
	mkdir -p $(PREDS)
	$(NEWSREC) submit --baseline
	cd $(PREDS) && sha256sum *.zip > SHA256SUMS
	unzip -l $(PREDS)/mind_submission.zip

# `all` is the one-command reproduce a grader runs, so every stage in it must
# complete without a GPU. `behaviour_features` qualifies (it only reads the
# already-built store and writes a summary); the A2 training stages do not —
# rerank/mlp.py requires CUDA and raises rather than degrading to a much slower
# device. Putting them here would turn "make all" into "make all, on a machine
# with an NVIDIA card", which is not what Q7.2 claims. They live in a2-all.
.PHONY: all
all: data bm25 semantic eval ablations behaviour_features submit

.PHONY: a2-all
a2-all: behaviour_features rerank nrms rerank-eval serving two-stage-ablation

# =============================================================================
# Quality gates
# =============================================================================
.PHONY: test
test:
	$(PYTEST)

.PHONY: test-full
test-full:
	@# Overrides the default marker filter from pyproject.toml. gpu stays
	@# excluded so this is still runnable on a CPU-only grading machine.
	@#
	@# BOTH flags are needed and they do different jobs. `-m "not gpu"` SELECTS
	@# the realdata-marked tests; `--realdata` points the store fixtures at the
	@# real data/processed/ instead of the tiny synthetic fixtures. With only the
	@# first, the realdata canaries run against the fixture store and assert real
	@# corpus counts against six rows -- which is how this target came to be
	@# failing from the moment it was written, unnoticed because the checks were
	@# being run as `pytest --realdata` by hand.
	$(PYTEST) -m "not gpu" --realdata

.PHONY: lint
lint:
	$(RUFF) check .
	$(RUFF) format --check .

.PHONY: format
format:
	$(RUFF) check --fix .
	$(RUFF) format .

.PHONY: check-no-data-files
check-no-data-files:
	$(PY) scripts/check_no_data_files.py

.PHONY: requirements
requirements:
	$(PY) scripts/check_requirements.py

# A2 has its own checklist. Separate target rather than a wider default, because
# `make requirements`' output is an A1 deliverable and should not change.
# ARGS lets a caller add --run or --question, e.g. `make a2-requirements ARGS=--run`.
.PHONY: a2-requirements
a2-requirements:
	$(PY) scripts/check_requirements.py --checklist agent_docs/a2_requirements.md $(ARGS)

# =============================================================================
# Q7.4 — AI usage log
# =============================================================================
.PHONY: ai-log
ai-log:
	$(PY) scripts/extract_ai_log.py

# =============================================================================
# Cleaning
#
# clean-derived deliberately does NOT touch $(RAW)/. Re-downloading ~460 MB to
# prove a rebuild works is not a rebuild test, it is a bandwidth test — the
# claim under Q1.5 is "rebuilds everything from raw files", and this is exactly
# that boundary.
# =============================================================================
.PHONY: clean-derived
clean-derived:
	rm -rf data $(ART) $(METRICS)
	@echo "removed data/ $(ART)/ $(METRICS)/ — $(RAW)/ untouched"

.PHONY: clean-all
clean-all: clean-derived
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
