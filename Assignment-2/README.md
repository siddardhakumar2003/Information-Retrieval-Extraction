# CS4.406 Assignment 1 — Lexical & Semantic Retrieval on MIND and EB-NeRD

A reproducible pipeline for news recommendation: raw files → unified feature store →
candidate generation (BM25 and embeddings) → offline evaluation → Codabench
prediction files. Both datasets, every stage.

| | |
|---|---|
| Datasets | MIND-small (English) · EB-NeRD demo + small (Danish) |
| Competitions | [MIND 13967](https://www.codabench.org/competitions/13967/) · [RecSys 2024 2469](https://www.codabench.org/competitions/2469/) |

---

## Reproduce everything

```bash
make setup                      # .venv, pinned deps, editable install, git hooks
make download                   # ~190 MB into Data/, idempotent, SHA256-verified
make clean-derived && make all  # rebuild every artefact and every number from raw files
```

`make all` runs `data → bm25 → semantic → eval → ablations → submit`. It needs no
GPU: `--device auto` detects CUDA and falls back to CPU.

`make clean-derived` removes `data/`, `artifacts/` and `reports/metrics/` but
deliberately **keeps `Data/`**. Re-downloading 190 MB to prove a rebuild works
would test bandwidth, not the pipeline; the claim being verified is Q1.5's
"rebuilds everything **from raw files**", and `Data/` is exactly that boundary.

Run `make` with no target for the full list.

### Prerequisites

Python 3.12+, `make`, `wget`, `unzip`, `curl`. On Debian/Ubuntu also
`sudo apt install python3.12-venv` — the stock `python3` can create a virtualenv
but cannot put `pip` in it. If that package is unavailable, `make setup` falls
back to a pip-less venv plus the official `get-pip` bootstrap and tells you so.

---

## Regenerating the prediction files

Prediction files are committed **gzipped** under `predictions/` (decision D6:
Q7.1 lists them as a deliverable while Q8 forbids large files in git; compressing
keeps both true). To regenerate them from scratch and confirm they match:

```bash
make clean-derived && make all      # exact command that produces predictions/
cd predictions && sha256sum --check SHA256SUMS
```

`predictions/SHA256SUMS` is written by `make submit` and committed alongside, so
a grader can verify the committed files are the ones the pipeline actually
produces rather than taking it on trust.

---

## Layout

```
src/newsrec/            the package — config.py is the single source of truth
  data/                 Q1: download, parse, unify, temporal split, feature store
  retrieval/            Q2: our own inverted index + BM25 · Q3: embeddings + ANN
  eval/                 Q4: metrics, beyond-accuracy, slices, bootstrap CIs
  submit/               Q5: Codabench prediction formats
scripts/                build/audit helpers the Makefile calls
tests/                  mirrors src/newsrec/; fixtures/ holds tiny synthetic data
Data/            (ignored)  immutable raw downloads — read, never written
data/            (ignored)  generated: interim, processed, feature_store
artifacts/       (ignored)  BM25 index, embedding matrices, ANN indexes
reports/         committed  small JSON/CSV metrics and figures
predictions/     committed  gzipped prediction files + SHA256SUMS
ai_usage/sessions/  committed  every prompt, verbatim
```

---

## Tests

```bash
make test        # fast only: no dataset, no GPU, target < 30 s
make test-full   # adds tests needing a built data/ store
```

`make test` is expected to pass in a **clean clone with no `data/` and no
`Data/`** — that is the state a grader clones into, so the default test run
depends only on the tiny synthetic fixtures committed under `tests/fixtures/`.

---

## Rules:

- **Temporal splits only.** Interaction data is never randomly split. The split
  type in `config.py` has no fraction or shuffle parameter, so a random split is
  unrepresentable rather than merely discouraged.
- **No future information.** A user's history for an impression contains only
  clicks *strictly* before that impression's timestamp. Post-click fields and
  dataset-period aggregates are usable only inside a labelled ablation, never in
  a headline number. `tests/test_leakage.py` asserts this, and a canary fixture
  proves that test can actually fail.
