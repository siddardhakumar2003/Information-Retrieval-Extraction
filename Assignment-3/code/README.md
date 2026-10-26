# Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction — Assignment 3.

Benchmarks exact search, HNSW (hnswlib), IVF-Flat and IVF-PQ (FAISS) on SIFT-128-Euclidean
and GloVe-100-Angular (ann-benchmarks datasets).

## Layout

```
code/
├── src/annbench/           core library (reused by every part's script)
│   ├── data.py             dataset download/load, GloVe normalisation, query split
│   ├── recall.py           our own tie-aware recall@k
│   ├── timing.py           single-query vs batched timing, median-of-repeats
│   ├── io_utils.py         results.csv append helper
│   ├── plotting.py         Pareto frontier + recall-vs-QPS plot
│   └── indexes/            ANNIndex wrappers: exact, hnsw, ivf_flat, ivf_pq
├── scripts/
│   ├── part1_exact_baseline.py
│   └── part2_benchmark.py
├── tests/test_recall.py    pytest unit tests for recall@k (incl. tie handling)
├── hypothesis.md           Part 3 / Part 4 hypotheses, committed before each experiment
├── requirements.txt
└── run_all.sh
```

Data lives one level up, at `../data/` (sibling of `code/`), and is **not** part of this repo
or the submission zip. If `../data/sift-128-euclidean.hdf5` or `../data/glove-100-angular.hdf5`
is missing, the scripts download it automatically from ann-benchmarks on first use.

Outputs are written to `../results/results.csv` and `../results/figures/` (also a sibling of
`code/`, matching the submission zip's top-level `results/` directory).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
bash run_all.sh            # full run: SIFT 1M / GloVe-100 1.18M, installs deps, runs tests,
                            # then Part 1 and Part 2, regenerating every number/figure/table.

bash run_all.sh --quick    # same pipeline on a 100K-vector train subset (first 100K rows),
                            # a smaller parameter grid, and fewer timing repeats.
                            # Finishes in well under 15 minutes on a laptop.
```

Each script is also runnable on its own, e.g. `python3 scripts/part2_benchmark.py --quick`.

Run just the unit tests with:

```bash
python3 -m pytest tests/ -v
```

## Methodology notes

- **Query split**: every dataset ships 10,000 queries. The first 5,000 ("tune") are used only
  to sanity-check that the swept search-parameter grid actually spans recall from low up to
  ~0.95+; the last 5,000 ("final") are what every reported recall/QPS number, plot and table
  is computed from. See `annbench.data.split_queries`.
- **GloVe normalisation**: GloVe-100's ground truth is angular/cosine. We L2-normalise every
  GloVe vector on load, after which Euclidean distance ranks identically to cosine distance
  for unit vectors, so every index just uses Euclidean distance throughout (`IndexFlatL2`,
  `IndexIVFFlat`/`IndexIVFPQ` with `METRIC_L2`, hnswlib `space="l2"`).
- **Recall@10 ties**: `recall.recall_at_k` compares *distances* against the ground truth's
  10th-nearest distance (with a small floating-point tolerance), not raw id membership, so an
  equidistant alternate neighbour at the boundary isn't counted as a miss.
- **Timing**: Part 2's QPS numbers are one query at a time, pinned to a single thread
  (`faiss.omp_set_num_threads(1)`), median of a few repeats. Part 1 additionally compares that
  against batched search (same thread count, then default thread count) to explain why batching
  is so much faster.

## Status

Parts 1 and 2 are implemented. Parts 3, 4 and the report are not yet started — `hypothesis.md`
has placeholder sections for them; each will get its hypothesis written and committed *before*
its experiment script is run, per the assignment's git-history requirement.
