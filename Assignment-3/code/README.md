# Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction — Assignment 3.

Benchmarks exact search, HNSW (hnswlib), IVF-Flat and IVF-PQ (FAISS) on SIFT-128-Euclidean
and GloVe-100-Angular (ann-benchmarks datasets), plus a scaling experiment (Part 3) and a
self-generated-data dimensionality experiment (Part 4). See `../report.pdf` for the write-up.

## Layout

```
code/
├── src/annbench/           core library (reused by every part's script)
│   ├── data.py              dataset download/load, GloVe normalisation, query split
│   ├── recall.py            our own tie-aware recall@k
│   ├── timing.py            single-query vs batched timing, median-of-repeats
│   ├── io_utils.py          results.csv append helper (idempotent per (part, dataset))
│   ├── paths.py             shared filesystem locations (data/, results/, figures/)
│   ├── plotting.py          Pareto frontier + recall-vs-QPS plot, qps_at_recall
│   ├── grids.py             HNSW/IVF-Flat/IVF-PQ build and search parameter grids
│   ├── benchmark.py         shared build/sweep/measure logic used by Parts 2 and 3
│   ├── gpu_utils.py         optional GPU-accelerated IVF training, CPU fallback
│   ├── synthetic.py         seeded uniform-random data generator for Part 4
│   ├── envinfo.py           machine specs + library version capture
│   └── indexes/             ANNIndex wrappers: exact, hnsw, ivf_flat, ivf_pq
├── scripts/
│   ├── part1_exact_baseline.py     Part 1: exact search, recall check, timing
│   ├── part2_benchmark.py          Part 2: HNSW / IVF-Flat / IVF-PQ sweep, both datasets
│   ├── part3_scaling.py            Part 3: SIFT at 100K/300K/1M vectors
│   ├── part4_dimensionality.py     Part 4: self-generated data, curse of dimensionality
│   └── capture_environment.py      writes results/environment.txt
├── tests/test_recall.py     pytest unit tests for recall@k (incl. tie handling)
├── hypothesis.md             Part 3 / Part 4 hypotheses, committed before each experiment
├── requirements.txt
├── run_all.sh                one-command local reproduction (all 4 parts)
├── run_ada.sh                SLURM submission script (IIIT-H Ada cluster)
└── run_ada_env.sh            SLURM script: capture environment.txt only, no benchmarks
```

Data lives one level up, at `../data/` (sibling of `code/`), and is **not** part of this repo
or the submission zip. If `../data/sift-128-euclidean.hdf5` or `../data/glove-100-angular.hdf5`
is missing, the scripts download it automatically from ann-benchmarks on first use — except on
Ada, whose compute nodes have no internet access, so the data files must be copied there
beforehand (see `run_ada.sh`'s header comment).

Outputs are written to `../results/results.csv`, `../results/figures/` and
`../results/environment.txt` (also siblings of `code/`, matching the submission zip's
top-level `results/` directory).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
bash run_all.sh            # full run: SIFT 1M / GloVe-100 1.18M, installs deps, runs tests,
                            # captures environment info, then Parts 1-4, regenerating every
                            # number/figure/table in the report.

bash run_all.sh --quick    # same pipeline on a 100K-vector train subset (first 100K rows),
                            # a smaller parameter grid, and fewer timing repeats.
                            # Finishes in well under 15 minutes on a laptop.
```

Each script is also runnable on its own, e.g. `python3 scripts/part2_benchmark.py --quick`.
Every part is independent (none reads another part's output, and `results.csv` rows are
replaced per `(part, dataset)` on write), so running them one at a time, in any order, any
number of times, is always safe — this is what `run_ada.sh part1`/`part2`/`part3`/`part4` relies
on to fit inside Ada's 6-hour job wall-clock limit.

Run just the unit tests with:

```bash
python3 -m pytest tests/ -v
```

### Running on the Ada cluster

```bash
sbatch run_ada.sh            # full pipeline (Part 1 -> 2 -> 3 -> 4)
sbatch run_ada.sh part3      # just Part 3 (same for part1 / part2 / part4)
sbatch run_ada_env.sh        # only captures results/environment.txt, runs no benchmarks
                              # (use this to refresh the environment record without
                              # re-running everything, e.g. after results are already final)
```

`run_ada.sh` activates the venv at `code/.venv` (must already exist on Ada with dependencies
installed — compute nodes have no internet, so `pip install` is skipped there), optionally uses
GPU acceleration for IVF-Flat/IVF-PQ *training* only (`ANNBENCH_USE_GPU=1`; benchmarked search
always stays single-thread CPU, per the assignment's methodology), and expects
`code/ada_logs/` to already exist for SLURM's output/error logs.

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
- **Timing**: every reported QPS number is one query at a time, pinned to a single thread
  (`faiss.omp_set_num_threads(1)`), median of 3 repeats (1 in `--quick` mode). Part 1
  additionally compares that against batched search (same thread count, then default thread
  count) to explain why batching is so much faster.
- **Part 3 (Scaling)**: repeats the Part 2 sweep for HNSW/IVF-Flat/exact on SIFT at 100K, 300K
  and 1M training vectors (ground truth recomputed by brute force at each truncated size), and
  tracks QPS at recall@10 >= 0.95 as a function of N.
- **Part 4 (Dimensionality)**: self-generated i.i.d. uniform-random data
  (`annbench.synthetic.generate_uniform`, seeded), dimensionality d in {32, 64, 128, 256} with
  N and the HNSW/IVF-Flat configuration held fixed, to isolate the curse-of-dimensionality
  effect on recall at a constant search budget. No tune/final split here, since there is no
  parameter sweep/selection happening (both indexes use one fixed, pre-decided config).
- **GPU use**: `annbench.gpu_utils` optionally runs IVF-Flat/IVF-PQ's *k-means training* step on
  GPU (opt-in via `ANNBENCH_USE_GPU=1`, automatic fallback to CPU on any error or if unavailable).
  All benchmarked search, on every machine, is always single-thread CPU.
- **Single-machine run**: all 4 parts ran on one IIIT-H Ada cluster node. Because the full,
  non-`--quick` sweep exceeds Ada's 6-hour job wall-clock limit in one run, the parts were
  submitted as separate jobs (`run_ada.sh part1`/`part2`/`part3`/`part4`) rather than one
  continuous job, but every reported number comes from the same machine. Specs and library
  versions are recorded in `../results/environment.txt`.

## Status

All 5 parts are complete, with every result generated from a single machine (an IIIT-H Ada
cluster node). `report.pdf` (at the project root) contains the write-up for Parts 1-4 and
answers the Part 5 discussion questions. `results/results.csv` and `results/figures/` hold
every number and figure behind it.
