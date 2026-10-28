# AI usage statement

Claude Code (Claude Sonnet 5) was used throughout this assignment's implementation, via an
interactive session. The full prompt/response log is in `prompts_and_responses.md` in this
folder, extracted directly from the session transcript.

All design decisions (which Part 3/Part 4 experiment to run, the dimensionality hypothesis,
resource/partition settings for the Ada cluster, etc.) were made by the student in that
conversation; the AI wrote the code implementing them and ran/verified it locally before each
result was reported back.

## Per-file classification

All files below are under `code/`.

| File | Classification |
|---|---|
| `run_ada.sh` | AI-generated, then human-edited (SLURM resource values, output/error log paths edited directly by the student on the Ada cluster) |
| `README.md` | AI-generated |
| `requirements.txt` | AI-generated |
| `hypothesis.md` | AI-generated (content for Part 3 and Part 4 hypotheses, written and reviewed before each experiment was run, per the assignment's requirement) |
| `run_all.sh` | AI-generated |
| `pytest.ini` | AI-generated |
| `.gitignore` | AI-generated |
| `scripts/part1_exact_baseline.py` | AI-generated |
| `scripts/part2_benchmark.py` | AI-generated |
| `scripts/part3_scaling.py` | AI-generated |
| `scripts/part4_dimensionality.py` | AI-generated |
| `scripts/capture_environment.py` | AI-generated |
| `src/annbench/__init__.py` | AI-generated |
| `src/annbench/data.py` | AI-generated |
| `src/annbench/recall.py` | AI-generated |
| `src/annbench/timing.py` | AI-generated |
| `src/annbench/io_utils.py` | AI-generated |
| `src/annbench/paths.py` | AI-generated |
| `src/annbench/plotting.py` | AI-generated |
| `src/annbench/grids.py` | AI-generated |
| `src/annbench/benchmark.py` | AI-generated |
| `src/annbench/gpu_utils.py` | AI-generated |
| `src/annbench/synthetic.py` | AI-generated |
| `src/annbench/envinfo.py` | AI-generated |
| `src/annbench/indexes/__init__.py` | AI-generated |
| `src/annbench/indexes/base.py` | AI-generated |
| `src/annbench/indexes/exact.py` | AI-generated |
| `src/annbench/indexes/hnsw.py` | AI-generated |
| `src/annbench/indexes/ivf_flat.py` | AI-generated |
| `src/annbench/indexes/ivf_pq.py` | AI-generated |
| `tests/test_recall.py` | AI-generated |

No file in `code/` was written by the student without AI assistance.
