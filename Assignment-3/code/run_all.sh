#!/usr/bin/env bash
# One-command reproduction. From inside code/:
#   bash run_all.sh            full run (SIFT 1M / GloVe-100 1.18M)
#   bash run_all.sh --quick    small subset (100K train vectors), <15 min
set -euo pipefail
cd "$(dirname "$0")"

QUICK_FLAG=""
if [ "${1:-}" = "--quick" ]; then
    QUICK_FLAG="--quick"
fi

if [ "${SKIP_DEPS_INSTALL:-0}" != "1" ]; then
    echo "== installing dependencies =="
    python3 -m pip install -q -r requirements.txt
else
    echo "== skipping dependency install (SKIP_DEPS_INSTALL=1) =="
fi

echo "== unit tests =="
python3 -m pytest tests/ -q

echo "== Part 1: exact baseline =="
python3 scripts/part1_exact_baseline.py $QUICK_FLAG

echo "== Part 2: benchmark HNSW / IVF-Flat / IVF-PQ =="
python3 scripts/part2_benchmark.py $QUICK_FLAG

echo "== Part 3: scaling experiment (SIFT) =="
python3 scripts/part3_scaling.py $QUICK_FLAG

echo "== Part 4: dimensionality experiment (self-generated data) =="
python3 scripts/part4_dimensionality.py $QUICK_FLAG

echo "== done =="
echo "results: ../results/results.csv"
echo "figures/tables: ../results/figures/"
