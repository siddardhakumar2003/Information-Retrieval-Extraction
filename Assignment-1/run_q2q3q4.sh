#!/bin/bash
# Run Q2, Q3, Q4 on real data (both datasets) from scratch, no synthetic data, no timeout

set -e

cd "$(dirname "$0")"
VENV_PYTHON=".venv/bin/python"

echo "====== Q2/Q3/Q4 Fresh Run ======"
echo "Start time: $(date)"
echo "Using Python: $($VENV_PYTHON --version)"

mkdir -p outputs/logs

echo ""
echo "====== Q2: EB-NeRD BM25 ======"
$VENV_PYTHON src/bm25_retrieval.py --k 50 100 200 --k1 1.5 --b 0.75 --data-dir data --out-dir outputs/lexical 2>&1 | tee outputs/logs/ebnerd_bm25.log

echo ""
echo "====== Q2: MIND BM25 ======"
$VENV_PYTHON src/mind_bm25_retrieval.py --k 50 100 200 --k1 1.5 --b 0.75 --out-dir outputs/mind_lexical 2>&1 | tee outputs/logs/mind_bm25.log

echo ""
echo "====== Q3: EB-NeRD Semantic ======"
$VENV_PYTHON src/semantic_retrieval.py --k 50 100 200 --nlist 100 --nprobe 10 --model paraphrase-multilingual-mpnet-base-v2 --out-dir outputs/semantic 2>&1 | tee outputs/logs/ebnerd_semantic.log

echo ""
echo "====== Q3: MIND Semantic ======"
$VENV_PYTHON src/mind_semantic_retrieval.py --k 50 100 200 --nlist 100 --nprobe 10 --model BAAI/bge-base-en-v1.5 --out-dir outputs/mind_semantic 2>&1 | tee outputs/logs/mind_semantic.log

echo ""
echo "====== Q4: Offline Evaluation (all 4 combos) ======"

echo "Q4.1: EB-NeRD BM25"
$VENV_PYTHON src/evaluate_models.py --dataset ebnerd --method bm25 2>&1 | tee outputs/logs/eval_ebnerd_bm25.log

echo ""
echo "Q4.2: EB-NeRD Semantic"
$VENV_PYTHON src/evaluate_models.py --dataset ebnerd --method semantic 2>&1 | tee outputs/logs/eval_ebnerd_semantic.log

echo ""
echo "Q4.3: MIND BM25"
$VENV_PYTHON src/evaluate_models.py --dataset mind --method bm25 2>&1 | tee outputs/logs/eval_mind_bm25.log

echo ""
echo "Q4.4: MIND Semantic"
$VENV_PYTHON src/evaluate_models.py --dataset mind --method semantic 2>&1 | tee outputs/logs/eval_mind_semantic.log

echo ""
echo "====== Complete ======"
echo "End time: $(date)"
echo "All outputs saved to outputs/ and logs to outputs/logs/"
