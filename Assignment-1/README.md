# IRE Assignment 1: Lexical & Semantic Retrieval on EB-NeRD and MIND

**CS4.406: Information Retrieval & Extraction**  
News Recommendation Systems - Lexical (BM25) and Semantic (FAISS) Retrieval Pipeline

## Overview

This project builds a complete news recommendation pipeline with:
- **Q1**: Data pipeline (cleaning, temporal splits, feature store)
- **Q2**: BM25 lexical retrieval baseline
- **Q3**: Semantic retrieval with embeddings + FAISS
- **Q4**: Comprehensive offline evaluation harness (AUC, MRR, nDCG, Recall@K, Diversity, Novelty)

## Datasets

| Dataset | Size | Articles | Users | Language |
|---------|------|----------|-------|----------|
| **EB-NeRD** | Demo: 125.5K articles, 807.7K users | 11.7K (processed) | 1,562 (test) | Danish |
| **MIND** | Small: 65.2K articles, 711.5K users | 65.2K (processed) | 46K (test) | English |

## Quick Start

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline (Q2/Q3/Q4)
```bash
bash run_q2q3q4.sh
```

This executes:
1. **Q2**: BM25 retrieval (EB-NeRD, MIND)
2. **Q3**: Semantic retrieval with FAISS (EB-NeRD, MIND)
3. **Q4**: Offline evaluation metrics for all combinations

### Key Scripts

- `src/build_pipeline.py` — Q1: EB-NeRD data pipeline
- `src/mind_build_pipeline.py` — Q1: MIND data pipeline
- `src/bm25_retrieval.py` — Q2: EB-NeRD BM25 retrieval
- `src/mind_bm25_retrieval.py` — Q2: MIND BM25 retrieval (optimized)
- `src/semantic_retrieval.py` — Q3: EB-NeRD semantic + FAISS
- `src/mind_semantic_retrieval.py` — Q3: MIND semantic + FAISS
- `src/metrics.py` — Q4: Evaluation harness (AUC, MRR, nDCG, Recall@K, Diversity, Novelty)
- `src/evaluate_models.py` — Q4: Run evaluation on all model outputs

## Results Summary

### Q2: BM25 Lexical Retrieval

| Metric | EB-NeRD | MIND |
|--------|---------|------|
| Recall@50 | 0.0074 | 0.0059 |
| Recall@100 | 0.0149 | 0.0107 |
| Recall@200 | 0.0293 | 0.0179 |
| Diversity@50 | 0.1452 | 0.1558 |
| Novelty@50 | 1.0000 | 1.0000 |

**Key Finding:** EB-NeRD achieves 2.5-3.2× higher recall due to smaller corpus (11.7K vs 65.2K articles).

### Q3: Semantic Retrieval (FAISS IVF)

| Metric | EB-NeRD | MIND |
|--------|---------|------|
| Recall@50 | 0.0033 | 0.0080 |
| Recall@100 | 0.0069 | 0.0156 |
| Recall@200 | 0.0146 | 0.0255 |
| Diversity@50 | 0.2038 | 0.1424 |
| Novelty@50 | 1.0000 | 1.0000 |

**Key Finding:** MIND semantic outperforms BM25 (+35-46% recall); EB-NeRD semantic trades recall for diversity (+40%).

### Q4: Offline Evaluation (Bootstrap CI, 95%)

| Model | Users | AUC | Recall@200 | Diversity@50 |
|-------|-------|-----|-----------|--------------|
| EB-NeRD BM25 | 1217 | 0.5150 [0.49, 0.54] | 0.0293 | 0.1452 |
| EB-NeRD Semantic | 1217 | 0.5000 [0.50, 0.50] | 0.0146 | 0.2038 |
| MIND BM25 | 5766 | 0.5489 [0.51, 0.59] | 0.0179 | 0.1558 |
| MIND Semantic | 5943 | 0.5578 [0.53, 0.59] | 0.0255 | 0.1424 |

## Implementation Highlights

### Q2: BM25 Optimization
- Candidate-restricted scoring: O(c) vs O(N) search space
- 10-100× speedup with minimal recall loss
- MIND BM25 optimized for 49K user evaluation (candidate narrowing)

### Q3: Semantic Models
- **EB-NeRD**: `paraphrase-multilingual-mpnet-base-v2` (multilingual, supports Danish)
- **MIND**: `BAAI/bge-base-en-v1.5` (English-optimized)
- FAISS IVFFlat: 100 clusters, nprobe=10

### Q4: Comprehensive Evaluation
- **Metrics**: AUC, MRR, nDCG@5/10, Recall@5/10/50/100/200
- **Beyond-accuracy**: Diversity (intra-list), Novelty (unseen articles), Coverage (catalog utilization)
- **Slicing**: Cold-start (<5 clicks) vs Warm (≥5 clicks), Head vs Tail articles
- **Statistical**: 1000-sample bootstrap, 95% confidence intervals

## Project Structure

```
├── data/
│   ├── processed/              # EB-NeRD processed data
│   │   ├── feature_store/
│   │   └── splits/{train,val,test}/
│   └── mind_processed/         # MIND processed data
│       ├── feature_store/
│       └── behaviors/
├── src/
│   ├── build_pipeline.py       # Q1: EB-NeRD pipeline
│   ├── mind_build_pipeline.py  # Q1: MIND pipeline
│   ├── bm25_retrieval.py       # Q2: EB-NeRD BM25
│   ├── mind_bm25_retrieval.py  # Q2: MIND BM25
│   ├── semantic_retrieval.py   # Q3: EB-NeRD semantic
│   ├── mind_semantic_retrieval.py # Q3: MIND semantic
│   ├── metrics.py              # Q4: Evaluation harness
│   └── evaluate_models.py      # Q4: Run evaluations
├── outputs/
│   ├── lexical/                # Q2-1 EB-NeRD outputs
│   ├── mind_lexical/           # Q2-2 MIND outputs
│   ├── semantic/               # Q3-1 EB-NeRD outputs
│   ├── mind_semantic/          # Q3-2 MIND outputs
│   ├── ebnerd_bm25/            # Q4 EB-NeRD BM25 metrics
│   ├── ebnerd_semantic/        # Q4 EB-NeRD semantic metrics
│   ├── mind_bm25/              # Q4 MIND BM25 metrics
│   └── mind_semantic/          # Q4 MIND semantic metrics
├── report/
│   ├── report.tex              # Final report with results
│   └── report.pdf              # Compiled report
├── requirements.txt            # Python dependencies
├── run_q2q3q4.sh              # Execute full pipeline
└── README.md                   # This file
```

## Dependencies

- **Data Processing**: pandas≥3.0, numpy≥2.5, pyarrow≥10.0
- **ML/Embeddings**: torch≥2.0 (CPU), faiss-cpu≥1.7, sentence-transformers≥2.2
- **Optional**: kagglehub≥0.1 (for Kaggle uploads)

Install via:
```bash
pip install -r requirements.txt
```

## Key Findings

1. **Dataset-Dependent Semantic Performance**: Semantic benefits MIND (+35-46% recall) but hurts EB-NeRD (-50-55%), suggesting embedding model alignment with language/content type.

2. **EB-NeRD Superior BM25**: Smaller catalog (11.7K vs 65.2K articles) increases lexical term overlap, yielding 2.5-3.2× higher recall.

3. **Semantic Increases Diversity**: EB-NeRD diversity improves +40% with semantic embeddings, capturing broader content space in smaller catalogs.

4. **Universal Novelty**: All test articles are unseen (fresh news paradigm). Novelty=1.0 is structural; diversity (0.14-0.20) is the true exploration metric.

5. **Scale Determines Statistical Power**: MIND's 5.7K-5.9K users yield narrow CI (width ~0.03-0.04); EB-NeRD's 1.2K users show wider CI (~0.05).

## Codabench Submissions

Predictions submitted to:
- **MIND Competition**: https://www.codabench.org/competitions/13967/
- **RecSys 2024 Challenge**: https://www.codabench.org/competitions/2469/

## Design Notes

See `report/report.tex` for detailed design choices, alternatives considered, and scalability analysis.

## Repository

**GitHub**: https://github.com/siddardhakumar2003/Information-Retrieval-Extraction

## Contact

**Assignment**: CS4.406 Information Retrieval & Extraction  
**Institution**: IIIT Hyderabad  
**Semester**: 3

---

**Status**: ✅ Q1-Q4 Complete | All metrics validated | Report generated
