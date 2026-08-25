# Q4 Metrics Issues & Fixes Summary

## Critical Issues Discovered

### 1. **Semantic Retrieval: Incorrect history_len Tracking**

**EB-NeRD Semantic Issue:**
- **Problem**: `history_len` was computed as `np.sum(profile_embedding != 0)` 
  - This counts non-zero dimensions in the embedding vector
  - For paraphrase-multilingual-mpnet-base-v2 (768-dim), almost all dimensions are non-zero
  - When no history existed, profile_embedding was all zeros → history_len = 0
  - **Result**: ALL users had history_len = 0, when they should have had 1-174 clicks
  
**MIND Semantic Issue:**  
- **Problem**: Same issue but with different embedding dimension
  - BGE embedding is 768-dimensional
  - **Result**: ALL users had history_len = 768 (embedding dimension), not actual click count

**Root Cause:**
- Both semantic retrieval scripts tried to read a non-existent `click_history` column
- When column doesn't exist, `row.get("click_history", None)` returns None for all rows
- This caused default zero profile to be used for all users
- The buggy line tried to compute history from embedding dimension as a "proxy"

### 2. **Data Schema Mismatch**

**What Happened:**
- Q1 creates behaviors data with columns: `article_ids_clicked`, `article_ids_inview`, etc.
- Semantic retrieval scripts looked for `click_history` (doesn't exist)
- Should have aggregated clicks from `article_ids_clicked` across all impressions per user
- BM25 correctly does this aggregation

## Fixes Applied

### semantic_retrieval.py (EB-NeRD)
```python
# BEFORE:
history_len": np.sum(profile_embedding != 0)  # WRONG

# AFTER:
# Aggregate clicks from impressions data
user_clicked_ids = defaultdict(list)
for _, row in users_train.iterrows():
    user_id = int(row.get("user_id"))
    clicked_articles = row.get("article_ids_clicked", None)
    if clicked_articles is not None and len(clicked_articles) > 0:
        user_clicked_ids[user_id].extend(clicked_articles)

# Store actual history length
user_history_lengths[user_id] = len(clicked_ids)
```

### mind_semantic_retrieval.py (MIND)
- Applied identical fix
- Now correctly aggregates click history from impressions data

## Impact on Results

### Metrics Now Correct For:
✅ EB-NeRD BM25: history_len = 17.93 (mean), range [1, 174]
✅ MIND BM25: history_len = 28.10 (mean), range [1, 768]

### Still Regenerating:
⏳ EB-NeRD Semantic: Embedding computation in progress (~15 min remaining)
⏳ MIND Semantic: Embedding computation in progress (~15 min remaining)

## Why This Matters

1. **Affects Slicing Metrics**: Cold-start (history < 5) vs warm users distinction relies on history_len
2. **Affects Novelty Computation**: Uses history_len to compute novelty score
3. **Skews Cold-Start Performance**: All users incorrectly classified when history_len = 0/768
4. **Confidence Intervals**: Bootstrap CI incorrectly averaged across misclassified users

## Next Steps

1. Wait for semantic embedding to complete (~10-15 min)
2. Re-run Q4 evaluation for both semantic methods
3. Compare corrected results:
   - EB-NeRD BM25 vs Semantic
   - MIND BM25 vs Semantic
4. Update report with corrected metrics

## Files Modified
- `src/semantic_retrieval.py` - Fixed history aggregation
- `src/mind_semantic_retrieval.py` - Fixed history aggregation
