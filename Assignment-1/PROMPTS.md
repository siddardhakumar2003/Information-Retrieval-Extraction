# Prompt Log

Log of prompts submitted to Claude Code for this project, kept for AI-usage disclosure / chat history export purposes.

---

### 2026-08-02

1. see whatever i give prompts first save them in separate file open a new file for prompts because i have to submit prompts for now on

2. all prompts, chat history exports, marking of AI-generated vs. human-written code. so save prompts i an file "name" of your choice but save from now on on this project

3. write them in separate file just

4. first explain me the datasets below and what are this benchmark dataset for: "MIND","EB-NeRD"

---

### 2026-08-03

1. read "data/ebnerd_demo.zip" and tell me about metadata of dataset

2. see i want all files metadata it is said that it has behaviour/impressions click history articles and more i want that type of metadata to read dataset

3. write a single script that clean this dataset and make a unified schema articles, behaviours, click history. Then split thae train folder in validation and train keep val folder for testing. Do time-based splitting e.g., last N days as validation, preceding M days as train.  small, reusable store for article features (title, abstract, body, category, entities, embeddings) and user features (click history, recency) -> file name is "build_pipeline.py"

---

### 2026-08-04

1. first tell me what we did in cleaning of dataset

2. BM25 retrieval baseline: Build inverted index from article titles+abstracts. For each user, concatenate their pre-train click history (title+abstract of each article), use as BM25 query. Retrieve top-K (50, 100, 150) articles. Measure Recall@K against actual user clicks in train window (union of all clicks per user). Output: recall_results.json + per_user_recall.parquet -> file name "src/bm25_retrieval.py"

3. Use processed feature store (cleaned articles, users_train_region with click_history, train behaviors) not raw zip files. Ground truth from train split only (1533 users have clicks, 57 excluded). BM25 parameters: k1=1.5, b=0.75, CLI-configurable.
