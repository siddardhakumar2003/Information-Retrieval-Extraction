"""
Q4: Offline Evaluation Harness
Computes per-impression metrics: AUC, MRR, nDCG@5/10, Recall@50/100/200,
Diversity@50, Novelty@50, Coverage@50 with 1000-sample bootstrap 95% CI and cold-start/warm slicing.

Usage:
    python3 src/offline_eval.py --dataset ebnerd --method bm25
    python3 src/offline_eval.py --dataset ebnerd --method semantic
    python3 src/offline_eval.py --dataset mind --method bm25
    python3 src/offline_eval.py --dataset mind --method semantic
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import OfflineMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation (Q4)")
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--method", choices=["bm25", "semantic"], required=True)
    args = parser.parse_args()

    dataset_name = f"{args.dataset}_{args.method}"
    output_dir = REPO_ROOT / "outputs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating {args.dataset} with {args.method} retrieval")

    # Load data from processed or mind_processed
    if args.dataset == "ebnerd":
        articles_path = REPO_ROOT / "data" / "processed" / "feature_store" / "articles.parquet"
        test_beh_path = REPO_ROOT / "data" / "processed" / "splits" / "test" / "behaviors.parquet"
        train_click_path = REPO_ROOT / "data" / "processed" / "splits" / "train" / "click_events.parquet"

        if args.method == "bm25":
            from bm25_retrieval import InvertedIndex
        else:
            from semantic_retrieval import SemanticRetriever

    else:  # mind
        articles_path = REPO_ROOT / "data" / "mind_processed" / "feature_store" / "articles.parquet"
        test_beh_path = REPO_ROOT / "data" / "mind_processed" / "feature_store" / "test_behaviors.parquet"
        train_click_path = None

        if args.method == "bm25":
            from mind_bm25_retrieval import MINDInvertedIndex as InvertedIndex
        else:
            from mind_semantic_retrieval import MINDSemanticRetriever as SemanticRetriever

    # Load data
    logger.info("Loading data...")
    articles = pd.read_parquet(articles_path)
    test_behaviors = pd.read_parquet(test_beh_path)
    if train_click_path:
        train_clicks = pd.read_parquet(train_click_path)
    else:
        train_clicks = None

    logger.info(f"  Articles: {len(articles)}")
    logger.info(f"  Test behaviors: {len(test_behaviors)}")

    n_articles = len(articles)

    # Create article_id to index mapping
    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"])}
    logger.info(f"Article ID mapping: {len(article_id_to_idx)} articles")

    # Build index/retriever
    logger.info(f"Building {args.method} index...")
    if args.method == "bm25":
        index = InvertedIndex.build(articles)
        logger.info(f"  Index: {len(index.postings)} terms, {index.N} docs")
    else:
        if args.dataset == "ebnerd":
            retriever = SemanticRetriever(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        else:
            retriever = SemanticRetriever(model="BAAI/bge-base-en-v1.5")
        retriever.build_index(articles)
        logger.info(f"  Embeddings: {retriever.embeddings.shape}")

    # Evaluate
    logger.info("Evaluating impressions...")
    metrics_harness = OfflineMetrics(k_values=[5, 10, 50, 100, 200])

    # Build category map
    article_categories = {}
    if "category" in articles.columns:
        for idx, row in articles.iterrows():
            article_categories[idx] = row.get("category", "unknown")

    results = []
    cold_start_results = []
    warm_results = []

    # Parse test behaviors based on dataset
    if args.dataset == "ebnerd":
        # Build user history from train clicks
        user_history_by_id = {}
        if train_clicks is not None:
            for _, row in train_clicks.iterrows():
                uid = row.get("user_id")
                aid = row.get("article_id")
                if uid not in user_history_by_id:
                    user_history_by_id[uid] = []
                user_history_by_id[uid].append(aid)

        for idx, row in test_behaviors.iterrows():
            if idx % 5000 == 0 and idx > 0:
                logger.info(f"  Processed {idx}/{len(test_behaviors)} impressions")

            user_id = row.get("user_id")
            candidates = row.get("article_ids_inview", []) if isinstance(row.get("article_ids_inview"), list) else []
            user_history = user_history_by_id.get(user_id, [])
            clicked = row.get("article_ids_clicked", []) if isinstance(row.get("article_ids_clicked"), list) else []

            if not candidates:
                continue

            # Create scores array
            scores = np.zeros(n_articles)

            # Map article IDs to indices for evaluation (convert to int for consistent dict lookup)
            candidate_indices = [article_id_to_idx[int(aid)] for aid in candidates if int(aid) in article_id_to_idx]
            clicked_indices = [article_id_to_idx[int(cid)] for cid in clicked if int(cid) in article_id_to_idx]
            user_history_indices = [article_id_to_idx[int(hid)] for hid in user_history if int(hid) in article_id_to_idx]

            if not candidate_indices:
                continue

            # Score candidates
            if args.method == "bm25":
                if user_history_indices:
                    # Build query term frequency counter from user history
                    from bm25_retrieval import tokenize
                    user_query_tokens = []
                    for hist_idx in user_history_indices:
                        article_text = articles.iloc[hist_idx]
                        text = f"{article_text.get('title', '')} {article_text.get('subtitle', '')}"
                        user_query_tokens.extend(tokenize(text))
                    qtf_counter = Counter(user_query_tokens)
                    # Score candidates (convert to int for consistent dict lookup)
                    candidate_article_ids = set(int(aid) for aid in candidates)
                    candidate_scores_dict = index.score_documents_batch(candidate_article_ids, qtf_counter)
                    for aid, score in candidate_scores_dict.items():
                        aid_int = int(aid)
                        if aid_int in article_id_to_idx:
                            idx = article_id_to_idx[aid_int]
                            scores[idx] = score
                else:
                    # No user history: use uniform scores
                    for idx in candidate_indices:
                        scores[idx] = 1.0 / len(candidate_indices)
            else:
                if user_history_indices:
                    # Semantic retrieval
                    cand_scores_dict = retriever.score_candidates(user_history_indices, candidate_indices)
                    for idx in candidate_indices:
                        scores[idx] = cand_scores_dict.get(idx, 0.0)
                else:
                    # No user history: use uniform scores
                    for idx in candidate_indices:
                        scores[idx] = 1.0 / len(candidate_indices)

            # Ground truth
            ground_truth = np.zeros(n_articles)
            for idx in clicked_indices:
                ground_truth[idx] = 1

            # Article IDs for diversity/novelty
            article_ids = list(range(n_articles))

            # Evaluate
            metrics = metrics_harness.evaluate_user(
                predictions=scores,
                ground_truth=ground_truth,
                article_ids=article_ids,
                article_categories=article_categories if article_categories else None,
                user_history=user_history,
            )
            metrics["impression_id"] = idx
            results.append(metrics)

            # Separate cold-start vs warm
            if len(user_history) < 5:
                cold_start_results.append(metrics)
            else:
                warm_results.append(metrics)

    else:  # MIND
        # MIND format: article_ids_shown, article_ids_clicked
        for idx, row in test_behaviors.iterrows():
            if idx % 10000 == 0 and idx > 0:
                logger.info(f"  Processed {idx}/{len(test_behaviors)} impressions")

            candidates = row.get("article_ids_shown", []) if isinstance(row.get("article_ids_shown"), list) else []
            clicked = row.get("article_ids_clicked", []) if isinstance(row.get("article_ids_clicked"), list) else []
            user_history = row.get("click_history", []) if isinstance(row.get("click_history"), list) else []

            if not candidates:
                continue

            # Create scores array
            scores = np.zeros(n_articles)

            # Score candidates
            if args.method == "bm25":
                if user_history and candidates:
                    cand_scores = index.score_candidates(user_history, candidates)
                    for aid, score in zip(candidates, cand_scores):
                        if isinstance(aid, str):
                            try:
                                aid = int(aid)
                            except:
                                continue
                        if aid < n_articles:
                            scores[aid] = score
            else:
                if user_history and candidates:
                    cand_scores = retriever.score_candidates(user_history, candidates)
                    for aid, score in zip(candidates, cand_scores):
                        if isinstance(aid, str):
                            try:
                                aid = int(aid)
                            except:
                                continue
                        if aid < n_articles:
                            scores[aid] = score

            # Ground truth
            ground_truth = np.zeros(n_articles)
            for cid in clicked:
                if isinstance(cid, str):
                    try:
                        cid = int(cid)
                    except:
                        continue
                if cid < n_articles:
                    ground_truth[cid] = 1

            # Article IDs for diversity/novelty
            article_ids = list(range(n_articles))

            # Evaluate
            metrics = metrics_harness.evaluate_user(
                predictions=scores,
                ground_truth=ground_truth,
                article_ids=article_ids,
                article_categories=article_categories if article_categories else None,
                user_history=user_history,
            )
            metrics["impression_id"] = idx
            results.append(metrics)

            # Separate cold-start vs warm
            if len(user_history) < 5:
                cold_start_results.append(metrics)
            else:
                warm_results.append(metrics)

    if not results:
        logger.warning("No impressions evaluated!")
        return

    metrics_df = pd.DataFrame(results)
    logger.info(f"Evaluated {len(metrics_df)} impressions")

    # Save per-impression results
    metrics_df.to_csv(output_dir / "offline_metrics_per_impression.csv", index=False)
    metrics_df.to_json(output_dir / "offline_metrics_per_impression.jsonl", orient="records", lines=True)
    logger.info(f"Saved per-impression metrics")

    # Aggregate metrics
    aggregated = metrics_harness.aggregate_metrics(metrics_df)
    logger.info(f"Aggregated {len(aggregated)} metrics")

    # Bootstrap CI
    ci = metrics_harness.bootstrap_ci(metrics_df, n_bootstrap=1000, ci=0.95)
    logger.info(f"Computed 95% bootstrap CI")

    # Cold-start vs warm
    cold_df = pd.DataFrame(cold_start_results) if cold_start_results else pd.DataFrame()
    warm_df = pd.DataFrame(warm_results) if warm_results else pd.DataFrame()

    cold_agg = metrics_harness.aggregate_metrics(cold_df) if len(cold_df) > 0 else {}
    warm_agg = metrics_harness.aggregate_metrics(warm_df) if len(warm_df) > 0 else {}

    logger.info(f"Cold-start: {len(cold_df)} impressions")
    logger.info(f"Warm users: {len(warm_df)} impressions")

    # Save results
    results_dict = {
        "config": {
            "name": f"{args.dataset.upper()} {args.method.upper()}",
            "n_impressions": len(metrics_df),
            "cold_threshold": 5,
        },
        "aggregated_metrics": aggregated,
        "confidence_intervals": {k: [float(v[0]), float(v[1])] for k, v in ci.items()},
        "slicing": {
            "cold_start": cold_agg,
            "warm_users": warm_agg,
        },
    }

    with open(output_dir / "offline_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved results to offline_metrics.json")

    # Print results
    print(f"\n{'='*70}")
    print(f"{args.dataset.upper()} {args.method.upper()} Evaluation Results ({len(metrics_df)} impressions)")
    print(f"{'='*70}")
    for metric, value in sorted(aggregated.items()):
        if pd.isna(value):
            print(f"{metric:20s}: N/A")
        else:
            if metric in ci:
                lower, upper = ci[metric]
                print(f"{metric:20s}: {value:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
            else:
                print(f"{metric:20s}: {value:.4f}")

    if cold_agg or warm_agg:
        print(f"\n{'='*70}")
        print("Cold-Start vs Warm Users")
        print(f"{'='*70}")
        all_metrics = set(cold_agg.keys()) | set(warm_agg.keys())
        for metric in sorted(all_metrics):
            val_cold = cold_agg.get(metric, np.nan)
            val_warm = warm_agg.get(metric, np.nan)
            if not pd.isna(val_cold) and not pd.isna(val_warm):
                diff = ((val_cold - val_warm) / val_warm * 100) if val_warm != 0 else 0
                print(f"{metric:20s}: {val_cold:.4f} (cold) | {val_warm:.4f} (warm) | {diff:+.1f}%")

    print(f"{'='*70}\n")
    logger.info("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
