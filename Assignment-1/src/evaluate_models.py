"""
Q4: Offline Evaluation Harness
Loads per-user predictions from Q2/Q3 (BM25 + semantic retrieval)
and computes AUC, MRR, nDCG, Recall, Diversity, Novelty, Coverage
with bootstrap CI and cold-start/warm-user + head/tail article slicing.

Usage:
    python src/evaluate_models.py --dataset ebnerd --method bm25
    python src/evaluate_models.py --dataset mind --method semantic
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from metrics import OfflineMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_predictions(dataset: str, method: str) -> pd.DataFrame:
    """Load per-user predictions from Q2/Q3 outputs."""
    if dataset == "ebnerd":
        if method == "bm25":
            output_dir = REPO_ROOT / "outputs" / "lexical"
        else:  # semantic
            output_dir = REPO_ROOT / "outputs" / "semantic"
    else:  # mind
        if method == "bm25":
            output_dir = REPO_ROOT / "outputs" / "mind_lexical"
        else:  # semantic
            output_dir = REPO_ROOT / "outputs" / "mind_semantic"

    preds_path = output_dir / "per_user_predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Per-user predictions not found: {preds_path}")

    return pd.read_parquet(preds_path)


def load_articles_metadata(dataset: str) -> Tuple[dict, dict]:
    """Load article metadata (categories and dense ID index) for diversity/novelty/coverage computation."""
    if dataset == "ebnerd":
        articles_path = REPO_ROOT / "data" / "processed" / "feature_store" / "articles.parquet"
    else:  # mind
        articles_path = REPO_ROOT / "data" / "mind_processed" / "feature_store" / "articles.parquet"

    articles = pd.read_parquet(articles_path)

    # Build category mapping
    article_categories = {}
    if "category" in articles.columns:
        for aid, cat in zip(articles["article_id"], articles["category"]):
            article_categories[aid] = cat

    # Build dense article ID to 0..N-1 index mapping (needed for compute_coverage)
    article_id_to_dense_idx = {aid: i for i, aid in enumerate(articles["article_id"])}

    return article_categories, article_id_to_dense_idx


def load_article_popularity(dataset: str) -> Dict:
    """Aggregate article click counts across train+val+test behaviors for head/tail slicing."""
    if dataset == "ebnerd":
        splits_dir = REPO_ROOT / "data" / "processed" / "splits"
        paths = [splits_dir / s / "behaviors.parquet" for s in ("train", "val", "test")]
    else:  # mind
        fs = REPO_ROOT / "data" / "mind_processed" / "feature_store"
        paths = [fs / f"{s}_behaviors.parquet" for s in ("train", "val", "test")]

    counts = Counter()
    for p in paths:
        df = pd.read_parquet(p, columns=["article_ids_clicked"])
        for clicked in df["article_ids_clicked"]:
            if clicked is not None and len(clicked) > 0:
                counts.update(clicked)

    return dict(counts)


def _bootstrap_coverage_ci(all_predictions_dense: list, k: int, metrics_harness: OfflineMetrics,
                           n_bootstrap: int = 200, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap CI for coverage (corpus-level metric, not per-user)."""
    n = len(all_predictions_dense)
    vals = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        resampled = [all_predictions_dense[i] for i in idx]
        vals.append(metrics_harness.compute_coverage(resampled, k=k))

    alpha = 1 - ci
    lo, hi = np.percentile(vals, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def evaluate(dataset: str, method: str) -> Tuple[dict, pd.DataFrame]:
    """Run Q4 evaluation on Q2/Q3 predictions."""
    logger.info(f"Evaluating {dataset.upper()} {method.upper()}")

    # Load predictions
    predictions_df = load_predictions(dataset, method)
    logger.info(f"Loaded predictions for {len(predictions_df)} users")

    # Fail fast: require history_ids column for real novelty
    if "history_ids" not in predictions_df.columns:
        raise RuntimeError(f"history_ids column missing in predictions — rerun Q2/Q3 scripts first")

    # Load article metadata and popularity
    article_categories, article_id_to_dense_idx = load_articles_metadata(dataset)
    article_popularity = load_article_popularity(dataset)

    # Initialize metrics harness
    metrics_harness = OfflineMetrics(k_values=[5, 10, 50, 100, 200])

    # Evaluate each user
    all_results = []
    cold_start_results = []
    warm_results = []
    all_predictions_dense = []
    ground_truth_list = []
    article_ids_list = []

    for _, row in predictions_df.iterrows():
        user_id = row["user_id"]
        retrieved_ids = row["retrieved_ids"]
        retrieved_scores = np.array(row["retrieved_scores"])
        hits = np.array(row["hits"])
        history_ids = row["history_ids"]
        history_len = row["history_len"]
        n_true_relevant = int(row["n_true_relevant"]) if "n_true_relevant" in predictions_df.columns else None

        # Skip if no predictions
        if len(retrieved_ids) == 0:
            continue

        # Build prediction and ground truth arrays (aligned to retrieved set)
        predictions = retrieved_scores
        ground_truth = hits

        # Convert history_ids to list if needed
        if history_ids is not None:
            if not isinstance(history_ids, list):
                history_ids = list(history_ids)
        else:
            history_ids = []

        # Evaluate with REAL novelty (actual clicked article IDs)
        metrics = metrics_harness.evaluate_user(
            predictions=predictions,
            ground_truth=ground_truth,
            article_ids=list(retrieved_ids),
            article_categories=article_categories if article_categories else None,
            user_history=history_ids,  # REAL history IDs (not fake range)
            n_true_relevant=n_true_relevant,
        )
        metrics["user_id"] = user_id
        metrics["history_len"] = history_len

        all_results.append(metrics)

        # Accumulate for batch-level metrics (coverage, head/tail slicing)
        retrieved_ids_list = list(retrieved_ids)
        all_predictions_dense.append([article_id_to_dense_idx[a] for a in retrieved_ids_list
                                       if a in article_id_to_dense_idx])
        ground_truth_list.append(ground_truth)
        article_ids_list.append(retrieved_ids_list)

        # Slice by cold-start vs warm (cold = <5 clicks, warm = >=5)
        if history_len < 5:
            cold_start_results.append(metrics)
        else:
            warm_results.append(metrics)

    if not all_results:
        logger.error("No results to evaluate!")
        return {}, pd.DataFrame()

    logger.info(f"Evaluated {len(all_results)} users")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Aggregate metrics
    aggregated = metrics_harness.aggregate_metrics(results_df)

    # Bootstrap CI
    ci = metrics_harness.bootstrap_ci(results_df, n_bootstrap=1000, ci=0.95)

    # Coverage metrics (corpus-level)
    logger.info("Computing coverage metrics...")
    coverage_50 = metrics_harness.compute_coverage(all_predictions_dense, k=50)
    coverage_100 = metrics_harness.compute_coverage(all_predictions_dense, k=100)
    cov_50_ci = _bootstrap_coverage_ci(all_predictions_dense, k=50, metrics_harness=metrics_harness, n_bootstrap=200)
    cov_100_ci = _bootstrap_coverage_ci(all_predictions_dense, k=100, metrics_harness=metrics_harness, n_bootstrap=200)

    aggregated["coverage@50"] = coverage_50
    aggregated["coverage@100"] = coverage_100
    ci["coverage@50"] = cov_50_ci
    ci["coverage@100"] = cov_100_ci

    # Head vs tail article slicing
    logger.info("Computing head/tail article slicing...")
    head_agg, tail_agg = metrics_harness.slice_by_article_popularity(
        results_df, article_popularity, ground_truth_list, article_ids_list, threshold=None
    )

    # Cold-start vs warm aggregation
    cold_agg = {}
    warm_agg = {}
    if cold_start_results:
        cold_df = pd.DataFrame(cold_start_results)
        cold_agg = metrics_harness.aggregate_metrics(cold_df)
    if warm_results:
        warm_df = pd.DataFrame(warm_results)
        warm_agg = metrics_harness.aggregate_metrics(warm_df)

    logger.info(f"Cold-start: {len(cold_start_results)} users, Warm: {len(warm_results)} users")
    logger.info(f"Head articles: {len(head_agg)} metrics, Tail articles: {len(tail_agg)} metrics")

    # Prepare output
    output_dict = {
        "config": {
            "dataset": dataset.upper(),
            "method": method.upper(),
            "n_users": len(results_df),
            "cold_threshold": 5,
        },
        "aggregated_metrics": {k: float(v) if pd.notna(v) else None for k, v in aggregated.items()},
        "confidence_intervals": {k: [float(v[0]), float(v[1])] for k, v in ci.items()},
        "slicing": {
            "cold_start": {k: float(v) if pd.notna(v) else None for k, v in cold_agg.items()},
            "warm_users": {k: float(v) if pd.notna(v) else None for k, v in warm_agg.items()},
            "head_articles": {k: float(v) if pd.notna(v) else None for k, v in head_agg.items()},
            "tail_articles": {k: float(v) if pd.notna(v) else None for k, v in tail_agg.items()},
        },
    }

    return output_dict, results_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--method", choices=["bm25", "semantic"], required=True)
    args = parser.parse_args()

    # Evaluate
    output_dict, results_df = evaluate(args.dataset, args.method)

    if output_dict is None or len(output_dict) == 0:
        logger.error("Evaluation failed, no output")
        return

    # Save results
    output_dir = REPO_ROOT / "outputs" / f"{args.dataset}_{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "offline_metrics.json", "w") as f:
        json.dump(output_dict, f, indent=2)

    results_df.to_csv(output_dir / "offline_metrics_per_user.csv", index=False)

    logger.info(f"Results saved to {output_dir}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"{args.dataset.upper()} {args.method.upper()} - Offline Evaluation Results")
    print(f"{'='*70}")
    agg = output_dict["aggregated_metrics"]
    ci = output_dict["confidence_intervals"]

    for metric in sorted(agg.keys()):
        val = agg[metric]
        if val is None:
            print(f"{metric:20s}: N/A")
        elif metric in ci:
            lower, upper = ci[metric]
            print(f"{metric:20s}: {val:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
        else:
            print(f"{metric:20s}: {val:.4f}")

    print(f"\n{'='*70}")
    print("Cold-Start vs Warm Users")
    print(f"{'='*70}")

    cold = output_dict["slicing"]["cold_start"]
    warm = output_dict["slicing"]["warm_users"]
    all_metrics = set(cold.keys()) | set(warm.keys())

    for metric in sorted(all_metrics):
        val_cold = cold.get(metric)
        val_warm = warm.get(metric)
        if val_cold is not None and val_warm is not None:
            diff = ((val_cold - val_warm) / val_warm * 100) if val_warm != 0 else 0
            print(f"{metric:20s}: {val_cold:.4f} (cold) | {val_warm:.4f} (warm) | {diff:+.1f}%")

    print(f"\n{'='*70}")
    print("Head vs Tail Articles")
    print(f"{'='*70}")

    head = output_dict["slicing"]["head_articles"]
    tail = output_dict["slicing"]["tail_articles"]
    all_metrics_ht = set(head.keys()) | set(tail.keys())

    for metric in sorted(all_metrics_ht):
        val_head = head.get(metric)
        val_tail = tail.get(metric)
        if val_head is not None and val_tail is not None:
            diff = ((val_head - val_tail) / val_tail * 100) if val_tail != 0 else 0
            print(f"{metric:20s}: {val_head:.4f} (head) | {val_tail:.4f} (tail) | {diff:+.1f}%")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
