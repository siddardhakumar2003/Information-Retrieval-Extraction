"""
Example usage of OfflineMetrics for evaluating retrieval systems.
Demonstrates all metrics: AUC, MRR, nDCG, Recall, Diversity, Novelty, Coverage.
"""

import numpy as np
import pandas as pd
from metrics import OfflineMetrics


def example_synthetic_data():
    """Create synthetic data for demonstration."""
    np.random.seed(42)

    # Simulate 100 users with 500 articles each
    n_users = 100
    n_articles = 500

    predictions_list = []
    ground_truth_list = []
    article_ids_list = []
    user_histories = []
    article_categories = {}

    for user_id in range(n_users):
        # Generate predictions (scores between 0-1)
        predictions = np.random.rand(n_articles)

        # Generate ground truth (sparse: ~2-3 clicks per user)
        ground_truth = np.zeros(n_articles)
        n_clicks = np.random.randint(2, 5)
        clicked_indices = np.random.choice(n_articles, n_clicks, replace=False)
        ground_truth[clicked_indices] = 1

        # Article categories
        for idx in range(n_articles):
            article_categories[idx] = np.random.choice(
                ['News', 'Sports', 'Tech', 'Business', 'Entertainment']
            )

        # User history (previous clicks)
        user_history = np.random.choice(n_articles, n_clicks, replace=False).tolist()

        predictions_list.append(predictions)
        ground_truth_list.append(ground_truth)
        article_ids_list.append(list(range(n_articles)))
        user_histories.append(user_history)

    return (predictions_list, ground_truth_list, article_ids_list,
            user_histories, article_categories)


def example_real_data():
    """Load real data from existing evaluation outputs."""
    import json
    from pathlib import Path

    # Load MIND BM25 results
    output_file = Path("outputs/mind_lexical/recall_results.json")
    if not output_file.exists():
        print(f"File not found: {output_file}")
        return None

    with open(output_file) as f:
        results = json.load(f)

    print("\nMIND BM25 Results (Summary):")
    print(f"  Users evaluated: {results['num_users_evaluated']}")
    print(f"  Config: k1={results['config']['k1']}, b={results['config']['b']}")
    for k in results['config']['k_values']:
        recall_key = f"recall@{k}"
        if recall_key in results:
            print(f"  Recall@{k}: {results[recall_key]['macro']:.4f}")

    return results


def example_basic_metrics():
    """Demonstrate basic metrics computation."""
    print("\n" + "="*60)
    print("Example 1: Basic Metrics Computation")
    print("="*60)

    evaluator = OfflineMetrics()

    # Single user example
    predictions = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    ground_truth = np.array([1, 0, 1, 0, 0])  # User clicked articles 0 and 2
    article_ids = [0, 1, 2, 3, 4]

    print("\nPrediction scores:", predictions)
    print("Ground truth labels:", ground_truth)
    print("Article IDs:", article_ids)

    # Rank articles by prediction score
    ranking = np.argsort(-predictions)
    ranked_labels = ground_truth[ranking]
    ranked_ids = [article_ids[i] for i in ranking]

    print("\nRanked order:", ranked_ids)
    print("Ranked labels:", ranked_labels)

    # Compute metrics
    auc = evaluator.compute_auc(ground_truth)
    mrr = evaluator.compute_mrr(ranked_labels)
    ndcg5 = evaluator.compute_ndcg(ranked_labels, k=5)
    ndcg10 = evaluator.compute_ndcg(ranked_labels, k=10)
    recall50 = evaluator.compute_recall(ranked_labels, k=50)

    print(f"\nMetrics:")
    print(f"  AUC:        {auc:.4f}")
    print(f"  MRR:        {mrr:.4f}")
    print(f"  nDCG@5:     {ndcg5:.4f}")
    print(f"  nDCG@10:    {ndcg10:.4f}")
    print(f"  Recall@50:  {recall50:.4f}")


def example_batch_evaluation():
    """Demonstrate batch evaluation on synthetic data."""
    print("\n" + "="*60)
    print("Example 2: Batch Evaluation with Synthetic Data")
    print("="*60)

    predictions_list, ground_truth_list, article_ids_list, user_histories, article_categories = (
        example_synthetic_data()
    )

    evaluator = OfflineMetrics()

    # Evaluate batch
    print(f"\nEvaluating {len(predictions_list)} users...")
    metrics_df = evaluator.evaluate_batch(
        predictions_list, ground_truth_list, article_ids_list,
        article_categories=article_categories,
        user_histories=user_histories
    )

    # Aggregate metrics
    aggregated = evaluator.aggregate_metrics(metrics_df)

    # Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")
    cis = evaluator.bootstrap_ci(metrics_df, n_bootstrap=1000)

    evaluator.print_results(aggregated, cis, name="Synthetic Data Evaluation")

    return metrics_df, aggregated, cis


def example_slicing(metrics_df, user_histories, ground_truth_list, article_ids_list):
    """Demonstrate sliced evaluation."""
    print("\n" + "="*60)
    print("Example 3: Sliced Evaluation (Cold-start vs Warm Users)")
    print("="*60)

    evaluator = OfflineMetrics()

    # Slice by user activity
    cold_metrics, warm_metrics = evaluator.slice_by_user_activity(
        metrics_df, user_histories, threshold=3
    )

    evaluator.print_sliced_results(
        cold_metrics, warm_metrics,
        "Cold-start (< 3 clicks)", "Warm (>= 3 clicks)"
    )


def example_diversity_novelty():
    """Demonstrate diversity and novelty metrics."""
    print("\n" + "="*60)
    print("Example 4: Diversity and Novelty Metrics")
    print("="*60)

    evaluator = OfflineMetrics()

    # Example predictions
    predictions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # article IDs in ranked order

    # Article categories
    article_categories = {
        0: "Sports", 1: "Sports", 2: "Sports",
        3: "News", 4: "News", 5: "Tech",
        6: "Tech", 7: "Business", 8: "Business", 9: "Entertainment"
    }

    # User history
    user_history = [0, 1, 3]  # User previously clicked these articles

    # Compute metrics
    diversity = evaluator.compute_diversity(predictions, article_categories, k=10)
    novelty = evaluator.compute_novelty(predictions, user_history, k=10)

    print(f"\nTop-10 predictions: {predictions}")
    print(f"User history: {user_history}")
    print(f"\nDiversity@10 (unique categories / 10): {diversity:.4f}")
    print(f"Novelty@10 (unseen articles / 10): {novelty:.4f}")

    # Show breakdown
    topk = set(predictions[:10])
    history = set(user_history)
    novel = topk - history
    print(f"\nBreakdown:")
    print(f"  Unique categories in top-10: {len(set(article_categories.get(a) for a in predictions[:10]))}")
    print(f"  Novel articles in top-10: {len(novel)} / 10")


def example_coverage():
    """Demonstrate coverage metric."""
    print("\n" + "="*60)
    print("Example 5: Coverage Metric")
    print("="*60)

    evaluator = OfflineMetrics()

    # 5 users with recommendations
    all_predictions = [
        [0, 1, 2, 3, 4],
        [1, 2, 5, 6, 7],
        [3, 4, 5, 8, 9],
        [0, 2, 8, 10, 11],
        [6, 7, 9, 10, 12]
    ]

    coverage = evaluator.compute_coverage(all_predictions, k=5)

    print(f"\n5 users with top-5 predictions:")
    for i, preds in enumerate(all_predictions):
        print(f"  User {i}: {preds}")

    recommended = set()
    for preds in all_predictions:
        recommended.update(preds[:5])

    print(f"\nUnique articles recommended: {sorted(recommended)}")
    print(f"Coverage@5: {coverage:.4f} ({len(recommended)} unique articles)")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("OFFLINE METRICS EVALUATION HARNESS - EXAMPLES")
    print("="*80)

    # Example 1: Basic metrics
    example_basic_metrics()

    # Example 2: Batch evaluation
    metrics_df, aggregated, cis = example_batch_evaluation()

    # Example 3: Slicing
    predictions_list, ground_truth_list, article_ids_list, user_histories, _ = (
        example_synthetic_data()
    )
    example_slicing(metrics_df, user_histories, ground_truth_list, article_ids_list)

    # Example 4: Diversity and novelty
    example_diversity_novelty()

    # Example 5: Coverage
    example_coverage()

    # Example 6: Real data (if available)
    print("\n" + "="*80)
    print("Example 6: Real Data from Existing Evaluation")
    print("="*80)
    real_results = example_real_data()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
