"""
Q4: Example evaluation script using OfflineMetrics harness.
Evaluates both BM25 and semantic retrieval results.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from metrics import OfflineMetrics
import argparse


def load_eval_data(dataset_name: str = "mind"):
    """Load evaluation data (ground truth + article metadata)."""
    if dataset_name.lower() == "mind":
        output_dir = Path("outputs/mind_lexical")

        # Load results
        with open(output_dir / "recall_results.json") as f:
            bm25_results = json.load(f)

        # Load article metadata if available
        article_file = Path("data/mind_processed/articles.parquet")
        if article_file.exists():
            articles_df = pd.read_parquet(article_file)
            article_categories = dict(zip(articles_df['article_id'], articles_df['category']))
        else:
            article_categories = None

        return bm25_results, article_categories

    elif dataset_name.lower() == "ebnerd":
        output_dir = Path("outputs/lexical")

        with open(output_dir / "recall_results.json") as f:
            bm25_results = json.load(f)

        article_file = Path("data/processed/articles.parquet")
        if article_file.exists():
            articles_df = pd.read_parquet(article_file)
            article_categories = dict(zip(articles_df['article_id'], articles_df['category']))
        else:
            article_categories = None

        return bm25_results, article_categories

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_retrieval_results(eval_data, dataset_name: str = "mind"):
    """
    Evaluate retrieval results using OfflineMetrics.
    Works with JSON format: {user_id: {"ground_truth": [...], "predictions": [...]}}
    """
    evaluator = OfflineMetrics()

    # Parse evaluation data
    user_ids = []
    predictions_list = []
    ground_truth_list = []

    if isinstance(eval_data, dict) and 'recall_by_k' in eval_data:
        # Summary format - compute metrics from recall
        print(f"Evaluation data format: Summary statistics")
        print(json.dumps(eval_data, indent=2))
        return

    # Try to load per-user format
    try:
        for user_id_str, user_data in eval_data.items():
            try:
                user_ids.append(int(user_id_str))

                # Normalize prediction scores to [0, 1]
                preds = np.array(user_data.get('predictions', []))
                if len(preds) > 0:
                    preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
                predictions_list.append(preds)

                # Ground truth labels
                labels = np.array(user_data.get('ground_truth', []))
                ground_truth_list.append(labels)
            except (KeyError, ValueError, TypeError):
                continue
    except (AttributeError, TypeError):
        print("Could not parse evaluation data format")
        return

    if len(predictions_list) == 0:
        print("No valid user data found for evaluation")
        return

    # Create article IDs list (indices)
    article_ids_list = [list(range(len(p))) for p in predictions_list]

    # Evaluate batch
    metrics_df = evaluator.evaluate_batch(
        predictions_list, ground_truth_list, article_ids_list
    )

    # Aggregate metrics
    aggregated = evaluator.aggregate_metrics(metrics_df)

    # Bootstrap confidence intervals
    cis = evaluator.bootstrap_ci(metrics_df, n_bootstrap=1000)

    return aggregated, cis, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval models")
    parser.add_argument("--dataset", default="mind", choices=["mind", "ebnerd"],
                       help="Dataset to evaluate on")
    parser.add_argument("--method", default="bm25", choices=["bm25", "semantic"],
                       help="Retrieval method to evaluate")
    parser.add_argument("--output", default=None,
                       help="Output file for results (JSON)")
    args = parser.parse_args()

    print(f"Loading {args.dataset.upper()} dataset evaluation data...")
    eval_data, article_categories = load_eval_data(args.dataset)

    print(f"Evaluating {args.method.upper()} retrieval on {args.dataset.upper()}...")
    results = evaluate_retrieval_results(eval_data, args.dataset)

    if results is None:
        print("Evaluation failed - no valid data found")
        return

    aggregated, cis, metrics_df = results

    # Create evaluator for printing
    evaluator = OfflineMetrics()
    evaluator.print_results(aggregated, cis, name=f"{args.method.upper()} ({args.dataset.upper()})")

    # Save results if requested
    if args.output:
        output_data = {
            "method": args.method,
            "dataset": args.dataset,
            "aggregated_metrics": aggregated,
            "confidence_intervals": {k: list(v) for k, v in cis.items()},
            "n_users": len(metrics_df)
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {output_path}")

    # Save detailed metrics
    metrics_output = f"outputs/{args.dataset}_{args.method}_metrics.csv"
    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_output, index=False)
    print(f"Detailed per-user metrics saved to {metrics_output}")


if __name__ == "__main__":
    main()
