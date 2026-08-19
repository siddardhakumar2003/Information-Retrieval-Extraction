"""
Q4: Offline Evaluation Harness
Computes AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage
with slicing and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings


class OfflineMetrics:
    """Offline evaluation harness for retrieval systems."""

    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: List of K values for ranking metrics (default: [5, 10, 50, 100])
        """
        self.k_values = k_values or [5, 10, 50, 100]
        self.results = {}

    def compute_auc(self, labels: np.ndarray) -> float:
        """
        Compute AUC-ROC for ranking.
        Args:
            labels: Binary labels (1 = clicked, 0 = not clicked)
        Returns:
            AUC score (0-1)
        """
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return np.nan

        # Count pairs where positive ranked higher than negative
        # Simpler: use ranking-based AUC
        ranks = np.argsort(-labels)  # Descending order
        auc = 0.0
        for i, rank in enumerate(ranks):
            if labels[rank] == 1:
                auc += (len(labels) - i - n_neg)

        auc = auc / (n_pos * n_neg) if n_pos * n_neg > 0 else np.nan
        return auc

    def compute_mrr(self, labels: np.ndarray) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first clicked article.
        Args:
            labels: Binary labels in ranked order
        Returns:
            MRR score (0-1)
        """
        for rank, label in enumerate(labels, 1):
            if label == 1:
                return 1.0 / rank
        return 0.0

    def compute_ndcg(self, labels: np.ndarray, k: int = 10) -> float:
        """
        nDCG@K: Normalized Discounted Cumulative Gain.
        Args:
            labels: Binary labels in ranked order
            k: Cutoff rank
        Returns:
            nDCG@K score (0-1)
        """
        labels = labels[:k]
        if len(labels) == 0 or np.sum(labels) == 0:
            return 0.0

        # DCG = sum(label_i / log2(i+1))
        dcg = np.sum(labels / np.log2(np.arange(2, len(labels) + 2)))

        # IDCG: ideal DCG with all positives at top
        ideal_labels = np.sort(labels)[::-1]
        idcg = np.sum(ideal_labels / np.log2(np.arange(2, len(ideal_labels) + 2)))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg

    def compute_recall(self, labels: np.ndarray, k: int = 50) -> float:
        """
        Recall@K: fraction of relevant items in top-K.
        Args:
            labels: Binary labels in ranked order
            k: Cutoff rank
        Returns:
            Recall@K (0-1)
        """
        n_relevant = np.sum(labels)
        if n_relevant == 0:
            return 0.0

        n_retrieved = np.sum(labels[:k])
        return n_retrieved / n_relevant

    def compute_diversity(self,
                         predictions: List[int],
                         article_categories: Dict[int, str],
                         k: int = 50) -> float:
        """
        Intra-list diversity: fraction of unique categories in top-K.
        Args:
            predictions: Ranked list of article IDs
            article_categories: Mapping of article ID to category
            k: Cutoff rank
        Returns:
            Diversity score (0-1)
        """
        topk_articles = predictions[:k]
        categories = set()

        for article_id in topk_articles:
            if article_id in article_categories:
                categories.add(article_categories[article_id])

        # Diversity = |unique_categories| / |top_k|
        diversity = len(categories) / len(topk_articles) if len(topk_articles) > 0 else 0.0
        return diversity

    def compute_novelty(self,
                       predictions: List[int],
                       user_history: List[int],
                       k: int = 50) -> float:
        """
        Novelty: fraction of articles not in user click history.
        Args:
            predictions: Ranked list of article IDs
            user_history: List of articles clicked by user
            k: Cutoff rank
        Returns:
            Novelty score (0-1)
        """
        topk_articles = set(predictions[:k])
        history_set = set(user_history)

        novel_articles = topk_articles - history_set
        novelty = len(novel_articles) / len(topk_articles) if len(topk_articles) > 0 else 0.0
        return novelty

    def compute_coverage(self,
                        all_predictions: List[List[int]],
                        k: int = 50) -> float:
        """
        Coverage: fraction of unique articles recommended across all users.
        Args:
            all_predictions: List of prediction lists for each user
            k: Cutoff rank
        Returns:
            Coverage score (0-1)
        """
        recommended_articles = set()
        for predictions in all_predictions:
            recommended_articles.update(predictions[:k])

        # Total unique articles in corpus
        corpus_size = max([max(pred) for pred in all_predictions if len(pred) > 0]) + 1

        coverage = len(recommended_articles) / corpus_size if corpus_size > 0 else 0.0
        return coverage

    def evaluate_user(self,
                     predictions: np.ndarray,
                     ground_truth: np.ndarray,
                     article_ids: List[int],
                     article_categories: Optional[Dict[int, str]] = None,
                     user_history: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate a single user's predictions.
        Args:
            predictions: Scores or probabilities for each article
            ground_truth: Binary labels for each article (1=clicked, 0=not)
            article_ids: Article IDs in same order as predictions
            article_categories: Optional category mapping
            user_history: Optional user's previous click history
        Returns:
            Dictionary of metrics
        """
        # Rank articles by prediction scores (descending)
        ranking_idx = np.argsort(-predictions)
        ranked_labels = ground_truth[ranking_idx]
        ranked_ids = [article_ids[i] for i in ranking_idx]

        metrics = {
            'auc': self.compute_auc(ground_truth),
            'mrr': self.compute_mrr(ranked_labels),
            'ndcg@5': self.compute_ndcg(ranked_labels, k=5),
            'ndcg@10': self.compute_ndcg(ranked_labels, k=10),
            'recall@50': self.compute_recall(ranked_labels, k=50),
            'recall@100': self.compute_recall(ranked_labels, k=100),
        }

        # Diversity
        if article_categories is not None:
            metrics['diversity@50'] = self.compute_diversity(
                ranked_ids, article_categories, k=50
            )
            metrics['diversity@100'] = self.compute_diversity(
                ranked_ids, article_categories, k=100
            )

        # Novelty
        if user_history is not None:
            metrics['novelty@50'] = self.compute_novelty(
                ranked_ids, user_history, k=50
            )
            metrics['novelty@100'] = self.compute_novelty(
                ranked_ids, user_history, k=100
            )

        return metrics

    def evaluate_batch(self,
                      predictions_list: List[np.ndarray],
                      ground_truth_list: List[np.ndarray],
                      article_ids_list: List[List[int]],
                      article_categories: Optional[Dict[int, str]] = None,
                      user_histories: Optional[List[List[int]]] = None) -> pd.DataFrame:
        """
        Evaluate multiple users' predictions.
        Args:
            predictions_list: List of prediction arrays
            ground_truth_list: List of ground truth arrays
            article_ids_list: List of article ID lists
            article_categories: Optional category mapping
            user_histories: Optional list of user histories
        Returns:
            DataFrame with metrics for each user
        """
        results = []

        for i, (preds, labels, article_ids) in enumerate(
            zip(predictions_list, ground_truth_list, article_ids_list)
        ):
            user_history = user_histories[i] if user_histories else None
            metrics = self.evaluate_user(
                preds, labels, article_ids,
                article_categories=article_categories,
                user_history=user_history
            )
            metrics['user_id'] = i
            results.append(metrics)

        return pd.DataFrame(results)

    def aggregate_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate metrics across users.
        Args:
            metrics_df: DataFrame with per-user metrics
        Returns:
            Dictionary of aggregated metrics (mean)
        """
        aggregated = {}
        for col in metrics_df.columns:
            if col != 'user_id':
                valid_vals = metrics_df[col].dropna()
                if len(valid_vals) > 0:
                    aggregated[col] = valid_vals.mean()

        return aggregated

    def bootstrap_ci(self,
                     metrics_df: pd.DataFrame,
                     n_bootstrap: int = 1000,
                     ci: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        Args:
            metrics_df: DataFrame with per-user metrics
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level (0-1)
        Returns:
            Dictionary mapping metric names to (lower, upper) CI bounds
        """
        confidence_intervals = {}
        alpha = 1 - ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        for col in metrics_df.columns:
            if col != 'user_id':
                valid_vals = metrics_df[col].dropna().values
                if len(valid_vals) == 0:
                    continue

                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(valid_vals, size=len(valid_vals), replace=True)
                    bootstrap_means.append(np.mean(sample))

                lower = np.percentile(bootstrap_means, lower_percentile)
                upper = np.percentile(bootstrap_means, upper_percentile)
                confidence_intervals[col] = (lower, upper)

        return confidence_intervals

    def slice_by_user_activity(self,
                               metrics_df: pd.DataFrame,
                               user_histories: List[List[int]],
                               threshold: int = 5) -> Tuple[Dict, Dict]:
        """
        Slice metrics by user activity (cold-start vs warm).
        Args:
            metrics_df: DataFrame with per-user metrics
            user_histories: List of user histories
            threshold: Number of clicks to separate cold/warm
        Returns:
            (cold_start_metrics, warm_user_metrics) dictionaries
        """
        cold_start_mask = np.array([len(h) < threshold for h in user_histories])
        warm_mask = ~cold_start_mask

        cold_start_df = metrics_df[cold_start_mask]
        warm_df = metrics_df[warm_mask]

        cold_start_metrics = self.aggregate_metrics(cold_start_df)
        warm_metrics = self.aggregate_metrics(warm_df)

        return cold_start_metrics, warm_metrics

    def slice_by_article_popularity(self,
                                    metrics_df: pd.DataFrame,
                                    article_clicks: Dict[int, int],
                                    ground_truth_list: List[np.ndarray],
                                    article_ids_list: List[List[int]],
                                    threshold: Optional[float] = None) -> Tuple[Dict, Dict]:
        """
        Slice metrics by article popularity (head vs tail).
        Args:
            metrics_df: DataFrame with per-user metrics
            article_clicks: Dictionary mapping article ID to click count
            ground_truth_list: List of ground truth arrays
            article_ids_list: List of article ID lists
            threshold: Percentile for head/tail split (default: 75th percentile)
        Returns:
            (head_metrics, tail_metrics) dictionaries
        """
        if threshold is None:
            threshold = np.percentile(list(article_clicks.values()), 75)

        # Create mask for whether ground truth contains head articles
        head_mask = []
        for i, (labels, article_ids) in enumerate(zip(ground_truth_list, article_ids_list)):
            has_head = False
            for j, article_id in enumerate(article_ids):
                if labels[j] == 1 and article_clicks.get(article_id, 0) >= threshold:
                    has_head = True
                    break
            head_mask.append(has_head)

        head_mask = np.array(head_mask)
        tail_mask = ~head_mask

        head_df = metrics_df[head_mask]
        tail_df = metrics_df[tail_mask]

        head_metrics = self.aggregate_metrics(head_df)
        tail_metrics = self.aggregate_metrics(tail_df)

        return head_metrics, tail_metrics

    def print_results(self,
                     aggregated_metrics: Dict[str, float],
                     confidence_intervals: Dict[str, Tuple[float, float]],
                     name: str = "Model"):
        """Print formatted evaluation results."""
        print(f"\n{'='*60}")
        print(f"{name} Evaluation Results")
        print(f"{'='*60}")

        for metric, value in sorted(aggregated_metrics.items()):
            if pd.isna(value):
                print(f"{metric:20s}: N/A")
            else:
                if metric in confidence_intervals:
                    lower, upper = confidence_intervals[metric]
                    print(f"{metric:20s}: {value:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
                else:
                    print(f"{metric:20s}: {value:.4f}")

        print(f"{'='*60}\n")

    def print_sliced_results(self,
                            slice1_metrics: Dict[str, float],
                            slice2_metrics: Dict[str, float],
                            slice1_name: str,
                            slice2_name: str):
        """Print sliced evaluation results."""
        print(f"\n{'='*60}")
        print(f"Sliced Evaluation Results")
        print(f"{'='*60}")

        all_metrics = set(slice1_metrics.keys()) | set(slice2_metrics.keys())

        print(f"\n{slice1_name:20s} | {slice2_name:20s} | Difference")
        print("-" * 70)

        for metric in sorted(all_metrics):
            val1 = slice1_metrics.get(metric, np.nan)
            val2 = slice2_metrics.get(metric, np.nan)

            if pd.isna(val1) or pd.isna(val2):
                print(f"{metric:20s}: N/A")
            else:
                diff = val1 - val2
                print(f"{metric:20s}: {val1:.4f} | {val2:.4f} | {diff:+.4f}")

        print(f"{'='*60}\n")


def load_predictions_and_labels(predictions_file: str,
                               labels_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load predictions and labels from files.
    Supports .npy, .csv, .parquet formats.
    """
    # Load predictions
    if predictions_file.endswith('.npy'):
        pred_array = np.load(predictions_file)
    elif predictions_file.endswith('.csv'):
        pred_df = pd.read_csv(predictions_file, index_col=0)
        pred_array = pred_df.values
    elif predictions_file.endswith('.parquet'):
        pred_df = pd.read_parquet(predictions_file)
        pred_array = pred_df.values
    else:
        raise ValueError(f"Unsupported file format: {predictions_file}")

    # Load labels
    if labels_file.endswith('.npy'):
        labels_array = np.load(labels_file)
    elif labels_file.endswith('.csv'):
        labels_df = pd.read_csv(labels_file, index_col=0)
        labels_array = labels_df.values
    elif labels_file.endswith('.parquet'):
        labels_df = pd.read_parquet(labels_file)
        labels_array = labels_df.values
    else:
        raise ValueError(f"Unsupported file format: {labels_file}")

    # Convert to list of arrays (one per user)
    predictions_list = [pred_array[i] for i in range(len(pred_array))]
    labels_list = [labels_array[i] for i in range(len(labels_array))]

    return predictions_list, labels_list


if __name__ == "__main__":
    # Example usage
    print("Offline Evaluation Harness loaded successfully")
    print("Import and use: from src.metrics import OfflineMetrics")
