"""BM25 lexical retrieval baseline for MIND news recommendation.

    python src/mind_bm25_retrieval.py [--k 50 100 150] [--k1 1.5] [--b 0.75] [--out-dir outputs/mind_lexical]

Builds inverted index from article titles+abstracts, retrieves top-K articles via BM25,
and measures recall against ground-truth clicks from training impressions.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Tokenize text (case-insensitive)."""
    return _TOKEN_RE.findall(text.lower())


class InvertedIndex:
    """Inverted-index + BM25 scoring."""

    def __init__(
        self,
        postings: dict[str, dict[int, int]],
        doc_len: dict[int, int],
        N: int,
        avgdl: float,
    ):
        self.postings = postings  # term -> {article_idx: term_freq}
        self.doc_len = doc_len    # article_idx -> token_count
        self.N = N                # total documents
        self.avgdl = avgdl        # average doc length

    @classmethod
    def build(cls, articles_df: pd.DataFrame) -> "InvertedIndex":
        postings = defaultdict(lambda: defaultdict(int))
        doc_len = {}
        article_indices = []

        for idx, (_, row) in enumerate(articles_df.iterrows()):
            if idx % 10000 == 0:
                logger.info(f"  Indexing article {idx}/{len(articles_df)}")

            # Combine title and abstract
            text = f"{row.get('title', '')} {row.get('abstract', '')}"
            tokens = tokenize(text)
            doc_len[idx] = len(tokens)

            # Index tokens (use set for efficiency, then count)
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                postings[token][idx] = count

            article_indices.append(row["article_id"])

        N = len(article_indices)
        avgdl = np.mean([doc_len[i] for i in range(N)])

        logger.info(f"  Built index: {N} articles, {len(postings)} unique terms")
        return cls(postings, doc_len, N, avgdl)

    def search(self, query_tokens: list[str], k: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        """BM25 search. Returns list of (article_idx, score)."""
        scores = defaultdict(float)

        for term in set(query_tokens):
            if term not in self.postings:
                continue

            idf = math.log((self.N - len(self.postings[term]) + 0.5) / (len(self.postings[term]) + 0.5) + 1)

            for doc_idx, freq in self.postings[term].items():
                dl = self.doc_len[doc_idx]
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (dl / self.avgdl))
                scores[doc_idx] += idf * (numerator / denominator)

        # Sort and return top-k
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


@dataclass
class Config:
    k_values: list[int]
    k1: float = 1.5
    b: float = 0.75
    out_dir: Path = Path("outputs/mind_lexical")


def _load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load articles, train behaviors for profiles, and val behaviors for evaluation."""
    feature_store = repo_root / "data" / "mind_processed" / "feature_store"

    articles = pd.read_parquet(feature_store / "articles.parquet")
    train_beh = pd.read_parquet(feature_store / "train_behaviors.parquet")  # For click history/profiles
    val_beh = pd.read_parquet(feature_store / "val_behaviors.parquet")      # For evaluation

    logger.info(f"Loaded {len(articles)} articles")
    logger.info(f"Loaded {len(train_beh)} train impressions, {train_beh['user_id'].nunique()} users (for profiles)")
    logger.info(f"Loaded {len(val_beh)} val impressions, {val_beh['user_id'].nunique()} users (for evaluation)")

    return articles, train_beh, val_beh


def _retrieve_and_evaluate(
    articles: pd.DataFrame,
    train_beh: pd.DataFrame,
    val_beh: pd.DataFrame,
    index: InvertedIndex,
    k_values: list[int],
) -> tuple[dict, pd.DataFrame]:
    """Retrieve top-K using train profiles and evaluate against val ground truth."""
    logger.info("Retrieving and evaluating...")

    # Map article_id to index
    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"].values)}
    idx_to_article_id = {v: k for k, v in article_id_to_idx.items()}

    # Build user profiles from train click history
    user_profiles = defaultdict(list)
    for _, row in train_beh.iterrows():
        user_id = row["user_id"]
        history = row.get("click_history", None)
        if history is not None and len(history) > 0:
            user_profiles[user_id].extend(history)

    # Build ground truth from val impressions
    user_clicks = defaultdict(set)
    for _, row in val_beh.iterrows():
        user_id = row["user_id"]
        clicked = row.get("article_ids_clicked", None)
        if clicked is not None and len(clicked) > 0:
            user_clicks[user_id].update(clicked)

    logger.info(f"Ground truth: {len(user_clicks)} users with clicks in validation")
    logger.info(f"User profiles: {len(user_profiles)} users with history in training")

    # Evaluate
    recall_results = {k: {"macro": 0.0, "num_users_with_nonzero_recall": 0} for k in k_values}
    per_user_recalls = []

    max_k = max(k_values)
    evaluated_count = 0

    for user_id in user_clicks.keys():
        # Skip users without profiles
        if user_id not in user_profiles:
            continue

        evaluated_count += 1
        user_all_clicks = user_clicks[user_id]

        # Get user's click history from training
        history = user_profiles[user_id]
        if not history:
            query_tokens = [""]
        else:
            # Combine titles of all articles in history
            history_texts = []
            for aid in history:
                if aid in article_id_to_idx:
                    art_idx = article_id_to_idx[aid]
                    art_title = articles.iloc[art_idx].get("title", "")
                    history_texts.append(str(art_title))
            query_text = " ".join(history_texts)
            query_tokens = tokenize(query_text) if query_text else [""]

        # Retrieve top-K
        retrieved = index.search(query_tokens, max_k)
        retrieved_ids = {idx_to_article_id[idx] for idx, _ in retrieved}

        # Calculate recall@k
        for k in k_values:
            top_k_ids = {idx_to_article_id[idx] for idx, _ in retrieved[:k]}
            recall = len(top_k_ids & user_all_clicks) / len(user_all_clicks)

            per_user_recalls.append({
                "user_id": user_id,
                "k": k,
                "recall": recall,
                "num_retrieved": len(top_k_ids),
                "num_ground_truth": len(user_all_clicks),
                "num_hits": len(top_k_ids & user_all_clicks),
            })

            recall_results[k]["macro"] += recall
            if recall > 0:
                recall_results[k]["num_users_with_nonzero_recall"] += 1

    # Normalize
    for k in k_values:
        recall_results[k]["macro"] /= evaluated_count

    logger.info(f"Evaluated {evaluated_count} users")

    per_user_df = pd.DataFrame(per_user_recalls)
    return recall_results, per_user_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 150], help="Retrieval cutoffs")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/mind_lexical"), help="Output directory")
    args = parser.parse_args()

    config = Config(
        k_values=sorted(args.k),
        k1=args.k1,
        b=args.b,
        out_dir=Path(args.out_dir),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: k={config.k_values}, k1={config.k1}, b={config.b}")

    # Load and build index
    articles, train_beh, val_beh = _load_data(REPO_ROOT)
    logger.info("Building BM25 index...")
    index = InvertedIndex.build(articles)

    # Retrieve and evaluate
    recall_results, per_user_df = _retrieve_and_evaluate(
        articles, train_beh, val_beh, index, config.k_values
    )

    # Save results
    logger.info("Saving results...")

    results_dict = {
        "config": {
            "k1": config.k1,
            "b": config.b,
            "k_values": config.k_values,
        },
        "num_users_evaluated": len(per_user_df["user_id"].unique()),
    }
    results_dict.update({f"recall@{k}": v for k, v in recall_results.items()})

    with open(config.out_dir / "recall_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    per_user_df.to_parquet(config.out_dir / "per_user_recall.parquet")

    logger.info(f"Results saved to {config.out_dir}")

    # Print summary
    print("\n" + "="*60)
    print("MIND BM25 RETRIEVAL RESULTS")
    print("="*60)
    for k in config.k_values:
        macro = recall_results[k]["macro"]
        nonzero = recall_results[k]["num_users_with_nonzero_recall"]
        print(f"Recall@{k:3d}: {macro:.4f} ({nonzero:4d} users with ≥1 hit)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
