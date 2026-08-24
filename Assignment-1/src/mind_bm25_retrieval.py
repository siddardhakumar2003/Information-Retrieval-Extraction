"""BM25 lexical retrieval baseline for MIND news recommendation.

    python src/mind_bm25_retrieval.py [--k 50 100 200] [--k1 1.5] [--b 0.75] [--out-dir outputs/mind_lexical]

Builds inverted index from article titles+abstracts. User profiles built from train+val
combined click history. Evaluates recall@K against test split ground truth.
Outputs: recall_results.json, per_user_recall.parquet, per_user_predictions.parquet (Q4 input).
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

    def idf(self, term: str) -> float:
        """Compute IDF for a term."""
        df = len(self.postings.get(term, {}))
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score_documents_batch(self, candidate_ids: set[int], qtf_counter: Counter,
                               k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
        """Score a set of candidate documents against a query."""
        scores = {}
        idf_cache = {term: self.idf(term) for term in qtf_counter}

        for doc_idx in candidate_ids:
            score = 0.0
            dl = self.doc_len.get(doc_idx, 0)
            norm_factor = 1 - b + b * (dl / self.avgdl) if self.avgdl > 0 else 1

            for term, qtf in qtf_counter.items():
                tf = self.postings.get(term, {}).get(doc_idx, 0)
                if tf == 0:
                    continue
                term_score = idf_cache[term] * (tf * (k1 + 1)) / (tf + k1 * norm_factor)
                score += qtf * term_score

            if score > 0:
                scores[doc_idx] = score

        return scores

    def search(self, query_tokens: list[str], k: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        """BM25 search. Returns list of (article_idx, score).
        Optimized: first narrows to candidate docs sharing query terms, then scores."""
        if not query_tokens or all(t == "" for t in query_tokens):
            return []

        query_terms = set(t for t in query_tokens if t)
        if not query_terms:
            return []

        # Gather candidate docs: union of posting lists for all query terms
        candidate_ids = set()
        for term in query_terms:
            if term in self.postings:
                candidate_ids.update(self.postings[term].keys())

        if not candidate_ids:
            return []

        # Score candidates
        qtf_counter = Counter(query_tokens)
        scores = self.score_documents_batch(candidate_ids, qtf_counter, k1, b)

        # Sort and return top-k
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


@dataclass
class Config:
    k_values: list[int]
    k1: float = 1.5
    b: float = 0.75
    out_dir: Path = Path("outputs/mind_lexical")


def _load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load articles, train+val behaviors for profiles, and test behaviors for evaluation."""
    feature_store = repo_root / "data" / "mind_processed" / "feature_store"

    articles = pd.read_parquet(feature_store / "articles.parquet")
    train_beh = pd.read_parquet(feature_store / "train_behaviors.parquet")
    val_beh = pd.read_parquet(feature_store / "val_behaviors.parquet")
    test_beh = pd.read_parquet(feature_store / "test_behaviors.parquet")

    logger.info(f"Loaded {len(articles)} articles")
    logger.info(f"Loaded {len(train_beh)} train impressions")
    logger.info(f"Loaded {len(val_beh)} val impressions")
    logger.info(f"Loaded {len(test_beh)} test impressions (for evaluation)")

    return articles, train_beh, val_beh, test_beh


def _retrieve_and_evaluate(
    articles: pd.DataFrame,
    train_beh: pd.DataFrame,
    val_beh: pd.DataFrame,
    test_beh: pd.DataFrame,
    index: InvertedIndex,
    k_values: list[int],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Retrieve top-K using train+val profiles and evaluate against test ground truth."""
    logger.info("Retrieving and evaluating...")

    # Map article_id to index
    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"].values)}
    idx_to_article_id = {v: k for k, v in article_id_to_idx.items()}

    # Build user profiles from train+val click history (vectorized via groupby)
    user_history = defaultdict(set)
    for uid, group in train_beh.groupby("user_id"):
        for articles_list in group["click_history"]:
            if articles_list is not None and len(articles_list) > 0:
                user_history[uid].update(articles_list)
    for uid, group in val_beh.groupby("user_id"):
        for articles_list in group["click_history"]:
            if articles_list is not None and len(articles_list) > 0:
                user_history[uid].update(articles_list)

    # Build ground truth from test impressions (vectorized via groupby)
    user_clicks = defaultdict(set)
    for uid, group in test_beh.groupby("user_id"):
        for clicked in group["article_ids_clicked"]:
            if clicked is not None and len(clicked) > 0:
                user_clicks[uid].update(clicked)

    logger.info(f"Ground truth: {len(user_clicks)} users with clicks in test")
    logger.info(f"User profiles: {len(user_history)} users with history in train+val")

    # Evaluate
    recall_results = {k: {"macro": 0.0, "num_users_with_nonzero_recall": 0} for k in k_values}
    per_user_recalls = []
    per_user_predictions = []

    max_k = max(k_values)
    evaluated_count = 0
    common_users = set(user_history.keys()) & set(user_clicks.keys())

    for user_id in sorted(common_users):
        if (evaluated_count + 1) % 1000 == 0:
            logger.info(f"  Evaluated {evaluated_count + 1}/{len(common_users)} users")

        evaluated_count += 1
        user_all_clicks = user_clicks[user_id]

        # Get user's click history from train+val
        history = list(user_history[user_id])
        if not history:
            query_tokens = [""]
        else:
            # Combine titles of all articles in history (vectorized)
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
        retrieved_ids_raw = [idx_to_article_id[idx] for idx, _ in retrieved]
        retrieved_scores = [score for _, score in retrieved]

        # Calculate recall@k
        for k in k_values:
            top_k_ids = set(retrieved_ids_raw[:k])
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

        # Persist top-K predictions for Q4 (scores + binary hit vector)
        hit_vector = [1 if aid in user_all_clicks else 0 for aid in retrieved_ids_raw]
        per_user_predictions.append({
            "user_id": user_id,
            "retrieved_ids": retrieved_ids_raw,
            "retrieved_scores": retrieved_scores,
            "hits": hit_vector,
            "history_len": len(history),
            "n_true_relevant": len(user_all_clicks),
        })

    # Normalize
    if evaluated_count > 0:
        for k in k_values:
            recall_results[k]["macro"] /= evaluated_count

    logger.info(f"Evaluated {evaluated_count} users")

    per_user_df = pd.DataFrame(per_user_recalls)
    per_user_preds_df = pd.DataFrame(per_user_predictions)
    return recall_results, per_user_df, per_user_preds_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200], help="Retrieval cutoffs")
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
    articles, train_beh, val_beh, test_beh = _load_data(REPO_ROOT)
    logger.info("Building BM25 index...")
    index = InvertedIndex.build(articles)

    # Retrieve and evaluate
    recall_results, per_user_df, per_user_preds_df = _retrieve_and_evaluate(
        articles, train_beh, val_beh, test_beh, index, config.k_values
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
    per_user_preds_df.to_parquet(config.out_dir / "per_user_predictions.parquet")

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
