"""BM25 lexical retrieval baseline for EB-NeRD news recommendation.

    python src/bm25_retrieval.py [--k 50 100 150] [--k1 1.5] [--b 0.75] [--out-dir data/processed/bm25]

Builds an inverted index from article titles+abstracts, represents each user
as a long profile query from their pre-train click history, retrieves top-K
articles via BM25, and measures recall against ground-truth clicks.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
_TOKEN_RE = re.compile(r"[a-zæøå0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize Danish text (case-insensitive, alphanumeric + Danish chars)."""
    return _TOKEN_RE.findall(text.lower())


class InvertedIndex:
    """Reusable inverted-index + BM25 scoring."""

    def __init__(
        self,
        postings: dict[str, dict[int, int]],
        doc_len: dict[int, int],
        N: int,
        avgdl: float,
    ):
        self.postings = postings  # term -> {article_id: term_freq}
        self.doc_len = doc_len    # article_id -> token_count
        self.N = N                # total documents
        self.avgdl = avgdl        # average doc length

    @classmethod
    def build(cls, articles_df: pd.DataFrame) -> "InvertedIndex":
        postings = defaultdict(lambda: defaultdict(int))
        doc_len = {}

        text = articles_df["title"].fillna("") + " " + articles_df["abstract"].fillna("")
        for article_id, txt in zip(articles_df["article_id"], text):
            tokens = tokenize(txt)
            doc_len[article_id] = len(tokens)
            for token in tokens:
                postings[token][article_id] += 1

        N = len(articles_df)
        avgdl = sum(doc_len.values()) / N if N > 0 else 0

        return cls(dict(postings), doc_len, N, avgdl)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "inverted_index.pkl", "wb") as f:
            pickle.dump((self.postings, self.doc_len, self.N, self.avgdl), f)

    @classmethod
    def load(cls, out_dir: Path) -> "InvertedIndex":
        with open(out_dir / "inverted_index.pkl", "rb") as f:
            postings, doc_len, N, avgdl = pickle.load(f)
        return cls(postings, doc_len, N, avgdl)

    def idf(self, term: str) -> float:
        """Non-negative BM25+ IDF (Lucene variant)."""
        df = len(self.postings.get(term, {}))
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score_documents_batch(
        self,
        candidate_ids: set[int],
        qtf_counter: Counter,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[int, float]:
        """Score multiple documents via BM25 with query-term-frequency weighting."""
        scores = {}

        for article_id in candidate_ids:
            score = 0.0
            doc_len = self.doc_len.get(article_id, 0)
            norm_factor = 1 - b + b * (doc_len / self.avgdl) if self.avgdl > 0 else 1

            for term, qtf in qtf_counter.items():
                tf = self.postings.get(term, {}).get(article_id, 0)
                if tf == 0:
                    continue
                idf = self.idf(term)
                term_score = idf * (tf * (k1 + 1)) / (tf + k1 * norm_factor)
                score += qtf * term_score

            if score > 0:
                scores[article_id] = score

        return scores


def bm25_score(
    qtf_counter: Counter,
    index: InvertedIndex,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[int, float]:
    """Score candidate articles via BM25, returning only scored documents."""
    candidate_docs = set()

    # Gather all candidate articles that match any query term.
    for term in qtf_counter.keys():
        if term in index.postings:
            candidate_docs.update(index.postings[term].keys())

    # Batch score all candidates.
    return index.score_documents_batch(candidate_docs, qtf_counter, k1, b)




def recall_at_k(retrieved_ids: list[int], truth_set: set[int], k: int) -> float:
    """Compute recall@k: |retrieved[:k] ∩ truth| / |truth|."""
    if not truth_set:
        return 0.0
    k_retrieved = set(retrieved_ids[:k])
    return len(k_retrieved & truth_set) / len(truth_set)


def run_evaluation(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    print("[1/6] Loading feature store")
    articles = pd.read_parquet(data_dir / "processed" / "feature_store" / "articles.parquet")
    users = pd.read_parquet(data_dir / "processed" / "feature_store" / "users_train_region.parquet")
    behaviors = pd.read_parquet(data_dir / "processed" / "splits" / "train" / "behaviors.parquet")

    print("[2/6] Building inverted index")
    index = InvertedIndex.build(articles)
    index.save(out_dir)
    print(f"  {len(index.postings)} unique terms, {len(articles)} documents")

    print("[3/6] Building user queries and ground truth")
    # Article lookup: article_id -> (title + " " + abstract)
    article_lookup = {}
    for _, row in articles.iterrows():
        article_lookup[row["article_id"]] = (
            (row["title"] or "") + " " + (row["abstract"] or "")
        )

    # Ground truth: per-user union of all clicked articles in train window
    ground_truth = defaultdict(set)
    for _, row in behaviors.iterrows():
        ground_truth[row["user_id"]].update(row["article_ids_clicked"])

    print(f"  {len(ground_truth)} users with train-window clicks")

    # Users from the feature store that have no train-window clicks.
    excluded_users = set(users["user_id"]) - set(ground_truth.keys())
    print(f"  {len(excluded_users)} users excluded (no train-window clicks)")

    print(f"[4/6] Scoring and ranking all users (k1={args.k1}, b={args.b})")
    per_user_results = []
    recall_by_k = defaultdict(list)

    # Convert to lists for fast access (avoid iterrows overhead)
    user_ids = users["user_id"].tolist()
    click_histories = users["click_history"].tolist()

    for idx in range(len(users)):
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(users)}] users scored", flush=True)

        user_id = int(user_ids[idx])
        click_history = click_histories[idx]

        # Skip users with no ground truth.
        if user_id not in ground_truth:
            continue

        # Build user query from their pre-train click history.
        # Manually build query without using a Series
        query_text = ""
        for article_id in click_history:
            article_id = int(article_id)
            if article_id in article_lookup:
                query_text += " " + article_lookup[article_id]

        tokens = tokenize(query_text)
        qtf = Counter(tokens)

        if not qtf:
            continue

        # Score and rank.
        scores = bm25_score(qtf, index, k1=args.k1, b=args.b)
        if not scores:
            continue

        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0]),  # descending score, tie-break by article_id
        )
        ranked_ids = [int(aid) for aid, _ in ranked]

        # Compute recall@K for each K, tracking per-user results.
        truth = ground_truth[user_id]
        history_len = int(len(click_history))
        truth_size = len(truth)
        user_recalls = {"user_id": user_id, "history_len": history_len, "truth_size": truth_size}
        recall_monotonic = True
        prev_recall = 0.0

        for k in args.k:
            recall = recall_at_k(ranked_ids, truth, k)
            user_recalls[f"recall@{k}"] = float(recall)

            # Check monotonicity.
            if recall < prev_recall - 1e-9:  # small epsilon for floating point
                recall_monotonic = False
            prev_recall = recall

            recall_by_k[k].append(recall)

        if not recall_monotonic:
            logging.warning(f"non-monotonic recall for user {user_id}")

        per_user_results.append(user_recalls)

    print(f"[5/6] Writing outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute macro and micro recall.
    results_summary = {
        "config": {"k1": args.k1, "b": args.b, "k_values": args.k},
        "num_users_evaluated": len(per_user_results),
        "num_users_excluded": len(excluded_users),
    }

    for k in args.k:
        recalls_k = recall_by_k[k]
        if recalls_k:
            macro = float(np.mean(recalls_k))
            num_nonzero = sum(1 for r in recalls_k if r > 0)
        else:
            macro = 0.0
            num_nonzero = 0

        results_summary[f"recall@{k}"] = {
            "macro": macro,
            "num_users_with_nonzero_recall": num_nonzero,
        }

    with open(out_dir / "recall_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Write per-user results.
    results_df = pd.DataFrame(per_user_results)
    results_df.to_parquet(out_dir / "per_user_recall.parquet", index=False)

    print(f"[6/6] Summary")
    print(f"\nRecall@K (macro-averaged over {len(per_user_results)} users):")
    for k in args.k:
        macro = np.mean(recall_by_k[k]) if recall_by_k[k] else 0.0
        print(f"  recall@{k}: {macro:.4f}")

    print(f"\nOutputs:")
    print(f"  {out_dir / 'inverted_index.pkl'}")
    print(f"  {out_dir / 'recall_results.json'}")
    print(f"  {out_dir / 'per_user_recall.parquet'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "lexical"))
    p.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[50, 100, 150],
        help="K values for recall@k evaluation",
    )
    p.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter")
    p.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
