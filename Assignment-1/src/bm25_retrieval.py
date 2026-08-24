"""BM25 lexical retrieval baseline for EB-NeRD news recommendation.

    python src/bm25_retrieval.py [--k 50 100 200] [--k1 1.5] [--b 0.75] [--out-dir outputs/lexical]

Builds an inverted index from article titles+abstracts. User profiles are built from
train+val combined click history. Evaluates recall@K against test split ground truth.
Outputs: recall_results.json, per_user_recall.parquet, per_user_predictions.parquet (Q4 input).
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

        # Use 'subtitle' for EB-NeRD (has title + subtitle), fallback to 'abstract' or 'body'
        subtitle_col = "subtitle" if "subtitle" in articles_df.columns else "abstract"
        text = articles_df["title"].fillna("") + " " + articles_df[subtitle_col].fillna("")
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

    print("[1/6] Loading feature store and splits (train+val for profiles, test for eval)")
    articles = pd.read_parquet(data_dir / "processed" / "feature_store" / "articles.parquet")

    # Load train+val behaviors for building user profiles
    train_beh = pd.read_parquet(data_dir / "processed" / "splits" / "train" / "behaviors.parquet")
    val_beh = pd.read_parquet(data_dir / "processed" / "splits" / "val" / "behaviors.parquet")

    # Load test behaviors for evaluation
    test_beh = pd.read_parquet(data_dir / "processed" / "splits" / "test" / "behaviors.parquet")

    print("[2/6] Building inverted index")
    index = InvertedIndex.build(articles)
    index.save(out_dir)
    print(f"  {len(index.postings)} unique terms, {len(articles)} documents")

    print("[3/6] Building user profiles from train+val click history and test ground truth")
    # Article lookup: article_id -> (title + " " + subtitle/abstract) [vectorized]
    article_lookup = {}
    subtitle_col = "subtitle" if "subtitle" in articles.columns else "abstract"
    for aid, title, subtitle in zip(articles["article_id"], articles["title"].fillna(""), articles[subtitle_col].fillna("")):
        article_lookup[aid] = f"{title} {subtitle}"

    # Build user profiles from train+val combined click history (vectorized via groupby)
    user_history = defaultdict(set)
    for uid, group in train_beh.groupby("user_id"):
        for articles in group["article_ids_clicked"]:
            if articles is not None and len(articles) > 0:
                user_history[uid].update(articles)
    for uid, group in val_beh.groupby("user_id"):
        for articles in group["article_ids_clicked"]:
            if articles is not None and len(articles) > 0:
                user_history[uid].update(articles)

    # Ground truth: per-user clicks in test split (vectorized via groupby)
    ground_truth = defaultdict(set)
    for uid, group in test_beh.groupby("user_id"):
        for articles in group["article_ids_clicked"]:
            if articles is not None and len(articles) > 0:
                ground_truth[uid].update(articles)

    print(f"  {len(user_history)} users with train+val click history")
    print(f"  {len(ground_truth)} users with test-split clicks (evaluation targets)")

    # Evaluate only users present in both
    common_users = set(user_history.keys()) & set(ground_truth.keys())
    print(f"  {len(common_users)} users in both (will evaluate)")
    excluded_users = set(ground_truth.keys()) - common_users
    print(f"  {len(excluded_users)} users excluded (no train+val history)")

    print(f"[4/6] Scoring and ranking all users (k1={args.k1}, b={args.b})")
    per_user_results = []
    per_user_predictions = []
    recall_by_k = defaultdict(list)
    max_k = max(args.k)

    evaluated_count = 0
    for user_id in sorted(common_users):
        if (evaluated_count + 1) % 100 == 0:
            print(f"  [{evaluated_count+1}/{len(common_users)}] users scored", flush=True)

        evaluated_count += 1

        # Build user query from train+val combined click history
        history = list(user_history[user_id])
        query_text = ""
        for article_id in history:
            article_id = int(article_id)
            if article_id in article_lookup:
                query_text += " " + article_lookup[article_id]

        tokens = tokenize(query_text)
        qtf = Counter(tokens)

        if not qtf:
            continue

        # Score and rank all articles.
        scores = bm25_score(qtf, index, k1=args.k1, b=args.b)
        if not scores:
            continue

        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0]),  # descending score, tie-break by article_id
        )
        ranked_ids = [aid for aid, _ in ranked]
        ranked_scores = [score for _, score in ranked]

        # Top-max_k for Q4 predictions
        top_k_ids = ranked_ids[:max_k]
        top_k_scores = ranked_scores[:max_k]

        # Ground truth for this user
        truth = ground_truth[user_id]

        # Compute recall@K for each K
        user_recalls = {"user_id": user_id, "history_len": len(history), "truth_size": len(truth)}
        recall_monotonic = True
        prev_recall = 0.0

        for k in args.k:
            recall = recall_at_k(ranked_ids, truth, k)
            user_recalls[f"recall@{k}"] = float(recall)

            if recall < prev_recall - 1e-9:
                recall_monotonic = False
            prev_recall = recall

            recall_by_k[k].append(recall)

        if not recall_monotonic:
            logging.warning(f"non-monotonic recall for user {user_id}")

        per_user_results.append(user_recalls)

        # Persist top-K predictions for Q4 (scores + binary hit vector)
        hit_vector = [1 if aid in truth else 0 for aid in top_k_ids]
        per_user_predictions.append({
            "user_id": user_id,
            "retrieved_ids": top_k_ids,
            "retrieved_scores": top_k_scores,
            "hits": hit_vector,
            "history_len": len(history),
            "n_true_relevant": len(truth),
        })

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

    # Write per-user predictions (for Q4 input)
    preds_df = pd.DataFrame(per_user_predictions)
    preds_df.to_parquet(out_dir / "per_user_predictions.parquet", index=False)

    print(f"[6/6] Summary")
    print(f"\nRecall@K (macro-averaged over {len(per_user_results)} users):")
    for k in args.k:
        macro = np.mean(recall_by_k[k]) if recall_by_k[k] else 0.0
        print(f"  recall@{k}: {macro:.4f}")

    print(f"\nOutputs:")
    print(f"  {out_dir / 'inverted_index.pkl'}")
    print(f"  {out_dir / 'recall_results.json'}")
    print(f"  {out_dir / 'per_user_recall.parquet'}")
    print(f"  {out_dir / 'per_user_predictions.parquet'} (Q4 input)")


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
        default=[50, 100, 200],
        help="K values for recall@k evaluation",
    )
    p.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter")
    p.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
