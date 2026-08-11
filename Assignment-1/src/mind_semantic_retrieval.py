"""Semantic embedding-based retrieval with FAISS IVF for MIND dataset.

    python src/mind_semantic_retrieval.py [--k 50 100 150] [--nlist 100] [--nprobe 10]

Uses BGE-base embeddings + FAISS IVF index:
1. Embed all articles (title + abstract)
2. Build IVF index (clusters vectors, searches relevant clusters)
3. Create user profiles: average embedding of training click history
4. Retrieve top-K articles per user from validation set
5. Evaluate: Recall@K against validation impressions
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    k_values: list[int]
    nlist: int  # number of IVF clusters
    nprobe: int  # number of clusters to search
    model_name: str = "BAAI/bge-base-en-v1.5"
    out_dir: Path = Path("outputs/mind_semantic")


def _load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load articles, train behaviors (for profiles), and val behaviors (for eval)."""
    feature_store = repo_root / "data" / "mind_processed" / "feature_store"

    articles = pd.read_parquet(feature_store / "articles.parquet")
    train_beh = pd.read_parquet(feature_store / "train_behaviors.parquet")
    val_beh = pd.read_parquet(feature_store / "val_behaviors.parquet")

    logger.info(f"Loaded {len(articles)} articles")
    logger.info(f"Loaded {len(train_beh)} train impressions for profiles")
    logger.info(f"Loaded {len(val_beh)} val impressions for evaluation")

    return articles, train_beh, val_beh


def _embed_articles(articles: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Embed article titles + abstracts using BGE-base."""
    logger.info("Embedding articles...")

    texts = []
    for _, row in articles.iterrows():
        title = str(row.get("title", "")) or ""
        abstract = str(row.get("abstract", "")) or ""
        text = f"{title} {abstract}".strip()
        texts.append(text if text else "[empty]")

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Embedded {len(embeddings)} articles, shape: {embeddings.shape}")

    return embeddings.astype(np.float32)


def _build_faiss_index(embeddings: np.ndarray, nlist: int, nprobe: int) -> faiss.IndexIVFFlat:
    """Build FAISS IVF index from article embeddings."""
    logger.info(f"Building FAISS IVF index (nlist={nlist}, nprobe={nprobe})...")

    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    # Train on sample
    index.train(embeddings[:min(100000, len(embeddings))])
    index.add(embeddings)
    index.nprobe = nprobe

    logger.info(f"Index built: {index.ntotal} vectors, {nlist} clusters")
    return index


def _create_user_profiles(
    articles: pd.DataFrame,
    train_beh: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create user profiles as average embedding of training click history."""
    logger.info("Creating user profiles from training click history...")

    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"].values)}
    user_profiles = {}

    for _, row in train_beh.iterrows():
        user_id = str(row.get("user_id"))
        history = row.get("click_history", None)

        if history is None or len(history) == 0:
            user_profiles[user_id] = np.zeros(embeddings.shape[1], dtype=np.float32)
            continue

        # Get embeddings of clicked articles
        valid_indices = []
        for aid in history:
            if aid in article_id_to_idx:
                valid_indices.append(article_id_to_idx[aid])

        if valid_indices:
            clicked_embeddings = embeddings[valid_indices]
            user_profiles[user_id] = np.mean(clicked_embeddings, axis=0).astype(np.float32)
        else:
            user_profiles[user_id] = np.zeros(embeddings.shape[1], dtype=np.float32)

    logger.info(f"Created profiles for {len(user_profiles)} users")
    return user_profiles


def _retrieve_and_evaluate(
    articles: pd.DataFrame,
    val_beh: pd.DataFrame,
    user_profiles: dict[str, np.ndarray],
    index: faiss.IndexIVFFlat,
    k_values: list[int],
) -> tuple[dict, pd.DataFrame]:
    """Retrieve top-K articles per user and evaluate against validation ground truth."""
    logger.info("Retrieving and evaluating...")

    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"].values)}
    idx_to_article_id = {v: k for k, v in article_id_to_idx.items()}

    # Build ground truth from validation impressions
    user_clicks = defaultdict(set)
    for _, row in val_beh.iterrows():
        user_id = str(row["user_id"])
        clicked_ids = row.get("article_ids_clicked", None)
        if clicked_ids is not None and len(clicked_ids) > 0:
            user_clicks[user_id].update(clicked_ids)

    logger.info(f"Ground truth: {len(user_clicks)} users with clicks in validation")

    # Evaluate
    recall_results = {k: {"macro": 0.0, "num_users_with_nonzero_recall": 0} for k in k_values}
    per_user_recalls = []

    max_k = max(k_values)
    evaluated_count = 0
    skipped_no_profile = 0
    skipped_no_clicks = 0

    for user_id in user_clicks.keys():
        if user_id not in user_profiles:
            skipped_no_profile += 1
            continue

        evaluated_count += 1
        profile_embedding = user_profiles[user_id]
        ground_truth = user_clicks[user_id]

        # Query FAISS
        if np.all(profile_embedding == 0):
            retrieved_indices = np.random.choice(len(article_id_to_idx), max_k, replace=False)
        else:
            query = profile_embedding.reshape(1, -1)
            _, retrieved_indices = index.search(query, max_k)
            retrieved_indices = retrieved_indices[0]

        retrieved_ids = set(idx_to_article_id.get(idx) for idx in retrieved_indices if idx >= 0)

        # Calculate recall@k
        for k in k_values:
            top_k_ids = set(list(retrieved_ids)[:k])
            recall = len(top_k_ids & ground_truth) / len(ground_truth) if ground_truth else 0.0

            per_user_recalls.append({
                "user_id": user_id,
                "k": k,
                "recall": recall,
                "num_retrieved": len(top_k_ids),
                "num_ground_truth": len(ground_truth),
                "num_hits": len(top_k_ids & ground_truth),
            })

            recall_results[k]["macro"] += recall
            if recall > 0:
                recall_results[k]["num_users_with_nonzero_recall"] += 1

    # Normalize
    if evaluated_count > 0:
        for k in k_values:
            recall_results[k]["macro"] /= evaluated_count

    logger.info(f"Skipped {skipped_no_profile} users (no profile)")
    logger.info(f"Evaluated {evaluated_count} users")

    per_user_df = pd.DataFrame(per_user_recalls)
    return recall_results, per_user_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 150], help="Retrieval cutoffs")
    parser.add_argument("--nlist", type=int, default=100, help="IVF clusters")
    parser.add_argument("--nprobe", type=int, default=10, help="Clusters to search")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="Embedding model")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/mind_semantic"), help="Output directory")
    args = parser.parse_args()

    config = Config(
        k_values=sorted(args.k),
        nlist=args.nlist,
        nprobe=args.nprobe,
        model_name=args.model,
        out_dir=Path(args.out_dir),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: k={config.k_values}, nlist={config.nlist}, nprobe={config.nprobe}")

    # Load data
    articles, train_beh, val_beh = _load_data(REPO_ROOT)

    # Load embedding model
    logger.info(f"Loading embedding model: {config.model_name}")
    model = SentenceTransformer(config.model_name)

    # Embed articles and build FAISS index
    article_embeddings = _embed_articles(articles, model)
    faiss_index = _build_faiss_index(article_embeddings, config.nlist, config.nprobe)

    # Create user profiles
    user_profiles = _create_user_profiles(articles, train_beh, article_embeddings)

    # Retrieve and evaluate
    recall_results, per_user_df = _retrieve_and_evaluate(
        articles, val_beh, user_profiles, faiss_index, config.k_values
    )

    # Save results
    logger.info("Saving results...")

    results_dict = {
        "config": {
            "model": config.model_name,
            "k_values": config.k_values,
            "nlist": config.nlist,
            "nprobe": config.nprobe,
        },
        "num_users_evaluated": len(user_profiles),
    }
    results_dict.update({f"recall@{k}": v for k, v in recall_results.items()})

    with open(config.out_dir / "recall_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    per_user_df.to_parquet(config.out_dir / "per_user_recall.parquet")

    with open(config.out_dir / "faiss_index.pkl", "wb") as f:
        pickle.dump(faiss_index, f)

    np.save(config.out_dir / "article_embeddings.npy", article_embeddings)
    with open(config.out_dir / "user_profiles.pkl", "wb") as f:
        pickle.dump(user_profiles, f)

    logger.info(f"Results saved to {config.out_dir}")

    # Print summary
    print("\n" + "="*60)
    print("MIND SEMANTIC RETRIEVAL RESULTS (BGE-base + FAISS IVF)")
    print("="*60)
    for k in config.k_values:
        macro = recall_results[k]["macro"]
        nonzero = recall_results[k]["num_users_with_nonzero_recall"]
        print(f"Recall@{k:3d}: {macro:.4f} ({nonzero:4d} users with ≥1 hit)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
