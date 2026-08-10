"""Semantic embedding-based retrieval with FAISS IVF for EB-NeRD.

    python src/semantic_retrieval.py [--k 50 100 150] [--nlist 100] [--nprobe 10] [--out-dir outputs/semantic]

Uses BGE-base embeddings + FAISS IVF index:
1. Embed all articles (title + abstract)
2. Build IVF index (clusters vectors, searches relevant clusters)
3. Create user profiles: average embedding of click history
4. Retrieve top-K articles per user
5. Evaluate: Recall@K against train impressions
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
    out_dir: Path = Path("outputs/semantic")


def _load_processed_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load articles, train behaviors, and user click history."""
    feature_store = repo_root / "data" / "processed" / "feature_store"

    articles = pd.read_parquet(feature_store / "articles.parquet")
    users_train = pd.read_parquet(feature_store / "users_train_region.parquet")

    splits = repo_root / "data" / "processed" / "splits"
    train_behaviors = pd.read_parquet(splits / "train" / "behaviors.parquet")

    logger.info(f"Loaded {len(articles)} articles")
    logger.info(f"Loaded {len(users_train)} users with click history")
    logger.info(f"Loaded {len(train_behaviors)} train impressions")

    return articles, users_train, train_behaviors


def _embed_articles(articles: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Embed article titles + abstracts using BGE-base."""
    logger.info("Embedding articles...")

    # Combine title and abstract for embedding
    texts = []
    for _, row in articles.iterrows():
        title = str(row.get("title", "")) or ""
        abstract = str(row.get("subtitle", "")) or ""  # subtitle is abstract
        text = f"{title} {abstract}".strip()
        texts.append(text if text else "[empty]")

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Embedded {len(embeddings)} articles, shape: {embeddings.shape}")

    return embeddings.astype(np.float32)


def _build_faiss_index(embeddings: np.ndarray, nlist: int, nprobe: int) -> faiss.IndexIVFFlat:
    """Build FAISS IVF index from article embeddings."""
    logger.info(f"Building FAISS IVF index (nlist={nlist}, nprobe={nprobe})...")

    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)  # L2 distance for inner product
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    # Training on a sample is sufficient for IVF
    index.train(embeddings[:min(100000, len(embeddings))])
    index.add(embeddings)
    index.nprobe = nprobe

    logger.info(f"Index built: {index.ntotal} vectors, {nlist} clusters")
    return index


def _create_user_profiles(
    articles: pd.DataFrame,
    users_train: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[int, np.ndarray]:
    """Create user profiles as average embedding of click history."""
    logger.info("Creating user profiles from click history...")

    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"].values)}
    user_profiles = {}

    for _, row in users_train.iterrows():
        user_id = int(row.get("user_id"))  # Get user_id from column, not index
        clicked_ids = row.get("click_history", None)  # Column name is "click_history", not "article_id_fixed"

        if clicked_ids is None or len(clicked_ids) == 0:
            # Default profile for users with no history
            user_profiles[user_id] = np.zeros(embeddings.shape[1], dtype=np.float32)
            continue

        # Get embeddings of clicked articles
        valid_indices = []
        for aid in clicked_ids:
            if aid in article_id_to_idx:
                valid_indices.append(article_id_to_idx[aid])

        if valid_indices:
            clicked_embeddings = embeddings[valid_indices]
            # Average embedding
            user_profiles[user_id] = np.mean(clicked_embeddings, axis=0).astype(np.float32)
        else:
            user_profiles[user_id] = np.zeros(embeddings.shape[1], dtype=np.float32)

    logger.info(f"Created profiles for {len(user_profiles)} users")
    return user_profiles


def _retrieve_and_evaluate(
    articles: pd.DataFrame,
    train_behaviors: pd.DataFrame,
    user_profiles: dict[int, np.ndarray],
    index: faiss.IndexIVFFlat,
    k_values: list[int],
) -> tuple[dict, pd.DataFrame]:
    """Retrieve top-K articles per user and evaluate against ground truth."""
    logger.info("Retrieving and evaluating...")

    article_id_to_idx = {aid: idx for idx, aid in enumerate(articles["article_id"].values)}
    idx_to_article_id = {v: k for k, v in article_id_to_idx.items()}

    # Build ground truth: user -> set of clicked article IDs
    user_clicks = defaultdict(set)
    for _, row in train_behaviors.iterrows():
        user_id = int(row["user_id"])  # Ensure consistent type
        clicked_ids = row.get("article_ids_clicked", None)
        if clicked_ids is not None and len(clicked_ids) > 0:
            user_clicks[user_id].update(clicked_ids)

    logger.info(f"Ground truth: {len(user_clicks)} users with clicks")

    # Evaluate
    recall_results = {k: {"macro": 0.0, "num_users_with_nonzero_recall": 0} for k in k_values}
    per_user_recalls = []

    max_k = max(k_values)
    evaluated_count = 0
    skipped_not_in_clicks = 0
    skipped_no_ground_truth = 0

    for user_id, profile_embedding in user_profiles.items():
        # Skip users with no click history (no ground truth)
        if user_id not in user_clicks:
            skipped_not_in_clicks += 1
            continue
        if len(user_clicks[user_id]) == 0:
            skipped_no_ground_truth += 1
            continue

        evaluated_count += 1

        # Query FAISS index
        if np.all(profile_embedding == 0):
            # User has no history, random retrieval
            retrieved_indices = np.random.choice(len(article_id_to_idx), max_k, replace=False)
        else:
            # Reshape for FAISS (expects 2D array)
            query = profile_embedding.reshape(1, -1)
            _, retrieved_indices = index.search(query, max_k)
            retrieved_indices = retrieved_indices[0]  # Get first (only) result

        retrieved_ids = set(idx_to_article_id.get(idx) for idx in retrieved_indices if idx >= 0)
        ground_truth = user_clicks[user_id]

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

    logger.info(f"Skipped {skipped_not_in_clicks} users (not in user_clicks)")
    logger.info(f"Skipped {skipped_no_ground_truth} users (empty ground truth)")
    logger.info(f"Evaluated {evaluated_count} users")

    # Normalize macro recall
    if evaluated_count > 0:
        for k in k_values:
            recall_results[k]["macro"] /= evaluated_count
    else:
        logger.error("No users evaluated! Check user_profiles and user_clicks overlap")

    per_user_df = pd.DataFrame(per_user_recalls)
    return recall_results, per_user_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 150],
                        help="Retrieval cutoffs")
    parser.add_argument("--nlist", type=int, default=100, help="IVF clusters")
    parser.add_argument("--nprobe", type=int, default=10, help="Clusters to search")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5",
                        help="Embedding model")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/semantic"),
                        help="Output directory")
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
    articles, users_train, train_behaviors = _load_processed_data(REPO_ROOT)

    # Load embedding model
    logger.info(f"Loading embedding model: {config.model_name}")
    model = SentenceTransformer(config.model_name)

    # Embed articles and build FAISS index
    article_embeddings = _embed_articles(articles, model)
    faiss_index = _build_faiss_index(article_embeddings, config.nlist, config.nprobe)

    # Create user profiles
    user_profiles = _create_user_profiles(articles, users_train, article_embeddings)

    # Retrieve and evaluate
    recall_results, per_user_df = _retrieve_and_evaluate(
        articles, train_behaviors, user_profiles, faiss_index, config.k_values
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

    # Save article embeddings and user profiles
    np.save(config.out_dir / "article_embeddings.npy", article_embeddings)
    with open(config.out_dir / "user_profiles.pkl", "wb") as f:
        pickle.dump(user_profiles, f)

    logger.info(f"Results saved to {config.out_dir}")

    # Print summary
    print("\n" + "="*60)
    print("SEMANTIC RETRIEVAL RESULTS (BGE-base + FAISS IVF)")
    print("="*60)
    for k in config.k_values:
        macro = recall_results[k]["macro"]
        nonzero = recall_results[k]["num_users_with_nonzero_recall"]
        print(f"Recall@{k:3d}: {macro:.4f} ({nonzero:4d} users with ≥1 hit)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
