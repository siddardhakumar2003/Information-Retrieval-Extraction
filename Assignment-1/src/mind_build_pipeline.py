"""One-command reproducible data pipeline for MIND dataset.

    python src/mind_build_pipeline.py

Raw TSV → clean → unified schema (articles / behaviors / click history)
→ time-based train/val/test split → feature store.

MIND structure:
  - news.tsv: article_id | category | subcategory | title | abstract | url | entities_json | entities_json
  - behaviors.tsv: impression_id | user_id | timestamp | history | impressions (with click labels)

Split semantics:
  - MINDsmall_train (92M) → split into train/val by time
  - MINDsmall_dev (42M) → use as final test set
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DIRS = {
    "train": REPO_ROOT / "data" / "MINDsmall_train",
    "test": REPO_ROOT / "data" / "MINDsmall_dev",
}


def _load_news(path: Path) -> pd.DataFrame:
    """Load news.tsv: id | category | subcategory | title | abstract | url | entities | entities."""
    df = pd.read_csv(
        path / "news.tsv",
        sep="\t",
        header=None,
        names=["article_id", "category", "subcategory", "title", "abstract", "url", "entities_title", "entities_body"],
        dtype={"article_id": str, "category": str, "subcategory": str},
    )
    return df


def _load_behaviors(path: Path) -> pd.DataFrame:
    """Load behaviors.tsv: impression_id | user_id | timestamp | history | impressions."""
    df = pd.read_csv(
        path / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "timestamp", "history", "impressions"],
        dtype={"impression_id": int, "user_id": str, "history": str, "impressions": str},
    )
    return df


def _parse_history(history_str: str) -> list[str]:
    """Parse space-separated article IDs."""
    if pd.isna(history_str) or not history_str:
        return []
    return history_str.split()


def _parse_impressions(impressions_str: str) -> tuple[list[str], list[str]]:
    """Parse space-separated article IDs with click labels (N123-1 N456-0)."""
    if pd.isna(impressions_str) or not impressions_str:
        return [], []

    shown = []
    clicked = []
    for item in impressions_str.split():
        aid, label = item.rsplit("-", 1)
        shown.append(aid)
        if label == "1":
            clicked.append(aid)
    return shown, clicked


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("[1/4] Loading MIND train and test sets")
    train_news = _load_news(RAW_DIRS["train"])
    train_behaviors = _load_behaviors(RAW_DIRS["train"])
    test_news = _load_news(RAW_DIRS["test"])
    test_behaviors = _load_behaviors(RAW_DIRS["test"])

    print(f"  Train: {len(train_news)} articles, {len(train_behaviors)} impressions")
    print(f"  Test: {len(test_news)} articles, {len(test_behaviors)} impressions")

    # Deduplicate articles across train and test
    print("[2/4] Deduplicating articles")
    all_news = pd.concat([train_news, test_news], ignore_index=True)
    all_news = all_news.drop_duplicates(subset=["article_id"], keep="first")
    print(f"  Total unique articles: {len(all_news)}")

    # Parse behaviors
    print("[3/4] Parsing behaviors and click history")

    # Train split: parse behaviors into separate impression/click events
    train_rows = []
    for _, row in train_behaviors.iterrows():
        shown, clicked = _parse_impressions(row["impressions"])
        history = _parse_history(row["history"])

        train_rows.append({
            "user_id": row["user_id"],
            "timestamp": pd.to_datetime(row["timestamp"]),
            "article_ids_shown": shown,
            "article_ids_clicked": clicked,
            "click_history": history,
        })

    train_beh_parsed = pd.DataFrame(train_rows)
    print(f"  Train impressions: {len(train_beh_parsed)}")
    print(f"  Train unique users: {train_beh_parsed['user_id'].nunique()}")

    # Test split
    test_rows = []
    for _, row in test_behaviors.iterrows():
        shown, clicked = _parse_impressions(row["impressions"])
        history = _parse_history(row["history"])

        test_rows.append({
            "user_id": row["user_id"],
            "timestamp": pd.to_datetime(row["timestamp"]),
            "article_ids_shown": shown,
            "article_ids_clicked": clicked,
            "click_history": history,
        })

    test_beh_parsed = pd.DataFrame(test_rows)
    print(f"  Test impressions: {len(test_beh_parsed)}")
    print(f"  Test unique users: {test_beh_parsed['user_id'].nunique()}")

    # Time-based split of train set
    print("[4/4] Creating time-based train/val split")
    train_beh_parsed = train_beh_parsed.sort_values("timestamp").reset_index(drop=True)

    val_days = args.val_days
    val_start = train_beh_parsed["timestamp"].max() - pd.Timedelta(days=val_days)

    split_idx = (train_beh_parsed["timestamp"] < val_start).sum()
    train_split = train_beh_parsed.iloc[:split_idx]
    val_split = train_beh_parsed.iloc[split_idx:]

    print(f"  Train: {len(train_split)} impressions, {train_split['user_id'].nunique()} users")
    print(f"  Val: {len(val_split)} impressions, {val_split['user_id'].nunique()} users")
    print(f"  Test: {len(test_beh_parsed)} impressions, {test_beh_parsed['user_id'].nunique()} users")

    # Save processed data
    feature_dir = out_dir / "feature_store"
    feature_dir.mkdir(exist_ok=True)

    print(f"\nSaving to {feature_dir}/")
    all_news.to_parquet(feature_dir / "articles.parquet")
    train_split.to_parquet(feature_dir / "train_behaviors.parquet")
    val_split.to_parquet(feature_dir / "val_behaviors.parquet")
    test_beh_parsed.to_parquet(feature_dir / "test_behaviors.parquet")

    # Save user click histories for train split
    train_users = train_split.groupby("user_id").agg({
        "click_history": lambda x: list(set([aid for hist in x for aid in hist])),  # Union of all histories
    }).reset_index()
    train_users.columns = ["user_id", "click_history"]
    train_users.to_parquet(feature_dir / "users_train.parquet")

    print(f"✓ Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/mind_processed", help="Output directory")
    parser.add_argument("--val-days", type=int, default=1, help="Days to use for validation split")
    args = parser.parse_args()
    run_pipeline(args)
