"""One-command reproducible data pipeline for EB-NeRD.

    python build_pipeline.py

Raw zip -> clean -> unified schema (articles / behaviors / click history)
-> time-based train/val/test split -> small on-disk feature store.

Split semantics (never random for interaction data):
  The raw archive ships two folders: `train/` (7 days) and `validation/` (the
  following 7 days). We keep the raw `validation/` folder untouched as the
  final held-out **test** set, and carve a new train/val pair out of the raw
  `train/` folder by time:
    - last  `--val-days`   days of the raw train window  -> new **val**
    - preceding `--train-days` days (default: everything before that) -> new **train**
  This keeps test fully unseen and gives a val split that is temporally
  contiguous with (and strictly after) train, exactly like the real
  train->test gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_TABLES = {
    "articles": "articles.parquet",
    "train_history": "train/history.parquet",
    "train_behaviors": "train/behaviors.parquet",
    "test_history": "validation/history.parquet",
    "test_behaviors": "validation/behaviors.parquet",
}


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _has_raw_tables(out_dir: Path) -> bool:
    return all((out_dir / rel).exists() for rel in RAW_TABLES.values())


def extract_zip(zip_path: Path, out_dir: Path, force: bool = False) -> Path:
    if not force and _has_raw_tables(out_dir):
        return out_dir  # already extracted (e.g. archive was unpacked outside this script)
    if not zip_path.exists():
        if _has_raw_tables(out_dir):
            return out_dir
        raise FileNotFoundError(
            f"neither {zip_path} nor an already-extracted dataset at {out_dir} was found"
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            zf.extract(name, out_dir)
    return out_dir


def load_raw(extracted_dir: Path) -> dict[str, pd.DataFrame]:
    return {key: pd.read_parquet(extracted_dir / rel) for key, rel in RAW_TABLES.items()}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["article_id"]).drop_duplicates(subset=["article_id"])
    for col in ("title", "subtitle", "body", "category_str"):
        df[col] = df[col].fillna("")
    df["premium"] = df["premium"].fillna(False)
    df["article_id"] = df["article_id"].astype("int64")
    return df.reset_index(drop=True)


def clean_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["impression_id", "user_id", "impression_time"])
    df = df.drop_duplicates(subset=["impression_id"])
    df = df[df["article_ids_inview"].apply(len) > 0].reset_index(drop=True)
    # anti-gaming invariant: every clicked article must be a shown candidate
    bad = ~df.apply(
        lambda r: set(r["article_ids_clicked"]).issubset(set(r["article_ids_inview"])), axis=1
    )
    if bad.any():
        df = df[~bad].reset_index(drop=True)
    return df


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"])
    fixed_cols = [
        "impression_time_fixed",
        "scroll_percentage_fixed",
        "article_id_fixed",
        "read_time_fixed",
    ]
    aligned = df[fixed_cols].apply(lambda r: len({len(v) for v in r}) == 1, axis=1)
    return df[aligned].reset_index(drop=True)


# --------------------------------------------------------------------------
# Unified schema
# --------------------------------------------------------------------------
# articles:  article_id, title, abstract, body, category, category_str, entities, published_time
# behaviors: impression_id, session_id, user_id, impression_time, article_ids_inview,
#            article_ids_clicked, device_type, is_subscriber
# click_events (long format, one row per read/click):
#            user_id, article_id, event_time, read_time, scroll_percentage, source

def unify_articles(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "article_id": df["article_id"],
        "title": df["title"],
        "abstract": df["subtitle"],  # EB-NeRD's "subtitle" == MIND's "abstract"
        "body": df["body"],
        "category": df["category"],
        "category_str": df["category_str"],
        "entities": [list(zip(nc, eg)) for nc, eg in zip(df["ner_clusters"], df["entity_groups"])],
        "published_time": df["published_time"],
    })
    return out


def unify_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    return df[[
        "impression_id", "session_id", "user_id", "impression_time",
        "article_ids_inview", "article_ids_clicked", "device_type", "is_subscriber",
    ]].reset_index(drop=True)


def explode_click_events(history_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> pd.DataFrame:
    hist_rows = []
    for r in history_df.itertuples(index=False):
        for aid, t, rt, sp in zip(
            r.article_id_fixed, r.impression_time_fixed, r.read_time_fixed, r.scroll_percentage_fixed
        ):
            hist_rows.append((r.user_id, aid, t, rt, sp, "history"))
    hist_df = pd.DataFrame(
        hist_rows, columns=["user_id", "article_id", "event_time", "read_time", "scroll_percentage", "source"]
    )

    # Note: behaviors.read_time/scroll_percentage describe the *context* article
    # (article_id), not the clicked candidates — attaching them to clicked
    # candidates would fabricate a signal that isn't in the raw data, so those
    # two columns are left null for impression-click events.
    click_rows = []
    for r in behaviors_df.itertuples(index=False):
        for aid in r.article_ids_clicked:
            click_rows.append((r.user_id, aid, r.impression_time, np.nan, np.nan, "impression_click"))
    click_df = pd.DataFrame(
        click_rows, columns=["user_id", "article_id", "event_time", "read_time", "scroll_percentage", "source"]
    )

    events = pd.concat([hist_df, click_df], ignore_index=True)
    return events.sort_values(["user_id", "event_time"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Time-based split
# --------------------------------------------------------------------------

@dataclass
class SplitBounds:
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # exclusive
    val_start: pd.Timestamp   # == train_end
    val_end: pd.Timestamp     # exclusive


def time_based_split(behaviors_df: pd.DataFrame, val_days: int, train_days: int | None) -> tuple[pd.DataFrame, pd.DataFrame, SplitBounds]:
    t = behaviors_df["impression_time"]
    window_start, window_end = t.min(), t.max()
    cutoff = window_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(days=val_days)
    floor = window_start if train_days is None else max(window_start, cutoff - pd.Timedelta(days=train_days))

    train = behaviors_df[(t >= floor) & (t < cutoff)].reset_index(drop=True)
    val = behaviors_df[t >= cutoff].reset_index(drop=True)
    bounds = SplitBounds(train_start=floor, train_end=cutoff, val_start=cutoff, val_end=window_end)
    return train, val, bounds


def assert_no_leakage(history_df: pd.DataFrame, train_beh: pd.DataFrame, val_beh: pd.DataFrame, bounds: SplitBounds) -> None:
    """Behaviour-window boundary check: no future-click leakage across splits."""
    if len(train_beh):
        assert train_beh["impression_time"].max() < bounds.val_start, "train impressions leak past the val cutoff"
    if len(val_beh):
        assert val_beh["impression_time"].min() >= bounds.val_start, "val impressions leak before the val cutoff"
    if len(train_beh) and len(val_beh):
        assert train_beh["impression_time"].max() < val_beh["impression_time"].min(), "train/val impression times overlap"
    max_hist_time = max(
        (max(row) for row in history_df["impression_time_fixed"] if len(row)), default=None
    )
    if max_hist_time is not None and len(train_beh):
        assert max_hist_time < train_beh["impression_time"].min(), "click history extends into the train window"


# --------------------------------------------------------------------------
# Feature store: articles
# --------------------------------------------------------------------------

def _hash_token(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return idx, sign


def hashed_text_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic, dependency-free feature-hashed bag-of-words vector.

    Placeholder for the article "semantic" feature until real embeddings
    (EB-NeRD word2vec/BERT artifacts, or a trained encoder) are plugged in
    for Q3 — kept here so every article always has a fixed-size vector.
    """
    vec = np.zeros(dim, dtype="float32")
    for token in text.lower().split():
        idx, sign = _hash_token(token, dim)
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class ArticleStore:
    """Reusable lookup for article content + entity + embedding features."""

    def __init__(self, table: pd.DataFrame, embeddings: np.ndarray, article_ids: np.ndarray):
        self.table = table.set_index("article_id", drop=False)
        self.embeddings = embeddings
        self._id_to_row = {aid: i for i, aid in enumerate(article_ids)}

    @classmethod
    def build(cls, articles: pd.DataFrame, embedding_dim: int) -> "ArticleStore":
        text = articles["title"] + " " + articles["abstract"] + " " + articles["body"]
        emb = np.stack([hashed_text_embedding(t, embedding_dim) for t in text])
        return cls(articles.reset_index(drop=True), emb, articles["article_id"].to_numpy())

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.table.reset_index(drop=True).to_parquet(out_dir / "articles.parquet")
        np.save(out_dir / "article_embeddings.npy", self.embeddings)
        np.save(out_dir / "article_ids.npy", np.array(list(self._id_to_row.keys())))

    @classmethod
    def load(cls, out_dir: Path) -> "ArticleStore":
        table = pd.read_parquet(out_dir / "articles.parquet")
        emb = np.load(out_dir / "article_embeddings.npy")
        ids = np.load(out_dir / "article_ids.npy")
        return cls(table, emb, ids)

    def get(self, article_id: int) -> pd.Series:
        return self.table.loc[article_id]

    def get_embedding(self, article_id: int) -> np.ndarray:
        return self.embeddings[self._id_to_row[article_id]]


# --------------------------------------------------------------------------
# Feature store: users
# --------------------------------------------------------------------------

class UserStore:
    """Reusable lookup for per-user click history + recency features."""

    def __init__(self, table: pd.DataFrame):
        self.table = table.set_index("user_id", drop=False)

    @classmethod
    def build(cls, history_df: pd.DataFrame, reference_time: pd.Timestamp, max_history: int) -> "UserStore":
        rows = []
        for r in history_df.itertuples(index=False):
            order = np.argsort(r.impression_time_fixed)
            aids = np.asarray(r.article_id_fixed)[order][-max_history:]
            read = np.asarray(r.read_time_fixed, dtype="float64")[order]
            scroll = np.asarray(r.scroll_percentage_fixed, dtype="float64")[order]
            last_click = max(r.impression_time_fixed) if len(r.impression_time_fixed) else None
            recency_seconds = (reference_time - last_click).total_seconds() if last_click is not None else np.nan
            rows.append({
                "user_id": r.user_id,
                "click_history": aids.tolist(),
                "history_length": len(r.article_id_fixed),
                "recency_seconds": recency_seconds,
                "avg_read_time": float(np.nanmean(read)) if len(read) else np.nan,
                "avg_scroll_percentage": float(np.nanmean(scroll)) if len(scroll) else np.nan,
            })
        return cls(pd.DataFrame(rows))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.reset_index(drop=True).to_parquet(path)

    @classmethod
    def load(cls, path: Path) -> "UserStore":
        return cls(pd.read_parquet(path))

    def get(self, user_id: int) -> pd.Series:
        return self.table.loc[user_id]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    raw_zip = Path(args.raw_zip)
    work_dir = Path(args.work_dir)
    extracted_dir = Path(args.extracted_dir) if args.extracted_dir else work_dir / "ebnerd_demo"
    out_dir = work_dir / "processed"

    print(f"[1/6] extracting {raw_zip} -> {extracted_dir}")
    extract_zip(raw_zip, extracted_dir, force=args.force)

    print("[2/6] loading raw parquet tables")
    raw = load_raw(extracted_dir)

    print("[3/6] cleaning")
    articles = clean_articles(raw["articles"])
    train_hist = clean_history(raw["train_history"])
    train_beh = clean_behaviors(raw["train_behaviors"])
    test_hist = clean_history(raw["test_history"])
    test_beh = clean_behaviors(raw["test_behaviors"])

    print("[4/6] unifying schema")
    articles_u = unify_articles(articles)
    train_beh_u = unify_behaviors(train_beh)
    test_beh_u = unify_behaviors(test_beh)

    print(f"[5/6] time-based split (val_days={args.val_days}, train_days={args.train_days})")
    new_train_beh, new_val_beh, bounds = time_based_split(train_beh_u, args.val_days, args.train_days)
    assert_no_leakage(train_hist, new_train_beh, new_val_beh, bounds)

    splits_dir = out_dir / "splits"
    for name, beh, hist in (
        ("train", new_train_beh, train_hist),
        ("val", new_val_beh, train_hist),
        ("test", test_beh_u, test_hist),
    ):
        split_dir = splits_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        beh.to_parquet(split_dir / "behaviors.parquet")
        explode_click_events(hist, beh).to_parquet(split_dir / "click_events.parquet")

    print("[6/6] building feature store")
    feature_dir = out_dir / "feature_store"
    article_store = ArticleStore.build(articles_u, args.embedding_dim)
    article_store.save(feature_dir)

    train_region_users = UserStore.build(train_hist, reference_time=bounds.train_start, max_history=args.max_history)
    train_region_users.save(feature_dir / "users_train_region.parquet")
    test_users = UserStore.build(test_hist, reference_time=test_beh_u["impression_time"].min(), max_history=args.max_history)
    test_users.save(feature_dir / "users_test.parquet")

    manifest = {
        "raw_zip": str(raw_zip),
        "config": {"val_days": args.val_days, "train_days": args.train_days,
                    "embedding_dim": args.embedding_dim, "max_history": args.max_history},
        "split_bounds": {k: str(v) for k, v in vars(bounds).items()},
        "row_counts": {
            "articles": len(articles_u),
            "train_behaviors": len(new_train_beh),
            "val_behaviors": len(new_val_beh),
            "test_behaviors": len(test_beh_u),
            "train_region_users": len(train_region_users.table),
            "test_users": len(test_users.table),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\ndone -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-zip", default="data/ebnerd_demo.zip")
    p.add_argument("--work-dir", default="data")
    p.add_argument("--extracted-dir", default=None, help="reuse an already-extracted dataset instead of unzipping")
    p.add_argument("--val-days", type=int, default=2, help="last N days of the raw train window -> new val")
    p.add_argument("--train-days", type=int, default=None, help="preceding M days -> new train (default: all remaining)")
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--max-history", type=int, default=50, help="max click-history length kept per user")
    p.add_argument("--force", action="store_true", help="re-extract raw zip even if already extracted")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
