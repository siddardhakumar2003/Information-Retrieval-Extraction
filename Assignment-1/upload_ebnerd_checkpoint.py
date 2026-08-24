#!/usr/bin/env python3
"""
Upload EB-NeRD checkpoint (predictions.txt + scores_all.parquet + embeddings) to Kaggle as a Dataset.

After Kaggle notebook timeout, download outputs from the "Output" tab, then run:
    python3 upload_ebnerd_checkpoint.py --checkpoint codabench_files/ebnerd --dataset-name ebnerd-checkpoint-v1

This creates a new Kaggle Dataset that can be referenced as /kaggle/input/<dataset-slug>/ in the next notebook run.
Requires: kagglehub configured with valid API key (~/.kaggle/kaggle.json).
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("ERROR: kagglehub not installed. Install with:")
    print("  pip install kagglehub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload EB-NeRD checkpoint to Kaggle Dataset using kagglehub"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g., codabench_files/ebnerd)",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Kaggle dataset slug (e.g., ebnerd-checkpoint-v1)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default: private)",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint).resolve()
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Check required files
    required_files = ["checkpoint.json", "predictions.txt"]
    missing = [f for f in required_files if not (checkpoint_dir / f).exists()]
    if missing:
        print(f"WARNING: Missing files: {missing}")
        print(f"  Available files: {list(checkpoint_dir.glob('*'))}")

    # Create metadata.json for dataset
    metadata_path = checkpoint_dir / "dataset-metadata.json"
    metadata = {
        "title": f"EB-NeRD Checkpoint - {args.dataset_name}",
        "id": f"sidhardhakumar2003/{args.dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
        "resources": [
            {
                "path": f"{checkpoint_dir.name}/checkpoint.json",
                "description": "Resume checkpoint (last_index, pred_lines, scores_rows)",
            },
            {
                "path": f"{checkpoint_dir.name}/predictions.txt",
                "description": "Codabench predictions file (append mode) - source of truth for resume",
            },
            {
                "path": f"{checkpoint_dir.name}/scores_all.parquet",
                "description": "Merged BM25 + semantic scores (single file)",
            },
            {
                "path": f"{checkpoint_dir.name}/embeddings.npy",
                "description": "Cached article embeddings",
            },
            {
                "path": f"{checkpoint_dir.name}/article_ids.npy",
                "description": "Article ID mapping for embeddings",
            },
        ],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Created dataset metadata: {metadata_path}")

    # Upload via kagglehub
    dataset_handle = f"kspsvlnsiddardha/{args.dataset_name}"
    print(f"\n🚀 Uploading checkpoint to Kaggle Dataset: {dataset_handle}")
    print(f"   Source directory: {checkpoint_dir}")

    try:
        kagglehub.dataset_upload(
            handle=dataset_handle,
            local_dataset_dir=str(checkpoint_dir),
        )
        print(f"\n✓ Upload complete!")
        print(f"Dataset: https://www.kaggle.com/datasets/{dataset_handle}")
        print(f"\n📌 In next Kaggle notebook run, set:")
        print(f"   RESUME_DIR = '/kaggle/input/sidhardhakumar2003-{args.dataset_name}/'")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print(f"\nNote: If dataset already exists, you may need to create a new version.")
        print(f"Check your Kaggle datasets: https://www.kaggle.com/settings/datasets")
        sys.exit(1)


if __name__ == "__main__":
    main()
