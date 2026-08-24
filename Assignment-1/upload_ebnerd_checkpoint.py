#!/usr/bin/env python3
"""
Upload EB-NeRD checkpoint (scores.parquet + predictions.txt + embeddings) to Kaggle as a Dataset.

After Kaggle notebook timeout, download outputs from the "Output" tab, then run:
    python3 upload_ebnerd_checkpoint.py --checkpoint codabench_files/ebnerd --dataset-name ebnerd-checkpoint-v1

This creates a new Kaggle Dataset that can be referenced as /kaggle/input/<dataset-slug>/ in the next notebook run.
Requires: kaggle CLI configured (~/.kaggle/kaggle.json) with API key.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    """Run shell command and return success."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload EB-NeRD checkpoint to Kaggle Dataset"
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

    # Create metadata.csv for dataset
    metadata_path = checkpoint_dir / "dataset-metadata.json"
    metadata = {
        "title": f"EB-NeRD Checkpoint - {args.dataset_name}",
        "id": f"sidhardhakumar2003/{args.dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
        "resources": [
            {
                "path": f"{checkpoint_dir.name}/checkpoint.json",
                "description": "Resume checkpoint (last_index, pred_lines, part count)",
            },
            {
                "path": f"{checkpoint_dir.name}/predictions.txt",
                "description": "Codabench predictions file (append mode)",
            },
            {
                "path": f"{checkpoint_dir.name}/embeddings.npy",
                "description": "Cached article embeddings (optional)",
            },
            {
                "path": f"{checkpoint_dir.name}/scores_parts",
                "description": "Parquet part files for scores",
            },
        ],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Created dataset metadata: {metadata_path}")

    # Upload via kaggle CLI
    print(f"\nUploading checkpoint to Kaggle as '{args.dataset_name}'...")
    cmd = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(checkpoint_dir),
        "-m",
    ]
    if not run_cmd(cmd):
        print("\nDataset may already exist. Try updating instead:")
        print(
            f"  kaggle datasets version -p {checkpoint_dir} -m 'Resume checkpoint v2'"
        )
        sys.exit(1)

    print(f"\n✓ Upload complete!")
    print(f"Dataset: https://www.kaggle.com/datasets/sidhardhakumar2003/{args.dataset_name}")
    print(f"\nIn next Kaggle notebook run, set:")
    print(f"  RESUME_DIR = '/kaggle/input/sidhardhakumar2003-{args.dataset_name}/'")


if __name__ == "__main__":
    main()
