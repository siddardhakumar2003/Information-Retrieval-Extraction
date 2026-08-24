#!/usr/bin/env python3
"""
Upload MIND evaluation scores to Kaggle as a Dataset for reference.

Usage:
    python3 upload_mind_scores.py --scores outputs/mind_semantic/offline_metrics.json

Requires: kaggle CLI configured (~/.kaggle/kaggle.json) with API key.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload MIND scores to Kaggle")
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to scores directory (e.g., outputs/mind_semantic)",
    )
    parser.add_argument(
        "--method",
        choices=["bm25", "semantic"],
        default="semantic",
        help="Retrieval method",
    )
    args = parser.parse_args()

    scores_dir = Path(args.scores).resolve()
    if not scores_dir.exists():
        print(f"ERROR: Scores directory not found: {scores_dir}")
        sys.exit(1)

    # Create metadata
    dataset_name = f"mind-{args.method}-scores"
    metadata_path = scores_dir.parent / "dataset-metadata.json"
    metadata = {
        "title": f"MIND {args.method.upper()} Evaluation Scores",
        "id": f"sidhardhakumar2003/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": f"MIND dataset evaluation results using {args.method} retrieval",
        "resources": [
            {"path": "offline_metrics.json", "description": "Aggregated metrics with 95% CI"},
            {
                "path": "offline_metrics_per_impression.csv",
                "description": "Per-impression evaluation records",
            },
        ],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Uploading MIND {args.method} scores to Kaggle...")

    # Upload via kaggle CLI
    cmd = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(scores_dir.parent),
        "-m",
        f"MIND {args.method.upper()} evaluation results",
    ]

    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    print(f"\n✓ Upload complete!")
    print(
        f"Dataset: https://www.kaggle.com/datasets/sidhardhakumar2003/{dataset_name}"
    )


if __name__ == "__main__":
    main()
