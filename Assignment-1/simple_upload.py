#!/usr/bin/env python3
"""
Simple utility to upload files/directories to Kaggle as a Dataset.

Usage:
    # Upload preprocessed data
    python3 simple_upload.py --path outputs/lexical --name ebnerd-bm25-results

    # Upload embedding cache
    python3 simple_upload.py --path outputs/semantic --name ebnerd-semantic-embeddings

Requires: kaggle CLI configured (~/.kaggle/kaggle.json) with API key.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload directory to Kaggle Dataset")
    parser.add_argument("--path", required=True, help="Path to upload (file or directory)")
    parser.add_argument("--name", required=True, help="Kaggle dataset slug")
    parser.add_argument(
        "--public", action="store_true", help="Make dataset public (default: private)"
    )
    parser.add_argument(
        "--description",
        default="Evaluation results",
        help="Dataset description",
    )
    args = parser.parse_args()

    upload_path = Path(args.path).resolve()
    if not upload_path.exists():
        print(f"ERROR: Path not found: {upload_path}")
        sys.exit(1)

    is_dir = upload_path.is_dir()
    if is_dir:
        file_count = len(list(upload_path.rglob("*")))
        print(f"Uploading directory: {upload_path} ({file_count} files)")
    else:
        size_mb = upload_path.stat().st_size / (1024 * 1024)
        print(f"Uploading file: {upload_path} ({size_mb:.1f} MB)")

    # Create metadata
    metadata_path = upload_path.parent / "dataset-metadata.json"
    metadata = {
        "title": args.name.replace("-", " ").title(),
        "id": f"sidhardhakumar2003/{args.name}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": args.description,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Upload via kaggle CLI
    cmd = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(upload_path.parent),
        "-m",
        args.description,
    ]

    if args.public:
        cmd.append("--public")

    print(f"\n  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        print("\nTo update existing dataset, use:")
        print(f"  kaggle datasets version -p {upload_path.parent} -m 'Updated results'")
        sys.exit(1)

    print(result.stdout)
    print(f"\n✓ Upload complete!")
    print(
        f"Dataset: https://www.kaggle.com/datasets/sidhardhakumar2003/{args.name}"
    )
    print(f"In Kaggle notebooks, reference via: /kaggle/input/sidhardhakumar2003-{args.name}/")


if __name__ == "__main__":
    main()
