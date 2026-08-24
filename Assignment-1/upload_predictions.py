#!/usr/bin/env python3
"""
Upload final predictions.txt to Kaggle Codabench competition.

After Stage 2 completion in Kaggle notebook, download predictions.txt, then run:
    python3 upload_predictions.py --predictions codabench_files/ebnerd/predictions.txt

Requires: kaggle CLI configured (~/.kaggle/kaggle.json) with API key.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload predictions to Kaggle Codabench")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.txt file",
    )
    parser.add_argument(
        "--competition",
        default="ebnerd-challenge",
        help="Kaggle competition slug (default: ebnerd-challenge)",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions).resolve()
    if not pred_path.exists():
        print(f"ERROR: Predictions file not found: {pred_path}")
        sys.exit(1)

    # Count lines for verification
    with open(pred_path) as f:
        n_lines = sum(1 for _ in f)
    print(f"Uploading {n_lines} predictions from {pred_path.name}...")

    # Submit via kaggle CLI
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        args.competition,
        "-f",
        str(pred_path),
        "-m",
        f"Automated submission: {n_lines} predictions",
    ]

    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    print(f"\n✓ Submission complete!")
    print(f"Check leaderboard: https://www.kaggle.com/competitions/{args.competition}/leaderboard")


if __name__ == "__main__":
    main()
