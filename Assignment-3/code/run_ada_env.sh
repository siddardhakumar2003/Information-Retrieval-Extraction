#!/bin/bash
# SLURM submission script: capture Ada's machine specs + library versions only.
# Use this instead of run_ada.sh when the benchmark results are already final
# and you just need results/environment.txt to reflect the machine the real
# run happened on, without re-running any part.
#
#   sbatch run_ada_env.sh
#
# Runs on a compute node (not the login node) so the captured CPU/GPU info
# matches where the benchmarks actually executed. Request the same GPU
# resource as run_ada.sh so "GPU visible to faiss" reflects a real run, not
# an idle CPU-only node.

#SBATCH --job-name=ann-bench-env
#SBATCH --output=ada_logs/env_output.out
#SBATCH --error=ada_logs/env_error.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

PROJECT_ROOT=/home2/cs4406.16/Assignment-3
CODE_DIR="$PROJECT_ROOT/code"
cd "$CODE_DIR"

echo "== activating venv =="
source "$CODE_DIR/.venv/bin/activate"

echo "== capturing environment (specs + library versions) =="
python3 "$CODE_DIR/scripts/capture_environment.py"
