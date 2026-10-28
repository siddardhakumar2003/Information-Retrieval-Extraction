#!/bin/bash
# SLURM submission script for the Ada cluster. Submit from inside code/ with:
#   sbatch run_ada.sh            runs the full pipeline (Part 1 -> 2 -> 3 -> 4)
#   sbatch run_ada.sh part1      runs only Part 1 (same for part2 / part3 / part4)
#
# Each part is independent -- none reads another part's output, and
# results.csv rows are replaced per (part, dataset) on write -- so running
# parts as separate job submissions (e.g. one per day, under the 6h wall
# limit) is always safe, in any order, any number of times.
#
# IMPORTANT: Ada's compute nodes have no internet access, so data.py's
# auto-download will NOT work here. Copy
# /home2/cs4406.16/Assignment-3/data/sift-128-euclidean.hdf5 and
# /home2/cs4406.16/Assignment-3/data/glove-100-angular.hdf5 onto Ada
# (e.g. via scp/rsync) BEFORE submitting this job. ada_logs/ must also
# already exist under code/ (this script no longer creates it).

#SBATCH --job-name=ann-benchmark
#SBATCH --output=ada_logs/output.out
#SBATCH --error=ada_logs/error.err
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00           # EDIT ME: wall-clock limit (HH:MM:SS); adjust to your partition's max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # matches your 4-cpu-per-task limit
#SBATCH --mem=28G                 # just under your 29G limit, leaves a little headroom

set -euo pipefail

PROJECT_ROOT=/home2/cs4406.16/Assignment-3
CODE_DIR="$PROJECT_ROOT/code"
cd "$CODE_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"  # keep FAISS's thread use within what SLURM actually allocated

# Only Part 2's IVF-Flat/IVF-PQ *training* step uses the GPU (via
# annbench.gpu_utils) — it falls back to CPU automatically if this isn't set,
# or if the venv doesn't actually have faiss-gpu installed. All benchmarked
# search is still plain CPU, single-thread, per the assignment's methodology.
export ANNBENCH_USE_GPU=1

echo "== activating venv =="
source "$CODE_DIR/.venv/bin/activate"

echo "== sanity check: data files present? =="
ls -la "$PROJECT_ROOT/data/sift-128-euclidean.hdf5" "$PROJECT_ROOT/data/glove-100-angular.hdf5"

echo "== sanity check: GPU visible to faiss? =="
python3 -c "import faiss; print('faiss sees', faiss.get_num_gpus(), 'GPU(s)')"

# venv already has the deps installed and compute nodes have no internet access,
# so skip run_all.sh's `pip install -r requirements.txt` step.
export SKIP_DEPS_INSTALL=1

PART="${1:-all}"
case "$PART" in
    all)
        echo "== running full pipeline (Part 1 -> 2 -> 3 -> 4) =="
        bash "$CODE_DIR/run_all.sh"
        ;;
    part1|part2|part3|part4)
        echo "== capturing environment (specs + library versions, this machine) =="
        python3 "$CODE_DIR"/scripts/capture_environment.py
        echo "== running only $PART =="
        python3 "$CODE_DIR"/scripts/"$PART"_*.py
        ;;
    *)
        echo "Unknown argument '$PART'. Use: all | part1 | part2 | part3 | part4" >&2
        exit 1
        ;;
esac
