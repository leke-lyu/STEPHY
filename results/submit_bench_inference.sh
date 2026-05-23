#!/bin/bash
# ==============================================================================
# Benchmark STEPHY single-tree inference latency on the cluster.
#
# Single (non-array) SLURM job, single-threaded by design — measures honest
# per-core forward-pass latency over the full test set for each label.
#
# Usage: bash results/submit_bench_inference.sh \
#            [<graphs_dir>] [<model_dir>]
#
# Defaults to /projects/lau_projects/simu/100k_diverse_population_result.
# Output: <graphs_dir>/logs/bench_stephy_<jobid>.out
# ==============================================================================
#SBATCH --job-name=bench_stephy
#SBATCH --partition=lau,week-long-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPHY_ROOT="$(dirname "$SCRIPT_DIR")"

GRAPHS_DIR="${1:-/projects/lau_projects/simu/100k_diverse_population_result}"
MODEL_DIR="${2:-${GRAPHS_DIR}/stephy}"
LOG_DIR="${GRAPHS_DIR}/logs"

if [ -z "$SLURM_JOB_ID" ]; then
    # --- Submission mode (called from login node) ---
    mkdir -p "$LOG_DIR"
    sbatch --output="${LOG_DIR}/bench_stephy_%j.out" \
           --error="${LOG_DIR}/bench_stephy_%j.err" \
           --export=ALL,GRAPHS_DIR="$GRAPHS_DIR",MODEL_DIR="$MODEL_DIR" \
           "$0" "$GRAPHS_DIR" "$MODEL_DIR"
    exit 0
fi

# --- Execution mode (inside SLURM) ---
source ~/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate stephy

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "  graphs_dir = ${GRAPHS_DIR}"
echo "  model_dir  = ${MODEL_DIR}"

python3 "${STEPHY_ROOT}/results/bench_stephy_inference.py" \
    --graphs_dir "$GRAPHS_DIR" \
    --model_dir  "$MODEL_DIR" \
    --labels     cls_as,reg_r0,reg_rr,reg_sss \
    --n_warmup   20
