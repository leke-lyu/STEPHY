#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --partition=lau
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=96:00:00

# ============================================================
# Usage:
#   bash submit_50k.sh <num_batches> <sims_per_batch> <base_dir>
#
# Example:
#   bash submit.sh 25 2000 /path/to/output
#   -> creates /path/to/output/50k/batch_0 .. batch_24
#
#   bash submit.sh 5 10 /path/to/output
#   -> creates /path/to/output/0.05k/batch_0 .. batch_4
# ============================================================

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Submission mode (run from login node) ---
    if [ $# -ne 3 ]; then
        echo "Usage: bash $0 <num_batches> <sims_per_batch> <base_dir>"
        exit 1
    fi

    NUM_BATCHES=$1
    SIMS_PER_BATCH=$2
    BASE_DIR=$3

    TOTAL=$((NUM_BATCHES * SIMS_PER_BATCH))
    TOTAL_K=$(awk "BEGIN {printf \"%g\", $TOTAL/1000}")
    FOLDER="${TOTAL_K}k"
    OUT_DIR="${BASE_DIR}/${FOLDER}"

    mkdir -p "${OUT_DIR}/logs"

    echo "Submitting $NUM_BATCHES jobs x $SIMS_PER_BATCH sims = $TOTAL total ($FOLDER)"
    echo "Output: $OUT_DIR"

    sbatch --array=0-$((NUM_BATCHES - 1)) \
           --output="${OUT_DIR}/logs/slurm_%A_%a.out" \
           --error="${OUT_DIR}/logs/slurm_%A_%a.err" \
           --export=ALL,SIMS_PER_BATCH="$SIMS_PER_BATCH",OUT_DIR="$OUT_DIR" \
           "$0"
else
    # --- Execution mode (inside SLURM array job) ---
    source ~/.bashrc
    conda activate stephy

    BATCH_DIR="${OUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$BATCH_DIR"

    sh simulate_and_extract.sh "$SIMS_PER_BATCH" "$BATCH_DIR"
fi
