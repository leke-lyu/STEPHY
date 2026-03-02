#!/bin/bash
# ============================================================
# SLURM BATCH SUBMISSION SCRIPT FOR SIMULATION PIPELINE
# ============================================================
#
# Submits simulate_and_extract.sh as a SLURM array job so that
# many independent batches of outbreak simulations can run in
# parallel on a cluster.
#
# Dual-mode script:
#   1. Submission mode (login node):  parses CLI arguments,
#      creates the output directory, and calls `sbatch --array`
#      to launch one task per batch.
#   2. Execution mode (compute node): activated when SLURM sets
#      SLURM_ARRAY_TASK_ID.  Each task runs
#      simulate_and_extract.sh <sims_per_batch> <batch_dir>.
#
# Usage:
#   bash submit.sh <num_batches> <sims_per_batch> <base_dir>
#
# Examples:
#   bash submit.sh 25 2000 /path/to/output
#   -> creates /path/to/output/50k/batch_0 .. batch_24
#
#   bash submit.sh 5 10 /path/to/output
#   -> creates /path/to/output/0.05k/batch_0 .. batch_4
#
# IMPORTANT: RANDOM_SEED in simulate_and_extract.sh must be "None"
# when using this script. A fixed seed makes every batch produce
# identical outbreak parameters (same R0, population sizes, etc.).
# ============================================================

#SBATCH --job-name=sim
#SBATCH --partition=lau
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=96:00:00

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

    # Guard: source simulate_and_extract.sh config to check RANDOM_SEED
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _SEED=$(grep -m1 '^RANDOM_SEED=' "$SCRIPT_DIR/simulate_and_extract.sh" | cut -d'"' -f2)
    if [ "$_SEED" != "None" ]; then
        echo "ERROR: RANDOM_SEED is set to '$_SEED' in simulate_and_extract.sh."
        echo "       A fixed seed will make every batch produce identical parameters."
        echo "       Set RANDOM_SEED=\"None\" before running multi-batch jobs."
        exit 1
    fi

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
