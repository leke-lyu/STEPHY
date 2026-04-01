#!/bin/bash
# ============================================================
# SLURM BATCH SUBMISSION SCRIPT FOR SIMULATION PIPELINE
# ============================================================
#
# Submits a simulate_and_extract.sh script as a SLURM array job
# so that many independent batches run in parallel on a cluster.
#
# Dual-mode script:
#   1. Submission mode (login node):  parses CLI arguments,
#      creates the output directory, and calls `sbatch --array`
#      to launch one task per batch.
#   2. Execution mode (compute node): activated when SLURM sets
#      SLURM_ARRAY_TASK_ID.  Each task runs the simulation script.
#
# Usage:
#   bash submit.sh <sim_script> <num_batches> <sims_per_batch> <base_dir>
#
# Examples:
#   # Generic simulation engine
#   bash simulate_and_extract/submit.sh \
#       simulate_and_extract/simulate_and_extract.sh 25 2000 /path/to/output
#
#   # Denmark simulation engine
#   bash simulate_and_extract/submit.sh \
#       simulate_and_extract_Denmark/simulate_and_extract.sh 25 2000 /path/to/output
#
# IMPORTANT: RANDOM_SEED in the simulation script must be "None"
# when using this script. A fixed seed makes every batch produce
# identical outbreak parameters (same R0, population sizes, etc.).
# ============================================================

#SBATCH --job-name=sim
#SBATCH --partition=lau,week-long-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=96:00:00

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Submission mode (run from login node) ---
    if [ $# -ne 4 ]; then
        echo "Usage: bash $0 <sim_script> <num_batches> <sims_per_batch> <base_dir>"
        exit 1
    fi

    SIM_SCRIPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    NUM_BATCHES=$2
    SIMS_PER_BATCH=$3
    BASE_DIR=$4

    if [ ! -f "$SIM_SCRIPT" ]; then
        echo "Error: simulation script not found: $SIM_SCRIPT"
        exit 1
    fi

    TOTAL=$((NUM_BATCHES * SIMS_PER_BATCH))
    TOTAL_K=$(awk "BEGIN {printf \"%g\", $TOTAL/1000}")
    FOLDER="${TOTAL_K}k"
    OUT_DIR="${BASE_DIR}/${FOLDER}"

    mkdir -p "${OUT_DIR}/logs"

    # Guard: RANDOM_SEED must be "None" for multi-batch runs
    _SEED=$(grep -m1 '^RANDOM_SEED=' "$SIM_SCRIPT" | cut -d'"' -f2)
    if [ "$_SEED" != "None" ]; then
        echo "ERROR: RANDOM_SEED is set to '$_SEED' in $(basename "$SIM_SCRIPT")."
        echo "       A fixed seed will make every batch produce identical parameters."
        echo "       Set RANDOM_SEED=\"None\" before running multi-batch jobs."
        exit 1
    fi

    echo "Submitting $NUM_BATCHES jobs x $SIMS_PER_BATCH sims = $TOTAL total ($FOLDER)"
    echo "Simulation: $SIM_SCRIPT"
    echo "Output: $OUT_DIR"

    sbatch --array=0-$((NUM_BATCHES - 1)) \
           --output="${OUT_DIR}/logs/slurm_%A_%a.out" \
           --error="${OUT_DIR}/logs/slurm_%A_%a.err" \
           --export=ALL,SIMS_PER_BATCH="$SIMS_PER_BATCH",OUT_DIR="$OUT_DIR",SIM_SCRIPT="$SIM_SCRIPT" \
           "$0"
else
    # --- Execution mode (inside SLURM array job) ---
    source ~/.bashrc
    conda activate stephy

    BATCH_DIR="${OUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$BATCH_DIR"

    sh "$SIM_SCRIPT" "$SIMS_PER_BATCH" "$BATCH_DIR"
fi
