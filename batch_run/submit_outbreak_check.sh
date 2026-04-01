#!/bin/bash
# ==============================================================================
# Parallel outbreak check via SLURM array + dependent merge job.
#
# Three modes (same script):
#   1. Submission (default): counts batch_* folders, submits array + merge job.
#   2. Array task (SLURM_ARRAY_TASK_ID set): processes one batch -> pickle.
#   3. Merge (MERGE_MODE=1): loads all pickles, prints summary.
#
# Usage: bash submit_outbreak_check.sh <data_dir> [SUB_TOP_PCT]
# ==============================================================================
#SBATCH --job-name=outbreak_check
#SBATCH --partition=lau,week-long-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -z "$SLURM_ARRAY_TASK_ID" ] && [ -z "$MERGE_MODE" ]; then
    # --- Submission mode ---
    if [ $# -lt 1 ]; then
        echo "Usage: bash $0 <data_dir> [SUB_TOP_PCT]"
        exit 1
    fi
    DATA_DIR="${1%/}"
    SUB_TOP_PCT="${2:-1}"
    OUTPUT_DIR="${DATA_DIR}_result"
    NUM_BATCHES=$(ls -d "${DATA_DIR}"/batch_* 2>/dev/null | wc -l)
    if [ "$NUM_BATCHES" -eq 0 ]; then
        echo "Error: No batch_* folders in ${DATA_DIR}"; exit 1
    fi
    mkdir -p "${OUTPUT_DIR}/logs"

    ARRAY_JOB=$(sbatch --parsable --array=0-$((NUM_BATCHES - 1)) \
        --output="${OUTPUT_DIR}/logs/outbreak_%A_%a.out" \
        --error="${OUTPUT_DIR}/logs/outbreak_%A_%a.err" \
        --export=ALL,DATA_DIR="$DATA_DIR",OUTPUT_DIR="$OUTPUT_DIR",SUB_TOP_PCT="$SUB_TOP_PCT",SCRIPT_DIR="$SCRIPT_DIR" \
        "$0")
    MERGE_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} \
        --output="${OUTPUT_DIR}/logs/outbreak_merge_%j.out" \
        --error="${OUTPUT_DIR}/logs/outbreak_merge_%j.err" \
        --export=ALL,DATA_DIR="$DATA_DIR",OUTPUT_DIR="$OUTPUT_DIR",SUB_TOP_PCT="$SUB_TOP_PCT",MERGE_MODE=1,SCRIPT_DIR="$SCRIPT_DIR" \
        "$0")
    echo "Submitted ${NUM_BATCHES} batch jobs (${ARRAY_JOB}), merge job (${MERGE_JOB})"

elif [ -n "$MERGE_MODE" ]; then
    source ~/.bashrc
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    conda activate stephy
    python3 "${SCRIPT_DIR}/outbreak_check.py" merge "$OUTPUT_DIR" "$SUB_TOP_PCT"

else
    source ~/.bashrc
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    conda activate stephy
    python3 "${SCRIPT_DIR}/outbreak_check.py" batch \
        "${DATA_DIR}/batch_${SLURM_ARRAY_TASK_ID}" \
        "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_outbreak.pkl"
fi
