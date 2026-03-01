#!/bin/bash
#SBATCH --job-name=build_graphs
#SBATCH --partition=lau
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=96:00:00

# Build graphs.pt per batch via SLURM array.
# Usage: bash submit_build_graphs.sh <data_dir> <subtree_width>
# Output: batch_*_graphs.pt in the output directory

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Submission mode ---
    if [ $# -ne 2 ]; then
        echo "Usage: bash $0 <data_dir> <subtree_width>"
        exit 1
    fi

    DATA_DIR="${1%/}"
    SUBTREE_WIDTH=$2
    OUTPUT_DIR="${DATA_DIR}_result"
    NUM_BATCHES=$(ls -d "${DATA_DIR}"/batch_* 2>/dev/null | wc -l)

    if [ "$NUM_BATCHES" -eq 0 ]; then
        echo "Error: No batch_* folders in ${DATA_DIR}"
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}/logs"
    echo "Submitting ${NUM_BATCHES} jobs (subtree_width=${SUBTREE_WIDTH})"

    sbatch --array=0-$((NUM_BATCHES - 1)) \
           --output="${OUTPUT_DIR}/logs/slurm_%A_%a.out" \
           --error="${OUTPUT_DIR}/logs/slurm_%A_%a.err" \
           --export=ALL,DATA_DIR="$DATA_DIR",SUBTREE_WIDTH="$SUBTREE_WIDTH",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
           "$0"
else
    # --- Execution mode (SLURM array task) ---
    source ~/.bashrc
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    conda activate stephy

    INPUT_DIR="${DATA_DIR}/batch_${SLURM_ARRAY_TASK_ID}"

    python3 "${SCRIPT_DIR}/build_graphs.py" \
        --input_dir "$INPUT_DIR" \
        --subtree_width "$SUBTREE_WIDTH" \
        --output "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_graphs.pt"
fi
