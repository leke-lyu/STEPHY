#!/bin/bash
# ==============================================================================
# Submit training jobs for all pipeline x label combinations via SLURM array.
#
# Launches 12 parallel SLURM tasks (3 pipelines x 4 labels).  Each task
# invokes the corresponding pipeline's train.py with the shared graphs.pt.
#
# Array index mapping:
#   task_id = pipeline_idx * 4 + label_idx
#   Pipelines: stephy2(0), CBLV-CNN2(1), CBLV-GAT2(2)
#   Labels:    R0(0), Recovery_Rate(1), Source_Sink_Score(2), Ancestral_State(3)
#
# Usage: bash submit_train.sh <graphs.pt or batch_dir> <num_locations>
# Output: <graphs_dir>/<pipeline>/<label_short>/
# ==============================================================================
#SBATCH --job-name=train
#SBATCH --partition=lau
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=96:00:00

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

PIPELINES=(stephy2 CBLV-CNN2 CBLV-GAT2)
LABELS=(R0 Recovery_Rate Source_Sink_Score Ancestral_State)
LABEL_SHORTS=(r0 rr sss as)

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Submission mode ---
    if [ $# -ne 2 ]; then
        echo "Usage: bash $0 <graphs.pt or batch_dir> <num_locations>"
        exit 1
    fi

    GRAPHS=$1
    NUM_LOCATIONS=$2
    GRAPHS_DIR="$(dirname "$GRAPHS")"

    if [ ! -e "$GRAPHS" ]; then
        echo "Error: ${GRAPHS} not found"
        exit 1
    fi

    # GRAPHS_DIR: parent if file, the dir itself if directory
    if [ -d "$GRAPHS" ]; then
        GRAPHS_DIR="$GRAPHS"
    fi

    mkdir -p "${GRAPHS_DIR}/logs"
    echo "Submitting 12 training jobs (graphs=${GRAPHS}, num_locations=${NUM_LOCATIONS})"

    sbatch --array=0-11 \
           --output="${GRAPHS_DIR}/logs/train_%A_%a.out" \
           --error="${GRAPHS_DIR}/logs/train_%A_%a.err" \
           --export=ALL,GRAPHS="$GRAPHS",NUM_LOCATIONS="$NUM_LOCATIONS",SCRIPT_DIR="$SCRIPT_DIR" \
           "$0"
else
    # --- Execution mode (SLURM array task) ---
    source ~/.bashrc
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    conda activate stephy

    PIPELINE_IDX=$((SLURM_ARRAY_TASK_ID / 4))
    LABEL_IDX=$((SLURM_ARRAY_TASK_ID % 4))

    PIPELINE="${PIPELINES[$PIPELINE_IDX]}"
    LABEL="${LABELS[$LABEL_IDX]}"
    LABEL_SHORT="${LABEL_SHORTS[$LABEL_IDX]}"

    if [ -d "$GRAPHS" ]; then
        GRAPHS_DIR="$GRAPHS"
    else
        GRAPHS_DIR="$(dirname "$GRAPHS")"
    fi
    OUTPUT_DIR="${GRAPHS_DIR}/${PIPELINE}/${LABEL_SHORT}"
    STEPHY_ROOT="$(dirname "$SCRIPT_DIR")"
    TRAIN_PY="${STEPHY_ROOT}/${PIPELINE}/train.py"

    mkdir -p "$OUTPUT_DIR"

    echo "Task ${SLURM_ARRAY_TASK_ID}: pipeline=${PIPELINE}, label=${LABEL}, output=${OUTPUT_DIR}"

    python3 "$TRAIN_PY" \
        --graphs "$GRAPHS" \
        --num_locations "$NUM_LOCATIONS" \
        --label "$LABEL" \
        --output_dir "$OUTPUT_DIR"
fi
