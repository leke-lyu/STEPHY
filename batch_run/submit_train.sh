#!/bin/bash
# ==============================================================================
# Submit training jobs for pipeline x label combinations via SLURM array.
#
# Each SLURM task invokes the corresponding pipeline's train.py with the
# shared graphs.pt.  Optional --pipeline and --labels flags select a subset;
# defaults to all 3 pipelines x 6 labels (18 jobs).
#
# Usage: bash submit_train.sh [--pipeline stephy|CBLV-CNN|CBLV-GAT] \
#            [--labels reg_r0,cls_r0,...] \
#            <graphs.pt or batch_dir> <num_locations>
# Output: <graphs_dir>/<pipeline>/<label>/
# ==============================================================================
#SBATCH --job-name=train
#SBATCH --partition=lau,week-long-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=128:00:00

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

ALL_PIPELINES="stephy,CBLV-CNN,CBLV-GAT"
ALL_LABELS="reg_r0,cls_r0,reg_rr,reg_sss,cls_sss,cls_as"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Submission mode ---

    # Parse optional flags
    SEL_PIPELINES="$ALL_PIPELINES"
    SEL_LABELS="$ALL_LABELS"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --pipeline) SEL_PIPELINES="$2"; shift 2 ;;
            --labels)   SEL_LABELS="$2";   shift 2 ;;
            *)          break ;;
        esac
    done

    if [ $# -ne 2 ]; then
        echo "Usage: bash $0 [--pipeline stephy|CBLV-CNN|CBLV-GAT] [--labels reg_r0,cls_r0,...] <graphs.pt or batch_dir> <num_locations>"
        exit 1
    fi

    GRAPHS=$1
    NUM_LOCATIONS=$2

    if [ ! -e "$GRAPHS" ]; then
        echo "Error: ${GRAPHS} not found"
        exit 1
    fi

    # GRAPHS_DIR: parent if file, the dir itself if directory
    if [ -d "$GRAPHS" ]; then
        GRAPHS_DIR="$GRAPHS"
    else
        GRAPHS_DIR="$(dirname "$GRAPHS")"
    fi

    # Build flat task list: "pipeline:label" pairs
    IFS=',' read -ra PIPELINES <<< "$SEL_PIPELINES"
    IFS=',' read -ra LABELS <<< "$SEL_LABELS"
    TASKS=()
    for P in "${PIPELINES[@]}"; do
        for L in "${LABELS[@]}"; do
            TASKS+=("${P}:${L}")
        done
    done
    NUM_TASKS=${#TASKS[@]}

    if [ "$NUM_TASKS" -eq 0 ]; then
        echo "Error: No tasks to submit"
        exit 1
    fi

    # Write task list so SLURM workers can look up their assignment
    TASK_FILE="${GRAPHS_DIR}/logs/train_tasks.txt"
    mkdir -p "${GRAPHS_DIR}/logs"
    printf '%s\n' "${TASKS[@]}" > "$TASK_FILE"

    echo "Submitting ${NUM_TASKS} training jobs (graphs=${GRAPHS}, num_locations=${NUM_LOCATIONS})"
    printf '  %s\n' "${TASKS[@]}"

    sbatch --array=0-$((NUM_TASKS - 1)) \
           --output="${GRAPHS_DIR}/logs/train_%A_%a.out" \
           --error="${GRAPHS_DIR}/logs/train_%A_%a.err" \
           --export=ALL,GRAPHS="$GRAPHS",NUM_LOCATIONS="$NUM_LOCATIONS",SCRIPT_DIR="$SCRIPT_DIR",TASK_FILE="$TASK_FILE" \
           "$0"
else
    # --- Execution mode (SLURM array task) ---
    source ~/.bashrc
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    conda activate stephy

    # Read assignment from task file
    TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
    PIPELINE="${TASK%%:*}"
    LABEL="${TASK##*:}"

    if [ -d "$GRAPHS" ]; then
        GRAPHS_DIR="$GRAPHS"
    else
        GRAPHS_DIR="$(dirname "$GRAPHS")"
    fi
    OUTPUT_DIR="${GRAPHS_DIR}/${PIPELINE}/${LABEL}"
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
