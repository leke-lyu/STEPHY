#!/bin/bash
# ==============================================================================
# Re-run RAPS calibration for already-trained classification models.
#
# One SLURM task per pipeline; each task handles all selected classification
# labels from a single graph load. No training is repeated -- this only re-runs
# the two post-hoc inference passes (calibration set -> q_hat, test set ->
# prediction sets) with the corrected RAPS conformity score.
#
# Usage: bash submit_recalibrate.sh [--pipeline stephy|CBLV-CNN|CBLV-GAT] \
#            [--labels cls_r0,cls_sss,cls_as] [--dry_run] \
#            <graphs.pt or batch_dir> <num_locations>
# Output: rewrites <graphs_dir>/<pipeline>/<label>/{cp_calibration.pt,
#         cp_metrics.json,test_predictions.csv}; originals kept as *.prebugfix.bak
# ==============================================================================
#SBATCH --job-name=recal_raps
#SBATCH --partition=lau,week-long-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

ALL_PIPELINES="stephy,CBLV-CNN,CBLV-GAT"
ALL_LABELS="cls_r0,cls_sss,cls_as"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # --- Submission mode ---

    SEL_PIPELINES="$ALL_PIPELINES"
    SEL_LABELS="$ALL_LABELS"
    DRY_RUN=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --pipeline) SEL_PIPELINES="$2"; shift 2 ;;
            --labels)   SEL_LABELS="$2";   shift 2 ;;
            --dry_run)  DRY_RUN="--dry_run"; shift ;;
            *)          break ;;
        esac
    done

    if [ $# -ne 2 ]; then
        echo "Usage: bash $0 [--pipeline stephy|CBLV-CNN|CBLV-GAT] [--labels cls_r0,...] [--dry_run] <graphs.pt or batch_dir> <num_locations>"
        exit 1
    fi

    GRAPHS=$1
    NUM_LOCATIONS=$2

    if [ ! -e "$GRAPHS" ]; then
        echo "Error: ${GRAPHS} not found"
        exit 1
    fi

    if [ -d "$GRAPHS" ]; then
        GRAPHS_DIR="$GRAPHS"
    else
        GRAPHS_DIR="$(dirname "$GRAPHS")"
    fi

    IFS=',' read -ra PIPELINES <<< "$SEL_PIPELINES"
    NUM_TASKS=${#PIPELINES[@]}

    if [ "$NUM_TASKS" -eq 0 ]; then
        echo "Error: No tasks to submit"
        exit 1
    fi

    TASK_FILE="${GRAPHS_DIR}/logs/recalibrate_tasks.txt"
    mkdir -p "${GRAPHS_DIR}/logs"
    printf '%s\n' "${PIPELINES[@]}" > "$TASK_FILE"

    echo "Submitting ${NUM_TASKS} re-calibration job(s)"
    echo "  graphs=${GRAPHS}  num_locations=${NUM_LOCATIONS}  labels=${SEL_LABELS}"
    [ -n "$DRY_RUN" ] && echo "  DRY RUN — no files will be modified"
    printf '  %s\n' "${PIPELINES[@]}"

    sbatch --array=0-$((NUM_TASKS - 1)) \
           --output="${GRAPHS_DIR}/logs/recalibrate_%A_%a.out" \
           --error="${GRAPHS_DIR}/logs/recalibrate_%A_%a.err" \
           --export=ALL,GRAPHS="$GRAPHS",NUM_LOCATIONS="$NUM_LOCATIONS",SCRIPT_DIR="$SCRIPT_DIR",TASK_FILE="$TASK_FILE",SEL_LABELS="$SEL_LABELS",DRY_RUN="$DRY_RUN" \
           "$0"
else
    # --- Execution mode (SLURM array task) ---
    source ~/.bashrc
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    conda activate stephy

    PIPELINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")

    echo "Task ${SLURM_ARRAY_TASK_ID}: pipeline=${PIPELINE}, labels=${SEL_LABELS}"

    python3 "${SCRIPT_DIR}/recalibrate_raps.py" \
        --graphs "$GRAPHS" \
        --pipeline "$PIPELINE" \
        --labels "$SEL_LABELS" \
        --num_locations "$NUM_LOCATIONS" \
        $DRY_RUN
fi
