#!/bin/bash
# ==============================================================================
# STEPHY Pipeline: End-to-end phylogenetic spatial transmission estimation.
#
# Shared entry point for all three pipelines (stephy, CBLV-CNN, CBLV-GAT).
# The --pipeline flag selects which model to use; default is stephy.
#
# Three steps per input dataset:
#   1. analyze_trees.py  -- Determine num_locations and subtree_width.
#                           (always uses stephy/analyze_trees.py)
#   2. build_graphs.py   -- Build DGL graphs with CBLV features and labels.
#                           (uses <pipeline>/build_graphs.py -> <pipeline>/data.py)
#   3. train.py          -- Train single-task models for selected labels.
#                           (uses <pipeline>/train.py -> <pipeline>/model.py)
#
# Labels (default: all):
#   reg_r0, cls_r0, reg_rr, reg_sss, cls_sss, cls_as
#
# Usage:
#   bash stephy/run_pipeline.sh [--pipeline stephy|CBLV-CNN|CBLV-GAT] \
#       [--labels reg_r0,cls_r0,...] \
#       <input_folder_0> [input_folder_1 ...] <output_folder>
#
# Examples:
#   bash stephy/run_pipeline.sh data/ output/
#   bash stephy/run_pipeline.sh --labels reg_r0,cls_r0 data/ output/
#   bash stephy/run_pipeline.sh --pipeline CBLV-CNN data/ output/
#   bash stephy/run_pipeline.sh --pipeline CBLV-GAT --labels cls_as data1/ data2/ output/
# ==============================================================================

set -e
export PYTHONDONTWRITEBYTECODE=1

# Directory containing this script (stephy/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root (parent of stephy/)
STEPHY_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Parse optional flags ---
PIPELINE="stephy"
ALL_LABELS="reg_r0,cls_r0,reg_rr,reg_sss,cls_sss,cls_as"
LABELS="$ALL_LABELS"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --labels)   LABELS="$2";   shift 2 ;;
        *)          break ;;
    esac
done

PIPELINE_DIR="$STEPHY_ROOT/$PIPELINE"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

# --- Parse positional args: input_folder(s) + output_folder ---
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 [--pipeline stephy|CBLV-CNN|CBLV-GAT] [--labels reg_r0,cls_r0,...] input_folder_0 [input_folder_1 ...] output_folder"
    exit 1
fi

ARGS=("$@")
NUM_ARGS=${#ARGS[@]}
OUT_FOLDER="${ARGS[$NUM_ARGS-1]}"
INPUT_FOLDERS=("${ARGS[@]:0:$NUM_ARGS-1}")

IFS=',' read -ra LABEL_ARRAY <<< "$LABELS"

echo "=== STEPHY: $PIPELINE ==="
echo "Labels: ${LABEL_ARRAY[*]}"
echo "Output: $OUT_FOLDER"
echo ""

mkdir -p "$OUT_FOLDER"

for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    DATASET_NAME=$(basename "$INPUT_FOLDER")
    WORK_DIR="$OUT_FOLDER/$DATASET_NAME"
    mkdir -p "$WORK_DIR"

    echo "--- $DATASET_NAME ---"

    # Step 1: Analyze trees (shared — always stephy/analyze_trees.py)
    ANALYZE_OUTPUT=$(python3 "$SCRIPT_DIR/analyze_trees.py" "$INPUT_FOLDER" 2>&1)
    NUM_LOCATIONS=$(echo "$ANALYZE_OUTPUT" | grep "\-\-num_locations" | awk '{print $2}')
    SUBTREE_WIDTH=$(echo "$ANALYZE_OUTPUT" | grep "\-\-subtree_width" | awk '{print $2}')

    if [ -z "$NUM_LOCATIONS" ] || [ -z "$SUBTREE_WIDTH" ]; then
        echo "  Error: Could not parse tree parameters. Skipping."
        continue
    fi
    echo "  Params: num_locations=$NUM_LOCATIONS, subtree_width=$SUBTREE_WIDTH"

    # Step 2: Build graphs (pipeline-specific build_graphs.py -> data.py)
    GRAPHS_FILE="$WORK_DIR/graphs.pt"
    echo "  Building graphs..."
    python3 "$PIPELINE_DIR/build_graphs.py" \
        --input_dir "$INPUT_FOLDER" \
        --subtree_width "$SUBTREE_WIDTH" \
        --output "$GRAPHS_FILE"

    # Step 3: Train selected labels (pipeline-specific train.py -> model.py)
    for LABEL in "${LABEL_ARRAY[@]}"; do
        echo "  Training $LABEL..."
        python3 "$PIPELINE_DIR/train.py" \
            --graphs "$GRAPHS_FILE" \
            --num_locations "$NUM_LOCATIONS" \
            --label "$LABEL" \
            --output_dir "$WORK_DIR/results_$LABEL"
    done

    echo "  Done: $WORK_DIR"
    echo ""
done

echo "=== Complete ==="
