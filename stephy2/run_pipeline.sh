#!/bin/bash
# ==============================================================================
# STEPHY Pipeline: End-to-end phylogenetic spatial transmission estimation.
#
# Shared entry point for all three pipelines (stephy2, CBLV-CNN2, CBLV-GAT2).
# The --pipeline flag selects which model to use; default is stephy2.
#
# Three steps per input dataset:
#   1. analyze_trees.py  -- Determine num_locations and subtree_width.
#                           (always uses stephy2/analyze_trees.py)
#   2. build_graphs.py   -- Build DGL graphs with CBLV features and labels.
#                           (uses <pipeline>/build_graphs.py -> <pipeline>/data.py)
#   3. train.py          -- Train four single-task models (R0, Recovery_Rate,
#                           Source_Sink_Score, Ancestral_State).
#                           (uses <pipeline>/train.py -> <pipeline>/model.py)
#
# Usage:
#   bash stephy2/run_pipeline.sh [--pipeline stephy2|CBLV-CNN2|CBLV-GAT2] \
#       <input_folder_0> [input_folder_1 ...] <output_folder>
#
# Examples:
#   bash stephy2/run_pipeline.sh data/ output/
#   bash stephy2/run_pipeline.sh --pipeline CBLV-CNN2 data/ output/
#   bash stephy2/run_pipeline.sh --pipeline CBLV-GAT2 data1/ data2/ output/
# ==============================================================================

set -e
export PYTHONDONTWRITEBYTECODE=1

# Directory containing this script (stephy2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root (parent of stephy2/)
STEPHY_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Parse --pipeline flag ---
PIPELINE="stephy2"
if [ "$1" = "--pipeline" ]; then
    PIPELINE="$2"
    shift 2
fi

PIPELINE_DIR="$STEPHY_ROOT/$PIPELINE"
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

# --- Parse positional args: input_folder(s) + output_folder ---
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 [--pipeline stephy2|CBLV-CNN2|CBLV-GAT2] input_folder_0 [input_folder_1 ...] output_folder"
    exit 1
fi

ARGS=("$@")
NUM_ARGS=${#ARGS[@]}
OUT_FOLDER="${ARGS[$NUM_ARGS-1]}"
INPUT_FOLDERS=("${ARGS[@]:0:$NUM_ARGS-1}")

echo "=== STEPHY: $PIPELINE ==="
echo "Output: $OUT_FOLDER"
echo ""

mkdir -p "$OUT_FOLDER"

for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    DATASET_NAME=$(basename "$INPUT_FOLDER")
    WORK_DIR="$OUT_FOLDER/$DATASET_NAME"
    mkdir -p "$WORK_DIR"

    echo "--- $DATASET_NAME ---"

    # Step 1: Analyze trees (shared — always stephy2/analyze_trees.py)
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

    # Step 3: Train all labels (pipeline-specific train.py -> model.py)
    for LABEL_PAIR in R0:results_r0 Recovery_Rate:results_rr \
                      Source_Sink_Score:results_sss Ancestral_State:results_as; do
        LABEL="${LABEL_PAIR%%:*}"
        OUT_SUBDIR="${LABEL_PAIR##*:}"
        echo "  Training $LABEL..."
        python3 "$PIPELINE_DIR/train.py" \
            --graphs "$GRAPHS_FILE" \
            --num_locations "$NUM_LOCATIONS" \
            --label "$LABEL" \
            --output_dir "$WORK_DIR/$OUT_SUBDIR"
    done

    echo "  Done: $WORK_DIR"
    echo ""
done

echo "=== Complete ==="
