#!/bin/bash
#
# STEPHY Pipeline: Phylogeny-only Model
# Usage: bash run_pipeline.sh inputfolder_0 [inputfolder_1 ...] outfolder

set -e
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 inputfolder_0 [inputfolder_1 ...] outfolder"
    exit 1
fi

ARGS=("$@")
NUM_ARGS=${#ARGS[@]}
OUT_FOLDER="${ARGS[$NUM_ARGS-1]}"
INPUT_FOLDERS=("${ARGS[@]:0:$NUM_ARGS-1}")

echo "=== STEPHY ==="
echo "Output: $OUT_FOLDER"
echo ""

mkdir -p "$OUT_FOLDER"

for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    DATASET_NAME=$(basename "$INPUT_FOLDER")
    WORK_DIR="$OUT_FOLDER/$DATASET_NAME"
    mkdir -p "$WORK_DIR"

    echo "--- $DATASET_NAME ---"

    # Step 1: Analyze trees
    ANALYZE_OUTPUT=$(python3 "$SCRIPT_DIR/analyze_trees.py" "$INPUT_FOLDER" 2>&1)
    NUM_LOCATIONS=$(echo "$ANALYZE_OUTPUT" | grep "\-\-num_locations" | awk '{print $2}')
    SUBTREE_WIDTH=$(echo "$ANALYZE_OUTPUT" | grep "\-\-subtree_width" | awk '{print $2}')

    if [ -z "$NUM_LOCATIONS" ] || [ -z "$SUBTREE_WIDTH" ]; then
        echo "  Error: Could not parse tree parameters. Skipping."
        continue
    fi
    echo "  Params: num_locations=$NUM_LOCATIONS, subtree_width=$SUBTREE_WIDTH"

    # Step 2: Build graphs
    GRAPHS_FILE="$WORK_DIR/graphs.pt"
    echo "  Building graphs..."
    python3 "$SCRIPT_DIR/build_graphs.py" \
        --input_dir "$INPUT_FOLDER" \
        --subtree_width "$SUBTREE_WIDTH" \
        --output "$GRAPHS_FILE"

    # Step 3: Train R0
    echo "  Training R0..."
    python3 "$SCRIPT_DIR/train.py" \
        --graphs "$GRAPHS_FILE" \
        --num_locations "$NUM_LOCATIONS" \
        --label R0 \
        --output_dir "$WORK_DIR/results_r0"

    # Step 4: Train Source_Sink_Score
    echo "  Training Source_Sink_Score..."
    python3 "$SCRIPT_DIR/train.py" \
        --graphs "$GRAPHS_FILE" \
        --num_locations "$NUM_LOCATIONS" \
        --label Source_Sink_Score \
        --output_dir "$WORK_DIR/results_sss"

    # Step 5: Train Recovery_Rate
    echo "  Training Recovery_Rate..."
    python3 "$SCRIPT_DIR/train.py" \
        --graphs "$GRAPHS_FILE" \
        --num_locations "$NUM_LOCATIONS" \
        --label Recovery_Rate \
        --output_dir "$WORK_DIR/results_rr"

    # Step 6: Train Ancestral_State (classification)
    echo "  Training Ancestral_State..."
    python3 "$SCRIPT_DIR/train.py" \
        --graphs "$GRAPHS_FILE" \
        --num_locations "$NUM_LOCATIONS" \
        --label Ancestral_State \
        --output_dir "$WORK_DIR/results_as"

    echo "  Done: $WORK_DIR"
    echo ""
done

echo "=== Complete ==="
