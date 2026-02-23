#!/bin/bash
#
# CBLV-CNN2 Pipeline: CNN + Aux Branch baseline (no graph structure)
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

echo "=== CBLV-CNN2 ==="
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

    # Step 3: Train all labels (label:output_dir pairs)
    for LABEL_PAIR in R0:results_r0 Recovery_Rate:results_rr \
                      Source_Sink_Score:results_sss Ancestral_State:results_as; do
        LABEL="${LABEL_PAIR%%:*}"
        OUT_SUBDIR="${LABEL_PAIR##*:}"
        echo "  Training $LABEL..."
        python3 "$SCRIPT_DIR/train.py" \
            --graphs "$GRAPHS_FILE" \
            --num_locations "$NUM_LOCATIONS" \
            --label "$LABEL" \
            --output_dir "$WORK_DIR/$OUT_SUBDIR"
    done

    echo "  Done: $WORK_DIR"
    echo ""
done

echo "=== Complete ==="
