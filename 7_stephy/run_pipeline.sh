#!/bin/bash
#
# STEPHY 7 Pipeline: CBLV-GAT with Epidemiological Features
#
# This pipeline integrates phylogenetic tree features (CBLV) with epidemiological
# data (Initial_Population, Epidemic_Peak, Peak_Timing, Accumulated_Infections)
# to predict R0 and Source/Sink scores.
#
# Usage:
#   bash run_pipeline.sh INPUT_FOLDER1 [INPUT_FOLDER2 ...] OUTPUT_FOLDER
#
# Example:
#   bash run_pipeline.sh /data/500_1_MM0.002 /data/500_1_MM0.005 /output/stephy7
#
# Requirements:
#   - Input folders must contain:
#     * *_beast2.trees files (phylogenetic trees)
#     * *_beast2.traj files (epidemic trajectories)
#     * *_parameter.csv files (ground truth parameters)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 INPUT_FOLDER1 [INPUT_FOLDER2 ...] OUTPUT_FOLDER"
    echo ""
    echo "Example:"
    echo "  $0 /data/500_1_MM0.002 /data/500_1_MM0.005 /output/stephy7"
    exit 1
fi

# Get all arguments
ALL_ARGS=("$@")
NUM_ARGS=${#ALL_ARGS[@]}

# Last argument is output folder
OUTPUT_BASE="${ALL_ARGS[$NUM_ARGS-1]}"

# Everything else is input folders
INPUT_FOLDERS=("${ALL_ARGS[@]:0:$NUM_ARGS-1}")

echo "============================================================"
echo "STEPHY 7 Pipeline: CBLV-GAT with Epidemiological Features"
echo "============================================================"
echo ""
echo "Input folders: ${#INPUT_FOLDERS[@]}"
for folder in "${INPUT_FOLDERS[@]}"; do
    echo "  - $folder"
done
echo "Output base: $OUTPUT_BASE"
echo ""

# Process each input folder
for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    DATASET_NAME=$(basename "$INPUT_FOLDER")
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET_NAME"

    echo ""
    echo "============================================================"
    echo "Processing: $DATASET_NAME"
    echo "============================================================"

    # Step 1: Preprocess trajectory data to generate *_nd.csv files
    echo ""
    echo "[Step 1/3] Preprocessing trajectory data..."
    echo "  Command: python3 $SCRIPT_DIR/preprocess.py $INPUT_FOLDER"
    python3 "$SCRIPT_DIR/preprocess.py" "$INPUT_FOLDER"

    # Step 2: Analyze trees to get parameters
    echo ""
    echo "[Step 2/3] Analyzing tree structure..."
    echo "  Command: python3 $SCRIPT_DIR/analyze_trees.py $INPUT_FOLDER"
    ANALYSIS_OUTPUT=$(python3 "$SCRIPT_DIR/analyze_trees.py" "$INPUT_FOLDER")
    echo "$ANALYSIS_OUTPUT"

    # Parse num_locations and subtree_width from output
    NUM_LOCATIONS=$(echo "$ANALYSIS_OUTPUT" | grep -oP '(?<=--num_locations )\d+')
    SUBTREE_WIDTH=$(echo "$ANALYSIS_OUTPUT" | grep -oP '(?<=--subtree_width )\d+')

    if [ -z "$NUM_LOCATIONS" ] || [ -z "$SUBTREE_WIDTH" ]; then
        echo "Error: Failed to parse parameters from analyze_trees.py"
        exit 1
    fi

    echo ""
    echo "  Extracted parameters:"
    echo "    num_locations: $NUM_LOCATIONS"
    echo "    subtree_width: $SUBTREE_WIDTH"

    # Step 3: Train model
    echo ""
    echo "[Step 3/3] Training CBLV-GAT model..."
    echo "  Command: python3 $SCRIPT_DIR/train.py \\"
    echo "           --input_dir $INPUT_FOLDER \\"
    echo "           --output_dir $OUTPUT_DIR \\"
    echo "           --num_locations $NUM_LOCATIONS \\"
    echo "           --subtree_width $SUBTREE_WIDTH"

    mkdir -p "$OUTPUT_DIR"

    python3 "$SCRIPT_DIR/train.py" \
        --input_dir "$INPUT_FOLDER" \
        --output_dir "$OUTPUT_DIR" \
        --num_locations "$NUM_LOCATIONS" \
        --subtree_width "$SUBTREE_WIDTH"

    echo ""
    echo "Completed: $DATASET_NAME"
    echo "Results saved to: $OUTPUT_DIR"
done

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Output directories:"
for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    DATASET_NAME=$(basename "$INPUT_FOLDER")
    echo "  - $OUTPUT_BASE/$DATASET_NAME"
done
