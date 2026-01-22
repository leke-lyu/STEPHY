#!/bin/bash
#
# Batch training pipeline for STEPHY (CBLV-GAT) R0 estimation - Base Model
#
# Usage:
#   bash run_pipeline.sh inputfolder_0 inputfolder_1 ... outfolder
#
# Example:
#   bash run_pipeline.sh /Users/lukelyu/Desktop/epidata/500_1_MM0.002 \
#                        /Users/lukelyu/Desktop/epidata/500_1_MM0.005 \
#                        /Users/lukelyu/Desktop/epidata/stephy
#

set -e  # Exit on error

# Prevent Python from creating __pycache__ folders
export PYTHONDONTWRITEBYTECODE=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check minimum arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 inputfolder_0 [inputfolder_1 ...] outfolder"
    echo "Example: bash $0 /path/to/epidata/500_1_MM0.002 /path/to/epidata/500_1_MM0.005 ./results"
    exit 1
fi

# Parse arguments: all but last are input folders, last is output folder
ARGS=("$@")
NUM_ARGS=${#ARGS[@]}
OUT_FOLDER="${ARGS[$NUM_ARGS-1]}"
INPUT_FOLDERS=("${ARGS[@]:0:$NUM_ARGS-1}")

echo "=============================================="
echo "STEPHY (CBLV-GAT) R0 Estimation - Base Model"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Output folder: $OUT_FOLDER"
echo "Number of datasets: ${#INPUT_FOLDERS[@]}"
echo ""

# Create output folder
mkdir -p "$OUT_FOLDER"

# Process each input folder
for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Processing: $INPUT_FOLDER"
    echo "=============================================="

    # Extract dataset name from input folder path
    DATASET_NAME=$(basename "$INPUT_FOLDER")
    WORK_DIR="$OUT_FOLDER/$DATASET_NAME"

    echo "Dataset name: $DATASET_NAME"
    echo "Working directory: $WORK_DIR"
    echo ""

    # Create working directory
    mkdir -p "$WORK_DIR"

    # ==========================================
    # Step 1: Analyze trees to get parameters
    # ==========================================
    echo "[Step 1/2] Analyzing trees..."

    ANALYZE_OUTPUT=$(python3 "$SCRIPT_DIR/analyze_trees.py" "$INPUT_FOLDER")
    echo "$ANALYZE_OUTPUT"

    # Parse num_locations and subtree_width from output
    NUM_LOCATIONS=$(echo "$ANALYZE_OUTPUT" | grep "\-\-num_locations" | awk '{print $2}')
    SUBTREE_WIDTH=$(echo "$ANALYZE_OUTPUT" | grep "\-\-subtree_width" | awk '{print $2}')

    if [ -z "$NUM_LOCATIONS" ] || [ -z "$SUBTREE_WIDTH" ]; then
        echo "Error: Could not parse parameters from analyze_trees.py output"
        echo "Skipping dataset: $DATASET_NAME"
        continue
    fi

    echo ""
    echo "Parsed parameters:"
    echo "  num_locations: $NUM_LOCATIONS"
    echo "  subtree_width: $SUBTREE_WIDTH"
    echo ""

    # ==========================================
    # Step 2: Train model
    # ==========================================
    echo "[Step 2/2] Training CBLV-GAT model..."

    python3 "$SCRIPT_DIR/train.py" \
        --num_locations "$NUM_LOCATIONS" \
        --subtree_width "$SUBTREE_WIDTH" \
        --input_dir "$INPUT_FOLDER" \
        --output_dir "$WORK_DIR"

    echo "Training complete."

    echo "=============================================="
    echo "Completed: $DATASET_NAME"
    echo "Results saved to: $WORK_DIR"
    echo "=============================================="
    echo ""
done

echo ""
echo "=============================================="
echo "All datasets processed!"
echo "=============================================="
echo "Output folder: $OUT_FOLDER"
