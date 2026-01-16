#!/bin/bash
#
# Batch training pipeline for STEPHY (CBLV-GAT) R0 estimation
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
echo "STEPHY (CBLV-GAT) R0 Estimation Pipeline"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Output folder: $OUT_FOLDER"
echo "Number of datasets: ${#INPUT_FOLDERS[@]}"
echo ""

# Create output folder
mkdir -p "$OUT_FOLDER"

# Arrays to store results for summary table
declare -a DATASET_NAMES
declare -a R2_VALUES
declare -a R_VALUES
declare -a MSE_VALUES
declare -a BEST_EPOCHS
DATASET_IDX=0

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

    echo ""
    echo "Training complete."
    echo ""

    # Extract results for summary table
    DATASET_NAMES[$DATASET_IDX]="$DATASET_NAME"

    # Parse summary.json for metrics
    METRICS=$(python3 -c "
import json
with open('$WORK_DIR/summary.json') as f:
    s = json.load(f)
print(f\"{s['test_r2']:.4f},{s['test_corr']:.4f},{s['test_mse']:.4f},{s['best_epoch']}\")
" 2>/dev/null || echo "N/A,N/A,N/A,N/A")

    R2_VALUES[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f1)
    R_VALUES[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f2)
    MSE_VALUES[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f3)
    BEST_EPOCHS[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f4)

    DATASET_IDX=$((DATASET_IDX + 1))

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
echo ""

# Print summary table
echo "Results Summary:"
echo "--------------------------------------------------------------------------------"
printf "%-25s | %8s | %8s | %10s | %10s\n" "Dataset" "R²" "r" "MSE" "Best Epoch"
echo "--------------------------------------------------------------------------------"
for i in "${!DATASET_NAMES[@]}"; do
    printf "%-25s | %8s | %8s | %10s | %10s\n" \
        "${DATASET_NAMES[$i]}" \
        "${R2_VALUES[$i]}" \
        "${R_VALUES[$i]}" \
        "${MSE_VALUES[$i]}" \
        "${BEST_EPOCHS[$i]}"
done
echo "--------------------------------------------------------------------------------"

# Save summary to CSV
SUMMARY_CSV="$OUT_FOLDER/summary_all.csv"
echo "Dataset,R2,r,MSE,Best_Epoch" > "$SUMMARY_CSV"
for i in "${!DATASET_NAMES[@]}"; do
    echo "${DATASET_NAMES[$i]},${R2_VALUES[$i]},${R_VALUES[$i]},${MSE_VALUES[$i]},${BEST_EPOCHS[$i]}" >> "$SUMMARY_CSV"
done
echo ""
echo "Summary saved to: $SUMMARY_CSV"
