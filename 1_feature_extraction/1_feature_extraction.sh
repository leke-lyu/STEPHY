#!/bin/bash
#
# Graph Feature Extraction Pipeline
# Extracts node and edge features from BEAST2 simulation outputs
#
# Usage: ./1_feature_extraction.sh <input_folder>
#

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

INPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Directory not found: $INPUT_DIR"
    exit 1
fi

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((width * current / total))

    printf "\r[%-${width}s] %d%% (%d/%d)" \
        "$(printf '=%.0s' $(seq 1 $filled))" \
        "$percent" "$current" "$total"
}

# Header
echo "========================================="
echo "GRAPH FEATURE EXTRACTION"
echo "========================================="
echo "Input: $INPUT_DIR"
echo ""

# Find trajectory files
cd "$INPUT_DIR" || exit 1
shopt -s nullglob
TRAJ_FILES=(*_beast2.traj)
shopt -u nullglob

if [ ${#TRAJ_FILES[@]} -eq 0 ]; then
    echo "ERROR: No *_beast2.traj files found in $INPUT_DIR"
    exit 1
fi

NUM_FILES=${#TRAJ_FILES[@]}
echo "Found $NUM_FILES outbreak(s)"
echo ""

# Process files - 3-step pipeline per outbreak
FAILED=0
for i in "${!TRAJ_FILES[@]}"; do
    prefix="${TRAJ_FILES[$i]%_beast2.traj}"
    FILE_FAILED=0

    # Step 1: Extract node features (population, R0, epidemic metrics)
    python3 "$SCRIPT_DIR/../utils/node_feature.py" \
        "${prefix}_parameter.csv" \
        "${prefix}_beast2.xml" \
        "${prefix}_beast2.traj" \
        "${prefix}_node.csv" >/dev/null 2>&1 || FILE_FAILED=1

    # Step 2: Extract edge features - migration rates
    python3 "$SCRIPT_DIR/../utils/edge_feature.py" \
        "${prefix}_parameter.csv" \
        "${prefix}_beast2.trees" \
        "${prefix}_edge.csv" >/dev/null 2>&1 || FILE_FAILED=1

    # Step 3: Add patristic distances to edge CSV (in-place update)
    Rscript "$SCRIPT_DIR/../utils/edge_feature.R" \
        "${prefix}_edge.csv" \
        "${prefix}_beast2.trees" >/dev/null 2>&1 || FILE_FAILED=1

    # Track failures and update progress
    [ $FILE_FAILED -eq 1 ] && FAILED=$((FAILED + 1))
    show_progress $((i + 1)) $NUM_FILES
done

echo ""
echo ""

# Summary
if [ $FAILED -eq 0 ]; then
    echo "✓ All $NUM_FILES file(s) processed successfully!"
else
    echo "⚠ Success: $((NUM_FILES - FAILED)), Failed: $FAILED"
fi

echo "Output: $INPUT_DIR/*_node.csv, *_edge.csv"
