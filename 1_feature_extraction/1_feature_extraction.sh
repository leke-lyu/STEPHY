#!/bin/bash

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

INPUT_DIR=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
TRAJ_FILES=($(ls *_beast2.traj 2>/dev/null))

if [ ${#TRAJ_FILES[@]} -eq 0 ]; then
    echo "ERROR: No *_beast2.traj files found"
    exit 1
fi

NUM_FILES=${#TRAJ_FILES[@]}
echo "Found $NUM_FILES outbreak(s)"
echo ""

# Process files
FAILED=0
for i in "${!TRAJ_FILES[@]}"; do
    prefix="${TRAJ_FILES[$i]%_beast2.traj}"

    python3 "$SCRIPT_DIR/../utils/node_feature.py" \
        "${prefix}_parameter.csv" \
        "${prefix}_beast2.xml" \
        "${prefix}_beast2.traj" \
        "${prefix}_node.csv" >/dev/null 2>&1 || FAILED=$((FAILED + 1))

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

echo "Output: $INPUT_DIR/*_node.csv"
