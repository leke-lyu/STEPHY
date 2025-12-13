#!/usr/bin/env bash
#
# Graph Feature Extraction Pipeline
# Extracts node and edge features from BEAST2 simulation outputs
#
# Usage: ./1_feature_extraction.sh <input_folder>

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

INPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Directory not found: $INPUT_DIR"
    exit 1
fi

# Find trajectory files
cd "$INPUT_DIR" || exit 1
shopt -s nullglob
TRAJ_FILES=(*_beast2.traj)
shopt -u nullglob

TOTAL=${#TRAJ_FILES[@]}

if [ $TOTAL -eq 0 ]; then
    echo "ERROR: No *_beast2.traj files found in $INPUT_DIR"
    exit 1
fi

# Progress bar function
progress_bar() {
    local current=$1
    local total=$2
    local prefix=$3
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    printf "\r[%-${width}s] %3d%% (%d/%d) %s" \
        "$(printf '#%.0s' $(seq 1 $filled 2>/dev/null) 2>/dev/null)" \
        "$percent" "$current" "$total" "$prefix"
}

# Track statistics
SUCCESS=0
FAILED=0
declare -a FAILED_LIST=()

echo "Feature Extraction Pipeline"
echo "==========================="
echo ""

# Process each outbreak
IDX=0
for traj in "${TRAJ_FILES[@]}"; do
    prefix="${traj%_beast2.traj}"
    ((IDX++))

    progress_bar $IDX $TOTAL "$prefix"

    # Run all steps, suppress output
    if python3 "$SCRIPT_DIR/../utils/node_feature.py" \
            "${prefix}_parameter.csv" \
            "${prefix}_beast2.xml" \
            "${prefix}_beast2.traj" \
            "${prefix}_node.csv" >/dev/null 2>&1 && \
       Rscript "$SCRIPT_DIR/../utils/node_feature.R" \
            "${prefix}_node.csv" \
            "${prefix}_beast2.trees" >/dev/null 2>&1 && \
       python3 "$SCRIPT_DIR/../utils/edge_feature.py" \
            "${prefix}_parameter.csv" \
            "${prefix}_beast2.trees" \
            "${prefix}_edge.csv" >/dev/null 2>&1 && \
       python3 "$SCRIPT_DIR/../utils/dtw.py" \
            "${prefix}_edge.csv" \
            "${prefix}_beast2.trees" >/dev/null 2>&1; then
        ((SUCCESS++))
    else
        ((FAILED++))
        FAILED_LIST+=("$prefix")
    fi
done

# Clear progress line and print summary
echo ""
echo ""
echo "==========================="
echo "Summary"
echo "==========================="
echo "Total:     $TOTAL"
echo "Success:   $SUCCESS"
echo "Failed:    $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed outbreaks:"
    for f in "${FAILED_LIST[@]}"; do
        echo "  - $f"
    done
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All outbreaks processed successfully!"
else
    echo "Completed with errors."
    exit 1
fi
