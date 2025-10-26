#!/bin/bash
# Script to remove taxa and Translate blocks from BEAST2 tree files

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tree_file>"
    exit 1
fi

TREE_FILE="$1"

# Check if file exists
if [ ! -f "$TREE_FILE" ]; then
    echo "Error: File $TREE_FILE not found"
    exit 1
fi

# Create a temporary file
TEMP_FILE="${TREE_FILE}.tmp"

# Use sed to:
# 1. Delete everything from "Begin taxa;" to "End;" (inclusive)
# 2. Delete everything from "Translate" to the semicolon after the last leaf entry
sed -e '/Begin taxa;/,/^End;$/d' \
    -e '/^\tTranslate$/,/;$/d' \
    "$TREE_FILE" > "$TEMP_FILE"

# Replace original file with edited version
mv "$TEMP_FILE" "$TREE_FILE"
