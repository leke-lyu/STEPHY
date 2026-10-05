#!/bin/bash
# ============================================================
# BEAST2 TREE FILE POST-PROCESSOR
# ============================================================
# Removes unnecessary blocks from BEAST2 tree files for cleaner output.
#
# What it removes:
#   1. Taxa block: Contains taxon definitions that are redundant for our use
#   2. Translate block: Maps numeric IDs to sample names (not needed)
#
# The resulting tree file is smaller and easier to parse for phylogenetic
# analysis while retaining all essential tree topology and timing information.
# ============================================================

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tree_file>" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  $0 0_beast2.trees" >&2
    exit 1
fi

TREE_FILE="$1"

# Validate input file exists
if [ ! -f "$TREE_FILE" ]; then
    echo "Error: File '$TREE_FILE' not found" >&2
    exit 1
fi

# Create temporary file for processing
TEMP_FILE="${TREE_FILE}.tmp"

# Remove taxa and translate blocks using sed
# Pattern 1: Delete from "Begin taxa;" to "End;" (inclusive)
# Pattern 2: Delete from "Translate" to semicolon (inclusive)
sed -e '/Begin taxa;/,/^End;$/d' \
    -e '/^\tTranslate$/,/;$/d' \
    "$TREE_FILE" > "$TEMP_FILE"

# Replace original file with cleaned version
mv "$TEMP_FILE" "$TREE_FILE"
