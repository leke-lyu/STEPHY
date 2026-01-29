#!/bin/bash
# Node Feature Extraction Script
# Usage: bash 1_nf_extraction.sh folder1 folder2 ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NF_SCRIPT="$SCRIPT_DIR/../utils/nf.py"

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <folder1> [folder2] ..."
    echo "Example: bash $0 /path/to/data1 /path/to/data2"
    exit 1
fi

if [ ! -f "$NF_SCRIPT" ]; then
    echo "Error: nf.py not found at $NF_SCRIPT"
    exit 1
fi

for folder in "$@"; do
    if [ -d "$folder" ]; then
        echo "Processing: $folder"
        python3 "$NF_SCRIPT" "$folder" --force
        echo ""
    else
        echo "Warning: Folder not found: $folder"
    fi
done

echo "Done."
