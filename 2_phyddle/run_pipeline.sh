#!/bin/bash
#
# Full Phyddle Pipeline for R0 Estimation
# - Uses phyddle library for formatting, training, and estimation
# - Includes auxiliary data and confidence prediction intervals (CPI)
#
# Usage:
#   bash run_pipeline.sh inputfolder_0 [inputfolder_1 ...] outfolder
#
# Example:
#   bash run_pipeline.sh /path/to/epidata/500_1_MM0.002 ./results
#

set -e  # Exit on error

# Get the directory where this script is located (contains Python scripts and configs)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check minimum arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 inputfolder_0 [inputfolder_1 ...] outfolder"
    echo "Example: bash $0 /path/to/epidata/500_1_MM0.002 /path/to/epidata/500_1_MM0.003 ./results"
    exit 1
fi

# Parse arguments: all but last are input folders, last is output folder
ARGS=("$@")
NUM_ARGS=${#ARGS[@]}
OUT_FOLDER="${ARGS[$NUM_ARGS-1]}"
INPUT_FOLDERS=("${ARGS[@]:0:$NUM_ARGS-1}")

echo "=============================================="
echo "Phyddle R0 Estimation Batch Pipeline"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Output folder: $OUT_FOLDER"
echo "Number of datasets: ${#INPUT_FOLDERS[@]}"
echo ""

# Create output folder if it doesn't exist
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
    SIM_DATA_DIR="$WORK_DIR/sim_data"

    echo "Dataset name: $DATASET_NAME"
    echo "Working directory: $WORK_DIR"
    echo ""

    # Create working directory structure
    mkdir -p "$WORK_DIR"
    mkdir -p "$SIM_DATA_DIR"

    # ==========================================
    # Step 1: Convert to phyddle format
    # ==========================================
    echo "[Step 1/4] Converting BEAST2 data to phyddle format..."
    python3 "$SCRIPT_DIR/convert_to_phyddle.py" \
        --input_dir "$INPUT_FOLDER" \
        --output_dir "$SIM_DATA_DIR" \
        --num_locations 16 \
        --max_trees_per_sim 1

    echo "Conversion complete."
    echo ""

    # ==========================================
    # Step 2: Get tree sizes
    # ==========================================
    echo "[Step 2/4] Analyzing tree sizes..."

    # Run tree_size.py and capture output
    TREE_SIZE_OUTPUT=$(python3 "$SCRIPT_DIR/tree_size.py" "$SIM_DATA_DIR")
    echo "$TREE_SIZE_OUTPUT"

    # Parse min and max tree sizes from output
    MIN_TAXA=$(echo "$TREE_SIZE_OUTPUT" | grep "^Min:" | awk '{print $2}')
    MAX_TAXA=$(echo "$TREE_SIZE_OUTPUT" | grep "^Max:" | awk '{print $2}')

    if [ -z "$MIN_TAXA" ] || [ -z "$MAX_TAXA" ]; then
        echo "Error: Could not parse tree sizes from tree_size.py output"
        echo "Skipping dataset: $DATASET_NAME"
        continue
    fi

    echo "Min taxa: $MIN_TAXA"
    echo "Max taxa: $MAX_TAXA"
    echo ""

    # ==========================================
    # Step 3: Create modified config file
    # ==========================================
    echo "[Step 3/4] Creating config file with tree size parameters..."

    # Create config.py for this dataset
    cat > "$WORK_DIR/config.py" << EOF
#!/usr/bin/env python3
"""
Phyddle config for R0 estimation.
Auto-generated for dataset: $DATASET_NAME
"""

args = {
    # Workspace
    'prefix'             : 'r0_est',
    'sim_prefix'         : 'sim',
    'sim_dir'            : './sim_data',
    'fmt_dir'            : './format_output',
    'trn_dir'            : './train_output',
    'est_dir'            : './estimate_output',

    # Analysis
    'use_parallel'       : 'T',
    'use_cuda'           : 'F',
    'num_proc'           : -1,
    'no_emp'             : 'T',

    # Format
    'encode_all_sim'     : 'T',
    'num_char'           : 16,
    'num_states'         : 2,
    'min_num_taxa'       : $MIN_TAXA,
    'max_num_taxa'       : $MAX_TAXA,
    'tree_width'         : $MAX_TAXA,
    'tree_encode'        : 'serial',
    'brlen_encode'       : 'height_brlen',
    'char_encode'        : 'integer',
    'char_format'        : 'nexus',
    'tensor_format'      : 'hdf5',

    # Parameters to estimate
    'param_est'          : {
        'log_R0_0'  : 'num',
        'log_R0_1'  : 'num',
        'log_R0_2'  : 'num',
        'log_R0_3'  : 'num',
        'log_R0_4'  : 'num',
        'log_R0_5'  : 'num',
        'log_R0_6'  : 'num',
        'log_R0_7'  : 'num',
        'log_R0_8'  : 'num',
        'log_R0_9'  : 'num',
        'log_R0_10' : 'num',
        'log_R0_11' : 'num',
        'log_R0_12' : 'num',
        'log_R0_13' : 'num',
        'log_R0_14' : 'num',
        'log_R0_15' : 'num',
    },
    'param_data'         : {},

    # Train
    'num_epochs'         : 300,
    'num_early_stop'     : 15,
    'trn_batch_size'     : 64,
    'prop_test'          : 0.2,
    'prop_val'           : 0.1,
    'prop_cal'           : 0.1,
    'cpi_coverage'       : 0.95,
    'cpi_asymmetric'     : 'T',
    'loss_numerical'     : 'mse',
    'optimizer'          : 'adam',
    'learning_rate'      : 0.001,
    'activation_func'    : 'relu',
    'log_offset'         : 1.0,

    # CNN architecture
    'phy_channel_plain'  : [32, 64, 128],
    'phy_channel_stride' : [32, 64],
    'phy_channel_dilate' : [32, 64],
    'phy_kernel_plain'   : [3, 5, 7],
    'phy_kernel_stride'  : [7, 9],
    'phy_kernel_dilate'  : [3, 5],
    'phy_stride_stride'  : [3, 6],
    'phy_dilate_dilate'  : [3, 5],
    'aux_channel'        : [128, 64, 32],
    'lbl_channel'        : [128, 64, 32],

    'verbose'            : 'T',
}
EOF

    echo "Config file created."
    echo ""

    # ==========================================
    # Step 4: Run phyddle (Format, Train, Estimate)
    # ==========================================
    echo "[Step 4/4] Running phyddle (Format, Train, Estimate)..."

    # Change to working directory and run phyddle
    cd "$WORK_DIR"
    python3 -m phyddle -c config.py -s FTE

    echo "Phyddle complete."

    # Return to original directory
    cd - > /dev/null

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