#!/bin/bash
#
# Simplified Phyddle Pipeline for R0 Estimation
# - Uses phyddle for data formatting (HDF5 tensors)
# - Custom CNN training (no auxiliary data, point estimates only)
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
    echo "[Step 1/5] Converting BEAST2 data to phyddle format..."
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
    echo "[Step 2/5] Analyzing tree sizes..."

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
    # Step 3: Create modified config files
    # ==========================================
    echo "[Step 3/5] Creating config files with tree size parameters..."

    # Create config_format.py for this dataset
    cat > "$WORK_DIR/config_format.py" << EOF
#!/usr/bin/env python3
"""
Phyddle config for Format step.
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

    # Data split
    'prop_test'          : 0.2,

    'verbose'            : 'T',
}
EOF

    # Create config.py for this dataset
    cat > "$WORK_DIR/config.py" << EOF
#!/usr/bin/env python3
"""
Training config for simplified R0 estimation CNN.
Auto-generated for dataset: $DATASET_NAME
"""

args = {
    # Workspace
    'prefix'             : 'r0_est',
    'sim_dir'            : './sim_data',
    'output_dir'         : './output',

    # Data
    'tree_width'         : $MAX_TAXA,
    'num_params'         : 16,
    'prop_val'           : 0.1,

    # Training
    'num_epochs'         : 300,
    'num_early_stop'     : 15,
    'batch_size'         : 64,
    'learning_rate'      : 0.001,
    'activation_func'    : 'relu',
    'num_proc'           : -1,

    # CNN architecture
    'phy_channel_plain'  : [32, 64, 128],
    'phy_channel_stride' : [32, 64],
    'phy_channel_dilate' : [32, 64],
    'phy_kernel_plain'   : [3, 5, 7],
    'phy_kernel_stride'  : [7, 9],
    'phy_kernel_dilate'  : [3, 5],
    'phy_stride_stride'  : [3, 6],
    'phy_dilate_dilate'  : [3, 5],
    'lbl_channel'        : [128, 64, 32],
}
EOF

    echo "Config files created."
    echo ""

    # ==========================================
    # Step 4: Run phyddle formatting
    # ==========================================
    echo "[Step 4/5] Running phyddle formatting (creating HDF5 tensors)..."

    # Change to working directory and run phyddle
    cd "$WORK_DIR"
    python3 -m phyddle -c config_format.py -s F

    echo "Formatting complete."
    echo ""

    # ==========================================
    # Step 5: Run training
    # ==========================================
    echo "[Step 5/5] Training neural network..."

    # Set PYTHONPATH so train.py imports config.py from current directory first
    PYTHONPATH="$WORK_DIR:$SCRIPT_DIR:$PYTHONPATH" python3 "$SCRIPT_DIR/train.py"

    echo "Training complete."
    echo ""

    # Extract results for summary table
    DATASET_NAMES[$DATASET_IDX]="$DATASET_NAME"

    # Parse training history for best epoch
    BEST_EPOCH=$(python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('./output/training_history.csv')
best_idx = np.argmin(df['val_loss'].values)
print(int(df['epoch'].values[best_idx]) + 1)
" 2>/dev/null || echo "N/A")
    BEST_EPOCHS[$DATASET_IDX]="$BEST_EPOCH"

    # Parse test predictions for R², r, MSE
    METRICS=$(python3 -c "
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
df = pd.read_csv('./output/test_predictions.csv')
true_cols = sorted([c for c in df.columns if c.endswith('_true')])
pred_cols = sorted([c for c in df.columns if c.endswith('_pred')])
all_true = np.concatenate([np.exp(df[c].values) for c in true_cols])
all_pred = np.concatenate([np.exp(df[c].values) for c in pred_cols])
r2 = r2_score(all_true, all_pred)
r = np.corrcoef(all_true, all_pred)[0, 1]
mse = np.mean((all_true - all_pred) ** 2)
print(f'{r2:.4f},{r:.4f},{mse:.4f}')
" 2>/dev/null || echo "N/A,N/A,N/A")

    R2_VALUES[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f1)
    R_VALUES[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f2)
    MSE_VALUES[$DATASET_IDX]=$(echo "$METRICS" | cut -d',' -f3)

    DATASET_IDX=$((DATASET_IDX + 1))

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
echo ""

# Print summary table
echo "Results Summary:"
echo "--------------------------------------------------------------------------------"
printf "%-25s | %8s | %8s | %8s | %10s\n" "Dataset" "R²" "r" "MSE" "Best Epoch"
echo "--------------------------------------------------------------------------------"
for i in "${!DATASET_NAMES[@]}"; do
    printf "%-25s | %8s | %8s | %8s | %10s\n" \
        "${DATASET_NAMES[$i]}" \
        "${R2_VALUES[$i]}" \
        "${R_VALUES[$i]}" \
        "${MSE_VALUES[$i]}" \
        "${BEST_EPOCHS[$i]}"
done
echo "--------------------------------------------------------------------------------"