#!/bin/bash
#
# Phyddle Pipeline for R0 and Source_Sink_Score Estimation
#
# Features:
#   - Trains separate models for R0 and SSS
#   - Format once, Train/Estimate separately (optimized)
#   - Auto-detects num_locations and tree sizes
#
# Usage:
#   bash run_pipeline.sh inputfolder_0 [inputfolder_1 ...] outfolder

set -e
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 inputfolder_0 [inputfolder_1 ...] outfolder"
    exit 1
fi

ARGS=("$@")
NUM_ARGS=${#ARGS[@]}
OUT_FOLDER="${ARGS[$NUM_ARGS-1]}"
INPUT_FOLDERS=("${ARGS[@]:0:$NUM_ARGS-1}")

echo "=============================================="
echo "Phyddle R0 + SSS Estimation Pipeline"
echo "=============================================="
echo "Output: $OUT_FOLDER"
echo "Datasets: ${#INPUT_FOLDERS[@]}"
echo ""

mkdir -p "$OUT_FOLDER"

# Generate param_est entries
generate_param_est() {
    local num_locs=$1 prefix=$2 result=""
    for ((i=0; i<num_locs; i++)); do
        result+="        '${prefix}_${i}': 'num',\n"
    done
    echo -e "$result"
}

generate_all_param_est() {
    local num_locs=$1 result=""
    for ((i=0; i<num_locs; i++)); do result+="        'R0_${i}': 'num',\n"; done
    for ((i=0; i<num_locs; i++)); do result+="        'SSS_${i}': 'num',\n"; done
    echo -e "$result"
}

# Generate training config
generate_train_config() {
    local param_est=$1
    cat << CONF
args = {
    'prefix'             : 'all_labels',
    'sim_prefix'         : 'sim',
    'sim_dir'            : '../sim_data',
    'fmt_dir'            : './format_output',
    'trn_dir'            : './train_output',
    'est_dir'            : './estimate_output',
    'use_parallel'       : 'T',
    'use_cuda'           : 'F',
    'num_proc'           : -1,
    'no_emp'             : 'T',
    'encode_all_sim'     : 'T',
    'num_char'           : $NUM_LOCATIONS,
    'num_states'         : 2,
    'min_num_taxa'       : $MIN_TAXA,
    'max_num_taxa'       : $MAX_TAXA,
    'tree_width'         : $MAX_TAXA,
    'tree_encode'        : 'serial',
    'brlen_encode'       : 'height_brlen',
    'char_encode'        : 'integer',
    'char_format'        : 'nexus',
    'tensor_format'      : 'hdf5',
    'param_est'          : {
${param_est}    },
    'param_data'         : {},
    'num_epochs'         : 500,
    'num_early_stop'     : 25,
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
CONF
}

# Process each input folder
for INPUT_FOLDER in "${INPUT_FOLDERS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Processing: $INPUT_FOLDER"
    echo "=============================================="

    DATASET_NAME=$(basename "$INPUT_FOLDER")
    WORK_DIR="$OUT_FOLDER/$DATASET_NAME"
    SIM_DATA_DIR="$WORK_DIR/sim_data"
    FORMAT_DIR="$WORK_DIR/format_output"

    mkdir -p "$WORK_DIR" "$SIM_DATA_DIR"

    # Step 1: Convert to phyddle format
    echo "[Step 1/5] Converting BEAST2 data..."
    CONVERT_OUTPUT=$(python3 "$SCRIPT_DIR/convert_to_phyddle.py" \
        --input_dir "$INPUT_FOLDER" \
        --output_dir "$SIM_DATA_DIR" \
        --max_trees_per_sim 1 2>&1)
    echo "$CONVERT_OUTPUT"

    NUM_LOCATIONS=$(echo "$CONVERT_OUTPUT" | grep "^PHYDDLE_PARAM num_locations" | awk '{print $3}')
    MIN_TAXA=$(echo "$CONVERT_OUTPUT" | grep "^PHYDDLE_PARAM min_taxa" | awk '{print $3}')
    MAX_TAXA=$(echo "$CONVERT_OUTPUT" | grep "^PHYDDLE_PARAM max_taxa" | awk '{print $3}')

    if [ -z "$NUM_LOCATIONS" ] || [ -z "$MIN_TAXA" ] || [ -z "$MAX_TAXA" ]; then
        echo "Error: Could not parse parameters. Skipping $DATASET_NAME"
        continue
    fi
    echo "Parsed: num_locations=$NUM_LOCATIONS, min_taxa=$MIN_TAXA, max_taxa=$MAX_TAXA"

    # Step 2: Format once
    echo ""
    echo "[Step 2/5] Formatting data..."
    ALL_PARAM_EST=$(generate_all_param_est "$NUM_LOCATIONS")
    cat > "$WORK_DIR/format_config.py" << EOF
#!/usr/bin/env python3
args = {
    'prefix'             : 'all_labels',
    'sim_prefix'         : 'sim',
    'sim_dir'            : './sim_data',
    'fmt_dir'            : './format_output',
    'trn_dir'            : './train_output',
    'est_dir'            : './estimate_output',
    'use_parallel'       : 'T',
    'use_cuda'           : 'F',
    'num_proc'           : -1,
    'no_emp'             : 'T',
    'encode_all_sim'     : 'T',
    'num_char'           : $NUM_LOCATIONS,
    'num_states'         : 2,
    'min_num_taxa'       : $MIN_TAXA,
    'max_num_taxa'       : $MAX_TAXA,
    'tree_width'         : $MAX_TAXA,
    'tree_encode'        : 'serial',
    'brlen_encode'       : 'height_brlen',
    'char_encode'        : 'integer',
    'char_format'        : 'nexus',
    'tensor_format'      : 'hdf5',
    'param_est'          : {
$ALL_PARAM_EST    },
    'param_data'         : {},
    'prop_test'          : 0.2,
    'verbose'            : 'T',
}
EOF
    cd "$WORK_DIR" && python3 -m phyddle -c format_config.py -s F && cd - > /dev/null

    # Step 3: Train R0 model
    echo ""
    echo "[Step 3/5] Training R0 model..."
    R0_DIR="$WORK_DIR/results_r0"
    mkdir -p "$R0_DIR"
    cp -r "$FORMAT_DIR" "$R0_DIR/format_output"
    R0_PARAM_EST=$(generate_param_est "$NUM_LOCATIONS" "R0")
    echo "#!/usr/bin/env python3" > "$R0_DIR/config.py"
    generate_train_config "$R0_PARAM_EST" >> "$R0_DIR/config.py"
    cd "$R0_DIR" && python3 -m phyddle -c config.py -s TE && cd - > /dev/null

    # Step 4: Train SSS model
    echo ""
    echo "[Step 4/5] Training SSS model..."
    SSS_DIR="$WORK_DIR/results_sss"
    mkdir -p "$SSS_DIR"
    cp -r "$FORMAT_DIR" "$SSS_DIR/format_output"
    SSS_PARAM_EST=$(generate_param_est "$NUM_LOCATIONS" "SSS")
    echo "#!/usr/bin/env python3" > "$SSS_DIR/config.py"
    generate_train_config "$SSS_PARAM_EST" >> "$SSS_DIR/config.py"
    cd "$SSS_DIR" && python3 -m phyddle -c config.py -s TE && cd - > /dev/null

    # Step 5: Summary
    echo ""
    echo "[Step 5/5] Complete: $DATASET_NAME"
    echo "  num_locations: $NUM_LOCATIONS, tree_width: $MAX_TAXA"
    echo "  R0:  $R0_DIR"
    echo "  SSS: $SSS_DIR"
done

echo ""
echo "=============================================="
echo "All datasets processed!"
echo "Output: $OUT_FOLDER"
echo "=============================================="
