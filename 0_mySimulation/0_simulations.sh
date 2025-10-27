#!/bin/bash

# ============================================================
# EPIDEMIC OUTBREAK SIMULATION PIPELINE
# Generates multiple outbreak scenarios with BEAST2
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

# Location and population settings
NUM_LOCS=10                        # Number of outbreak locations/demes
POP_MIN=1000                       # Minimum population size
POP_MAX=9000                     # Maximum population size
SHARED_POP_SIZE=false              # true = uniform, false = random per location
SEED_LOCATION="None"               # "None" = random, or integer (0 to NUM_LOCS-1)

# R0 settings (basic reproduction number)
R0_MIN=2
R0_MAX=5
SHARED_R0=false                    # true = uniform, false = random per location
MAX_R0_DIFF=1                      # Maximum difference across locations

# Recovery rate settings (mu)
MU_MIN=0.02
MU_MAX=0.05
SHARED_RECOVERY_RATE=false         # true = uniform, false = random per location
MAX_MU_DIFF=0.01                   # Maximum difference across locations

# Sample rate settings (shared across all locations)
SAMPLE_MIN=0.0003
SAMPLE_MAX=0.0004

# Migration rate settings
MIGRATION_MIN=0.0001
MIGRATION_MAX=0.002
SHARED_MIGRATION_RATE=false        # true = uniform, false = random per pair

# Simulation time settings
SIM_TIME_MIN=4
SIM_TIME_MAX=8
TIME_UNITS="recovery_period"       # "recovery_period" or "arbitrary"

# Random seed and simulation count
RANDOM_SEED="None"                 # "None" = random, or integer for reproducibility
NUM_SIMS=1                        # Number of BEAST2 simulations per outbreak

# Export configuration
export NUM_LOCS SEED_LOCATION RANDOM_SEED TIME_UNITS NUM_SIMS
export POP_MIN POP_MAX SHARED_POP_SIZE
export R0_MIN R0_MAX SHARED_R0 MAX_R0_DIFF
export MU_MIN MU_MAX SHARED_RECOVERY_RATE MAX_MU_DIFF
export SAMPLE_MIN SAMPLE_MAX
export MIGRATION_MIN MIGRATION_MAX SHARED_MIGRATION_RATE
export SIM_TIME_MIN SIM_TIME_MAX


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_outbreaks> [output_folder]"
    echo ""
    echo "Examples:"
    echo "  $0 10          # Generate 10 outbreaks in current directory"
    echo "  $0 10 results  # Generate 10 outbreaks in results/ folder"
    exit 1
fi

NUM_OUTBREAKS=$1
OUTPUT_DIR=${2:-.}


# ============================================================
# SETUP
# ============================================================

# Save the script directory before changing directories
SCRIPT_DIR="$(pwd)"

mkdir -p "$OUTPUT_DIR/logs"

echo "Running $NUM_OUTBREAKS outbreak(s) (each with $NUM_SIMS BEAST2 simulations)..."
echo "Output directory: $OUTPUT_DIR"
echo ""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))

    printf "\rProgress: ["
    printf "%${completed}s" | tr ' ' '='
    printf "%${remaining}s" | tr ' ' '-'
    printf "] %d%% (%d/%d)" $percentage $current $total
}

run_outbreak_simulation() {
    local idx=$1
    local xml_file="${idx}_beast2.xml"
    local param_file="${idx}_parameter.csv"
    local log_file="logs/sim_${idx}.log"
    local tree_file="${idx}_beast2.trees"
    local traj_file="${idx}_beast2.traj"
    local edge_file="${idx}_edge.csv"
    local node_file="${idx}_node.csv"

    # Generate parameters and XML
    python3 "$SCRIPT_DIR/scripts/xml_Generation.py" "$xml_file" "$param_file" > "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Run BEAST2 (generates .traj and .trees files in current directory)
    beast2 "$xml_file" >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Edit tree file (remove taxa and Translate blocks)
    bash "$SCRIPT_DIR/scripts/edit_tree.sh" "$tree_file" >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Get edge feature
    Rscript "$SCRIPT_DIR/scripts/edge_feature.R" "$tree_file" "$edge_file" >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Get node feature (only if edge feature succeeded)
    python3 "$SCRIPT_DIR/scripts/node_feature.py" "$xml_file" "$traj_file" "$node_file" >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    return 0
}


# ============================================================
# MAIN LOOP
# ============================================================

FAILED_OUTBREAKS=()

for i in $(seq 0 $((NUM_OUTBREAKS - 1))); do
    # Change to output directory so BEAST2 files are created there
    # Redirect stderr to /dev/null to suppress segfault messages
    (cd "$OUTPUT_DIR" && run_outbreak_simulation $i) 2>/dev/null

    if [ $? -ne 0 ]; then
        FAILED_OUTBREAKS+=($i)
    fi

    show_progress $((i + 1)) $NUM_OUTBREAKS
done


# ============================================================
# COMPLETION
# ============================================================

echo ""
echo ""

SUCCESSFUL=$((NUM_OUTBREAKS - ${#FAILED_OUTBREAKS[@]}))

if [ ${#FAILED_OUTBREAKS[@]} -eq 0 ]; then
    echo "✓ All $NUM_OUTBREAKS outbreak(s) completed successfully!"
else
    echo "⚠ Completed with $SUCCESSFUL successful and ${#FAILED_OUTBREAKS[@]} failed outbreak(s)"
    echo "Failed outbreaks: ${FAILED_OUTBREAKS[*]}"
    echo "Check logs in $OUTPUT_DIR/logs for details"
fi

echo "Output: $OUTPUT_DIR"
echo "Logs:   $OUTPUT_DIR/logs"
