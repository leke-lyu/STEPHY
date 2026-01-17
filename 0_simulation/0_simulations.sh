#!/bin/bash
# ============================================================
# EPIDEMIC OUTBREAK SIMULATION PIPELINE
# ============================================================
# Generates epidemic outbreak scenarios using BEAST2/ReMaster
#
# This script:
#   1. Generates epidemic parameters and XML configuration
#   2. Runs BEAST2 simulations to produce trajectory and tree files
#   3. Post-processes tree files to remove unnecessary blocks
#
# Output files per outbreak:
#   - {i}_beast2.xml: BEAST2 configuration
#   - {i}_parameter.csv: Epidemic parameters
#   - {i}_beast2.traj: Trajectory data (population dynamics)
#   - {i}_beast2.trees: Phylogenetic trees
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

# Location and population settings
NUM_LOCS=16                        # Number of outbreak locations/demes
POP_MIN=300                      # Minimum population size per location
POP_MAX=3000                       # Maximum population size per location
SHARED_POP_SIZE=false              # true = same size for all locations, false = random
SEED_LOCATION="None"               # Starting location: "None" = random, or 0 to NUM_LOCS-1

# R0 settings (basic reproduction number)
R0_MIN=2                           # Minimum R0 value
R0_MAX=6                           # Maximum R0 value
SHARED_R0=false                    # true = same R0 for all locations, false = random
MAX_R0_DIFF=2                      # Maximum difference in R0 across locations

# Recovery rate settings (mu - rate at which infected individuals recover)
MU_MIN=0.01                        # Minimum recovery rate
MU_MAX=0.04                        # Maximum recovery rate
SHARED_RECOVERY_RATE=false         # true = same rate for all locations, false = random
MAX_MU_DIFF=0.005                   # Maximum difference in recovery rate across locations

# Sample rate settings (rate of sampling infected individuals for sequencing)
SAMPLE_MIN=0.003                  # Minimum sampling rate (shared across all locations)
SAMPLE_MAX=0.003                 # Maximum sampling rate (shared across all locations)

# Migration rate settings (rate of movement between locations)
MIGRATION_MIN=0.0001               # Minimum migration rate
MIGRATION_MAX=0.008              # Maximum migration rate
SHARED_MIGRATION_RATE=false        # true = same rate for all pairs, false = random

# Simulation time settings
SIM_TIME_MIN=10                     # Minimum simulation time
SIM_TIME_MAX=20                     # Maximum simulation time
TIME_UNITS="recovery_period"       # "recovery_period" = scaled by 1/mean(mu), "arbitrary" = absolute

# Random seed and simulation count
RANDOM_SEED="None"                 # "None" = random seed, or integer for reproducibility
NUM_SIMS=1                        # Number of BEAST2 simulations to generate per outbreak

# Export all configuration variables to environment
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

SCRIPT_DIR="$(pwd)"
mkdir -p "$OUTPUT_DIR/logs"

echo "========================================="
echo "EPIDEMIC OUTBREAK SIMULATION"
echo "========================================="
echo "Outbreaks:       $NUM_OUTBREAKS"
echo "Sims per outbreak: $NUM_SIMS"
echo "Locations:       $NUM_LOCS"
echo "Output directory: $OUTPUT_DIR"
echo "========================================="
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

    # Step 1: Generate epidemic parameters and BEAST2 XML configuration
    python3 "$SCRIPT_DIR/../utils/xml_generation.py" "$xml_file" "$param_file" > "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Step 2: Run BEAST2 simulation (generates .traj and .trees files)
    beast2 "$xml_file" >> "$log_file" 2>&1
    if [ $? -ne 0 ]; then
        return 1
    fi

    # Step 3: Clean up tree file by removing taxa and translate blocks
    bash "$SCRIPT_DIR/../utils/edit_tree.sh" "$tree_file" >> "$log_file" 2>&1
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
    (cd "$OUTPUT_DIR" && run_outbreak_simulation $i)

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
