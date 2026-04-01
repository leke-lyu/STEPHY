#!/bin/bash
# ============================================================
# COMBINED SIMULATION AND FEATURE EXTRACTION PIPELINE
# ============================================================
#
# Merges 0_simulations.sh and 1_feature_extraction.sh into a
# single loop that simulates one outbreak, immediately checks
# the tip-count filter, extracts features, and deletes bulky
# intermediate files (.xml, .traj) before moving to the next.
#
# Loop logic:
#   1. Generate parameters + BEAST2 XML   (xml_generation.py)
#   2. Run BEAST2 simulation              (beast2)
#   3. Clean tree file                    (edit_tree.sh)
#   4. Filter + extract features          (process_simulation)
#   On success  -> keep .trees, _nf.csv, _parameter.csv
#   On failure  -> delete everything, retry same index
#
# Usage:
#   bash simulate_and_extract.sh <target_successful> [output_folder]
#
# Examples:
#   bash simulate_and_extract.sh 100
#   bash simulate_and_extract.sh 100 results
# ============================================================

set -uo pipefail

# ============================================================
# CONFIGURATION
# ============================================================

# Locations / populations — 5 Danish regions (fixed)
# 0=Hovedstaden, 1=Midtjylland, 2=Syddanmark, 3=Sjaelland, 4=Nordjylland
NUM_LOCS=5
FIXED_POP_SIZES="1860000 1330000 1220000 840000 590000"
SEED_LOCATION="None"              # "None" = random, or 0..NUM_LOCS-1

# R0 — wide range covering Alpha(1–1.3), Delta(1.3–2.4), Omicron(2.8–4.4)
R0_MIN=0.5
R0_MAX=4
SHARED_R0=false
MAX_R0_DIFF=1
R0_DISTRIBUTION="beta"            # "uniform" or "beta"
R0_ALPHA=2                        # Beta shape α (only used when R0_DISTRIBUTION=beta)
R0_BETA=3.5                       # Beta shape β (only used when R0_DISTRIBUTION=beta)
                                  # Beta(2, 3.5) on [0.5, 4] → mode ≈ 1.5, mean ≈ 1.8

# Recovery rate (mu, per day) — covers Alpha–Omicron infectious periods
MU_MIN=0.07
MU_MAX=0.23
SHARED_RECOVERY_RATE=false
MAX_MU_DIFF=0.01

# Sampling rate (per day, shared across all locations)
SAMPLE_MIN=0.001
SAMPLE_MAX=0.01

# Migration rate
MIGRATION_MIN=0.001
MIGRATION_MAX=0.012
SHARED_MIGRATION_RATE=false

# Simulation time (days, absolute)
SIM_TIME_MIN=30
SIM_TIME_MAX=270
TIME_UNITS="arbitrary"            # "recovery_period" or "arbitrary"

# Early termination: stop simulation when sampled tips reach this count
ENDS_WHEN="sample>=8000"

# Misc
RANDOM_SEED="None"                # "None" = random (OS entropy), or integer for
                                  # single-batch reproducibility.
                                  # WARNING: A fixed seed makes EVERY call to
                                  # xml_generation.py produce identical parameters.
                                  # When running multiple batches via submit.sh,
                                  # keep this set to "None" so each batch draws
                                  # independent random parameters.
NUM_SIMS=1

# Tip-count filter: every location must have more than this many tips
MIN_TIPS=50

# Safety: abort if a single index fails this many times in a row
MAX_ATTEMPTS_PER_OUTBREAK=50

# Export configuration for xml_generation.py (reads via os.getenv())
export NUM_LOCS FIXED_POP_SIZES SEED_LOCATION RANDOM_SEED TIME_UNITS NUM_SIMS
export R0_MIN R0_MAX SHARED_R0 MAX_R0_DIFF R0_DISTRIBUTION R0_ALPHA R0_BETA
export MU_MIN MU_MAX SHARED_RECOVERY_RATE MAX_MU_DIFF
export SAMPLE_MIN SAMPLE_MAX
export MIGRATION_MIN MIGRATION_MAX SHARED_MIGRATION_RATE
export SIM_TIME_MIN SIM_TIME_MAX
export ENDS_WHEN

# ============================================================
# ARGUMENTS
# ============================================================

if [ $# -eq 0 ]; then
    echo "Usage: $0 <target_successful> [output_folder]"
    echo ""
    echo "  target_successful  Number of successful outbreaks to produce"
    echo "  output_folder      Output directory (default: current directory)"
    exit 1
fi

TARGET=$1
OUTPUT_DIR=${2:-.}

# ============================================================
# SETUP
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="$(dirname "$SCRIPT_DIR")/simulate_and_extract"

mkdir -p "$OUTPUT_DIR/logs"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "========================================="
echo "COMBINED SIMULATION + FEATURE EXTRACTION"
echo "========================================="
echo "Target successful: $TARGET"
echo "Locations:         $NUM_LOCS"
echo "Output directory:  $OUTPUT_DIR"
echo "Max retries/index: $MAX_ATTEMPTS_PER_OUTBREAK"
echo "========================================="
echo ""

# ============================================================
# MAIN LOOP
# ============================================================

consecutive_fails=0

cd "$OUTPUT_DIR"

# Resume: count existing successful outbreaks (_nf.csv = success marker)
success_count=$(find . -maxdepth 1 -name '*_nf.csv' | wc -l | tr -d ' ')
total_attempts=$(find logs -maxdepth 1 -name 'attempt_*.log' 2>/dev/null | wc -l | tr -d ' ')

if [ $success_count -gt 0 ]; then
    echo "Resuming: found $success_count existing successful outbreaks, starting at index $success_count"
    echo "  (previous attempts: $total_attempts)"
fi

while [ $success_count -lt $TARGET ]; do
    idx=$success_count
    log_file="logs/attempt_${total_attempts}.log"

    # File names (relative to OUTPUT_DIR)
    xml_file="${idx}_beast2.xml"
    param_file="${idx}_parameter.csv"
    tree_file="${idx}_beast2.trees"
    traj_file="${idx}_beast2.traj"
    nf_file="${idx}_nf.csv"

    # Run the four pipeline steps; stop at first failure
    step_ok=true

    # Step 1: Generate parameters + XML
    if $step_ok; then
        python3 "$SCRIPT_DIR/xml_generation.py" "$xml_file" "$param_file" \
            > "$log_file" 2>&1 || step_ok=false
    fi

    # Step 2: Run BEAST2 simulation
    if $step_ok; then
        beast2 -overwrite "$xml_file" >> "$log_file" 2>&1 || step_ok=false
    fi

    # Step 3: Clean tree file
    if $step_ok; then
        bash "$SHARED_DIR/edit_tree.sh" "$tree_file" >> "$log_file" 2>&1
    fi

    # Step 4: Tip-count filter + feature extraction
    #         (calls the same process_simulation used by 1_feature_extraction)
    if $step_ok; then
        python3 -c "
import sys, pathlib
sys.path.insert(0, '$SHARED_DIR')
from characterizing_outbreak import process_simulation
result = process_simulation(
    pathlib.Path('$tree_file'), pathlib.Path('$traj_file'),
    pathlib.Path('$param_file'), pathlib.Path('$nf_file'),
    min_tips=$MIN_TIPS)
sys.exit(0 if result == 'processed' else 1)
" >> "$log_file" 2>&1 || step_ok=false
    fi

    # ----------------------------------------------------------
    # Handle result
    # ----------------------------------------------------------
    if $step_ok; then
        # Success: keep .trees, _nf.csv, _parameter.csv
        rm -f "$xml_file" "$traj_file"
        success_count=$((success_count + 1))
        consecutive_fails=0
    else
        # Failed: delete all files for this index, retry
        rm -f "$xml_file" "$param_file" "$tree_file" "$traj_file" "$nf_file"
        consecutive_fails=$((consecutive_fails + 1))
        if [ $consecutive_fails -ge $MAX_ATTEMPTS_PER_OUTBREAK ]; then
            echo ""
            echo "ERROR: $consecutive_fails consecutive failures at index $idx. Aborting."
            exit 1
        fi
    fi

    total_attempts=$((total_attempts + 1))
    printf "\rSuccessful: %d/%d | Attempts: %d | Consecutive fails: %d  " \
        $success_count $TARGET $total_attempts $consecutive_fails
done

# ============================================================
# DONE
# ============================================================

echo ""
echo ""
echo "========================================="
echo "  DONE: $success_count successful outbreaks"
echo "  Total attempts: $total_attempts"
echo "  Output: $OUTPUT_DIR"
echo "========================================="
