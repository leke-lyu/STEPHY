#!/bin/bash
# ==============================================================================
# Benchmark STEPHY inference latency on the 5 Denmark clade ML trees.
#
# Single (non-array) SLURM job, single-threaded by design -- measures honest
# per-core forward-pass latency for each clade x task (r0/rr/sss/as) over the
# five real Nextstrain-clade ML trees.
#
# Usage: bash results/submit_denmark_inference.sh \
#            [<input_dir>] [<model_dir>]
#
# Defaults to /projects/lau_projects/denmark/... -- the Denmark stephy_input
# trees and pe_old/stephy2 model must already be staged there.
# Output: <input_dir>/../logs/bench_denmark_<jobid>.out
# ==============================================================================
#SBATCH --job-name=bench_denmark
#SBATCH --partition=lau,week-long-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STEPHY_ROOT="${STEPHY_ROOT:-$(dirname "$SCRIPT_DIR")}"

INPUT_DIR="${1:-/projects/lau_projects/denmark/stephy_input}"
MODEL_DIR="${2:-/projects/lau_projects/denmark/pe_old/stephy2}"
LOG_DIR="$(dirname "$INPUT_DIR")/logs"

if [ -z "$SLURM_JOB_ID" ]; then
    # --- Submission mode (called from login node via `bash`) ---
    mkdir -p "$LOG_DIR"
    sbatch --output="${LOG_DIR}/bench_denmark_%j.out" \
           --error="${LOG_DIR}/bench_denmark_%j.err" \
           --export=ALL,SCRIPT_DIR="$SCRIPT_DIR",STEPHY_ROOT="$STEPHY_ROOT",INPUT_DIR="$INPUT_DIR",MODEL_DIR="$MODEL_DIR" \
           "$0" "$INPUT_DIR" "$MODEL_DIR"
    exit 0
fi

# Sanity check: STEPHY_ROOT must point to the repo, not the SLURM spool dir.
if [ ! -f "${STEPHY_ROOT}/results/denmark_stephy_inference.py" ]; then
    echo "ERROR: cannot find denmark_stephy_inference.py at ${STEPHY_ROOT}/results/" >&2
    echo "  STEPHY_ROOT resolved to: ${STEPHY_ROOT}" >&2
    echo "  This usually means you ran 'sbatch ...' instead of 'bash ...'." >&2
    echo "  Re-run with:  bash results/submit_denmark_inference.sh" >&2
    exit 1
fi

# --- Execution mode (inside SLURM) ---
source ~/.bashrc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate stephy

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "  input_dir = ${INPUT_DIR}"
echo "  model_dir = ${MODEL_DIR}"

python3 "${STEPHY_ROOT}/results/denmark_stephy_inference.py" \
    --input_dir "$INPUT_DIR" \
    --model_dir "$MODEL_DIR" \
    --clades   20I,21I,21J,21K,21L \
    --tasks    r0,rr,sss,as \
    --n_warmup 20 \
    --n_repeat 100
