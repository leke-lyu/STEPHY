#!/usr/bin/env python3
"""
Preprocessing Script for 7_stephy.

Extracts epidemiological features from trajectory files and labels from parameter CSVs,
then writes summary files (*_nd.csv) for each simulation.

Input files (per simulation):
    - *_beast2.traj: Trajectory time-series data
    - *_parameter.csv: Ground truth parameters (R0, Initial_Population)

Output files (per simulation):
    - *_nd.csv: Node data summary with columns:
        Initial_Population, Epidemic_Peak, Peak_Timing, Accumulated_Infections, R0, Source_Sink_Score

Usage:
    python3 preprocess.py /path/to/data [--force]

    --force: Overwrite existing *_nd.csv files (default: skip existing)
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add utils to path
SCRIPT_DIR = Path(__file__).parent
UTILS_DIR = SCRIPT_DIR.parent / 'utils'
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from trajectory_utils import (
    load_trajectory_wide,
    get_sample_data,
    calculate_epidemic_peaks,
    load_R0_and_population_from_csv
)


def calculate_accumulated_infections(sample_data, num_nodes):
    """
    Calculate total accumulated infections for each node.

    This is approximated by the final R (recovered) count for each location,
    which represents all individuals who were infected and recovered.

    For SIR models: Accumulated_Infections ≈ Initial_S - Final_S

    Args:
        sample_data: Pre-filtered and sorted DataFrame for a single sample
        num_nodes: Total number of nodes in the network

    Returns:
        Dictionary mapping node_id to accumulated infections count
    """
    accumulated = {}

    # Get initial and final S values for each node
    first_row = sample_data.iloc[0]
    last_row = sample_data.iloc[-1]

    for node_id in range(num_nodes):
        s_col = f'S[{node_id}]'
        if s_col in sample_data.columns:
            initial_s = first_row[s_col]
            final_s = last_row[s_col]
            # Accumulated infections = people who left S compartment
            accumulated[node_id] = int(initial_s - final_s)
        else:
            accumulated[node_id] = 0

    return accumulated


def calculate_source_sink_score(peak_timings, num_nodes):
    """
    Calculate Source/Sink score for each node based on epidemic timing.

    Source nodes have early epidemics (exporters), Sink nodes have late epidemics (importers).
    Score is normalized to [-1, 1] range:
        -1 = Latest epidemic (pure sink)
        +1 = Earliest epidemic (pure source)

    Args:
        peak_timings: Dictionary mapping node_id to peak timing
        num_nodes: Total number of nodes

    Returns:
        Dictionary mapping node_id to source/sink score
    """
    scores = {}

    # Get all valid peak timings (> 0)
    valid_timings = [(nid, t) for nid, t in peak_timings.items() if t > 0]

    if len(valid_timings) < 2:
        # Not enough data for meaningful comparison
        return {nid: 0.0 for nid in range(num_nodes)}

    times = [t for _, t in valid_timings]
    min_time = min(times)
    max_time = max(times)
    time_range = max_time - min_time

    if time_range < 1e-8:
        # All epidemics at same time
        return {nid: 0.0 for nid in range(num_nodes)}

    for node_id in range(num_nodes):
        timing = peak_timings.get(node_id, 0)
        if timing > 0:
            # Normalize: early = +1 (source), late = -1 (sink)
            normalized = (timing - min_time) / time_range  # 0 to 1 (early to late)
            scores[node_id] = 1.0 - 2.0 * normalized       # +1 to -1 (source to sink)
        else:
            # No epidemic at this location
            scores[node_id] = 0.0

    return scores


def process_single_simulation(traj_file, param_file, output_file, force=False):
    """
    Process a single simulation and create node data CSV.

    Args:
        traj_file: Path to trajectory file (*_beast2.traj)
        param_file: Path to parameter file (*_parameter.csv)
        output_file: Path to output file (*_nd.csv)
        force: If True, overwrite existing output file

    Returns:
        True if processed successfully, False if skipped or failed
    """
    # Skip if output exists and not forcing
    if output_file.exists() and not force:
        return False

    try:
        # Load parameter data (R0, Initial_Population)
        params = load_R0_and_population_from_csv(str(param_file))
        num_nodes = len(params['R0'])

        # Load trajectory data
        df_wide = load_trajectory_wide(str(traj_file))
        sample_data = get_sample_data(df_wide, sample_id=0)

        # Calculate epidemic metrics
        peak_metrics = calculate_epidemic_peaks(sample_data, num_nodes)
        accumulated = calculate_accumulated_infections(sample_data, num_nodes)

        # Extract peak timings for source/sink calculation
        peak_timings = {nid: metrics['peak_time'] for nid, metrics in peak_metrics.items()}
        source_sink = calculate_source_sink_score(peak_timings, num_nodes)

        # Build output dataframe
        rows = []
        for node_id in range(num_nodes):
            rows.append({
                'Initial_Population': params['Initial_Population'].get(node_id, 0),
                'Epidemic_Peak': peak_metrics[node_id]['peak'],
                'Peak_Timing': peak_metrics[node_id]['peak_time'],
                'Accumulated_Infections': accumulated[node_id],
                'R0': params['R0'].get(node_id, 0.0),
                'Source_Sink_Score': source_sink[node_id]
            })

        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_file, index=False)

        return True

    except Exception as e:
        print(f"  Error processing {traj_file.name}: {e}")
        return False


def preprocess_dataset(input_folder, force=False):
    """
    Preprocess all simulations in a dataset folder.

    Args:
        input_folder: Path to dataset folder
        force: If True, overwrite existing *_nd.csv files

    Returns:
        Tuple of (processed_count, skipped_count, error_count)
    """
    input_folder = Path(input_folder)

    # Find all trajectory files
    traj_files = sorted(input_folder.glob('*_beast2.traj'))

    if not traj_files:
        print(f"No trajectory files found in {input_folder}")
        return 0, 0, 0

    print(f"Found {len(traj_files)} trajectory files")

    processed = 0
    skipped = 0
    errors = 0

    for traj_file in tqdm(traj_files, desc="Preprocessing"):
        # Derive file names
        prefix = traj_file.stem.replace('_beast2', '')
        param_file = input_folder / f"{prefix}_parameter.csv"
        output_file = input_folder / f"{prefix}_nd.csv"

        if not param_file.exists():
            print(f"  Warning: No parameter file for {traj_file.name}")
            errors += 1
            continue

        result = process_single_simulation(traj_file, param_file, output_file, force)

        if result:
            processed += 1
        elif output_file.exists():
            skipped += 1
        else:
            errors += 1

    return processed, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess trajectory data for 7_stephy'
    )
    parser.add_argument('input_folder', help='Input data directory')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing *_nd.csv files')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)

    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)

    print(f"Preprocessing: {input_folder}")
    print(f"Force overwrite: {args.force}")
    print()

    processed, skipped, errors = preprocess_dataset(input_folder, args.force)

    print()
    print("=" * 50)
    print(f"Processed: {processed}")
    print(f"Skipped (existing): {skipped}")
    print(f"Errors: {errors}")
    print("=" * 50)


if __name__ == '__main__':
    main()
