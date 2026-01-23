#!/usr/bin/env python3
"""
Node Feature Extraction Script.

Input:  *_beast2.traj, *_beast2.xml, *_parameter.csv
Output: *_nf.csv (Location, Initial_Population, Epidemic_Peak, Peak_Timing,
                  Accumulated_Infections, R0, Source_Sink_Score)

Usage:  python3 nf.py /path/to/data [--force]
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numba import jit
from tqdm import tqdm

I_PATTERN = re.compile(r'I\[(\d+)\]')


@jit(nopython=True, cache=True)
def _find_nonzero_changes(diff_arr):
    """JIT-compiled function to find non-zero elements in diff array."""
    total = 0
    for i in range(diff_arr.shape[0]):
        for j in range(diff_arr.shape[1]):
            if diff_arr[i, j] != 0:
                total += 1

    row_indices = np.empty(total, dtype=np.int32)
    col_indices = np.empty(total, dtype=np.int32)
    values = np.empty(total, dtype=np.int32)

    idx = 0
    for i in range(diff_arr.shape[0]):
        for j in range(diff_arr.shape[1]):
            if diff_arr[i, j] != 0:
                row_indices[idx] = i
                col_indices[idx] = j
                values[idx] = diff_arr[i, j]
                idx += 1

    return row_indices, col_indices, values


def _parse_reaction_side(side):
    """Parse one side of a reaction string into species counts."""
    species = defaultdict(int)
    for term in side.strip().split('+'):
        match = re.match(r'^(\d+)?(.+)$', term.strip())
        if match:
            coef = int(match.group(1)) if match.group(1) else 1
            species[match.group(2)] += coef
    return species


def load_trajectory_wide(traj_file):
    """Load trajectory file and pivot to wide format using Polars."""
    df = pl.read_csv(traj_file, separator='\t')
    if df.is_empty():
        raise ValueError(f"Trajectory file '{traj_file}' is empty")

    df = df.with_columns(
        pl.when(pl.col('population').is_in(['R', 'sample']))
        .then(pl.col('population'))
        .otherwise(pl.col('population') + '[' + pl.col('index').cast(pl.Int32).cast(pl.Utf8) + ']')
        .alias('species')
    )
    return df.pivot(on='species', index=['Sample', 't'], values='value').fill_null(0).to_pandas()


def load_parameters(param_file):
    """Load R0 and Initial_Population from parameter CSV."""
    df = pd.read_csv(param_file)
    r0 = {int(c.replace('R0_loc_', '')): float(df[c].iloc[0])
          for c in df.columns if c.startswith('R0_loc_')}
    pop = {int(c.replace('population_loc_', '')): int(df[c].iloc[0])
           for c in df.columns if c.startswith('population_loc_')}
    return {'R0': r0, 'Initial_Population': pop}


def load_reactions(xml_file):
    """Load reaction definitions from BEAST2 XML file."""
    root = ET.parse(xml_file).getroot()
    lookup = {}

    for reaction in root.findall('.//reaction'):
        reaction_str = (reaction.text or '').strip()
        if '->' not in reaction_str:
            continue

        parts = reaction_str.split('->')
        if len(parts) != 2:
            continue

        reactants = _parse_reaction_side(parts[0])
        products = _parse_reaction_side(parts[1])
        all_species = set(reactants) | set(products)
        net_change = {s: products.get(s, 0) - reactants.get(s, 0) for s in all_species}
        lookup[tuple(sorted(net_change.items()))] = reaction_str

    return lookup


def calculate_epidemic_peaks(sample_data, num_nodes):
    """Calculate epidemic peak and timing for each node."""
    metrics = {n: {'peak': 0, 'peak_time': 0.0} for n in range(num_nodes)}
    i_cols = {c for c in sample_data.columns if c.startswith('I[')}

    for node_id in range(num_nodes):
        col = f'I[{node_id}]'
        if col in i_cols:
            peak = int(sample_data[col].max())
            if peak > 0:
                idx = sample_data[col].idxmax()
                metrics[node_id] = {'peak': peak, 'peak_time': float(sample_data.loc[idx, 't'])}
    return metrics


def classify_events(sample_data, reaction_lookup, species_cols):
    """Classify events by matching population changes to reactions (Numba-accelerated)."""
    diff_arr = sample_data[species_cols].diff().values[1:].astype(np.int32)
    col_names = list(species_cols)

    row_indices, col_indices, values = _find_nonzero_changes(diff_arr)
    if len(row_indices) == 0:
        return pd.Series(dtype=int)

    event_types = []
    current_row = -1
    current_changes = []

    for i in range(len(row_indices)):
        if row_indices[i] != current_row:
            if current_changes:
                event_types.append(reaction_lookup.get(tuple(sorted(current_changes)), "Unknown"))
            current_row = row_indices[i]
            current_changes = []
        current_changes.append((col_names[col_indices[i]], int(values[i])))

    if current_changes:
        event_types.append(reaction_lookup.get(tuple(sorted(current_changes)), "Unknown"))

    return pd.Series(event_types).value_counts() if event_types else pd.Series(dtype=int)


def calculate_node_metrics(event_counts, num_nodes):
    """Calculate accumulated_infections and source_sink_score for each node."""
    infections = np.zeros(num_nodes, dtype=int)
    exports = np.zeros(num_nodes, dtype=int)
    imports = np.zeros(num_nodes, dtype=int)

    for event_type, freq in event_counts.items():
        if '->' not in event_type:
            continue

        left, right = [s.strip() for s in event_type.split('->', 1)]
        left_nodes = [int(n) for n in I_PATTERN.findall(left)]
        right_nodes = [int(n) for n in I_PATTERN.findall(right)]

        for node_id in right_nodes:
            if node_id < num_nodes:
                infections[node_id] += freq

        if (len(left_nodes) == 1 and len(right_nodes) == 1 and
                left == f"I[{left_nodes[0]}]" and right == f"I[{right_nodes[0]}]"):
            src, dst = left_nodes[0], right_nodes[0]
            if src != dst:
                if src < num_nodes:
                    exports[src] += freq
                if dst < num_nodes:
                    imports[dst] += freq

    total = exports + imports
    source_sink = np.divide(exports - imports, total, out=np.zeros(num_nodes), where=total > 0)

    return {n: {'accumulated_infections': int(infections[n]), 'source_sink_score': float(source_sink[n])}
            for n in range(num_nodes)}


def process_simulation(traj_file, param_file, output_file, force=False):
    """Process one simulation and write node features to CSV."""
    if output_file.exists() and not force:
        return False

    try:
        params = load_parameters(str(param_file))
        num_nodes = len(params['R0'])

        df_wide = load_trajectory_wide(str(traj_file))
        sample_data = df_wide[df_wide['Sample'] == 0].sort_values('t').reset_index(drop=True)

        xml_file = traj_file.with_name(traj_file.name.replace('_beast2.traj', '_beast2.xml'))
        if not xml_file.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file}")
        reaction_lookup = load_reactions(str(xml_file))

        species_cols = [c for c in df_wide.columns if c not in ('Sample', 't')]
        event_counts = classify_events(sample_data, reaction_lookup, species_cols)
        node_metrics = calculate_node_metrics(event_counts, num_nodes)
        peak_metrics = calculate_epidemic_peaks(sample_data, num_nodes)

        rows = [{
            'Location': n,
            'Initial_Population': params['Initial_Population'].get(n, 0),
            'Epidemic_Peak': peak_metrics[n]['peak'],
            'Peak_Timing': peak_metrics[n]['peak_time'],
            'Accumulated_Infections': node_metrics[n]['accumulated_infections'],
            'R0': params['R0'].get(n, 0.0),
            'Source_Sink_Score': node_metrics[n]['source_sink_score'],
        } for n in range(num_nodes)]

        pd.DataFrame(rows).to_csv(output_file, index=False)
        return True

    except Exception as e:
        print(f"  Error processing {traj_file.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract node features from trajectory data')
    parser.add_argument('input_folder', help='Input data directory')
    parser.add_argument('--force', action='store_true', help='Overwrite existing *_nf.csv files')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)

    traj_files = sorted(input_folder.glob('*_beast2.traj'))
    if not traj_files:
        print(f"No trajectory files found in {input_folder}")
        sys.exit(0)

    print(f"Extracting node features: {input_folder}")
    print(f"Found {len(traj_files)} trajectory files\n")

    processed = skipped = errors = 0
    for traj_file in tqdm(traj_files, desc="Processing"):
        prefix = traj_file.stem.replace('_beast2', '')
        param_file = input_folder / f"{prefix}_parameter.csv"
        output_file = input_folder / f"{prefix}_nf.csv"

        if not param_file.exists():
            print(f"  Warning: No parameter file for {traj_file.name}")
            errors += 1
            continue

        result = process_simulation(traj_file, param_file, output_file, args.force)
        if result:
            processed += 1
        elif output_file.exists():
            skipped += 1
        else:
            errors += 1

    print(f"\nProcessed: {processed} | Skipped: {skipped} | Errors: {errors}")


if __name__ == '__main__':
    main()
