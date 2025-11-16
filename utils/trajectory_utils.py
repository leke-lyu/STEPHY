#!/usr/bin/env python3
"""
Shared Utility Functions for Trajectory Analysis
"""

import pandas as pd
import xml.etree.ElementTree as ET
import re
import sys
from collections import defaultdict


# ==============================================================================
# Trajectory File Functions
# ==============================================================================

def load_trajectory_wide(traj_file):
    """
    Load trajectory file and convert to wide format for easy plotting.

    Args:
        traj_file: Path to .traj file (tab-separated)

    Returns:
        DataFrame with columns: Sample, t, S[0], S[1], ..., I[0], I[1], ..., R, sample
        Each row represents a time point for a specific sample

    Raises:
        FileNotFoundError: If trajectory file doesn't exist
        pd.errors.EmptyDataError: If file is empty

    Example:
        >>> df = load_trajectory_wide('0_beast2.traj')
        >>> df.columns
        Index(['Sample', 't', 'S[0]', 'S[1]', 'I[0]', 'I[1]', 'R', 'sample'], dtype='object')
    """
    try:
        df = pd.read_csv(traj_file, sep='\t')
    except Exception as e:
        raise FileNotFoundError(f"Failed to read trajectory file '{traj_file}': {e}")

    if df.empty:
        raise pd.errors.EmptyDataError(f"Trajectory file '{traj_file}' is empty")

    # Create standardized species names using vectorized operations (much faster)
    # For R and sample: just use population name (ignore index)
    # For S and I: use population[index] format
    needs_index = ~df['population'].isin(['R', 'sample'])
    df['species'] = df['population'].copy()
    df.loc[needs_index, 'species'] = (
        df.loc[needs_index, 'population'] + '[' +
        df.loc[needs_index, 'index'].astype(int).astype(str) + ']'
    )

    # Convert to wide format (time series ready for plotting)
    df_wide = df.pivot_table(
        index=['Sample', 't'],
        columns='species',
        values='value',
        fill_value=0
    ).reset_index()

    return df_wide

def get_compartment_groups(df_wide):
    """
    Organize species columns into compartment groups (S, I, R).

    Args:
        df_wide: Wide-format trajectory DataFrame

    Returns:
        Dictionary with keys 'S', 'I', 'R', 'sample' containing lists of column names

    Example:
        >>> df = load_trajectory_wide('0_beast2.traj')
        >>> groups = get_compartment_groups(df)
        >>> groups['I']
        ['I[0]', 'I[1]', 'I[2]', ...]
    """
    groups = {
        'S': [],
        'I': [],
        'R': [],
        'sample': []
    }

    for col in df_wide.columns:
        if col in ['Sample', 't']:
            continue
        elif col.startswith('S['):
            groups['S'].append(col)
        elif col.startswith('I['):
            groups['I'].append(col)
        elif col == 'R':
            groups['R'].append(col)
        elif col == 'sample':
            groups['sample'].append(col)

    # Sort location-based compartments by index
    for key in ['S', 'I']:
        groups[key] = sorted(groups[key], key=lambda x: int(re.search(r'\[(\d+)\]', x).group(1)))

    return groups

def get_sample_data(df_wide, sample_id):
    """
    Extract data for a specific simulation sample.

    Args:
        df_wide: Wide-format trajectory DataFrame
        sample_id: Sample ID to extract

    Returns:
        DataFrame with only the specified sample, sorted by time

    Example:
        >>> df = load_trajectory_wide('0_beast2.traj')
        >>> sample_0 = get_sample_data(df, 0)
    """
    return df_wide[df_wide['Sample'] == sample_id].sort_values('t').reset_index(drop=True)

def calculate_epidemic_peaks(sample_data, num_nodes):
    """
    Calculate epidemic peak and peak timing for each node.
    If multiple time points have the same peak value, returns the earliest timing.

    Args:
        sample_data: Pre-filtered and sorted DataFrame for a single sample
        num_nodes: Total number of nodes in the network

    Returns:
        Dictionary mapping node_id to {'peak': max_value, 'peak_time': time_of_max}
        Nodes with no I[x] column or all zeros get {'peak': 0, 'peak_time': 0.0}

    Example:
        >>> df = load_trajectory_wide('0_beast2.traj')
        >>> sample_data = df[df['Sample'] == 0].sort_values('t').reset_index(drop=True)
        >>> metrics = calculate_epidemic_peaks(sample_data, num_nodes=10)
        >>> metrics[0]
        {'peak': 150, 'peak_time': 45.2}
    """
    # Initialize metrics for ALL nodes (peak=0, time=0.0 by default)
    metrics = {node_id: {'peak': 0, 'peak_time': 0.0} for node_id in range(num_nodes)}

    # Get available I[x] columns once (cache for performance)
    available_i_cols = {col for col in sample_data.columns if col.startswith('I[')}

    # Process each node
    for node_id in range(num_nodes):
        i_col = f'I[{node_id}]'

        # Check if column exists using cached set (faster than checking columns)
        if i_col in available_i_cols:
            peak_value = int(sample_data[i_col].max())

            if peak_value > 0:
                # Find earliest peak occurrence (data is already sorted by time)
                # Use idxmax for fast lookup, which returns first occurrence
                earliest_peak_idx = sample_data[i_col].idxmax()
                peak_time = float(sample_data.loc[earliest_peak_idx, 't'])

                metrics[node_id] = {
                    'peak': peak_value,
                    'peak_time': peak_time
                }

    return metrics


# ==============================================================================
# XML Reaction Functions
# ==============================================================================

def parse_reaction(reaction_str):
    """
    Parse reaction string into reactants and products with coefficients.

    Args:
        reaction_str: Reaction string (e.g., 'S[0] + I[0] -> 2I[0]')

    Returns:
        Tuple of (reactants_dict, products_dict) or (None, None) if parsing fails

    Example:
        >>> reactants, products = parse_reaction('S[0] + I[0] -> 2I[0]')
        >>> reactants
        {'S[0]': 1, 'I[0]': 1}
        >>> products
        {'I[0]': 2}
    """
    parts = reaction_str.split('->')
    if len(parts) != 2:
        return None, None

    def parse_side(side):
        species_dict = defaultdict(int)
        for term in side.strip().split('+'):
            match = re.match(r'^(\d+)?(.+)$', term.strip())
            if match:
                coef = int(match.group(1)) if match.group(1) else 1
                species_dict[match.group(2)] += coef
        return species_dict

    return parse_side(parts[0]), parse_side(parts[1])

def load_reactions_from_xml(xml_file):
    """
    Load all reactions from BEAST2 XML file.

    Args:
        xml_file: Path to .xml file

    Returns:
        Dictionary mapping reaction net changes to reaction strings

    Raises:
        FileNotFoundError: If XML file doesn't exist
        ET.ParseError: If XML is malformed

    Example:
        >>> reactions = load_reactions_from_xml('0_beast2.xml')
        >>> len(reactions)
        120  # For 10 locations: 10 infection + 10 recovery + 10 sampling + 90 migration
    """
    try:
        root = ET.parse(xml_file).getroot()
    except Exception as e:
        raise FileNotFoundError(f"Failed to read XML file '{xml_file}': {e}")

    reaction_lookup = {}

    for reaction in root.findall('.//reaction'):
        reaction_str = reaction.text.strip() if reaction.text else ""
        reactants, products = parse_reaction(reaction_str)

        if reactants is not None and products is not None:
            all_species = set(reactants.keys()) | set(products.keys())
            net_change = {s: products.get(s, 0) - reactants.get(s, 0) for s in all_species}
            change_key = tuple(sorted(net_change.items()))
            reaction_lookup[change_key] = reaction_str

    return reaction_lookup


# ==============================================================================
# Parameter CSV Functions
# ==============================================================================

def load_R0_and_population_from_csv(parameter_file):
    """
    Load R0 and Initial_Population from parameter CSV file.

    Args:
        parameter_file: Path to parameter CSV file

    Returns:
        Dictionary with keys 'R0' and 'Initial_Population', each mapping node_id to value

    Raises:
        SystemExit: If parameter file cannot be read

    Example:
        >>> params = load_R0_and_population_from_csv('0_parameter.csv')
        >>> params['R0'][0]
        2.84
        >>> params['Initial_Population'][0]
        9388
    """
    try:
        params_df = pd.read_csv(parameter_file)
    except Exception as e:
        print(f"ERROR: Failed to read parameter file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract R0 values (vectorized for performance)
    r0_cols = [col for col in params_df.columns if col.startswith('R0_loc_')]
    r0_dict = {
        int(col.replace('R0_loc_', '')): float(params_df[col].values[0])
        for col in r0_cols
    }

    # Extract Initial Population values (vectorized for performance)
    pop_cols = [col for col in params_df.columns if col.startswith('population_loc_')]
    pop_dict = {
        int(col.replace('population_loc_', '')): int(params_df[col].values[0])
        for col in pop_cols
    }

    return {'R0': r0_dict, 'Initial_Population': pop_dict}


def load_migration_rate_from_csv(parameter_file):
    """
    Load migration rates from parameter CSV file.

    Args:
        parameter_file: Path to parameter CSV file

    Returns:
        Nested dictionary mapping {from_node: {to_node: rate}}
        Example: migration_rates[0][1] = rate from location 0 to location 1

    Raises:
        SystemExit: If parameter file cannot be read

    Example:
        >>> migration_rates = load_migration_rate_from_csv('0_parameter.csv')
        >>> migration_rates[0][1]  # Rate from loc 0 to loc 1
        0.000717818
        >>> migration_rates[1][0]  # Rate from loc 1 to loc 0
        0.000998965
    """
    try:
        params_df = pd.read_csv(parameter_file)
    except Exception as e:
        print(f"ERROR: Failed to read parameter file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract migration rate columns (format: migration_loc_x_to_loc_y)
    migration_cols = [col for col in params_df.columns if col.startswith('migration_loc_')]

    # Build nested dictionary: {from_node: {to_node: rate}}
    migration_dict = {}
    pattern = re.compile(r'migration_loc_(\d+)_to_loc_(\d+)')

    # Parse all migration columns
    for col in migration_cols:
        match = pattern.match(col)
        if match:
            from_node = int(match.group(1))
            to_node = int(match.group(2))
            rate = float(params_df[col].values[0])

            # Initialize nested dict if needed
            if from_node not in migration_dict:
                migration_dict[from_node] = {}

            migration_dict[from_node][to_node] = rate

    return migration_dict
