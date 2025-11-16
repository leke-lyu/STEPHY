#!/usr/bin/env python3
"""
Extract node-level features from epidemic trajectory simulations for phyloGNN.

Generates CSV with deterministic dimensions (num_nodes × num_samples) containing:
- Static parameters: Initial_Population, R0
- Epidemic metrics: Epidemic_Peak, Peak_Timing
- Event metrics: Accumulated_Infections, Num_Samples, Source_Sink_Score
"""

import pandas as pd
import re
import sys
from trajectory_utils import (
    load_trajectory_wide,
    load_reactions_from_xml,
    load_R0_and_population_from_csv,
    calculate_epidemic_peaks
)

# Pre-compile regex patterns for performance
I_PATTERN = re.compile(r'I\[(\d+)\]')


def classify_events_for_sample(sample_data, reaction_lookup, species_cols):
    """
    Classify all events for a single sample by matching population changes to reactions.

    Args:
        sample_data: Wide-format trajectory data for one sample
        reaction_lookup: Dictionary mapping net changes to reaction strings
        species_cols: List of species column names

    Returns:
        DataFrame with columns ['time', 'event_type']
    """
    diff_data = sample_data[species_cols].diff()

    # Pre-extract time values for faster access
    time_values = sample_data['t'].values

    events = []
    # Use iloc for faster integer-based indexing (avoid loc overhead)
    for i in range(1, len(sample_data)):
        # Use iloc instead of loc for performance
        row = diff_data.iloc[i]
        non_zero = row[row != 0]

        if len(non_zero) > 0:
            change_key = tuple(sorted(non_zero.to_dict().items()))
            events.append({
                'time': time_values[i],
                'event_type': reaction_lookup.get(change_key, "Unknown")
            })

    return pd.DataFrame(events)


def calculate_node_event_metrics(event_counts, num_nodes):
    """
    Calculate event-based metrics for all nodes from reaction event counts.

    Args:
        event_counts: Series of event type counts (can be empty)
        num_nodes: Total number of nodes in the network

    Returns:
        Dictionary for each node_id (0 to num_nodes-1) containing:
        - accumulated_infections: Total infections (I[x] appears in products), 0 if no events
        - num_samples: Count of I[x] -> R + sample events, 0 if no events
        - source_sink_score: (exports - imports) / (exports + imports), NA if no imports/exports
    """
    # Initialize metrics for ALL nodes
    metrics = {node_id: {
        'accumulated_infections': 0,
        'num_samples': 0,
        'export_count': 0,
        'import_count': 0,
        'source_sink_score': float('nan')  # NA by default
    } for node_id in range(num_nodes)}

    # Parse each event once and update relevant populations
    for event_type, freq in event_counts.items():
        if '->' not in event_type:
            continue

        left, right = event_type.split('->', 1)
        left, right = left.strip(), right.strip()

        # Find all I[x] populations in left and right (with node IDs)
        left_matches = I_PATTERN.findall(left)
        right_matches = I_PATTERN.findall(right)

        left_node_ids = [int(nid) for nid in left_matches]
        right_node_ids = [int(nid) for nid in right_matches]

        # Accumulated infections: I[x] appears in products
        for node_id in right_node_ids:
            if node_id < num_nodes:
                metrics[node_id]['accumulated_infections'] += freq

        # Number of samples: I[x] -> R + sample
        if right == 'R + sample' and len(left_node_ids) == 1:
            left_i = f"I[{left_node_ids[0]}]"
            if left == left_i:
                node_id = left_node_ids[0]
                if node_id < num_nodes:
                    metrics[node_id]['num_samples'] += freq

        # Export: I[x] -> I[y] where x != y
        if len(left_node_ids) == 1 and len(right_node_ids) == 1:
            left_node_id = left_node_ids[0]
            right_node_id = right_node_ids[0]
            left_i = f"I[{left_node_id}]"
            right_i = f"I[{right_node_id}]"

            if left == left_i and right == right_i and left_node_id != right_node_id:
                if left_node_id < num_nodes:
                    metrics[left_node_id]['export_count'] += freq
                if right_node_id < num_nodes:
                    metrics[right_node_id]['import_count'] += freq

    # Calculate source-sink scores (only if node has imports or exports)
    for node_id in range(num_nodes):
        export_count = metrics[node_id]['export_count']
        import_count = metrics[node_id]['import_count']
        total = export_count + import_count
        if total > 0:
            metrics[node_id]['source_sink_score'] = (export_count - import_count) / total
        # else: remains NaN

    return metrics


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 node_feature.py <parameter_file> <xml_file> <traj_file> [output_file]", file=sys.stderr)
        print("Example: python3 node_feature.py 0_parameter.csv 0_beast2.xml 0_beast2.traj 0_node.csv", file=sys.stderr)
        sys.exit(1)

    parameter_file = sys.argv[1]
    xml_file = sys.argv[2]
    traj_file = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        # Extract file prefix for graph_id
        file_prefix = re.sub(r'_beast2\.traj$', '', traj_file.split('/')[-1])

        # Load parameters (R0 and Initial_Population) - determines number of nodes
        parameters = load_R0_and_population_from_csv(parameter_file)

        # Determine number of nodes from parameters
        num_nodes_r0 = len(parameters['R0'])
        num_nodes_pop = len(parameters['Initial_Population'])

        if num_nodes_r0 != num_nodes_pop:
            print(f"ERROR: Mismatch in parameter counts - R0 has {num_nodes_r0} nodes, Initial_Population has {num_nodes_pop} nodes", file=sys.stderr)
            sys.exit(1)

        num_nodes = num_nodes_r0
        print(f"Number of nodes: {num_nodes}")

        # Load reactions
        reaction_lookup = load_reactions_from_xml(xml_file)

        # Load trajectory data using utility function
        df_wide = load_trajectory_wide(traj_file)

        species_cols = [col for col in df_wide.columns if col not in ['Sample', 't']]

        # Get all unique sample IDs
        sample_ids = sorted(df_wide['Sample'].unique())
        print(f"Number of samples: {len(sample_ids)}")

        # Collect results for all nodes × all samples
        all_results = []

        # Group by sample for efficient iteration (avoids repeated filtering)
        grouped = df_wide.groupby('Sample', sort=True)

        # Process each sample
        for sample_id, sample_data in grouped:
            sample_data = sample_data.sort_values('t').reset_index(drop=True)

            # Classify events for this sample
            events_df = classify_events_for_sample(sample_data, reaction_lookup, species_cols)

            # Get event counts (may be empty)
            if events_df.empty:
                event_counts = pd.Series(dtype=int)
            else:
                event_counts = events_df['event_type'].value_counts()

            # Calculate metrics for ALL nodes (even those with no events)
            node_metrics = calculate_node_event_metrics(event_counts, num_nodes)
            epidemic_metrics = calculate_epidemic_peaks(sample_data, num_nodes)

            # Create graph_id
            graph_id = f"{file_prefix}_{sample_id}"

            # Add results for EVERY node (0 to num_nodes-1)
            for node_id in range(num_nodes):
                all_results.append({
                    'graph_id': graph_id,
                    'node': node_id,
                    'Initial_Population': parameters['Initial_Population'].get(node_id, 0),
                    'R0': parameters['R0'].get(node_id, 0.0),
                    'Epidemic_Peak': epidemic_metrics[node_id]['peak'],
                    'Peak_Timing': epidemic_metrics[node_id]['peak_time'],
                    'Accumulated_Infections': node_metrics[node_id]['accumulated_infections'],
                    'Num_Samples': node_metrics[node_id]['num_samples'],
                    'Source_Sink_Score': node_metrics[node_id]['source_sink_score']
                })

        # Create output dataframe
        result_df = pd.DataFrame(all_results)

        # Write output
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Successfully processed {len(sample_ids)} sample(s) × {num_nodes} nodes = {len(result_df)} rows")
            print(f"Output written to: {output_file}")
        else:
            # Print to stdout in CSV format
            print(result_df.to_csv(index=False))

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Notes:
# - Uses trajectory_utils.py for shared functionality (load_trajectory_wide, load_reactions_from_xml, load_R0_and_population_from_csv, calculate_epidemic_peaks)
# - Number of nodes is determined from parameter CSV file (length of R0 columns = length of Initial_Population columns)
# - Output dimensions are deterministic: num_nodes × num_samples rows, 9 columns
# - EVERY node gets a record for EVERY sample (even if no events occurred)
# - R0 and Initial_Population are loaded from parameter CSV file (R0_loc_x and population_loc_x columns)
# - Epidemic_Peak and Peak_Timing are calculated from max(I[x]) across the trajectory
#   - If multiple time points have the same peak value, the earliest timing is used
#   - Nodes with no I[x] column or all zeros get peak=0, peak_time=0.0
# - Accumulated_Infections may be 0 (if node never had infections in reaction products)
# - Num_Samples may be 0 (if no I[x] -> R + sample events occurred)
# - Source_Sink_Score is NaN if node has no imports/exports (not 0.0)