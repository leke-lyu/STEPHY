#!/usr/bin/env python3
"""Extract node-level features from epidemic trajectory simulations."""

import sys
import re
import pandas as pd
from trajectory_utils import (
    load_trajectory_wide,
    load_reactions_from_xml,
    load_R0_and_population_from_csv,
    calculate_epidemic_peaks
)

I_PATTERN = re.compile(r'I\[(\d+)\]')


def classify_events_for_sample(sample_data, reaction_lookup, species_cols):
    """Classify all events for a single sample by matching population changes to reactions."""
    diff_data = sample_data[species_cols].diff()
    time_values = sample_data['t'].values

    events = []
    for i in range(1, len(sample_data)):
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
    """Calculate event-based metrics for all nodes from reaction event counts."""
    metrics = {node_id: {
        'accumulated_infections': 0,
        'num_samples': 0,
        'export_count': 0,
        'import_count': 0,
        'source_sink_score': float('nan')
    } for node_id in range(num_nodes)}

    for event_type, freq in event_counts.items():
        if '->' not in event_type:
            continue

        left, right = [s.strip() for s in event_type.split('->', 1)]
        left_node_ids = [int(nid) for nid in I_PATTERN.findall(left)]
        right_node_ids = [int(nid) for nid in I_PATTERN.findall(right)]

        # Accumulated infections: I[x] appears in products
        for node_id in right_node_ids:
            if node_id < num_nodes:
                metrics[node_id]['accumulated_infections'] += freq

        # Number of samples: I[x] -> R + sample
        if right == 'R + sample' and len(left_node_ids) == 1:
            node_id = left_node_ids[0]
            if left == f"I[{node_id}]" and node_id < num_nodes:
                metrics[node_id]['num_samples'] += freq

        # Export/Import: I[x] -> I[y] where x != y
        if len(left_node_ids) == 1 and len(right_node_ids) == 1:
            src, dst = left_node_ids[0], right_node_ids[0]
            if left == f"I[{src}]" and right == f"I[{dst}]" and src != dst:
                if src < num_nodes:
                    metrics[src]['export_count'] += freq
                if dst < num_nodes:
                    metrics[dst]['import_count'] += freq

    # Calculate source-sink scores
    for node_id in range(num_nodes):
        total = metrics[node_id]['export_count'] + metrics[node_id]['import_count']
        if total > 0:
            metrics[node_id]['source_sink_score'] = (
                metrics[node_id]['export_count'] - metrics[node_id]['import_count']
            ) / total

    return metrics


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 node_feature.py <parameter_file> <xml_file> <traj_file> [output_file]", file=sys.stderr)
        sys.exit(1)

    parameter_file, xml_file, traj_file = sys.argv[1:4]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    file_prefix = re.sub(r'_beast2\.traj$', '', traj_file.split('/')[-1])

    # Load data
    parameters = load_R0_and_population_from_csv(parameter_file)
    num_nodes = len(parameters['R0'])
    reaction_lookup = load_reactions_from_xml(xml_file)
    df_wide = load_trajectory_wide(traj_file)
    species_cols = [col for col in df_wide.columns if col not in ['Sample', 't']]

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of samples: {df_wide['Sample'].nunique()}")

    # Process each sample
    all_results = []
    for sample_id, sample_data in df_wide.groupby('Sample', sort=True):
        sample_data = sample_data.sort_values('t').reset_index(drop=True)
        events_df = classify_events_for_sample(sample_data, reaction_lookup, species_cols)
        event_counts = events_df['event_type'].value_counts() if not events_df.empty else pd.Series(dtype=int)

        node_metrics = calculate_node_event_metrics(event_counts, num_nodes)
        epidemic_metrics = calculate_epidemic_peaks(sample_data, num_nodes)
        graph_id = f"{file_prefix}_{sample_id}"

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

    result_df = pd.DataFrame(all_results)

    if output_file:
        result_df.to_csv(output_file, index=False)
    else:
        print(result_df.to_csv(index=False))


if __name__ == "__main__":
    main()
