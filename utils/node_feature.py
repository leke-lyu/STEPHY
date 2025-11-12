import pandas as pd
import re
import sys
from trajectory_utils import (
    load_trajectory_wide,
    load_reactions_from_xml,
    parse_reaction,
    load_parameters_from_csv,
    calculate_epidemic_metrics
)


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
    events = []

    for i in range(1, len(sample_data)):
        non_zero = diff_data.loc[i][diff_data.loc[i] != 0]
        if len(non_zero) > 0:
            change_key = tuple(sorted(non_zero.to_dict().items()))
            events.append({
                'time': sample_data.loc[i, 't'],
                'event_type': reaction_lookup.get(change_key, "Unknown")
            })

    return pd.DataFrame(events)


def calculate_node_metrics(event_counts, i_populations):
    """
    Calculate node-level metrics from event counts.

    Args:
        event_counts: Series of event type counts
        i_populations: List of I[x] population names

    Returns:
        Dictionary for each I[x] population containing:
        - accumulated_infections: Total infections (I[x] appears in products)
        - num_samples: Count of I[x] -> R + sample events
        - source_sink_score: (exports - imports) / (exports + imports)
    """
    metrics = {i_pop: {
        'accumulated_infections': 0,
        'num_samples': 0,
        'export_count': 0,
        'import_count': 0,
        'source_sink_score': 0.0
    } for i_pop in i_populations}

    # Parse each event once and update relevant populations
    i_pattern = re.compile(r'I\[\d+\]')

    for event_type, freq in event_counts.items():
        if '->' not in event_type:
            continue

        left, right = event_type.split('->', 1)
        left, right = left.strip(), right.strip()

        # Find all I[x] populations in left and right
        left_i_pops = i_pattern.findall(left)
        right_i_pops = i_pattern.findall(right)

        # Accumulated infections: I[x] appears in products
        for i_pop in right_i_pops:
            if i_pop in metrics:
                metrics[i_pop]['accumulated_infections'] += freq

        # Number of samples: I[x] -> R + sample
        if right == 'R + sample' and len(left_i_pops) == 1 and left == left_i_pops[0]:
            metrics[left_i_pops[0]]['num_samples'] += freq

        # Export: I[x] -> I[y] where x != y
        if len(left_i_pops) == 1 and len(right_i_pops) == 1:
            left_pop = left_i_pops[0]
            right_pop = right_i_pops[0]
            if left == left_pop and right == right_pop and left_pop != right_pop:
                if left_pop in metrics:
                    metrics[left_pop]['export_count'] += freq
                if right_pop in metrics:
                    metrics[right_pop]['import_count'] += freq

    # Calculate source-sink scores
    for i_pop in i_populations:
        export_count = metrics[i_pop]['export_count']
        import_count = metrics[i_pop]['import_count']
        total = export_count + import_count
        if total > 0:
            metrics[i_pop]['source_sink_score'] = (export_count - import_count) / total

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

        # Load parameters (R0 and Initial_Population)
        parameters = load_parameters_from_csv(parameter_file)

        # Load reactions
        reaction_lookup = load_reactions_from_xml(xml_file)

        # Load trajectory data using utility function
        df_wide = load_trajectory_wide(traj_file)

        species_cols = [col for col in df_wide.columns if col not in ['Sample', 't']]

        # Get all unique sample IDs
        sample_ids = sorted(df_wide['Sample'].unique())

        # Collect all results
        all_results = []

        # Process each sample
        for sample_id in sample_ids:
            sample_data = df_wide[df_wide['Sample'] == sample_id].sort_values('t').reset_index(drop=True)

            # Classify events for this sample
            events_df = classify_events_for_sample(sample_data, reaction_lookup, species_cols)

            if events_df.empty:
                continue

            event_counts = events_df['event_type'].value_counts()

            # Extract unique I populations
            i_populations = set()
            for event_type in events_df['event_type'].unique():
                matches = re.findall(r'I\[\d+\]', event_type)
                i_populations.update(matches)

            if not i_populations:
                continue

            i_populations = sorted(i_populations, key=lambda x: int(re.search(r'\[(\d+)\]', x).group(1)))

            # Calculate metrics for this sample
            node_metrics = calculate_node_metrics(event_counts, i_populations)
            epidemic_metrics = calculate_epidemic_metrics(sample_data)

            # Create graph_id
            graph_id = f"{file_prefix}_{sample_id}"

            # Add results for each node
            for i_pop in i_populations:
                node_id = int(re.search(r'\[(\d+)\]', i_pop).group(1))
                all_results.append({
                    'graph_id': graph_id,
                    'node': node_id,
                    'Initial_Population': parameters['Initial_Population'].get(node_id, 0),
                    'R0': parameters['R0'].get(node_id, 0.0),
                    'Epidemic_Peak': epidemic_metrics.get(node_id, {}).get('peak', 0),
                    'Peak_Timing': epidemic_metrics.get(node_id, {}).get('peak_time', 0.0),
                    'Accumulated_Infections': node_metrics[i_pop]['accumulated_infections'],
                    'Num_Samples': node_metrics[i_pop]['num_samples'],
                    'Source_Sink_Score': node_metrics[i_pop]['source_sink_score']
                })

        if not all_results:
            print("ERROR: No valid data to write - all samples failed processing", file=sys.stderr)
            sys.exit(1)

        # Create output dataframe
        result_df = pd.DataFrame(all_results)

        # Write output
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Successfully processed {len(sample_ids)} sample(s)")
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
# - Uses trajectory_utils.py for shared functionality (load_trajectory_wide, load_reactions_from_xml, parse_reaction, calculate_epidemic_metrics)
# - R0 and Initial_Population are loaded from parameter CSV file (population_loc_x and R0_loc_x columns)
# - Epidemic_Peak and Peak_Timing are calculated from max(I[x]) across the trajectory
# - Populations that never appear in any events are excluded entirely from the results
# - Populations with no imports/exports get source_sink_score values of 0.0
# - Populations with no samples get num_samples values of 0