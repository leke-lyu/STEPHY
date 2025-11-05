#!/usr/bin/env python3

import pandas as pd
import xml.etree.ElementTree as ET
import re
import sys
from collections import defaultdict

def parse_reaction(reaction_str):
    """Parse reaction string into reactants and products with coefficients."""
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


def get_net_change(reactants, products):
    """Calculate net change for each species."""
    all_species = set(reactants.keys()) | set(products.keys())
    return {s: products.get(s, 0) - reactants.get(s, 0) for s in all_species}


def load_reactions_from_xml(xml_file):
    """Load reactions from XML and create lookup dictionary."""
    try:
        root = ET.parse(xml_file).getroot()
    except Exception as e:
        print(f"ERROR: Failed to read XML file: {e}", file=sys.stderr)
        sys.exit(1)

    reaction_lookup = {}

    for reaction in root.findall('.//reaction'):
        reaction_str = reaction.text.strip() if reaction.text else ""
        reactants, products = parse_reaction(reaction_str)

        if reactants is not None and products is not None:
            net_change = get_net_change(reactants, products)
            change_key = tuple(sorted(net_change.items()))
            reaction_lookup[change_key] = reaction_str

    return reaction_lookup


def create_species_name(pop, idx):
    """Create species name from population and index."""
    if pop in ['R', 'sample'] or pd.isna(idx):
        return pop
    return f"{pop}[{int(idx)}]"


def classify_events_for_sample(sample_data, reaction_lookup, species_cols):
    """Classify all events for a single sample."""
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


def calculate_initial_populations(df, i_populations, sample_id):
    """Calculate initial population sizes (S[x] + I[x] at t=0) for a specific sample."""
    df_t0 = df[(df['t'] == 0.0) & (df['Sample'] == sample_id)]
    initial_populations = {}

    for i_pop in i_populations:
        idx = int(re.search(r'\[(\d+)\]', i_pop).group(1))
        s_val = df_t0[(df_t0['population'] == 'S') & (df_t0['index'] == idx)]['value'].values
        i_val = df_t0[(df_t0['population'] == 'I') & (df_t0['index'] == idx)]['value'].values
        initial_populations[i_pop] = int((s_val[0] if len(s_val) > 0 else 0) +
                                          (i_val[0] if len(i_val) > 0 else 0))

    return initial_populations


def calculate_node_metrics(event_counts, i_populations):
    """
    Calculate all node metrics in a single pass for efficiency.

    Returns a dictionary for each I[x] population containing:
    - accumulated_infections: Total infections accumulated (I[x] appears in products)
    - num_samples: Count of I[x] -> R + sample reactions
    - source_sink_score: (exports - imports) / (exports + imports)
    """
    metrics = {i_pop: {
        'accumulated_infections': 0,
        'num_samples': 0,
        'export_count': 0,
        'import_count': 0
    } for i_pop in i_populations}

    # Single pass through all events
    for event_type, freq in event_counts.items():
        if '->' not in event_type:
            continue

        parts = event_type.split('->')
        left = parts[0].strip()
        right = parts[1].strip()

        for i_pop in i_populations:
            # Accumulated infections: I[x] appears in products
            if i_pop in right:
                metrics[i_pop]['accumulated_infections'] += freq

            # Number of samples: I[x] -> R + sample
            if left == i_pop and right == 'R + sample':
                metrics[i_pop]['num_samples'] += freq

            # Export: I[x] -> I[others]
            if i_pop in left and 'I[' in right and right.startswith('I[') and right != i_pop:
                metrics[i_pop]['export_count'] += freq

            # Import: I[others] -> I[x]
            if 'I[' in left and i_pop in right:
                left_i_pops = re.findall(r'I\[\d+\]', left)
                if left_i_pops and i_pop not in left_i_pops:
                    metrics[i_pop]['import_count'] += freq

    # Calculate source-sink scores
    for i_pop in i_populations:
        export_count = metrics[i_pop]['export_count']
        import_count = metrics[i_pop]['import_count']
        total = export_count + import_count

        if total > 0:
            metrics[i_pop]['source_sink_score'] = (export_count - import_count) / total
        else:
            metrics[i_pop]['source_sink_score'] = 0.0

    return metrics


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 node_feature.py <xml_file> <traj_file> [output_file]", file=sys.stderr)
        print("Example: python3 node_feature.py 0_beast2.xml 0_beast2.traj 0_node.csv", file=sys.stderr)
        sys.exit(1)

    xml_file = sys.argv[1]
    traj_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        # Validate input file exists
        try:
            df = pd.read_csv(traj_file, sep='\t')
        except Exception as e:
            print(f"ERROR: Failed to read trajectory file: {e}", file=sys.stderr)
            sys.exit(1)

        if df.empty:
            print("ERROR: Trajectory file is empty", file=sys.stderr)
            sys.exit(1)

        # Extract file prefix for graph_id
        file_prefix = re.sub(r'_beast2\.traj$', '', traj_file.split('/')[-1])

        # Load reactions
        reaction_lookup = load_reactions_from_xml(xml_file)

        # Create species names
        df['species'] = df.apply(lambda row: create_species_name(row['population'], row['index']), axis=1)

        # Pivot data
        df_wide = df.pivot_table(
            index=['Sample', 't'], columns='species', values='value', fill_value=0
        ).reset_index()

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
            initial_populations = calculate_initial_populations(df, i_populations, sample_id)
            node_metrics = calculate_node_metrics(event_counts, i_populations)

            # Create graph_id
            graph_id = f"{file_prefix}_{sample_id}"

            # Add results for each node
            for i_pop in i_populations:
                node_id = re.search(r'\[(\d+)\]', i_pop).group(1)
                all_results.append({
                    'graph_id': graph_id,
                    'node': node_id,
                    'Initial_Population': initial_populations[i_pop],
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
# - Populations that never appear in any events are excluded entirely from the results.
# - Populations with no imports/exports get source_sink_score values of 0.0
# - Populations with no samples get num_samples values of 0