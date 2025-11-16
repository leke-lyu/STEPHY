#!/usr/bin/env python3
"""
Extract edge-level features from phylogenetic tree states for phyloGNN.

Generates CSV with deterministic dimensions (num_edges × num_states) containing:
- Graph identification: graph_id
- Edge topology: src, dst
- Edge attributes: migration_rate
"""

import os
import sys
import re
import pandas as pd
from trajectory_utils import load_migration_rate_from_csv


def count_tree_states(trees_file):
    """
    Count the number of tree states in a BEAST2 .trees file.

    Args:
        trees_file: Path to .trees file

    Returns:
        Number of tree STATE_ entries

    Example:
        >>> num_states = count_tree_states('0_beast2.trees')
        >>> num_states
        10
    """
    count = 0
    try:
        with open(trees_file, 'r') as f:
            for line in f:
                if line.startswith('tree STATE_'):
                    count += 1
    except Exception as e:
        print(f"ERROR: Failed to read trees file: {e}", file=sys.stderr)
        sys.exit(1)

    return count


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 edge_feature.py <parameter_file> <trees_file> <output_file>", file=sys.stderr)
        print("Example: python3 edge_feature.py 0_parameter.csv 0_beast2.trees 0_edge.csv", file=sys.stderr)
        sys.exit(1)

    parameter_file = sys.argv[1]
    trees_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        # Extract file prefix for graph_id
        file_prefix = re.sub(r'_beast2\.trees$', '', os.path.basename(trees_file))

        # Load migration rates (same for all graphs)
        migration_rates = load_migration_rate_from_csv(parameter_file)

        # Determine number of nodes from migration rates
        # num_nodes = number of unique nodes (sources and destinations)
        num_nodes = len(migration_rates)
        print(f"Number of nodes: {num_nodes}")

        # Count number of tree states (graphs)
        num_states = count_tree_states(trees_file)
        print(f"Number of tree states: {num_states}")

        if num_states == 0:
            print("ERROR: No tree states found in trees file", file=sys.stderr)
            sys.exit(1)

        # Calculate expected output dimensions
        # Include self-loops: num_nodes × num_nodes edges per graph
        num_edges_per_graph = num_nodes * num_nodes
        total_edges = num_edges_per_graph * num_states
        print(f"Edges per graph (including self-loops): {num_edges_per_graph}")
        print(f"Total edges: {total_edges}")

        # Collect all edge features
        all_edges = []

        # For each tree state (graph)
        for state_id in range(num_states):
            graph_id = f"{file_prefix}_{state_id}"

            # For each source node
            for src in range(num_nodes):
                # For each destination node (including self-loops)
                for dst in range(num_nodes):
                    if src == dst:
                        # Self-loop: migration_rate is NaN
                        migration_rate = float('nan')
                    else:
                        # Regular edge: get migration_rate from parameter CSV
                        migration_rate = migration_rates[src][dst]

                    all_edges.append({
                        'graph_id': graph_id,
                        'src': src,
                        'dst': dst,
                        'migration_rate': migration_rate
                    })

        # Create output dataframe and write to CSV
        result_df = pd.DataFrame(all_edges)
        result_df.to_csv(output_file, index=False)

        print(f"Successfully processed {num_states} graph(s) × {num_edges_per_graph} edges = {len(result_df)} rows")
        print(f"Output written to: {output_file}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
