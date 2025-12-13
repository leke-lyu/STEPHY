#!/usr/bin/env python3
"""Extract edge-level features from phylogenetic tree states."""

import sys
import re
import os
import pandas as pd
from trajectory_utils import load_migration_rate_from_csv


def count_tree_states(trees_file):
    """Count the number of tree states in a BEAST2 .trees file."""
    with open(trees_file, 'r') as f:
        return sum(1 for line in f if line.startswith('tree STATE_'))


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 edge_feature.py <parameter_file> <trees_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    parameter_file, trees_file, output_file = sys.argv[1:4]
    file_prefix = re.sub(r'_beast2\.trees$', '', os.path.basename(trees_file))

    migration_rates = load_migration_rate_from_csv(parameter_file)
    num_nodes = len(migration_rates)
    num_states = count_tree_states(trees_file)

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of tree states: {num_states}")

    all_edges = []
    for state_id in range(num_states):
        graph_id = f"{file_prefix}_{state_id}"
        for src in range(num_nodes):
            for dst in range(num_nodes):
                migration_rate = float('nan') if src == dst else migration_rates[src][dst]
                all_edges.append({
                    'graph_id': graph_id,
                    'src': src,
                    'dst': dst,
                    'migration_rate': migration_rate
                })

    result_df = pd.DataFrame(all_edges)
    result_df.to_csv(output_file, index=False)
    print(f"Output: {len(result_df)} rows -> {output_file}")


if __name__ == "__main__":
    main()
