#!/usr/bin/env python3
"""
Graph Loading Script for DGL

Load node and edge CSV files and create DGL graphs.

Usage:
    python graph_loading.py --input_folder /path/to/folder --output graphs.bin
"""

import os
import argparse
import traceback
import pickle
import pandas as pd
import dgl
import torch
from pathlib import Path


def find_graph_files(input_folder):
    """Find all node and edge CSV file pairs."""
    input_path = Path(input_folder)
    node_files = list(input_path.glob("*_node.csv"))

    print(f"Found {len(node_files)} node files")

    graph_files = {}
    for node_file in node_files:
        prefix = node_file.stem.replace("_node", "")
        edge_file = input_path / f"{prefix}_edge.csv"

        if edge_file.exists():
            graph_files[prefix] = (str(node_file), str(edge_file))
        else:
            print(f"Warning: No matching edge file for {node_file.name}")

    return graph_files


def load_graph_from_csv(node_file, edge_file):
    """Load graphs from a node and edge CSV file pair."""
    node_df = pd.read_csv(node_file)
    edge_df = pd.read_csv(edge_file)

    unique_graph_ids = node_df['graph_id'].unique()
    graphs = {}

    for graph_id in unique_graph_ids:
        graph_nodes = node_df[node_df['graph_id'] == graph_id].copy()
        graph_edges = edge_df[edge_df['graph_id'] == graph_id].copy()

        # Create node ID mapping (ensure sequential 0, 1, 2, ...)
        node_ids = sorted(graph_nodes['node'].unique())
        node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Map to sequential indices
        src_nodes = graph_edges['src'].map(node_id_map).astype(int).values
        dst_nodes = graph_edges['dst'].map(node_id_map).astype(int).values

        # Create DGL graph
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_ids))

        # Add node features (sort to align with node indices)
        graph_nodes = graph_nodes.sort_values('node')
        node_feature_cols = ['Initial_Population', 'R0', 'Epidemic_Peak',
                            'Peak_Timing', 'Accumulated_Infections',
                            'Num_Samples', 'Source_Sink_Score']

        for col in node_feature_cols:
            if col in graph_nodes.columns:
                g.ndata[col] = torch.tensor(graph_nodes[col].values, dtype=torch.float32)

        # Add edge features
        if 'migration_rate' in graph_edges.columns:
            g.edata['migration_rate'] = torch.tensor(
                graph_edges['migration_rate'].values,
                dtype=torch.float32
            )

        graphs[graph_id] = g

    return graphs


def load_all_graphs(input_folder):
    """Load all graphs from input folder."""
    graph_files = find_graph_files(input_folder)

    if not graph_files:
        print("No graph files found!")
        return {}

    all_graphs = {}
    for prefix, (node_file, edge_file) in graph_files.items():
        try:
            graphs = load_graph_from_csv(node_file, edge_file)
            for graph_id, graph in graphs.items():
                all_graphs[graph_id] = graph
        except Exception as e:
            print(f"Error loading graph {prefix}: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*50}")
    print(f"Successfully loaded {len(all_graphs)} graphs")
    print(f"{'='*50}")

    return all_graphs


def save_graphs(graphs, output_path):
    """Save graphs to DGL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_ids = list(graphs.keys())
    graph_list = list(graphs.values())

    dgl.save_graphs(str(output_path), graph_list)

    metadata = {'num_graphs': len(graphs), 'graph_ids': graph_ids}
    metadata_path = output_path.with_suffix('.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✓ Saved graphs to {output_path}")
    print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Load graphs from CSV files into DGL format')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to folder containing node and edge CSV files')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the graphs')

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return

    graphs = load_all_graphs(args.input_folder)

    if graphs:
        print("\nGraph Summary:")
        for graph_id, g in list(graphs.items())[:5]:
            print(f"  {graph_id}: {g.num_nodes()} nodes, {g.num_edges()} edges")
            print(f"    Node features: {list(g.ndata.keys())}")
            print(f"    Edge features: {list(g.edata.keys())}")

        if len(graphs) > 5:
            print(f"  ... and {len(graphs) - 5} more graphs")

    if args.output and graphs:
        save_graphs(graphs, args.output)

    return graphs


if __name__ == "__main__":
    main()
