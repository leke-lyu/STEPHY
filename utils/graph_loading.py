#!/usr/bin/env python3
"""
Graph Loading Module for DGL

Load node and edge CSV files and create DGL graphs.
"""

import traceback
import pandas as pd
import dgl
import torch
import numpy as np
from pathlib import Path


def parse_patristic_distance_array(distance_string):
    """
    Parse patristic distance array string and compute summary statistics.

    Returns dict with: min, q25, median, q75, max, mean, std, count
    If input is NaN/empty, all values are NaN.
    """
    nan_stats = {
        'min': np.nan, 'q25': np.nan, 'median': np.nan, 'q75': np.nan,
        'max': np.nan, 'mean': np.nan, 'std': np.nan, 'count': np.nan
    }

    # Handle NaN or empty values
    if pd.isna(distance_string) or distance_string == '':
        return nan_stats

    try:
        # Remove brackets and split by whitespace/tabs
        clean_string = str(distance_string).strip('[]')
        values = np.array([float(x) for x in clean_string.split()])

        return {
            'min': np.min(values),
            'q25': np.percentile(values, 25),
            'median': np.median(values),
            'q75': np.percentile(values, 75),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    except:
        return nan_stats


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
                            'Num_Samples', 'Source_Sink_Score',
                            'Monophyletic_Groups', 'Earliest_Sample_Time',
                            'Median_Sample_Time', 'Latest_Sample_Time']

        for col in node_feature_cols:
            if col in graph_nodes.columns:
                g.ndata[col] = torch.tensor(graph_nodes[col].values, dtype=torch.float32)

        # Add edge features
        if 'migration_rate' in graph_edges.columns:
            g.edata['migration_rate'] = torch.tensor(
                graph_edges['migration_rate'].values,
                dtype=torch.float32
            )

        if 'patristic_distance' in graph_edges.columns:
            # Parse patristic distance arrays and compute summary statistics
            stats_list = graph_edges['patristic_distance'].apply(parse_patristic_distance_array)

            # Add each statistic as a separate edge feature
            stat_names = ['min', 'q25', 'median', 'q75', 'max', 'mean', 'std', 'count']
            for stat in stat_names:
                g.edata[f'patristic_distance_{stat}'] = torch.tensor(
                    [s[stat] for s in stats_list], dtype=torch.float32
                )

        # Add DTW-based edge features
        dtw_feature_cols = ['dtw_distance', 'dtw_lag_mean', 'dtw_lag_std']
        for col in dtw_feature_cols:
            if col in graph_edges.columns:
                g.edata[col] = torch.tensor(
                    graph_edges[col].values, dtype=torch.float32
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
