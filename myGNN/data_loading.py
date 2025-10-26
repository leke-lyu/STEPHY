import argparse
import pandas as pd
import os
from glob import glob
import numpy as np
import dgl
import torch

def parse_edge_feature(edge_feature_str):
    """Parse edge feature string format [xx\tyy\tzz] into list of floats."""
    vector_str = str(edge_feature_str).strip('[]')
    return [float(v) for v in vector_str.split('\t')]

def load_and_convert_to_dgl(input_folder, num_of_nodes, min_length, max_length):
    """
    Load graph data from CSV files, filter, and convert to DGL format.

    Args:
        input_folder: Path to folder containing *_node.csv and *_edge.csv files
        num_of_nodes: Number of nodes for fully connected graph selection
        min_length: Minimum edge feature vector length
        max_length: Maximum edge feature vector length

    Returns:
        list: List of DGL graph objects
        dict: Metadata (max_length, max_value, num_graphs)
    """
    # Load all CSV files
    node_files = glob(os.path.join(input_folder, "*_node.csv"))
    edge_files = glob(os.path.join(input_folder, "*_edge.csv"))

    print(f"Found {len(node_files)} node files and {len(edge_files)} edge files")

    if not node_files or not edge_files:
        print("No data files found!")
        return [], {}

    all_nodes = pd.concat([pd.read_csv(f) for f in node_files], ignore_index=True)
    all_edges = pd.concat([pd.read_csv(f) for f in edge_files], ignore_index=True)

    print(f"Total nodes: {len(all_nodes)}, Total edges: {len(all_edges)}")

    # Filter Step 1: Fully connected graphs
    expected_edges = num_of_nodes + num_of_nodes * (num_of_nodes - 1) // 2
    node_counts = all_nodes.groupby('graph_id').size()
    edge_counts = all_edges.groupby('graph_id').size()

    valid_ids = {
        gid for gid in node_counts.index
        if node_counts[gid] == num_of_nodes and edge_counts.get(gid, 0) == expected_edges
    }

    print(f"Graphs with {num_of_nodes} nodes and {expected_edges} edges: {len(valid_ids)}")

    # Filter Step 2: Edge vector length constraints
    # Parse all edge features once and compute stats
    edge_data_by_graph = {}
    actual_max_length = 0
    all_values = []

    for graph_id in valid_ids:
        graph_edges = all_edges[all_edges['graph_id'] == graph_id]
        edge_vectors = [parse_edge_feature(ef) for ef in graph_edges['edge_features']]

        lengths = [len(v) for v in edge_vectors]
        min_len, max_len = min(lengths), max(lengths)

        # Check length constraints
        if min_len >= min_length and max_len <= max_length:
            edge_data_by_graph[graph_id] = edge_vectors
            actual_max_length = max(actual_max_length, max_len)
            all_values.extend([val for vec in edge_vectors for val in vec])

    print(f"Graphs after length filter ({min_length} <= min_len and max_len <= {max_length}): {len(edge_data_by_graph)}")

    if not edge_data_by_graph:
        return [], {}

    max_value = max(all_values)
    print(f"\nEdge feature padding: MAX_LENGTH={actual_max_length}, MAX_VALUE={max_value:.2f}")

    # Convert to DGL graphs
    dgl_graphs = []

    for graph_id in edge_data_by_graph.keys():
        node_df = all_nodes[all_nodes['graph_id'] == graph_id].sort_values('node').reset_index(drop=True)
        edge_df = all_edges[all_edges['graph_id'] == graph_id]

        # Node features and labels
        node_features = node_df[['Initial_Population', 'Accumulated_Infections']].values.astype(np.float32)
        node_labels = node_df['Source_Sink_Score'].values.astype(np.float32)

        # Edge list and features
        src_nodes = edge_df['node1'].values.astype(np.int64)
        dst_nodes = edge_df['node2'].values.astype(np.int64)

        # Pad edge features
        padded_features = [
            vec + [max_value] * (actual_max_length - len(vec))
            for vec in edge_data_by_graph[graph_id]
        ]

        # Create DGL graph
        g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
        
        # Assign node features
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        g.ndata['label'] = torch.tensor(node_labels, dtype=torch.float32)
        
        # Store edge features
        edge_feat_tensor = torch.tensor(padded_features, dtype=torch.float32)
        g.edata['feat'] = edge_feat_tensor
        
        # Convert to bidirectional (undirected) graph
        g = dgl.to_bidirected(g, copy_ndata=True)
        
        # After to_bidirected, we need to manually handle edge features
        # because they may not be copied automatically in older DGL versions
        if 'feat' not in g.edata:
            # Edge features were lost, need to reconstruct them
            # Get the edge list from the bidirected graph
            src_bi, dst_bi = g.edges()
            
            # Create a mapping from original edges to their features
            edge_to_feat = {}
            for i, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
                edge_to_feat[(s, d)] = edge_feat_tensor[i]
            
            # Assign features to all edges in the bidirected graph
            new_edge_feats = []
            for s, d in zip(src_bi.numpy(), dst_bi.numpy()):
                # Check if this edge was in the original graph
                if (s, d) in edge_to_feat:
                    new_edge_feats.append(edge_to_feat[(s, d)])
                # Otherwise it's a reverse edge, use the original edge's features
                elif (d, s) in edge_to_feat:
                    new_edge_feats.append(edge_to_feat[(d, s)])
                else:
                    # This shouldn't happen, but handle it gracefully
                    print(f"Warning: Edge ({s}, {d}) not found in original graph")
                    new_edge_feats.append(torch.zeros(actual_max_length))
            
            g.edata['feat'] = torch.stack(new_edge_feats)
        
        g.graph_id = graph_id

        dgl_graphs.append(g)

    print(f"Converted {len(dgl_graphs)} graphs to DGL format")
    print(f"  Node feature dim: 2, Edge feature dim: {actual_max_length}")

    metadata = {
        'max_length': actual_max_length,
        'max_value': max_value,
        'num_graphs': len(dgl_graphs)
    }

    return dgl_graphs, metadata

def main(input_folder, num_of_nodes, min_length, max_length):
    """Main entry point for data loading script."""
    dgl_graphs, metadata = load_and_convert_to_dgl(input_folder, num_of_nodes, min_length, max_length)

    if dgl_graphs:
        # Calculate edges after bidirectionalization
        # Original: n self-loops + n*(n-1)/2 undirected edges
        # After to_bidirected: n self-loops + n*(n-1) directed edges
        edges_after_bidirected = num_of_nodes + num_of_nodes * (num_of_nodes - 1)
        
        print(f"\n{'='*50}")
        print("Summary:")
        print(f"  Total graphs: {len(dgl_graphs)}")
        print(f"  Nodes per graph: {num_of_nodes}")
        print(f"  Edges per graph (after bidirectionalization): {edges_after_bidirected}")
        print(f"  Node features: [Initial_Population, Accumulated_Infections]")
        print(f"  Node label: Source_Sink_Score")
        print(f"  Edge feature dim: {metadata['max_length']} (padded with {metadata['max_value']:.2f})")
        print(f"{'='*50}")

        # ═══════════════════════════════════════════════════════════════
        # ║  START INSPECTION CODE - SAFE TO DELETE THIS ENTIRE BLOCK  ║
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'='*50}")
        print("DETAILED INSPECTION OF FIRST GRAPH:")
        print(f"{'='*50}")

        sample_graph = dgl_graphs[0]

        print(f"\n📊 Graph ID: {sample_graph.graph_id}")
        print(f"   Number of nodes: {sample_graph.num_nodes()}")
        print(f"   Number of edges: {sample_graph.num_edges()}")

        print(f"\n🔵 Node Features (first 3 nodes):")
        print(f"   Shape: {sample_graph.ndata['feat'].shape}")
        print(f"   Data:\n{sample_graph.ndata['feat'][:3]}")

        print(f"\n🎯 Node Labels (first 3 nodes):")
        print(f"   Shape: {sample_graph.ndata['label'].shape}")
        print(f"   Data: {sample_graph.ndata['label'][:3]}")

        print(f"\n🔗 Edge Features (first 3 edges):")
        print(f"   Shape: {sample_graph.edata['feat'].shape}")
        print(f"   First edge feature vector (first 20 values):")
        print(f"   {sample_graph.edata['feat'][0][:20]}")

        print(f"\n✅ Bidirectionality Check:")
        has_0_to_1 = sample_graph.has_edges_between(0, 1)
        has_1_to_0 = sample_graph.has_edges_between(1, 0)
        print(f"   Edge 0→1 exists: {has_0_to_1}")
        print(f"   Edge 1→0 exists: {has_1_to_0}")

        print(f"\n📈 Edge Feature Statistics:")
        all_edge_feats = sample_graph.edata['feat']
        print(f"   Min value: {all_edge_feats.min().item():.2f}")
        print(f"   Max value: {all_edge_feats.max().item():.2f}")
        print(f"   Mean value: {all_edge_feats.mean().item():.2f}")
        print(f"   Padding value: {metadata['max_value']:.2f}")

        print(f"\n{'='*50}")
        print("✨ Inspection complete!")
        print(f"{'='*50}\n")
        # ═══════════════════════════════════════════════════════════════
        # ║    END INSPECTION CODE - DELETE FROM START MARKER TO HERE  ║
        # ═══════════════════════════════════════════════════════════════

    return dgl_graphs, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and convert graph data to DGL format")
    parser.add_argument("input_folder", type=str, help="Path to the input folder")
    parser.add_argument("num_of_nodes", type=int, help="Number of nodes")
    parser.add_argument("min_length", type=int, help="Minimum edge vector length")
    parser.add_argument("max_length", type=int, help="Maximum edge vector length")

    args = parser.parse_args()
    main(args.input_folder, args.num_of_nodes, args.min_length, args.max_length)
