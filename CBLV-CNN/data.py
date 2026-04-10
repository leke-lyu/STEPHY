#!/usr/bin/env python3
"""
Data loading for CBLV-CNN — CBLV node features only, no edges.

Graphs are used for batching only; no edges or edge features are computed.
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import dgl

sys.path.append(str(Path(__file__).resolve().parent.parent / 'stephy'))
from beast2_parser import count_trees
from tree_core import load_tree, VirtualSubtreeEncoder, load_labels


def build_graph(tree_file, tree_idx, subtree_width, input_folder, cblv_scale='tree_height'):
    """Build a DGL graph with CBLV node features only (no edges)."""
    phy = load_tree(tree_file, tree_idx)
    tree_height = max(nd.root_distance for nd in phy.leaf_node_iter())

    encoder = VirtualSubtreeEncoder(phy, tree_height)
    locations = encoder.get_all_locations()
    n_nodes = len(locations)

    node_cblv = np.zeros((n_nodes, subtree_width, 4))
    node_aux = np.zeros((n_nodes, 5))
    for i, loc in enumerate(locations):
        cblv, aux_stats = encoder.encode_cblv(loc, subtree_width=subtree_width, cblv_scale=cblv_scale)
        node_cblv[i] = cblv
        node_aux[i] = aux_stats

    node_cblv = np.transpose(node_cblv, (0, 2, 1))

    file_prefix = Path(tree_file).stem.replace('_beast2', '')
    labels = load_labels(input_folder, file_prefix, n_nodes)

    g = dgl.graph(([], []), num_nodes=n_nodes)
    g.ndata['cblv'] = torch.tensor(node_cblv, dtype=torch.float32)
    g.ndata['aux'] = torch.tensor(node_aux, dtype=torch.float32)
    g.ndata['location'] = torch.tensor(locations, dtype=torch.long)
    for name, vals in labels.items():
        g.ndata[name] = torch.tensor(vals, dtype=torch.float32)

    return g, locations, tree_height


def build_all_graphs(input_folder, subtree_width, file_pattern='*_beast2.trees', verbose=True, cblv_scale='tree_height'):
    """Build DGL graphs from all trees in folder."""
    input_folder = Path(input_folder)
    tree_files = sorted(input_folder.glob(file_pattern))
    graphs = []

    iterator = tqdm(tree_files, desc="Building graphs") if verbose else tree_files
    for tree_file in iterator:
        file_prefix = tree_file.stem.replace('_beast2', '')
        with open(tree_file) as f:
            n_trees = count_trees(f.read())

        for idx in range(n_trees):
            try:
                g, locs, height = build_graph(str(tree_file), idx, subtree_width, input_folder, cblv_scale=cblv_scale)
                graphs.append((g, f"{file_prefix}_{idx}", locs, height))
            except Exception as e:
                if verbose:
                    print(f"Error {tree_file.name}[{idx}]: {e}")

    return graphs
