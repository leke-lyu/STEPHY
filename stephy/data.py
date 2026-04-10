#!/usr/bin/env python3
"""
Data loading for STEPHY — CBLV node features + DTW edge features.

Builds fully connected DGL graphs (no self-loops) from BEAST2 trees.
DTW edge features capture temporal lag between location epidemic curves.
"""

import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
import dgl
from scipy.stats import gaussian_kde
from numba import jit

from config import DATA_ARGS
from beast2_parser import parse_trees, count_trees
from tree_core import load_tree, get_location, is_sample, VirtualSubtreeEncoder, load_labels


# ==============================================================================
# DTW Edge Features
# ==============================================================================

def extract_tip_times(tree_file, tree_idx=0):
    """Extract tip sampling times grouped by location from a BEAST2 tree file."""
    with open(tree_file) as f:
        trees = parse_trees(f.read())
    if not trees or tree_idx >= len(trees):
        return {}

    tips = defaultdict(list)
    for loc, time in re.findall(r'\d+\[&type="I\{(\d+)\}",samp="sample",time=([\d.]+)\]', trees[tree_idx]):
        tips[int(loc)].append(float(time))
    return dict(tips)


def tips_to_curves(tips_by_loc, num_points=None):
    """Convert tip sampling times to KDE-smoothed epidemic curves."""
    if num_points is None:
        num_points = DATA_ARGS['dtw_num_points']
    all_times = [t for times in tips_by_loc.values() for t in times]
    if not all_times:
        return {}, 0

    t_grid = np.linspace(min(all_times), max(all_times), num_points)
    dt = (max(all_times) - min(all_times)) / (num_points - 1)

    curves = {}
    for loc, times in tips_by_loc.items():
        curves[loc] = gaussian_kde(times)(t_grid) * len(times) if len(times) >= 2 else np.zeros(num_points)
    return curves, dt


@jit(nopython=True, cache=True)
def _dtw_forward(curve_a, curve_b):
    """Numba-optimized DTW forward pass. Returns DTW matrix."""
    n, m = len(curve_a), len(curve_b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (curve_a[i-1] - curve_b[j-1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1])
    return dtw


@jit(nopython=True, cache=True)
def _dtw_backtrack(dtw):
    """Numba-optimized DTW backtrack. Returns lag mean and std."""
    n, m = dtw.shape[0] - 1, dtw.shape[1] - 1
    i, j = n, m
    lag_sum = 0.0
    lag_sq_sum = 0.0
    count = 0

    while i > 0 and j > 0:
        lag = j - i
        lag_sum += lag
        lag_sq_sum += lag * lag
        count += 1

        diag = dtw[i-1, j-1]
        up = dtw[i-1, j]
        left = dtw[i, j-1]

        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i, j = i - 1, j
        else:
            i, j = i, j - 1

    if count == 0:
        return 0.0, 0.0

    mean = lag_sum / count
    variance = (lag_sq_sum / count) - (mean * mean)
    return mean, np.sqrt(max(variance, 0.0))


def dtw_with_lag(curve_a, curve_b):
    """Compute DTW distance and lag statistics between two curves."""
    dtw = _dtw_forward(curve_a, curve_b)
    lag_mean, lag_std = _dtw_backtrack(dtw)
    return dtw[-1, -1], lag_mean, lag_std


def compute_dtw_edge_features(tree_file, tree_idx=0):
    """Compute DTW-based edge features for all ordered location pairs."""
    tips_by_loc = extract_tip_times(tree_file, tree_idx)
    if not tips_by_loc:
        return None, None, None, []

    curves, dt = tips_to_curves(tips_by_loc)
    if not curves:
        return None, None, None, []

    locations = sorted(curves.keys())
    curve_arr = np.array([curves[loc] for loc in locations])

    src, dst, feats = [], [], []
    for i in range(len(locations)):
        for j in range(len(locations)):
            if i != j:
                dist, lag_mean, lag_std = dtw_with_lag(curve_arr[i], curve_arr[j])
                src.append(i)
                dst.append(j)
                feats.append([dist, lag_mean * dt, lag_std * dt])

    return src, dst, np.array(feats), locations


# ==============================================================================
# Graph Building
# ==============================================================================

def _encode_nodes(tree_file, tree_idx, subtree_width, cblv_scale):
    """Load tree and encode CBLV node features. Shared by build_graph."""
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

    # Transpose to (n_nodes, 4, subtree_width) for Conv1d
    node_cblv = np.transpose(node_cblv, (0, 2, 1))
    return node_cblv, node_aux, locations, tree_height


def build_graph(tree_file, tree_idx, subtree_width, input_folder, cblv_scale='tree_height'):
    """Build a fully connected DGL graph with CBLV nodes and DTW edges."""
    node_cblv, node_aux, locations, tree_height = _encode_nodes(
        tree_file, tree_idx, subtree_width, cblv_scale)
    n_nodes = len(locations)

    # DTW edge features
    src, dst, edge_feats, dtw_locs = compute_dtw_edge_features(tree_file, tree_idx)
    if locations != dtw_locs:
        raise ValueError(f"Location mismatch: CBLV={locations}, DTW={dtw_locs}")

    # Labels
    file_prefix = Path(tree_file).stem.replace('_beast2', '')
    labels = load_labels(input_folder, file_prefix, n_nodes)

    # Assemble graph
    g = dgl.graph((src, dst), num_nodes=n_nodes)
    g.ndata['cblv'] = torch.tensor(node_cblv, dtype=torch.float32)
    g.ndata['aux'] = torch.tensor(node_aux, dtype=torch.float32)
    g.ndata['location'] = torch.tensor(locations, dtype=torch.long)
    for name, vals in labels.items():
        g.ndata[name] = torch.tensor(vals, dtype=torch.float32)
    g.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)

    return g, locations, tree_height


def build_all_graphs(input_folder, subtree_width, file_pattern='*_beast2.trees', verbose=True, cblv_scale='tree_height'):
    """Build DGL graphs from all BEAST2 tree files in a folder."""
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
