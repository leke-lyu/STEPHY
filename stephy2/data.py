#!/usr/bin/env python3
"""
Data Loading and Preprocessing for STEPHY2 (Phylogeny-only Model + Aux Branch).

Handles data loading and graph construction for the STEPHY GAT model with DTW
edge features. Loads phylogenetic trees (*_beast2.trees) and labels (*_nf.csv),
encodes per-location tree topology via CBLV (Compact Branch-Length Vector),
computes DTW-based edge features from KDE-smoothed epidemic curves, and
assembles everything into DGL graphs.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
import dgl
import dendropy as dp
from scipy.stats import gaussian_kde
from numba import jit

from config import DATA_ARGS
from beast2_parser import parse_trees, count_trees


# ==============================================================================
# Tree Loading Utilities
# ==============================================================================

def load_tree(tree_file, tree_idx=0):
    """Load and preprocess a single tree from a BEAST2 file."""
    tree_list = dp.TreeList.get(
        path=str(tree_file), schema='nexus',
        suppress_internal_node_taxa=True, suppress_leaf_node_taxa=True
    )
    phy = tree_list[tree_idx]
    phy.is_rooted = True
    phy.suppress_unifurcations()
    phy.calc_node_root_distances()
    return phy


def get_location(node):
    """Extract location ID from node annotation (e.g., 'I{7}' -> 7)."""
    annot = node.annotations.get_value('type') if node.annotations else None
    if annot and '{' in str(annot):
        annot_str = str(annot)
        return int(annot_str.split('{')[1].split('}')[0])
    return None


def is_sample(node):
    """Check if node is an actual sample (not ancestral reconstruction)."""
    return node.annotations and node.annotations.get_value('samp') == 'sample'


# ==============================================================================
# CBLV Encoder (Compact Branch-Length Vector)
#
# Encodes tree topology as a (subtree_width x 4) matrix per location via
# virtual in-order traversal. No tree copying required.
#
# Matrix columns:
#   0: Leaf distances from previous branch point
#   1: Internal node depths (root distances)
#   2: Accumulated edge lengths to leaves
#   3: Accumulated edge lengths to internal nodes
# ==============================================================================

class VirtualSubtreeEncoder:
    """
    CBLV encoder using virtual subtree traversal (no tree copying).

    For each location, performs a virtual in-order traversal of the full tree,
    visiting only nodes belonging to that location's subtree. This avoids the
    cost of copying/pruning the tree for each location. Produces a
    (subtree_width, 4) matrix encoding branch lengths, node depths, and
    accumulated edge lengths, plus 5 auxiliary statistics (MRCA depth,
    earliest/latest tip times, average branch length, tip count).
    """

    def __init__(self, phy, tree_height):
        self.phy = phy
        self.tree_height = tree_height
        self._precompute_location_stats()

    def _precompute_location_stats(self):
        """Bottom-up pass: compute location counts and max distances per node."""
        for nd in self.phy.postorder_node_iter():
            if nd.is_leaf():
                loc = get_location(nd)
                if is_sample(nd) and loc is not None:
                    nd.loc_counts = {loc: 1}
                    nd.loc_max_dist = {loc: nd.root_distance}
                else:
                    nd.loc_counts, nd.loc_max_dist = {}, {}
            else:
                nd.loc_counts, nd.loc_max_dist = {}, {}
                for child in nd.child_nodes():
                    for loc, cnt in child.loc_counts.items():
                        nd.loc_counts[loc] = nd.loc_counts.get(loc, 0) + cnt
                    for loc, dist in child.loc_max_dist.items():
                        nd.loc_max_dist[loc] = max(nd.loc_max_dist.get(loc, 0), dist)

    def _find_mrca(self, loc):
        """Find MRCA of all tips with target location."""
        total = self.phy.seed_node.loc_counts.get(loc, 0)
        if total < 2:
            return None
        mrca = self.phy.seed_node
        while True:
            children = [c for c in mrca.child_nodes() if c.loc_counts.get(loc, 0) == total]
            if len(children) == 1:
                mrca = children[0]
            else:
                break
        return mrca

    def _is_branch_point(self, node, loc):
        """Check if node is a branching point for target location."""
        return not node.is_leaf() and sum(1 for c in node.child_nodes() if c.loc_counts.get(loc, 0) > 0) >= 2

    def _find_parent_branch(self, node, loc, mrca):
        """Find nearest ancestor that is a branching point for target location."""
        current = node.parent_node
        while current and current != mrca:
            if self._is_branch_point(current, loc):
                return current
            current = current.parent_node
        return mrca

    def _accum_edge(self, node, parent_branch):
        """Compute accumulated edge length from node up to parent branching point."""
        total, current = 0, node
        while current and current != parent_branch:
            total += current.edge.length or 0
            current = current.parent_node
        return total

    def _virtual_inorder(self, node, loc, last_branch_dist, mrca):
        """Generator for virtual in-order traversal of location subtree."""
        if node.is_leaf():
            if is_sample(node) and get_location(node) == loc:
                parent = self._find_parent_branch(node, loc, mrca)
                yield ('leaf', node.root_distance - last_branch_dist, self._accum_edge(node, parent))
        elif self._is_branch_point(node, loc):
            children = sorted(
                [c for c in node.child_nodes() if c.loc_counts.get(loc, 0) > 0],
                key=lambda c: c.loc_max_dist.get(loc, 0), reverse=True
            )
            yield from self._virtual_inorder(children[0], loc, last_branch_dist, mrca)

            accum = (node.edge.length or 0) if node == mrca else self._accum_edge(node, self._find_parent_branch(node, loc, mrca))
            yield ('internal', node.root_distance, accum)

            for child in children[1:]:
                yield from self._virtual_inorder(child, loc, node.root_distance, mrca)
        else:
            relevant = [c for c in node.child_nodes() if c.loc_counts.get(loc, 0) > 0]
            if relevant:
                yield from self._virtual_inorder(relevant[0], loc, last_branch_dist, mrca)

    def encode_cblv(self, loc, subtree_width=None, cblv_scale='tree_height'):
        """
        Encode CBLV matrix and auxiliary statistics for a target location.

        Args:
            loc: Target location ID
            subtree_width: Pad/truncate to this many rows
            cblv_scale: 'tree_height' (divide by tree height -> [0,1]) or 'log1p' (log(x+1))

        Returns: (heights, aux_stats)
            heights: (subtree_width, 4) CBLV matrix, scaled according to cblv_scale
            aux_stats: [mrca_depth, earliest_tip_time, latest_tip_time, avg_branch_length, n_tips]
        """
        mrca = self._find_mrca(loc)
        n_tips = self.phy.seed_node.loc_counts.get(loc, 0)
        if mrca is None:
            return np.zeros((subtree_width or 1, 4)), [0.0, 0.0, 0.0, 0.0, float(n_tips)]

        mrca_depth = mrca.root_distance
        n_tips = mrca.loc_counts.get(loc, 0)

        # Compute avg path length from MRCA to each tip and collect tip sampling times
        tip_dists = []
        tip_times = []
        for nd in mrca.leaf_iter():
            if is_sample(nd) and get_location(nd) == loc:
                tip_dists.append(nd.root_distance - mrca_depth)
                tip_times.append(float(nd.annotations.get_value('time')))

        aux_stats = [mrca_depth, min(tip_times), max(tip_times),
                     float(np.mean(tip_dists)), float(n_tips)]

        # Fill CBLV matrix from virtual in-order traversal.
        # idx tracks the current leaf position; internal nodes advance idx.
        # Columns: 0=leaf dist from branch, 1=internal depth, 2=leaf accum edge, 3=internal accum edge
        heights = np.zeros((n_tips, 4))
        idx = 0

        for event_type, val1, val2 in self._virtual_inorder(mrca, loc, mrca.root_distance, mrca):
            if idx >= n_tips:
                break
            if event_type == 'leaf':
                # First leaf gets absolute distance (includes mrca_depth); others get relative
                heights[idx, 0] = val1 + (mrca_depth if idx == 0 else 0)
                heights[idx, 2] = val2  # accumulated edge length to this leaf
            else:
                # Internal node: store at next leaf's row, then advance position
                if idx + 1 < n_tips:
                    heights[idx + 1, 1] = val1  # internal node depth
                    heights[idx + 1, 3] = val2  # accumulated edge length to internal node
                idx += 1

        if cblv_scale == 'tree_height':
            heights /= self.tree_height
        elif cblv_scale == 'log1p':
            heights = np.log1p(heights)

        # Pad to subtree_width
        if subtree_width and n_tips != subtree_width:
            padded = np.zeros((subtree_width, 4))
            padded[:min(n_tips, subtree_width)] = heights[:min(n_tips, subtree_width)]
            heights = padded

        return heights, aux_stats

    def get_all_locations(self):
        """Get sorted list of all sample locations in the tree."""
        return sorted({get_location(nd) for nd in self.phy.leaf_node_iter()
                      if is_sample(nd) and get_location(nd) is not None})


# ==============================================================================
# DTW Edge Features
# ==============================================================================

def extract_tip_times(tree_file, tree_idx=0):
    """
    Extract tip sampling times grouped by location from a BEAST2 tree file.

    Parses the raw NEXUS tree string with regex to find annotated sample tips,
    returning a dict mapping location ID to list of sampling times.
    """
    with open(tree_file) as f:
        trees = parse_trees(f.read())
    if not trees or tree_idx >= len(trees):
        return {}

    tips = defaultdict(list)
    for loc, time in re.findall(r'\d+\[&type="I\{(\d+)\}",samp="sample",time=([\d.]+)\]', trees[tree_idx]):
        tips[int(loc)].append(float(time))
    return dict(tips)


def tips_to_curves(tips_by_loc, num_points=None):
    """
    Convert tip sampling times to KDE-smoothed epidemic curves.

    Uses Gaussian KDE to produce a smooth density estimate for each location,
    evaluated on a shared time grid. Locations with fewer than 2 tips get a
    zero curve. Returns the curves dict and the time step (dt) for lag scaling.
    """
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

        # Find min of three neighbors
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
    std = np.sqrt(max(variance, 0.0))

    return mean, std


def dtw_with_lag(curve_a, curve_b):
    """
    Compute DTW distance and lag statistics between two curves.
    Uses numba JIT compilation for ~50-100x speedup.
    """
    dtw = _dtw_forward(curve_a, curve_b)
    lag_mean, lag_std = _dtw_backtrack(dtw)
    return dtw[-1, -1], lag_mean, lag_std


def compute_dtw_edge_features(tree_file, tree_idx=0):
    """
    Compute DTW-based edge features for all ordered location pairs.

    For each pair (i, j) where i != j, computes three features:
    DTW distance, lag mean (scaled by dt), and lag std (scaled by dt).
    Returns source/dest index lists, feature array, and sorted location IDs.
    """
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
# Label Loading
# ==============================================================================

def load_labels(input_folder, file_prefix, num_nodes):
    """
    Load labels from *_nf.csv file.

    Args:
        input_folder: Path to data folder
        file_prefix: File prefix (e.g., '0' for '0_nf.csv')
        num_nodes: Expected number of nodes (locations)

    Returns:
        labels: dict with 'R0', 'Source_Sink_Score', 'Recovery_Rate', and
                'Ancestral_State' arrays
    """
    input_folder = Path(input_folder)
    nf_file = input_folder / f"{file_prefix}_nf.csv"

    if not nf_file.exists():
        raise FileNotFoundError(f"Label file not found: {nf_file}")

    df = pd.read_csv(nf_file)

    if len(df) != num_nodes:
        raise ValueError(f"Label data has {len(df)} rows, expected {num_nodes}")

    # Extract labels
    labels = {
        'R0': df['R0'].values.astype(np.float32),
        'Source_Sink_Score': df['Source_Sink_Score'].values.astype(np.float32),
        'Recovery_Rate': df['Recovery_Rate'].values.astype(np.float32),
        'Ancestral_State': df['Ancestral_State'].values.astype(np.float32),
    }

    return labels


# ==============================================================================
# Graph Building
# ==============================================================================

def build_graph(tree_file, tree_idx, subtree_width, input_folder, cblv_scale='tree_height'):
    """
    Build a single DGL graph from a BEAST2 tree.

    Constructs a fully connected graph (no self-loops) where each node is a
    geographic location. Node features are CBLV matrices (4, subtree_width) plus
    5-dim auxiliary stats. Edge features are DTW-derived (distance, lag_mean,
    lag_std). Labels are loaded from the corresponding *_nf.csv file.

    Returns:
        g: DGL graph with node features (cblv, aux, labels) and edge features
        locations: Sorted list of location IDs
        tree_height: Height of the tree (max root distance among leaves)
    """
    phy = load_tree(tree_file, tree_idx)
    tree_height = max(nd.root_distance for nd in phy.leaf_node_iter())

    # Node features (CBLV)
    encoder = VirtualSubtreeEncoder(phy, tree_height)
    locations = encoder.get_all_locations()
    n_nodes = len(locations)

    # CBLV shape: (n_nodes, subtree_width, 4) -> transpose to (n_nodes, 4, subtree_width)
    node_cblv = np.zeros((n_nodes, subtree_width, 4))
    node_aux = np.zeros((n_nodes, 5))

    for i, loc in enumerate(locations):
        cblv, aux_stats = encoder.encode_cblv(loc, subtree_width=subtree_width, cblv_scale=cblv_scale)
        node_cblv[i] = cblv
        node_aux[i] = aux_stats

    # Transpose to (n_nodes, 4, subtree_width) for Conv1d
    node_cblv = np.transpose(node_cblv, (0, 2, 1))

    # Edge features (DTW)
    src, dst, edge_feats, dtw_locs = compute_dtw_edge_features(tree_file, tree_idx)
    if locations != dtw_locs:
        raise ValueError(f"Location mismatch: CBLV={locations}, DTW={dtw_locs}")

    # Load labels from preprocessed CSV
    file_prefix = Path(tree_file).stem.replace('_beast2', '')
    labels = load_labels(input_folder, file_prefix, n_nodes)

    # Construct graph
    g = dgl.graph((src, dst), num_nodes=n_nodes)
    g.ndata['cblv'] = torch.tensor(node_cblv, dtype=torch.float32)
    g.ndata['aux'] = torch.tensor(node_aux, dtype=torch.float32)
    g.ndata['location'] = torch.tensor(locations, dtype=torch.long)
    g.ndata['R0'] = torch.tensor(labels['R0'], dtype=torch.float32)
    g.ndata['Source_Sink_Score'] = torch.tensor(labels['Source_Sink_Score'], dtype=torch.float32)
    g.ndata['Recovery_Rate'] = torch.tensor(labels['Recovery_Rate'], dtype=torch.float32)
    g.ndata['Ancestral_State'] = torch.tensor(labels['Ancestral_State'], dtype=torch.float32)
    g.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)

    return g, locations, tree_height


def build_all_graphs(input_folder, subtree_width, file_pattern='*_beast2.trees', verbose=True, cblv_scale='tree_height'):
    """
    Build DGL graphs from all BEAST2 tree files in a folder.

    Iterates over all matching tree files, and for each file processes every
    tree (some files contain multiple posterior trees). Errors on individual
    trees are caught and reported rather than aborting the entire batch.

    Returns:
        List of (graph, graph_id, locations, tree_height) tuples
    """
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
