#!/usr/bin/env python3
"""
Shared tree loading, CBLV encoding, and label utilities.

Used by all three pipelines (stephy, CBLV-CNN, CBLV-GAT).
No heavy dependencies (no torch, dgl, scipy, numba).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import dendropy as dp


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
    visiting only nodes belonging to that location's subtree. Produces a
    (subtree_width, 4) matrix plus 5 auxiliary statistics.
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

        Returns: (heights, aux_stats)
            heights: (subtree_width, 4) CBLV matrix
            aux_stats: [mrca_depth, earliest_tip_time, latest_tip_time, avg_branch_length, n_tips]
        """
        mrca = self._find_mrca(loc)
        n_tips = self.phy.seed_node.loc_counts.get(loc, 0)
        if mrca is None:
            return np.zeros((subtree_width or 1, 4)), [0.0, 0.0, 0.0, 0.0, float(n_tips)]

        mrca_depth = mrca.root_distance
        n_tips = mrca.loc_counts.get(loc, 0)

        tip_dists = []
        tip_times = []
        for nd in mrca.leaf_iter():
            if is_sample(nd) and get_location(nd) == loc:
                tip_dists.append(nd.root_distance - mrca_depth)
                tip_times.append(float(nd.annotations.get_value('time')))

        aux_stats = [mrca_depth, min(tip_times), max(tip_times),
                     float(np.mean(tip_dists)), float(n_tips)]

        heights = np.zeros((n_tips, 4))
        idx = 0

        for event_type, val1, val2 in self._virtual_inorder(mrca, loc, mrca.root_distance, mrca):
            if idx >= n_tips:
                break
            if event_type == 'leaf':
                heights[idx, 0] = val1 + (mrca_depth if idx == 0 else 0)
                heights[idx, 2] = val2
            else:
                if idx + 1 < n_tips:
                    heights[idx + 1, 1] = val1
                    heights[idx + 1, 3] = val2
                idx += 1

        if cblv_scale == 'tree_height':
            heights /= self.tree_height
        elif cblv_scale == 'log1p':
            heights = np.log1p(heights)

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
# Label Loading
# ==============================================================================

def load_labels(input_folder, file_prefix, num_nodes):
    """
    Load labels from *_nf.csv file.

    Returns dict with regression (reg_r0, reg_rr, reg_sss) and
    classification (cls_r0, cls_sss, cls_as) arrays.
    """
    nf_file = Path(input_folder) / f"{file_prefix}_nf.csv"
    if not nf_file.exists():
        raise FileNotFoundError(f"Label file not found: {nf_file}")

    df = pd.read_csv(nf_file)
    if len(df) != num_nodes:
        raise ValueError(f"Label data has {len(df)} rows, expected {num_nodes}")

    reg_r0 = df['R0'].values.astype(np.float32)
    reg_rr = df['Recovery_Rate'].values.astype(np.float32)
    reg_sss = df['Source_Sink_Score'].values.astype(np.float32)

    cls_r0 = np.zeros(num_nodes, dtype=np.float32)
    cls_r0[np.argmax(reg_r0)] = 1.0

    cls_sss = np.zeros(num_nodes, dtype=np.float32)
    cls_sss[np.argmax(reg_sss)] = 1.0

    cls_as = df['Ancestral_State'].values.astype(np.float32)

    return {
        'reg_r0': reg_r0, 'cls_r0': cls_r0,
        'reg_rr': reg_rr,
        'reg_sss': reg_sss, 'cls_sss': cls_sss,
        'cls_as': cls_as,
    }
