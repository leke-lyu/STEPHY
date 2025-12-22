#!/usr/bin/env python3
"""
Extract CBLV (Compact Bijective Ladderized Vector) node features from phylogenetic trees.

Uses VirtualSubtreeEncoder (Option 3) for efficient encoding without tree copying.

Usage:
    python3 cblv_feature.py <tree_file> [output_file] [--tree_width N]

Output columns:
    graph_id, node, cblv_0, cblv_1, ..., cblv_{tree_width*4-1}
"""

import sys
import argparse
import numpy as np
import pandas as pd
import dendropy as dp


# ============================================================
# Helper functions
# ============================================================

def get_location(node):
    """Extract location number from 'type' annotation like 'I{7}'"""
    type_annot = node.annotations.get_value('type') if node.annotations else None
    if type_annot and '{' in str(type_annot):
        loc = str(type_annot).split('{')[1].split('}')[0]
        return int(loc)
    return None


def is_sample(node):
    """Check if node is an actual sample (not ancestral reconstruction)"""
    samp_annot = node.annotations.get_value('samp') if node.annotations else None
    return samp_annot == 'sample'


# ============================================================
# VirtualSubtreeEncoder (Option 3 from cblv_demo.ipynb)
# ============================================================

class VirtualSubtreeEncoder:
    """
    Efficient CBLV encoder that avoids creating subtree objects.

    Preprocessing (once): O(N) to annotate each node with location counts
    Per-location encoding: O(M) where M is number of tips in that location
    """

    def __init__(self, phy, tree_height):
        self.phy = phy
        self.tree_height = tree_height
        phy.calc_node_root_distances()
        self._preprocess()

    def _preprocess(self):
        """Compute location counts and max distances for each node."""
        for nd in self.phy.postorder_node_iter():
            if nd.is_leaf():
                loc = get_location(nd)
                if is_sample(nd) and loc is not None:
                    nd.loc_counts = {loc: 1}
                    nd.loc_max_dist = {loc: nd.root_distance}
                else:
                    nd.loc_counts = {}
                    nd.loc_max_dist = {}
            else:
                nd.loc_counts = {}
                nd.loc_max_dist = {}
                for child in nd.child_nodes():
                    for loc, count in getattr(child, 'loc_counts', {}).items():
                        nd.loc_counts[loc] = nd.loc_counts.get(loc, 0) + count
                    for loc, dist in getattr(child, 'loc_max_dist', {}).items():
                        nd.loc_max_dist[loc] = max(nd.loc_max_dist.get(loc, 0), dist)

    def _find_mrca(self, target_loc):
        """Find MRCA for a target location."""
        total_count = self.phy.seed_node.loc_counts.get(target_loc, 0)
        if total_count < 2:
            return None
        mrca = self.phy.seed_node
        while True:
            children_with_all = [c for c in mrca.child_nodes()
                                 if c.loc_counts.get(target_loc, 0) == total_count]
            if len(children_with_all) == 1:
                mrca = children_with_all[0]
            else:
                break
        return mrca

    def _is_branching_point(self, node, target_loc):
        """Check if node is a branching point for target location."""
        if node.is_leaf():
            return False
        relevant_children = [c for c in node.child_nodes()
                            if c.loc_counts.get(target_loc, 0) > 0]
        return len(relevant_children) >= 2

    def _find_parent_branching_point(self, node, target_loc, mrca):
        """Find the nearest ancestor that is a branching point for target location."""
        current = node.parent_node
        while current is not None:
            if current == mrca or self._is_branching_point(current, target_loc):
                return current
            current = current.parent_node
        return mrca

    def _compute_accumulated_edge(self, node, parent_branch):
        """Compute accumulated edge length from node up to (but not including) parent_branch."""
        total = 0
        current = node
        while current != parent_branch and current is not None:
            total += current.edge.length if current.edge.length else 0
            current = current.parent_node
        return total

    def _virtual_inorder(self, node, target_loc, last_branch_root_dist, mrca):
        """Generator for virtual inorder traversal."""
        if node.is_leaf():
            loc = get_location(node)
            if is_sample(node) and loc == target_loc:
                parent_branch = self._find_parent_branching_point(node, target_loc, mrca)
                accum_edge = self._compute_accumulated_edge(node, parent_branch)
                dist_from_last = node.root_distance - last_branch_root_dist
                yield ('leaf', node, dist_from_last, accum_edge)

        elif self._is_branching_point(node, target_loc):
            relevant_children = [c for c in node.child_nodes()
                                if c.loc_counts.get(target_loc, 0) > 0]
            relevant_children.sort(key=lambda c: c.loc_max_dist.get(target_loc, 0),
                                  reverse=True)

            first_child = relevant_children[0]
            yield from self._virtual_inorder(first_child, target_loc, last_branch_root_dist, mrca)

            if node == mrca:
                accum_edge = node.edge.length if node.edge.length else 0
            else:
                parent_branch = self._find_parent_branching_point(node, target_loc, mrca)
                accum_edge = self._compute_accumulated_edge(node, parent_branch)
            yield ('internal', node, node.root_distance, accum_edge)

            for child in relevant_children[1:]:
                yield from self._virtual_inorder(child, target_loc, node.root_distance, mrca)
        else:
            relevant_children = [c for c in node.child_nodes()
                                if c.loc_counts.get(target_loc, 0) > 0]
            if relevant_children:
                child = relevant_children[0]
                yield from self._virtual_inorder(child, target_loc, last_branch_root_dist, mrca)

    def encode_cblv(self, target_loc, tree_width=None, rescale=True):
        """
        Compute CBLV for a target location without creating subtree.

        Args:
            target_loc: Location ID to encode
            tree_width: If provided, pad output to this many rows (default: actual n_tips)
            rescale: Whether to normalize by tree height

        Returns:
            (cblv_matrix, stem_distance, n_tips)
        """
        mrca = self._find_mrca(target_loc)
        if mrca is None:
            n_tips = self.phy.seed_node.loc_counts.get(target_loc, 0)
            if tree_width:
                return np.zeros((tree_width, 4)), 0, n_tips
            return None, 0, n_tips

        stem_distance = mrca.root_distance
        n_tips = mrca.loc_counts.get(target_loc, 0)
        heights = np.zeros((n_tips, 4))
        height_idx = 0

        for event in self._virtual_inorder(mrca, target_loc, mrca.root_distance, mrca):
            if height_idx >= n_tips:
                break

            if event[0] == 'leaf':
                _, node, dist_from_last_branch, accum_edge_len = event
                if height_idx == 0:
                    heights[height_idx, 0] = dist_from_last_branch + stem_distance
                else:
                    heights[height_idx, 0] = dist_from_last_branch
                heights[height_idx, 2] = accum_edge_len

            elif event[0] == 'internal':
                _, node, node_root_dist, accum_edge_len = event
                if height_idx + 1 < n_tips:
                    heights[height_idx + 1, 1] = node_root_dist
                    heights[height_idx + 1, 3] = accum_edge_len
                height_idx += 1

        if rescale:
            heights = heights / self.tree_height

        # Pad to tree_width if specified
        if tree_width and tree_width > n_tips:
            padded = np.zeros((tree_width, 4))
            padded[:n_tips, :] = heights
            heights = padded
        elif tree_width and tree_width < n_tips:
            # Truncate if more tips than tree_width
            heights = heights[:tree_width, :]

        return heights, stem_distance, n_tips

    def get_all_locations(self):
        """Return sorted list of all locations in the tree."""
        all_locs = set()
        for nd in self.phy.leaf_node_iter():
            loc = get_location(nd)
            if loc is not None and is_sample(nd):
                all_locs.add(loc)
        return sorted(all_locs)


# ============================================================
# Main function
# ============================================================

def extract_cblv_features(tree_file, tree_width=300):
    """
    Extract CBLV features for all locations in all trees.

    Args:
        tree_file: Path to BEAST2 tree file (NEXUS format)
        tree_width: Number of rows to pad CBLV matrix (default: 300)

    Returns:
        DataFrame with columns: graph_id, node, cblv_0, cblv_1, ..., cblv_{tree_width*4-1}
    """
    # Load trees
    tree_list = dp.TreeList.get(
        path=tree_file,
        schema='nexus',
        suppress_internal_node_taxa=True,
        suppress_leaf_node_taxa=True
    )
    print(f"Loaded {len(tree_list)} trees from {tree_file}", file=sys.stderr)

    # Extract file prefix for graph_id
    file_prefix = tree_file.split('/')[-1]
    file_prefix = file_prefix.replace('_beast2.trees', '').replace('.trees', '')

    all_results = []

    for tree_idx, phy in enumerate(tree_list):
        phy.is_rooted = True
        phy.suppress_unifurcations()
        phy.calc_node_root_distances()

        tree_height = max(nd.root_distance for nd in phy.leaf_node_iter())
        encoder = VirtualSubtreeEncoder(phy, tree_height)
        locations = encoder.get_all_locations()

        graph_id = f"{file_prefix}_{tree_idx}"

        for loc in locations:
            cblv, stem, n_tips = encoder.encode_cblv(loc, tree_width=tree_width, rescale=True)

            # Flatten CBLV matrix to 1D: (tree_width, 4) -> (tree_width * 4,)
            cblv_flat = cblv.flatten()

            row = {
                'graph_id': graph_id,
                'node': loc,
                'n_tips': n_tips,
                'stem_distance': stem / tree_height if tree_height > 0 else 0,
            }

            # Add CBLV columns
            for i, val in enumerate(cblv_flat):
                row[f'cblv_{i}'] = val

            all_results.append(row)

        print(f"  Tree {tree_idx}: {len(locations)} locations, height={tree_height:.2f}", file=sys.stderr)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description='Extract CBLV node features from phylogenetic trees.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 cblv_feature.py tree.trees output.csv
    python3 cblv_feature.py tree.trees output.csv --tree_width 500
    python3 cblv_feature.py tree.trees  # prints to stdout
        """
    )
    parser.add_argument('tree_file', help='BEAST2 tree file (NEXUS format)')
    parser.add_argument('output_file', nargs='?', help='Output CSV file (default: stdout)')
    parser.add_argument('--tree_width', type=int, default=300,
                        help='Pad CBLV matrix to this many rows (default: 300)')

    args = parser.parse_args()

    # Extract features
    result_df = extract_cblv_features(args.tree_file, tree_width=args.tree_width)

    # Output
    if args.output_file:
        result_df.to_csv(args.output_file, index=False)
        print(f"Saved {len(result_df)} rows to {args.output_file}", file=sys.stderr)
    else:
        print(result_df.to_csv(index=False))


if __name__ == "__main__":
    main()
