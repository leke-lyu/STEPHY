#!/usr/bin/env python3
"""
Generate CBLV (Compact Bijective Ladderized Vector) encoding for BEAST2 trees.

Usage:
    python generate_cblv.py <tree_file> [--tree_index 0] [--tree_width 500] [--rescale] [--output output.npy]

Notes:
    - Single-child nodes (migration events in BEAST2) are automatically collapsed
    - Tree is ladderized before encoding (deepest subtrees first)
"""

import argparse
import dendropy as dp
import numpy as np


def encode_cblv(phy, tree_width, tree_encode_type='height_brlen', rescale=True):
    """
    Encode Compact Bijective Ladderized Vector (CBLV) array.

    Columns:
    - Col 0: leaf node-to-last internal node distance
    - Col 1: internal node root-distance
    - Col 2: leaf node branch length (if height_brlen)
    - Col 3: internal node branch length (if height_brlen)

    Arguments:
        phy (dendropy.Tree):     phylogenetic tree
        tree_width (int):        number of rows (max num taxa)
        tree_encode_type (str):  'height_only' (2 cols) or 'height_brlen' (4 cols)
        rescale (bool):          normalize to [0,1] if True

    Returns:
        numpy.ndarray: The encoded CBLV tensor of shape (tree_width, num_cols)
    """

    # Determine number of columns
    if tree_encode_type == 'height_only':
        num_tree_col = 2
    elif tree_encode_type == 'height_brlen':
        num_tree_col = 4
    else:
        raise ValueError(f"Unknown tree_encode_type: {tree_encode_type}")

    # Collapse single-child nodes (migration events in BEAST2)
    # A -> X (single child) -> B  becomes  A -> B
    phy.suppress_unifurcations()

    # Calculate root distances for all nodes
    phy.calc_node_root_distances(return_leaf_distances_only=False)

    # Initialize matrix
    heights = np.zeros((tree_width, num_tree_col))
    height_idx = 0

    # --- Postorder traversal: ladderize tree ---
    for nd in phy.postorder_node_iter():
        if nd.is_leaf():
            nd.max_root_distance = nd.root_distance
        else:
            children = nd.child_nodes()
            ch_max_root_distance = [ch.max_root_distance for ch in children]
            ch_max_root_distance_rank = np.argsort(ch_max_root_distance)[::-1]
            children_reordered = [children[i] for i in ch_max_root_distance_rank]
            nd.max_root_distance = max(ch_max_root_distance)
            nd.set_children(children_reordered)

    # --- Inorder traversal: fill matrix ---
    last_int_node = phy.seed_node
    if last_int_node.edge.length is None:
        last_int_node.edge.length = 0

    for nd in phy.inorder_node_iter():
        if height_idx >= tree_width:
            break

        if nd.is_leaf():
            # Leaf: fill columns 0 and 2
            heights[height_idx, 0] = nd.root_distance - last_int_node.root_distance
            if tree_encode_type == 'height_brlen':
                heights[height_idx, 2] = nd.edge.length if nd.edge.length else 0
        else:
            # Internal: fill columns 1 and 3 of next row, then increment
            if height_idx + 1 < tree_width:
                heights[height_idx + 1, 1] = nd.root_distance
                if tree_encode_type == 'height_brlen':
                    heights[height_idx + 1, 3] = nd.edge.length if nd.edge.length else 0
            last_int_node = nd
            height_idx += 1

    # Rescale if requested
    if rescale:
        max_val = np.max(heights)
        if max_val > 0:
            heights = heights / max_val

    return heights


def main():
    parser = argparse.ArgumentParser(description='Generate CBLV encoding for BEAST2 trees')
    parser.add_argument('tree_file', help='Path to NEXUS tree file')
    parser.add_argument('--tree_index', type=int, default=0, help='Which tree to encode (default: 0)')
    parser.add_argument('--tree_width', type=int, default=None, help='Number of rows (default: num leaves)')
    parser.add_argument('--rescale', action='store_true', help='Rescale to [0,1]')
    parser.add_argument('--encode_type', choices=['height_only', 'height_brlen'], default='height_brlen')
    parser.add_argument('--output', '-o', default=None, help='Output file (.npy or .csv)')

    args = parser.parse_args()

    # Read trees
    print(f"Reading: {args.tree_file}")
    trees = dp.TreeList.get(
        path=args.tree_file,
        schema="nexus",
        suppress_leaf_node_taxa=True,
        suppress_internal_node_taxa=True
    )
    print(f"Found {len(trees)} trees")

    # Get tree
    phy = trees[args.tree_index]
    print(f"Tree: {phy.label}")

    # Count leaves before and after suppressing unifurcations
    num_leaves = len([nd for nd in phy.leaf_node_iter()])
    print(f"Leaves: {num_leaves}")

    # Tree width
    tree_width = args.tree_width if args.tree_width else num_leaves

    # Encode
    cblv = encode_cblv(phy, tree_width, args.encode_type, args.rescale)

    print(f"\nCBLV shape: {cblv.shape}")

    # Show first 20 rows
    print(f"\n{'Row':<5} {'Col0':<15} {'Col1':<15} {'Col2':<15} {'Col3':<15}")
    print("-" * 70)
    for i in range(min(20, tree_width)):
        print(f"{i:<5} {cblv[i,0]:<15.4f} {cblv[i,1]:<15.4f} {cblv[i,2]:<15.4f} {cblv[i,3]:<15.4f}")

    print(f"\nMax: {np.max(cblv):.4f}, Min: {np.min(cblv):.4f}")

    # Save
    if args.output:
        if args.output.endswith('.csv'):
            np.savetxt(args.output, cblv, delimiter=',')
        else:
            np.save(args.output if args.output.endswith('.npy') else args.output + '.npy', cblv)
        print(f"Saved: {args.output}")

    return cblv


if __name__ == '__main__':
    main()
