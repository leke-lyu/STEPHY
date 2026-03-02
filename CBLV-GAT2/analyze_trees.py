#!/usr/bin/env python3
"""
Analyze BEAST2 tree files to determine graph-building parameters.

Parses ``*_beast2.trees`` files in a given folder and reports the number of
distinct locations (``num_locations``) and the maximum number of tips belonging
to any single location across all trees (``subtree_width``).  These two values
are required by ``build_graphs.py`` and ``train.py``.

Usage:
    python3 analyze_trees.py /path/to/epidata/folder
"""

import sys
import re
from collections import defaultdict
from pathlib import Path

from beast2_parser import parse_trees


def analyze_beast2_trees(input_folder):
    """Analyze BEAST2 tree files and return statistics."""
    input_folder = Path(input_folder)
    tree_files = sorted(input_folder.glob('*_beast2.trees'))

    if not tree_files:
        print(f"Error: No *_beast2.trees files found in {input_folder}")
        sys.exit(1)

    # Track statistics
    tree_stats = []  # (tree_id, total_tips, loc_counts)

    for tree_file in tree_files:
        file_prefix = tree_file.stem.replace('_beast2', '')
        with open(tree_file) as f:
            content = f.read()

        trees = parse_trees(content)

        for idx, tree_str in enumerate(trees):
            tree_id = f"{file_prefix}_{idx}"
            tips = re.findall(r'\d+\[&type="I\{(\d+)\}",samp="sample"', tree_str)

            loc_counts = defaultdict(int)
            for loc in tips:
                loc_counts[int(loc)] += 1

            tree_stats.append((tree_id, len(tips), dict(loc_counts)))

    return tree_stats


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 analyze_trees.py /path/to/epidata/folder')
        sys.exit(1)

    input_folder = sys.argv[1]

    if not Path(input_folder).is_dir():
        print(f"Error: {input_folder} is not a directory")
        sys.exit(1)

    # Analyze
    tree_stats = analyze_beast2_trees(input_folder)

    # Compute statistics
    num_trees = len(tree_stats)

    # tree_width: total tips per tree
    min_tree = min(tree_stats, key=lambda t: t[1])
    max_tree = max(tree_stats, key=lambda t: t[1])

    # subtree_width: max tips per location per tree
    subtree_stats = [(tree_id, loc, count)
                     for tree_id, _, loc_counts in tree_stats
                     for loc, count in loc_counts.items()]
    min_subtree = min(subtree_stats, key=lambda x: x[2])
    max_subtree = max(subtree_stats, key=lambda x: x[2])

    # num_locations
    all_locations = {loc for _, _, loc_counts in tree_stats for loc in loc_counts}
    num_locations = len(all_locations)

    # Output
    print(f"Input: {input_folder}")
    print(f"Trees analyzed: {num_trees}")

    print("\n" + "-" * 50)
    print("tree_width (total tips per tree)")
    print("-" * 50)
    print(f"  Range: [{min_tree[1]}, {max_tree[1]}]")
    print(f"  Min: {min_tree[0]} ({min_tree[1]} tips)")
    print(f"  Max: {max_tree[0]} ({max_tree[1]} tips)")

    print("\n" + "-" * 50)
    print("subtree_width (tips per location)")
    print("-" * 50)
    print(f"  Range: [{min_subtree[2]}, {max_subtree[2]}]")
    print(f"  Min: tree={min_subtree[0]}, loc={min_subtree[1]} ({min_subtree[2]} tips)")
    print(f"  Max: tree={max_subtree[0]}, loc={max_subtree[1]} ({max_subtree[2]} tips)")

    print("\n" + "-" * 50)
    print("FOR train.py:")
    print(f"  --num_locations {num_locations}")
    print(f"  --subtree_width {max_subtree[2]}")


if __name__ == '__main__':
    main()
