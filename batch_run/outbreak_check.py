#!/usr/bin/env python3
"""Inspect BEAST2 tree datasets for tree_width and subtree_width.

Recursively finds *_beast2.trees under the input folder, supporting:
    folder/*_beast2.trees           (single batch)
    folder/batch_*/*_beast2.trees   (multi-batch)

Reports min/max/mean statistics and recommends --num_locations and
--subtree_width parameters for the build_graphs step.

Usage:
    python3 outbreak_check.py /path/to/epidata/folder [SUB_TOP_PCT]

Arguments:
    folder        Path to the dataset directory.
    SUB_TOP_PCT   Top percentile of subtree_width to discard (default: 1).
"""

import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Regex to extract individual trees and tip locations from BEAST2 .trees files
TREE_RE = re.compile(r'tree STATE_\d+ = (.+?)(?=\ntree |\nEnd;|$)', re.DOTALL)
TIP_RE  = re.compile(r'\d+\[&type="I\{(\d+)\}",samp="sample"')


# NOTE: This parse_trees() duplicates regex logic from beast2_parser.py but has
# a different interface — it walks the filesystem (takes a root directory path)
# and returns per-tree statistics, whereas beast2_parser.parse_trees() operates
# on in-memory file content and returns raw tree strings.
def parse_trees(root):
    """Parse all *_beast2.trees files under `root`.

    Returns list of (tree_id, total_tips, {location: tip_count}).
    """
    root = Path(root)
    stats = []
    for tf in sorted(root.rglob('*_beast2.trees')):
        batch  = tf.parent.name
        prefix = tf.stem.replace('_beast2', '')
        tid    = f'{batch}/{prefix}' if batch != root.name else prefix
        for idx, tree_str in enumerate(TREE_RE.findall(tf.read_text())):
            lc = defaultdict(int)
            for loc in TIP_RE.findall(tree_str):
                lc[int(loc)] += 1
            stats.append((f'{tid}_{idx}', sum(lc.values()), dict(lc)))
    return stats


def print_stats(stats):
    """Print min/max/mean for tree_width (total tips) and subtree_width (tips per location)."""
    sizes = np.array([s[1] for s in stats])
    min_t = min(stats, key=lambda s: s[1])
    max_t = max(stats, key=lambda s: s[1])

    subs      = [(tid, loc, n) for tid, _, lc in stats for loc, n in lc.items()]
    sub_sizes = np.array([s[2] for s in subs])
    min_s     = min(subs, key=lambda s: s[2])
    max_s     = max(subs, key=lambda s: s[2])

    print(f'  tree_width     — Range: [{min_t[1]}, {max_t[1]}],  Mean: {sizes.mean():.1f}')
    print(f'    Min: {min_t[0]} ({min_t[1]} tips)')
    print(f'    Max: {max_t[0]} ({max_t[1]} tips)')
    print(f'  subtree_width  — Range: [{min_s[2]}, {max_s[2]}],  Mean: {sub_sizes.mean():.1f}')
    print(f'    Min: tree={min_s[0]}, loc={min_s[1]} ({min_s[2]} tips)')
    print(f'    Max: tree={max_s[0]}, loc={max_s[1]} ({max_s[2]} tips)')


def main():
    """CLI entry point: parse args, load trees, print summary, and recommend parameters."""
    if len(sys.argv) < 2:
        print('Usage: python3 outbreak_check.py /path/to/epidata/folder [SUB_TOP_PCT]')
        sys.exit(1)

    dataset = sys.argv[1]
    sub_top_pct = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    if not Path(dataset).is_dir():
        print(f'Error: {dataset} is not a directory')
        sys.exit(1)

    stats = parse_trees(dataset)
    if not stats:
        print(f'Error: No *_beast2.trees files found in {dataset}')
        sys.exit(1)

    locations = sorted({loc for _, _, lc in stats for loc in lc})

    # --- Original summary ------------------------------------------------------
    print(f'Dataset:   {dataset}')
    print(f'Locations: {locations}')
    print(f'\n=== All trees ({len(stats)} trees) ===')
    print_stats(stats)

    # --- Filter by subtree_width (outbreak removal) ----------------------------
    all_sub_sizes = np.array([n for _, _, lc in stats for n in lc.values()])
    sub_cutoff = int(np.percentile(all_sub_sizes, 100 - sub_top_pct, method='lower'))

    filtered = [s for s in stats if all(n <= sub_cutoff for n in s[2].values())]

    print(f'\n=== Discarding trees with any subtree_width > {sub_cutoff} '
          f'(top {sub_top_pct}% subtree threshold) ===')
    print(f'    {len(stats)} -> {len(filtered)} trees remain\n')
    print_stats(filtered)

    max_subtree = max(n for _, _, lc in filtered for n in lc.values())
    print(f'\n--num_locations {len(locations)} --subtree_width {max_subtree}')


if __name__ == '__main__':
    main()
