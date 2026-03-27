#!/usr/bin/env python3
"""Inspect BEAST2 tree datasets for tree_width and subtree_width.

Subcommands:
    check  <folder> [SUB_TOP_PCT]   Sequential scan (all batches at once).
    batch  <folder> <output.pkl>    Process one batch, save stats as pickle.
    merge  <pickle_dir> [SUB_TOP_PCT]  Load batch pickles, print merged stats.

The 'check' subcommand is the original all-in-one mode.  For large datasets,
use 'batch' (parallelised via SLURM) + 'merge' instead.
"""

import re
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

TREE_RE = re.compile(r'tree STATE_\d+ = (.+?)(?=\ntree |\nEnd;|$)', re.DOTALL)
TIP_RE  = re.compile(r'\d+\[&type="I\{(\d+)\}",samp="sample"')


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def parse_trees(root):
    """Parse all *_beast2.trees files under *root*.

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


def summarise(stats, dataset_label, sub_top_pct=1):
    """Print tree/subtree statistics, filter outliers, and recommend params."""
    locations = sorted({loc for _, _, lc in stats for loc in lc})

    print(f'Dataset:   {dataset_label}')
    print(f'Locations: {locations}')
    print(f'\n=== All trees ({len(stats)} trees) ===')
    _print_stats(stats)

    # Filter by subtree_width (outbreak removal)
    all_sub = np.array([n for _, _, lc in stats for n in lc.values()])
    cutoff  = int(np.percentile(all_sub, 100 - sub_top_pct, method='lower'))
    filtered = [s for s in stats if all(n <= cutoff for n in s[2].values())]

    print(f'\n=== Discarding trees with any subtree_width > {cutoff} '
          f'(top {sub_top_pct}% subtree threshold) ===')
    print(f'    {len(stats)} -> {len(filtered)} trees remain\n')
    _print_stats(filtered)

    max_sub = max(n for _, _, lc in filtered for n in lc.values())
    print(f'\n--num_locations {len(locations)} --subtree_width {max_sub}')


def _print_stats(stats):
    """Print min/max/mean for tree_width and subtree_width."""
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


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_check(args):
    """Sequential scan — original all-in-one mode."""
    if len(args) < 1:
        sys.exit('Usage: outbreak_check.py check <folder> [SUB_TOP_PCT]')
    folder = args[0]
    sub_top_pct = int(args[1]) if len(args) >= 2 else 1
    if not Path(folder).is_dir():
        sys.exit(f'Error: {folder} is not a directory')
    stats = parse_trees(folder)
    if not stats:
        sys.exit(f'Error: No *_beast2.trees files found in {folder}')
    summarise(stats, folder, sub_top_pct)


def cmd_batch(args):
    """Process one batch folder and pickle the stats."""
    if len(args) != 2:
        sys.exit('Usage: outbreak_check.py batch <folder> <output.pkl>')
    folder, output = args
    if not Path(folder).is_dir():
        sys.exit(f'Error: {folder} is not a directory')
    stats = parse_trees(folder)
    if not stats:
        sys.exit(f'Error: No *_beast2.trees files found in {folder}')
    with open(output, 'wb') as f:
        pickle.dump(stats, f)
    print(f'{folder}: {len(stats)} trees -> {output}')


def cmd_merge(args):
    """Load all batch pickles and print merged summary."""
    if len(args) < 1:
        sys.exit('Usage: outbreak_check.py merge <pickle_dir> [SUB_TOP_PCT]')
    pkl_dir = Path(args[0])
    sub_top_pct = int(args[1]) if len(args) >= 2 else 1
    pkl_files = sorted(pkl_dir.glob('batch_*_outbreak.pkl'))
    if not pkl_files:
        sys.exit(f'Error: No batch_*_outbreak.pkl files in {pkl_dir}')
    stats = []
    for pf in pkl_files:
        with open(pf, 'rb') as f:
            stats.extend(pickle.load(f))
    summarise(stats, str(pkl_dir), sub_top_pct)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {'check': cmd_check, 'batch': cmd_batch, 'merge': cmd_merge}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f'Usage: outbreak_check.py <{"|".join(COMMANDS)}> [args...]')
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()
