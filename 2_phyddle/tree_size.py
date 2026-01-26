#!/usr/bin/env python3
"""
Check tree sizes (number of taxa) in a folder of Newick files.

Usage:
    python3 tree_size.py /path/to/sim_data
"""

import os, sys
import dendropy as dp


def get_tree_sizes(folder):
    """Get (filename, num_taxa) for each .tre file, excluding downsampled."""
    sizes = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.tre') and 'downsampled' not in f:
            try:
                tree = dp.Tree.get(path=os.path.join(folder, f), schema='newick')
                sizes.append((f, len(tree.leaf_nodes())))
            except Exception as e:
                print(f'Warning: {f}: {e}')
    return sizes


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 tree_size.py /path/to/folder')
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f'Error: {folder} is not a directory')
        sys.exit(1)

    sizes = get_tree_sizes(folder)
    if not sizes:
        print('No .tre files found')
        sys.exit(1)

    counts = [s[1] for s in sizes]
    print(f'Folder: {folder}')
    print(f'Trees:  {len(sizes)}')
    print(f'Min:    {min(counts)} ({sizes[counts.index(min(counts))][0]})')
    print(f'Max:    {max(counts)} ({sizes[counts.index(max(counts))][0]})')
    print(f'Suggested tree_width: {max(counts)}')


if __name__ == '__main__':
    main()
