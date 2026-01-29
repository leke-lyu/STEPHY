#!/usr/bin/env python3
"""
Graph Label Extraction Script.

Input:  *_beast2.trees, *_parameter.csv
Output: *_gl.csv (Location, Ancester_State)

Usage:  python3 gl.py /path/to/data [--force]
"""

import argparse
import re
import sys
from pathlib import Path

import dendropy
import pandas as pd
from tqdm import tqdm


def convert_annotation(match):
    """Convert BEAST2 annotation format to Newick comment metadata."""
    text = match.group(0)
    loc = re.search(r'I\{(\d+)\}', text).group(1)
    time = re.search(r'time=([\d.eE+-]+)', text).group(1)
    return f'[&location={loc},time={time}]'


def load_tree(filepath):
    """Load a BEAST2 tree file and return a dendropy Tree."""
    with open(filepath) as f:
        content = f.read()
    tree_str = re.search(r'tree\s+\w+\s*=\s*(.+)', content).group(1)
    converted = re.sub(r'\[&[^\]]+\]', convert_annotation, tree_str)
    tree = dendropy.Tree.get(
        data=converted, schema="newick", extract_comment_metadata=True
    )
    tree.is_rooted = True
    return tree


def load_parameters(param_file):
    """Load R0 and Initial_Population from parameter CSV."""
    df = pd.read_csv(param_file)
    r0 = {int(c.replace('R0_loc_', '')): float(df[c].iloc[0])
          for c in df.columns if c.startswith('R0_loc_')}
    pop = {int(c.replace('population_loc_', '')): int(df[c].iloc[0])
           for c in df.columns if c.startswith('population_loc_')}
    return {'R0': r0, 'Initial_Population': pop}


def get_mrca_location(tree):
    """Get the location of the MRCA of all sampled tips."""
    tips = list(tree.leaf_node_iter())
    mrca = tree.mrca(taxon_labels=[t.taxon.label for t in tips])
    return int(mrca.annotations.get_value('location'))


def process_simulation(tree_file, param_file, output_file, force=False):
    """Process one simulation and write graph label to CSV."""
    if output_file.exists() and not force:
        return False

    try:
        params = load_parameters(str(param_file))
        num_nodes = len(params['R0'])

        tree = load_tree(str(tree_file))
        mrca_loc = get_mrca_location(tree)

        rows = [{
            'Location': n,
            'Ancester_State': int(n == mrca_loc),
        } for n in range(num_nodes)]

        pd.DataFrame(rows).to_csv(output_file, index=False)
        return True

    except Exception as e:
        print(f"  Error processing {tree_file.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract graph labels (ancestor state) from trees')
    parser.add_argument('input_folder', help='Input data directory')
    parser.add_argument('--force', action='store_true', help='Overwrite existing *_gl.csv files')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)

    tree_files = sorted(input_folder.glob('*_beast2.trees'))
    if not tree_files:
        print(f"No tree files found in {input_folder}")
        sys.exit(0)

    print(f"Extracting graph labels: {input_folder}")
    print(f"Found {len(tree_files)} tree files\n")

    processed = skipped = errors = 0
    for tree_file in tqdm(tree_files, desc="Processing"):
        prefix = tree_file.stem.replace('_beast2', '')
        param_file = input_folder / f"{prefix}_parameter.csv"
        output_file = input_folder / f"{prefix}_gl.csv"

        if not param_file.exists():
            print(f"  Warning: No parameter file for {tree_file.name}")
            errors += 1
            continue

        result = process_simulation(tree_file, param_file, output_file, args.force)
        if result:
            processed += 1
        elif output_file.exists():
            skipped += 1
        else:
            errors += 1

    print(f"\nProcessed: {processed} | Skipped: {skipped} | Errors: {errors}")


if __name__ == '__main__':
    main()
