#!/usr/bin/env python3
"""
Convert BEAST2/ReMaster simulation output to phyddle format.

Input:  {i}_beast2.trees (NEXUS), {i}_nf.csv (R0 and Source_Sink_Score per location)
Output: sim.{i}.tre (Newick), sim.{i}.dat.nex (states), sim.{i}.labels.csv

Auto-detects num_locations from _nf.csv files.
Outputs tree size statistics for phyddle config.

Usage:
    python3 convert_to_phyddle.py --input_dir /path/to/folder --output_dir ./sim_data
"""

import os
import re
import argparse
import pandas as pd
import dendropy as dp


def parse_args():
    p = argparse.ArgumentParser(description='Convert BEAST2 data to phyddle format')
    p.add_argument('--input_dir', required=True, help='Input directory')
    p.add_argument('--output_dir', required=True, help='Output directory')
    p.add_argument('--prefix', default='sim', help='Output file prefix')
    p.add_argument('--max_trees_per_sim', type=int, default=1, help='Trees per simulation')
    return p.parse_args()


def find_simulations(input_dir):
    """Find simulation indices with both {i}_beast2.trees and {i}_nf.csv."""
    tree_idx = {int(m.group(1)) for f in os.listdir(input_dir)
                if (m := re.match(r'(\d+)_beast2\.trees', f))}
    nf_idx = {int(m.group(1)) for f in os.listdir(input_dir)
              if (m := re.match(r'(\d+)_nf\.csv', f))}
    return sorted(tree_idx & nf_idx)


def detect_num_locations(input_dir, sim_indices):
    """Detect num_locations from first _nf.csv file."""
    if not sim_indices:
        return None
    return len(pd.read_csv(os.path.join(input_dir, f'{sim_indices[0]}_nf.csv')))


def count_trees(nexus_file):
    """Count trees in a NEXUS file."""
    with open(nexus_file) as f:
        return len(re.findall(r'tree\s+\S+\s*=', f.read(), re.IGNORECASE))


def extract_trees(nexus_file, max_trees):
    """Extract Newick trees from NEXUS."""
    with open(nexus_file) as f:
        content = f.read()
    pattern = r'tree\s+(\S+)\s*=\s*(.+?)(?=tree\s+\S+\s*=|;?\s*End\s*;|$)'
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    return [(n.strip(), t.strip().rstrip(';')) for n, t in matches[:max_trees]]


def convert_annotations(tree_str):
    """Convert BEAST2 annotations: 1[&type="I{6}",...] -> t1[&type="I",location="6",...]"""
    taxon_locations = {}

    def replace_tip(m):
        tid, loc, rest = m.groups()
        taxon_locations[f't{tid}'] = int(loc)
        return f't{tid}[&type="I",location="{loc}"{rest}]'

    tree_str = re.sub(r'(\d+)\[\&type="I\{(\d+)\}"(.*?)\]', replace_tip, tree_str)
    tree_str = re.sub(r'\[\&type="I\{(\d+)\}"(.*?)\]', r'[&type="I",location="\1"\2]', tree_str)
    return tree_str, taxon_locations


def save_tree(tree_str, output_file):
    """Save tree as binary Newick. Returns number of taxa."""
    tree_str = tree_str if tree_str.endswith(';') else tree_str + ';'
    try:
        tree = dp.Tree.get(data=tree_str, schema='newick')
        tree.suppress_unifurcations()
        tree.resolve_polytomies()
        num_taxa = len(tree.leaf_nodes())
        tree.write(path=output_file, schema='newick')
        return num_taxa
    except Exception as e:
        print(f'Warning: {e}')
        with open(output_file, 'w') as f:
            f.write(tree_str)
        return 0


def save_states(taxon_locations, num_locations, output_file):
    """Save location states as NEXUS (one-hot encoding)."""
    lines = ['taxa  ' + '0' * num_locations]
    for taxon in sorted(taxon_locations, key=lambda x: int(x[1:])):
        state = ['0'] * num_locations
        loc = taxon_locations[taxon]
        if loc < num_locations:
            state[loc] = '1'
        lines.append(f'{taxon}  {"".join(state)}')

    with open(output_file, 'w') as f:
        f.write(f'#NEXUS\nBegin DATA;\nDimensions NTAX={len(lines)} NCHAR={num_locations};\n'
                f'Format MISSING=? GAP=- DATATYPE=STANDARD SYMBOLS="01";\nMatrix\n'
                f'{chr(10).join(lines)}\n;\nEND;\n')


def save_labels(nf_file, output_file, num_locations):
    """Save R0 and SSS as labels CSV (no log transform)."""
    df = pd.read_csv(nf_file)
    if len(df) != num_locations:
        raise ValueError(f"Expected {num_locations} rows in {nf_file}, got {len(df)}")

    labels = {}
    for i in range(num_locations):
        labels[f'R0_{i}'] = df['R0'].values[i]
        labels[f'SSS_{i}'] = df['Source_Sink_Score'].values[i]
    pd.DataFrame([labels]).to_csv(output_file, index=False)


def process_simulation(sim_idx, input_dir, output_dir, prefix, num_locations, max_trees, start_idx):
    """Process one simulation. Returns (success, num_trees, taxa_counts)."""
    tree_file = os.path.join(input_dir, f'{sim_idx}_beast2.trees')
    nf_file = os.path.join(input_dir, f'{sim_idx}_nf.csv')

    if not os.path.exists(tree_file) or not os.path.exists(nf_file):
        return False, 0, []

    trees = extract_trees(tree_file, max_trees)
    if not trees:
        return False, 0, []

    count, taxa_counts = 0, []
    for i, (_, tree_str) in enumerate(trees):
        idx = start_idx + i
        converted, taxon_locs = convert_annotations(tree_str)
        if not taxon_locs:
            continue

        num_taxa = save_tree(converted, os.path.join(output_dir, f'{prefix}.{idx}.tre'))
        taxa_counts.append(num_taxa)
        save_states(taxon_locs, num_locations, os.path.join(output_dir, f'{prefix}.{idx}.dat.nex'))
        save_labels(nf_file, os.path.join(output_dir, f'{prefix}.{idx}.labels.csv'), num_locations)
        count += 1

    return True, count, taxa_counts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Input:  {args.input_dir}')
    print(f'Output: {args.output_dir}')

    sim_indices = find_simulations(args.input_dir)
    if not sim_indices:
        print('No simulations found (need both _beast2.trees and _nf.csv)')
        return

    num_locations = detect_num_locations(args.input_dir, sim_indices)
    if num_locations is None:
        print('Error: Could not detect num_locations')
        return

    trees_available = count_trees(os.path.join(args.input_dir, f'{sim_indices[0]}_beast2.trees'))
    print(f'Simulations: {len(sim_indices)}, Trees/file: {trees_available}, Extracting: {args.max_trees_per_sim}')
    print(f'Detected num_locations: {num_locations}')

    mapping, global_idx, total, all_taxa = [], 0, 0, []
    for sim_idx in sim_indices:
        success, n, taxa = process_simulation(
            sim_idx, args.input_dir, args.output_dir, args.prefix,
            num_locations, args.max_trees_per_sim, global_idx)
        if success:
            mapping.extend({'output_idx': global_idx + t, 'sim': sim_idx, 'tree': t} for t in range(n))
            global_idx += n
            total += n
            all_taxa.extend(taxa)

    pd.DataFrame(mapping).to_csv(os.path.join(args.output_dir, 'file_mapping.csv'), index=False)

    min_taxa = min(all_taxa) if all_taxa else 0
    max_taxa = max(all_taxa) if all_taxa else 0

    print(f'Converted: {total} trees from {len(sim_indices)} simulations')
    print('')
    # Machine-readable output for run_pipeline.sh
    print(f'PHYDDLE_PARAM num_locations {num_locations}')
    print(f'PHYDDLE_PARAM min_taxa {min_taxa}')
    print(f'PHYDDLE_PARAM max_taxa {max_taxa}')


if __name__ == '__main__':
    main()
