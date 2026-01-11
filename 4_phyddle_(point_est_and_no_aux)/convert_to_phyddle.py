#!/usr/bin/env python3
"""
Convert BEAST2/ReMaster simulation output to phyddle format.

Input:  {i}_beast2.trees (NEXUS), {i}_parameter.csv
Output: sim.{i}.tre (Newick), sim.{i}.dat.nex (states), sim.{i}.labels.csv

Usage:
    # Extract 1 tree per simulation (default)
    python3 convert_to_phyddle.py --input_dir /path/to/epidata/folder --output_dir ./sim_data

    # Extract all 5 trees per simulation
    python3 convert_to_phyddle.py --input_dir /path/to/epidata/folder --output_dir ./sim_data --max_trees_per_sim 5
"""

import os, re, argparse
import numpy as np
import pandas as pd
import dendropy as dp


def parse_args():
    p = argparse.ArgumentParser(description='Convert BEAST2 data to phyddle format')
    p.add_argument('--input_dir', required=True, help='Input directory')
    p.add_argument('--output_dir', required=True, help='Output directory')
    p.add_argument('--prefix', default='sim', help='Output file prefix')
    p.add_argument('--num_locations', type=int, default=16, help='Number of locations')
    p.add_argument('--max_trees_per_sim', type=int, default=1, help='Trees to extract per simulation')
    return p.parse_args()


def find_simulations(input_dir):
    """Find simulation indices from {i}_beast2.trees files."""
    indices = [int(m.group(1)) for f in os.listdir(input_dir)
               if (m := re.match(r'(\d+)_beast2\.trees', f))]
    return sorted(set(indices))


def count_trees(nexus_file):
    """Count trees in a NEXUS file."""
    with open(nexus_file) as f:
        return len(re.findall(r'tree\s+\S+\s*=', f.read(), re.IGNORECASE))


def extract_trees(nexus_file, max_trees):
    """Extract Newick trees from NEXUS. Returns [(name, newick_str), ...]."""
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
    """Save tree as binary Newick."""
    tree_str = tree_str if tree_str.endswith(';') else tree_str + ';'
    try:
        tree = dp.Tree.get(data=tree_str, schema='newick')
        tree.suppress_unifurcations()
        tree.resolve_polytomies()
        tree.write(path=output_file, schema='newick')
    except Exception as e:
        print(f'Warning: {e}')
        with open(output_file, 'w') as f:
            f.write(tree_str)


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


def save_labels(param_file, output_file, num_locations):
    """Save log-transformed R0 values as labels CSV."""
    df = pd.read_csv(param_file)
    labels = {f'log_R0_{i}': np.log(df[f'R0_loc_{i}'].values[0])
              for i in range(num_locations) if f'R0_loc_{i}' in df.columns}
    pd.DataFrame([labels]).to_csv(output_file, index=False)


def process_simulation(sim_idx, input_dir, output_dir, prefix, num_locations, max_trees, start_idx):
    """Process one simulation. Returns (success, num_trees_processed)."""
    tree_file = os.path.join(input_dir, f'{sim_idx}_beast2.trees')
    param_file = os.path.join(input_dir, f'{sim_idx}_parameter.csv')

    if not os.path.exists(tree_file) or not os.path.exists(param_file):
        return False, 0

    trees = extract_trees(tree_file, max_trees)
    if not trees:
        return False, 0

    count = 0
    for i, (_, tree_str) in enumerate(trees):
        idx = start_idx + i
        converted, taxon_locs = convert_annotations(tree_str)
        if not taxon_locs:
            continue
        save_tree(converted, os.path.join(output_dir, f'{prefix}.{idx}.tre'))
        save_states(taxon_locs, num_locations, os.path.join(output_dir, f'{prefix}.{idx}.dat.nex'))
        save_labels(param_file, os.path.join(output_dir, f'{prefix}.{idx}.labels.csv'), num_locations)
        count += 1
    return True, count


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Input:  {args.input_dir}')
    print(f'Output: {args.output_dir}')

    sim_indices = find_simulations(args.input_dir)
    if not sim_indices:
        print('No simulations found')
        return

    trees_available = count_trees(os.path.join(args.input_dir, f'{sim_indices[0]}_beast2.trees'))
    print(f'Simulations: {len(sim_indices)}, Trees/file: {trees_available}, Extracting: {args.max_trees_per_sim}')

    mapping, global_idx, total = [], 0, 0
    for sim_idx in sim_indices:
        success, n = process_simulation(sim_idx, args.input_dir, args.output_dir, args.prefix,
                                        args.num_locations, args.max_trees_per_sim, global_idx)
        if success:
            mapping.extend({'output_idx': global_idx + t, 'sim': sim_idx, 'tree': t} for t in range(n))
            global_idx += n
            total += n

    pd.DataFrame(mapping).to_csv(os.path.join(args.output_dir, 'file_mapping.csv'), index=False)
    print(f'Converted: {total} trees from {len(sim_indices)} simulations')


if __name__ == '__main__':
    main()
