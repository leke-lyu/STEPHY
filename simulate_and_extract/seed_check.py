#!/usr/bin/env python3
"""Verify that the multi-seed stress-test sets really have several introductions.

The ``index_loc_k`` configs seed k distinct locations from an unsampled origin
``X`` at t = 0 (see xml_generation.generate_xml). For every
``{sim_id}_beast2.trees`` / ``{sim_id}_parameter.csv`` / ``{sim_id}_nf.csv``
triple this script reports:

    k_configured      seeds listed in seed_location_index (1 = single seed,
                      i.e. the multi-seed code did not run)
    root_type         type annotation of the tree root ('X' when seeded via
                      the origin, 'I{n}' otherwise)
    n_founders        children of the root = introductions that left sampled
                      descendants (only meaningful when root_type == 'X')
    ancestral_labels  sum of Ancestral_State in the nf.csv (0 when the MRCA
                      is X, 1 when a single founder survived)

Usage:
    python3 seed_check.py <dataset_dir> [<dataset_dir> ...] [--out DIR] [--workers N]
"""

import argparse
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

TREE_RE = re.compile(r'tree STATE_\d+ = (.+?)(?=\ntree |\nEnd;|$)', re.DOTALL)
ANNOT_RE = re.compile(r'\[&[^\]]*\]')
TYPE_RE = re.compile(r'\[&type="([^"]+)"')


def root_children(newick):
    """Number of child clades of the root node of a Newick string."""
    plain = ANNOT_RE.sub('', newick).strip().rstrip(';')
    if not plain.startswith('('):
        return 0
    depth, children = 0, 1
    for ch in plain:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                break
        elif ch == ',' and depth == 1:
            children += 1
    return children


def scan_batch(batch_dir):
    """Return one row per tree in *batch_dir* (see module docstring)."""
    batch_dir = Path(batch_dir)
    rows = []
    for tree_file in sorted(batch_dir.glob('*_beast2.trees')):
        sim_id = tree_file.name.replace('_beast2.trees', '')
        param_file = batch_dir / f'{sim_id}_parameter.csv'
        nf_file = batch_dir / f'{sim_id}_nf.csv'
        if not (param_file.exists() and nf_file.exists()):
            continue
        seeds = str(pd.read_csv(param_file).iloc[0]['seed_location_index'])
        ancestral = int(pd.read_csv(nf_file)['Ancestral_State'].sum())

        for tree_idx, newick in enumerate(TREE_RE.findall(tree_file.read_text())):
            types = TYPE_RE.findall(newick)
            rows.append({
                'batch': batch_dir.name,
                'sim_id': sim_id,
                'tree_idx': tree_idx,
                'k_configured': len(seeds.split(';')),
                'seed_locations': seeds,
                'root_type': types[-1] if types else '',
                'n_founders': root_children(newick),
                'ancestral_labels': ancestral,
            })
    return rows


def scan_dataset(dataset_dir, workers):
    dataset_dir = Path(dataset_dir)
    batches = sorted(p for p in dataset_dir.glob('batch_*') if p.is_dir()) or [dataset_dir]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(scan_batch, batches)
    else:
        results = (scan_batch(b) for b in batches)
    return pd.DataFrame([r for batch_rows in results for r in batch_rows])


def _dist(counter):
    return ', '.join(f'{k}: {v}' for k, v in sorted(counter.items()))


def summarise(df, name):
    n = len(df)
    print(f'\n=== {name}  ({n} trees) ===')
    print(f'k configured (seeds in parameter CSV):  {_dist(Counter(df["k_configured"]))}')
    print(f'Root type:                              {_dist(Counter(df["root_type"]))}')
    if (df['k_configured'] > 1).all() and (df['root_type'] == 'X').all():
        print(f'Realised founders (children of X):      {_dist(Counter(df["n_founders"]))}')
        print(f'Ancestral_State all zero:               {int((df["ancestral_labels"] == 0).sum())} / {n}')
        print('Multi-seed run: OK')
    else:
        print('Multi-seed run: FAILED - single seed and/or no origin X; the multi-seed '
              'xml_generation.py did not run for these trees.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_dirs', nargs='+', help='dataset folders holding batch_*/')
    parser.add_argument('--out', default='.', help='where the per-tree CSVs go (default: cwd)')
    parser.add_argument('--workers', type=int, default=1, help='parallel batch folders')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in args.dataset_dirs:
        name = Path(dataset_dir).resolve().name
        df = scan_dataset(dataset_dir, args.workers)
        if df.empty:
            print(f'\n=== {name}: no trees found ===')
            continue
        csv_path = out_dir / f'{name}_seeds.csv'
        df.to_csv(csv_path, index=False)
        summarise(df, name)
        print(f'Per-tree table: {csv_path}')


if __name__ == '__main__':
    main()
