#!/usr/bin/env python3
"""Check how much of the time-varying sampling schedule each outbreak saw.

The ``heterogeneous_sampling_b`` configs step the sampling rate through three
phases over equal thirds of ``simulation_time``, but a simulation also stops
as soon as the sample cap (``ENDS_WHEN="sample>=5000"``) is reached. An
outbreak that hits the cap inside the first third never sees a rate change.

For every ``{sim_id}_beast2.trees`` / ``{sim_id}_parameter.csv`` pair this
script reads each tip's forward sampling time from the ReMaster annotation
(``time=...``) and reports per tree:

    n_tips           sampled tips; == cap means the cap stopped the run,
                     fewer means the run reached maxTime (simulation_time)
    ended_by         'cap' or 'max_time'
    phases_reached   1 + number of phase boundaries before the last tip
    tips_phase_k     tips sampled while phase k's rate was active

Usage:
    python3 sampling_phase_check.py <dataset_dir> [<dataset_dir> ...]
                                    [--cap 5000] [--out DIR] [--workers N]

<dataset_dir> holds batch_*/ sub-folders (or the tree files directly). One
``<dataset>_sampling_phases.csv`` per dataset is written to --out, and a
summary is printed for each.
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TREE_RE = re.compile(r'tree STATE_\d+ = (.+?)(?=\ntree |\nEnd;|$)', re.DOTALL)
TIP_TIME_RE = re.compile(r'\d+\[&type="I\{\d+\}",samp="sample",time=([\d.eE+-]+)\]')


# ---------------------------------------------------------------------------
# Per-tree extraction
# ---------------------------------------------------------------------------

def scan_batch(batch_dir, cap):
    """Return one row per tree in *batch_dir* (see module docstring)."""
    batch_dir = Path(batch_dir)
    rows = []
    for tree_file in sorted(batch_dir.glob('*_beast2.trees')):
        sim_id = tree_file.name.replace('_beast2.trees', '')
        param_file = batch_dir / f'{sim_id}_parameter.csv'
        if not param_file.exists():
            continue
        params = pd.read_csv(param_file).iloc[0]
        sim_time = float(params['simulation_time'])
        phase_rates = [params[c] for c in sorted(params.index)
                       if c.startswith('sample_rate_phase_')]
        num_phases = max(len(phase_rates), 1)
        boundaries = [sim_time * k / num_phases for k in range(1, num_phases)]

        for tree_idx, newick in enumerate(TREE_RE.findall(tree_file.read_text())):
            times = np.array([float(t) for t in TIP_TIME_RE.findall(newick)])
            n_tips = len(times)
            if n_tips == 0:
                continue
            t_last = times.max()
            phase_of_tip = np.searchsorted(boundaries, times, side='right')
            row = {
                'batch': batch_dir.name,
                'sim_id': sim_id,
                'tree_idx': tree_idx,
                'n_tips': n_tips,
                'ended_by': 'cap' if n_tips >= cap else 'max_time',
                'simulation_time': sim_time,
                't_first_tip': times.min(),
                't_last_tip': t_last,
                'phases_reached': 1 + int(np.sum(t_last > np.array(boundaries))),
            }
            for k in range(num_phases):
                row[f'tips_phase_{k}'] = int(np.sum(phase_of_tip == k))
            for k, rate in enumerate(phase_rates):
                row[f'sample_rate_phase_{k}'] = rate
            rows.append(row)
    return rows


def scan_dataset(dataset_dir, cap, workers):
    """Scan every batch_* folder (or the folder itself) into one DataFrame."""
    dataset_dir = Path(dataset_dir)
    batches = sorted(p for p in dataset_dir.glob('batch_*') if p.is_dir()) or [dataset_dir]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(scan_batch, batches, [cap] * len(batches))
    else:
        results = (scan_batch(b, cap) for b in batches)
    rows = [r for batch_rows in results for r in batch_rows]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _pct(k, n):
    return f'{k} ({100 * k / n:.1f}%)' if n else '0'


def summarise(df, name, cap, min_share):
    """Print two highlights: what stopped each run, and how balanced the tips
    of the time-stopped runs are across the sampling phases."""
    n = len(df)
    phase_cols = sorted(c for c in df.columns if c.startswith('tips_phase_'))
    by_cap = df[df['ended_by'] == 'cap']
    by_time = df[df['ended_by'] == 'max_time']

    print(f'\n=== {name}  ({n} trees) ===')
    stopped_in = ' | '.join(f'{k}: {int((by_cap["phases_reached"] == k).sum())}'
                           for k in range(1, len(phase_cols) + 1))
    print(f'{f"Stopped by size ({cap} tips):":<30}{_pct(len(by_cap), n):<16} cap hit in phase {stopped_in}')

    line = f'{"Stopped by time:":<30}{_pct(len(by_time), n):<16}'
    if len(by_time):
        shares = by_time[phase_cols].div(by_time['n_tips'], axis=0)
        median = ' / '.join(f'{v:.2f}' for v in shares.median())
        worst_idx = shares.min(axis=1).idxmin()
        worst = by_time.loc[worst_idx]
        worst_shares = ' / '.join(f'{v:.2f}' for v in shares.loc[worst_idx])
        unbalanced = shares.min(axis=1) < min_share
        line += (f'median tip share per phase {median}; '
                 f'{_pct(int(unbalanced.sum()), len(by_time))} have a phase below '
                 f'{100 * min_share:g}% (worst {worst["batch"]}/{worst["sim_id"]}: {worst_shares})')
    print(line)

    # `time=` must be forward simulation time: no tip past simulation_time, none at 0.
    late = int((df['t_last_tip'] > df['simulation_time'] * (1 + 1e-9)).sum())
    at_zero = int((df['t_first_tip'] == 0).sum())
    status = 'ok' if late == 0 and at_zero == 0 else f'FAILED ({late} past simulation_time, {at_zero} at time 0)'
    print(f'Tip-time check: {status}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_dirs', nargs='+', help='dataset folders holding batch_*/')
    parser.add_argument('--cap', type=int, default=5000,
                        help='sample cap from ENDS_WHEN (default 5000)')
    parser.add_argument('--out', default='.', help='where the per-tree CSVs go (default: cwd)')
    parser.add_argument('--workers', type=int, default=1, help='parallel batch folders')
    parser.add_argument('--min_share', type=float, default=0.05,
                        help='a time-stopped tree with any phase below this tip share is flagged unbalanced')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in args.dataset_dirs:
        name = Path(dataset_dir).resolve().name
        df = scan_dataset(dataset_dir, args.cap, args.workers)
        if df.empty:
            print(f'\n=== {name}: no trees found ===')
            continue
        csv_path = out_dir / f'{name}_sampling_phases.csv'
        df.to_csv(csv_path, index=False)
        summarise(df, name, args.cap, args.min_share)
        print(f'Per-tree table: {csv_path}')


if __name__ == '__main__':
    main()
