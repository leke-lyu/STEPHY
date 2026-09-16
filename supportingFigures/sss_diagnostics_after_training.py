#!/usr/bin/env python3
"""
SSS diagnostics (after training) — what does the true Source/Sink Score track
on the test split, and does the trained model beat the simulation-parameter
oracles?

Predictors, ranked per graph against true_reg_sss:

  A  R0                  per location, from {batch}/{sim_id}_nf.csv
  B  MigIdx              (outflow - inflow) / (outflow + inflow) from the
                         migration matrix in {batch}/{sim_id}_parameter.csv;
                         same sign convention as SSS (+1 source, -1 sink)
  C  Initial_Population  per location, from {batch}/{sim_id}_nf.csv
  D  pred_SSS            the model's prediction (only with --prediction_result)

Per graph and predictor: the rank of the true top-SSS location in the
predictor's ordering (1 = match) and the Spearman rho between predictor and
true SSS across the graph's locations.

Outputs next to this script:
  sss_diagnostics_after_training.pdf  2 x {3|4} grid of rank / rho histograms
  sss_diagnostics_after_training.csv  one row per graph, both metrics per predictor
  sss_diagnostics_after_training.out  tee'd stdout (true-SSS summary + table)

Defaults resolve from STEPHY_MODELS, the root of the Zenodo model archive
(see ../zenodo/README.md), which must contain simulation_benchmark/:
  --sss-predictions  $STEPHY_MODELS/simulation_benchmark/100k_diverse_population_result/stephy/reg_sss/test_predictions.csv
  --nf-root          $STEPHY_MODELS/simulation_benchmark/100k_diverse_population

Usage:
    export STEPHY_MODELS=/path/to/trained_model
    python3 sss_diagnostics_after_training.py --prediction_result
    python3 sss_diagnostics_after_training.py --prediction_result \\
        --sss-predictions /path/to/reg_sss/test_predictions.csv \\
        --nf-root         /path/to/batches/
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from _paths import under

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6,
    'axes.labelsize': 6,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

KEY = ['batch', 'sim_id', 'tree_idx', 'location_idx']
GROUP = ['batch', 'sim_id', 'tree_idx']
_MIG_RE = re.compile(r'^migration_loc_(\d+)_to_loc_(\d+)$')

# (panel name, column in the enriched frame, suffix in the per-graph CSV)
ORACLES = [
    ('R0', 'r0', 'r0'),
    ('MigIdx', 'mig_idx', 'migidx'),
    ('Initial_Population', 'init_pop', 'pop'),
]
MODEL = ('pred_SSS', 'pred_reg_sss', 'pred')
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#8172B2']


class _Tee:
    """Mirror writes across streams (tees stdout to sss_diagnostics_after_training.out)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


# -- Per-graph metrics -------------------------------------------------------

def per_graph_stats(df, value_col, target_col='true_reg_sss'):
    """
    Score one predictor against the target within every graph.

    Returns one row per (batch, sim_id, tree_idx): the rank the target's
    top location receives in `value_col` (1 = the predictor picks the same
    top location) and the Spearman rho between target and predictor. Rho is
    NaN when either column is constant within the graph.
    """
    rows = []
    for _, grp in df.groupby(GROUP, sort=False):
        rank = grp[value_col].rank(ascending=False, method='min')
        degenerate = grp[value_col].nunique() < 2 or grp[target_col].nunique() < 2
        rows.append({
            **{k: grp[k].iloc[0] for k in GROUP},
            'num_locations': len(grp),
            'top_sss_rank_in_x': int(rank.loc[grp[target_col].idxmax()]),
            'spearman_rho': np.nan if degenerate
                            else spearmanr(grp[target_col], grp[value_col])[0],
        })
    return pd.DataFrame(rows)


# -- Data loading ------------------------------------------------------------

def _load_migration_index(parameter_path):
    """
    Per-location MigIdx from the migration matrix in one *_parameter.csv.

    MigIdx = (outflow - inflow) / (outflow + inflow), so +1 is a pure source
    and -1 a pure sink, matching the SSS convention. NaN where a location has
    no migration at all.
    """
    row = pd.read_csv(parameter_path).iloc[0]
    inflow, outflow = {}, {}
    for col, val in row.items():
        m = _MIG_RE.match(col)
        if m:
            src, tgt = int(m.group(1)), int(m.group(2))
            outflow[src] = outflow.get(src, 0.0) + float(val)
            inflow[tgt] = inflow.get(tgt, 0.0) + float(val)
    if not inflow:
        raise ValueError(f'No migration_loc_X_to_loc_Y columns in {parameter_path}')
    locs = sorted(set(inflow) | set(outflow))
    total = np.array([outflow.get(a, 0.0) + inflow.get(a, 0.0) for a in locs])
    net = np.array([outflow.get(a, 0.0) - inflow.get(a, 0.0) for a in locs])
    return pd.DataFrame({'Location': locs,
                         'mig_idx': np.where(total > 0, net / np.where(total > 0, total, 1), np.nan)})


def _join_per_sim(sss_df, nf_root, filename_tpl, value_cols, loader):
    """
    Attach per-location values, loaded from one file per outbreak, onto sss_df.

    `loader(path)` must return a DataFrame with columns ['Location', *value_cols].
    Raises if a file is missing or any location lacks a value.
    """
    pieces = []
    for (batch, sim_id), grp in sss_df.groupby(['batch', 'sim_id'], sort=False):
        path = Path(nf_root) / str(batch) / filename_tpl.format(sim_id=sim_id)
        if not path.exists():
            raise FileNotFoundError(f'Missing file: {path}')
        merged = grp.merge(loader(path), left_on='location_idx',
                           right_on='Location', how='left')
        if merged[value_cols].isna().any().any():
            raise ValueError(f'Missing {value_cols} for some locations in ({batch},{sim_id})')
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def attach_oracles(sss_df, nf_root):
    """Join r0, init_pop (from _nf.csv) and mig_idx (from _parameter.csv) onto sss_df."""
    enriched = _join_per_sim(
        sss_df, nf_root, '{sim_id}_nf.csv', ['R0', 'Initial_Population'],
        lambda p: pd.read_csv(p)[['Location', 'R0', 'Initial_Population']])
    enriched = _join_per_sim(
        enriched, nf_root, '{sim_id}_parameter.csv', ['mig_idx'],
        _load_migration_index)
    return enriched.rename(columns={'R0': 'r0', 'Initial_Population': 'init_pop'})


# -- Plotting ----------------------------------------------------------------

def plot(stats, output_path):
    """
    Draw the 2 x N diagnostic grid, one column per predictor in `stats`.

    `stats` is a list of (name, per_graph_stats frame). Top row: histogram of
    the rank the true top-SSS location gets in that predictor's ordering.
    Bottom row: histogram of per-graph Spearman rho, median as a red line.
    """
    n = len(stats)
    fig, axes = plt.subplots(2, n, figsize=(1.75 * n, 3.6), sharey='row',
                             squeeze=False)
    n_loc = int(stats[0][1]['num_locations'].mode().iloc[0])
    bins_rank = np.arange(0.5, n_loc + 1.5)
    bins_rho = np.linspace(-1, 1, 41)
    box = dict(boxstyle='round', facecolor='white', alpha=0.85)

    for j, (name, df) in enumerate(stats):
        color = PALETTE[j % len(PALETTE)]
        ax = axes[0, j]
        ax.hist(df['top_sss_rank_in_x'], bins=bins_rank, color=color,
                edgecolor='black', linewidth=0.3)
        ax.set_xticks(range(1, n_loc + 1))
        ax.set_xlabel(f'Rank of true-SSS top-1 in\n{name} ordering (1 = match)')
        ax.set_title(f'SSS vs {name}')
        ax.text(0.97, 0.95,
                f'Rank-1: {(df["top_sss_rank_in_x"] == 1).mean():.1%}\nN: {len(df)}',
                transform=ax.transAxes, ha='right', va='top', bbox=box)
        ax.text(-0.15, 1.08, chr(ord('a') + j), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')

        ax = axes[1, j]
        rhos = df['spearman_rho'].dropna()
        ax.hist(rhos, bins=bins_rho, color=color, edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.4)
        ax.axvline(rhos.median(), color='red', linewidth=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(f'Spearman ρ (SSS, {name})')
        ax.text(0.97, 0.95,
                f'median: {rhos.median():.3f}\nmean:   {rhos.mean():.3f}\n'
                f'ρ > 0:  {(rhos > 0).mean():.1%}',
                transform=ax.transAxes, ha='right', va='top',
                family='monospace', bbox=box)

    for ax in axes[:, 0]:
        ax.set_ylabel('Number of test graphs')
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# -- Main --------------------------------------------------------------------

def main():
    """
    Rank the three simulation-parameter oracles, and optionally the model's
    own prediction, against true SSS on the test split; write the figure, the
    per-graph CSV and the summary table next to this script.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sss-predictions', type=Path,
                   default=under('models', 'simulation_benchmark',
                                 '100k_diverse_population_result', 'stephy',
                                 'reg_sss', 'test_predictions.csv'),
                   help='reg_sss/test_predictions.csv from a trained pipeline')
    p.add_argument('--nf-root', type=Path,
                   default=under('models', 'simulation_benchmark',
                                 '100k_diverse_population'),
                   help='Root containing batch_*/{sim_id}_nf.csv and {sim_id}_parameter.csv')
    p.add_argument('--prediction_result', action='store_true',
                   help='Also score the model prediction (panel d) alongside the three oracles')
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent
    stem = out_dir / 'sss_diagnostics_after_training'
    log = open(stem.with_suffix('.out'), 'w')
    sys.stdout = _Tee(sys.__stdout__, log)

    sss_df = pd.read_csv(args.sss_predictions)
    predictors = ORACLES + ([MODEL] if args.prediction_result else [])
    required = KEY + ['true_reg_sss'] + ([MODEL[1]] if args.prediction_result else [])
    missing = set(required) - set(sss_df.columns)
    if missing:
        raise ValueError(f'{args.sss_predictions} lacks columns {sorted(missing)}')

    enriched = attach_oracles(sss_df, args.nf_root)
    stats = [(name, per_graph_stats(enriched, col)) for name, col, _ in predictors]
    plot(stats, stem.with_suffix('.pdf'))

    per_graph = stats[0][1][GROUP].copy()
    for (_, _, short), (_, df) in zip(predictors, stats):
        per_graph[f'rank_in_{short}'] = df['top_sss_rank_in_x'].values
        per_graph[f'rho_{short}'] = df['spearman_rho'].values
    per_graph.to_csv(stem.with_suffix('.csv'), index=False)

    sss = enriched['true_reg_sss'].dropna()
    print(f'True SSS  N_graphs={enriched.groupby(GROUP, sort=False).ngroups}  '
          f'N_loc={len(sss)}  range=[{sss.min():.4g}, {sss.max():.4g}]  '
          f'mean={sss.mean():.3f}  median={sss.median():.3f}')
    print(f'{"Predictor":<20} {"N_graphs":>9} {"N_loc":>7} {"range":>26} '
          f'{"rank-1":>8} {"ρ median":>10} {"ρ mean":>9} {"ρ>0":>7}')
    for (name, col, _), (_, df) in zip(predictors, stats):
        rhos = df['spearman_rho'].dropna()
        v = enriched[col].dropna()
        print(f'{name:<20} {len(df):>9} {len(v):>7} '
              f'{f"[{v.min():.4g}, {v.max():.4g}]":>26} '
              f'{(df["top_sss_rank_in_x"] == 1).mean():>8.1%} '
              f'{rhos.median():>10.3f} {rhos.mean():>9.3f} {(rhos > 0).mean():>7.1%}')
    for ext in ('.pdf', '.csv', '.out'):
        print(f'Saved: {stem.with_suffix(ext)}')
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == '__main__':
    main()
