#!/usr/bin/env python3
"""
SSS diagnostics — what does the true Source/Sink Score actually track, and
does our GNN beat the two "simulation-parameter oracle" baselines?

One figure, three columns:
    col 1 — true_SSS vs Initial_Population  (sim param, NOT seen by GNN)
    col 2 — true_SSS vs true_reg_r0         (sim param, NOT seen by GNN)
    col 3 — true_SSS vs pred_reg_sss        (our GNN's prediction)

Two rows, same framing per column:
    row A — histogram of the rank (in the X ordering) of the location with the
            largest true_SSS in each graph. Rank 1 means X picks the true top.
    row B — histogram of per-graph Spearman ρ between true_SSS and X, i.e.
            corr(rank_SSS, rank_X) across the N locations of each graph.

Population data is read per-location from {batch}/{sim_id}_nf.csv (the same
source fig3 originally used).

Usage:
    python3 fig3_sss_diagnostics.py \
        --sss-predictions /path/to/reg_sss/test_predictions.csv \
        --r0-predictions  /path/to/reg_r0/test_predictions.csv \
        --nf-root         /path/to/batches/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

KEY = ['batch', 'sim_id', 'tree_idx', 'location_idx']
GROUP = ['batch', 'sim_id', 'tree_idx']


def per_graph_stats(df, value_col, target_col='true_reg_sss'):
    """For each graph: (a) rank of top-target location in value ordering,
    (b) Spearman ρ(target, value)."""
    rows = []
    for keys, grp in df.groupby(GROUP, sort=False):
        value_rank = grp[value_col].rank(ascending=False, method='min')
        top_target_idx = grp[target_col].idxmax()

        if grp[value_col].nunique() < 2 or grp[target_col].nunique() < 2:
            rho = np.nan
        else:
            rho, _ = spearmanr(grp[target_col], grp[value_col])

        rows.append({
            'num_locations': len(grp),
            'top_sss_rank_in_x': int(value_rank.loc[top_target_idx]),
            'spearman_rho': rho,
        })
    return pd.DataFrame(rows)


def build_population_column(sss_df, nf_root):
    cache = {}
    pieces = []
    for (batch, sim_id), grp in sss_df.groupby(['batch', 'sim_id'], sort=False):
        key = (batch, sim_id)
        if key not in cache:
            path = Path(nf_root) / str(batch) / f'{sim_id}_nf.csv'
            if not path.exists():
                raise FileNotFoundError(f"Missing nf file: {path}")
            cache[key] = pd.read_csv(path)[['Location', 'Initial_Population']]
        nf = cache[key]
        merged = grp.merge(nf, left_on='location_idx', right_on='Location',
                           how='left')
        if merged['Initial_Population'].isna().any():
            missing = merged.loc[merged['Initial_Population'].isna(),
                                 'location_idx'].tolist()
            raise ValueError(f"Missing locations {missing} in {batch}/{sim_id}_nf.csv")
        pieces.append(merged)
    joined = pd.concat(pieces, ignore_index=True)
    return per_graph_stats(joined, value_col='Initial_Population')


def build_r0_column(sss_df, r0_path):
    r0 = pd.read_csv(r0_path)[KEY + ['true_reg_r0']]
    merged = sss_df[KEY + ['true_reg_sss']].merge(
        r0, on=KEY, how='inner', validate='one_to_one')
    if len(merged) != len(sss_df):
        raise ValueError(
            f"Join row-count mismatch: sss={len(sss_df)} joined={len(merged)}")
    return per_graph_stats(merged, value_col='true_reg_r0')


def build_pred_column(sss_df):
    return per_graph_stats(sss_df, value_col='pred_reg_sss')


def plot(columns, titles, subtitles, output_path):
    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(4.6 * ncols, 7), sharey='row',
                             squeeze=False)
    palette = ['#4C72B0', '#DD8452', '#55A868']

    n_loc_mode = int(columns[0]['num_locations'].mode().iloc[0])
    bins_rank = np.arange(0.5, n_loc_mode + 1.5, 1)
    bins_rho = np.linspace(-1, 1, 41)

    for j, (df, title, subtitle) in enumerate(zip(columns, titles, subtitles)):
        color = palette[j % len(palette)]

        axA = axes[0, j]
        axA.hist(df['top_sss_rank_in_x'], bins=bins_rank,
                 edgecolor='black', color=color)
        axA.set_xticks(range(1, n_loc_mode + 1))
        axA.set_xlabel(f'Rank of true-SSS top-1 in\n{subtitle} ordering (1 = match)')
        if j == 0:
            axA.set_ylabel('Number of test graphs')
        axA.set_title(title)
        frac1 = (df['top_sss_rank_in_x'] == 1).mean()
        axA.text(0.97, 0.95, f'Rank-1: {frac1:.1%}\nN: {len(df)}',
                 transform=axA.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        axB = axes[1, j]
        rhos = df['spearman_rho'].dropna()
        axB.hist(rhos, bins=bins_rho, edgecolor='black', color=color)
        axB.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        axB.axvline(rhos.median(), color='red', linestyle='-',
                    linewidth=1.5, label=f'median = {rhos.median():.3f}')
        axB.set_xlim(-1.05, 1.05)
        axB.set_xlabel(f'Spearman ρ (rank_SSS, rank_{subtitle})')
        if j == 0:
            axB.set_ylabel('Number of test graphs')
        axB.legend(loc='upper left')
        axB.text(0.97, 0.95,
                 f'median: {rhos.median():.3f}\n'
                 f'mean:   {rhos.mean():.3f}\n'
                 f'ρ > 0:  {(rhos > 0).mean():.1%}',
                 transform=axB.transAxes, ha='right', va='top',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.suptitle(
        'What does the true Source/Sink Score track?\n'
        'Simulation-parameter oracles (Pop, R0) vs GNN prediction',
        y=1.00, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sss-predictions', required=True, type=Path)
    p.add_argument('--r0-predictions', required=True, type=Path)
    p.add_argument('--nf-root', required=True, type=Path,
                   help='Root of {batch}/{sim_id}_nf.csv files.')
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or args.sss_predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sss_df = pd.read_csv(args.sss_predictions)
    for col in KEY + ['true_reg_sss', 'pred_reg_sss']:
        if col not in sss_df.columns:
            raise ValueError(f"reg_sss predictions missing '{col}'")

    pop_col = build_population_column(sss_df, args.nf_root)
    r0_col = build_r0_column(sss_df, args.r0_predictions)
    pred_col = build_pred_column(sss_df)

    cols = [pop_col, r0_col, pred_col]
    titles = ['A. SSS vs Population', 'B. SSS vs R0', 'C. SSS vs our prediction']
    subtitles = ['Population', 'R0', 'pred_SSS']

    fig_path = out_dir / 'fig3_sss_diagnostics.pdf'
    plot(cols, titles, subtitles, fig_path)

    # One consolidated per-graph CSV
    per_graph = sss_df.drop_duplicates(GROUP)[GROUP].reset_index(drop=True)
    per_graph['rank_in_pop']  = pop_col['top_sss_rank_in_x'].values
    per_graph['rho_pop']      = pop_col['spearman_rho'].values
    per_graph['rank_in_r0']   = r0_col['top_sss_rank_in_x'].values
    per_graph['rho_r0']       = r0_col['spearman_rho'].values
    per_graph['rank_in_pred'] = pred_col['top_sss_rank_in_x'].values
    per_graph['rho_pred']     = pred_col['spearman_rho'].values
    csv_path = out_dir / 'fig3_sss_diagnostics.csv'
    per_graph.to_csv(csv_path, index=False)

    print(f"{'Predictor':<12} {'N':>6} {'rank-1':>8} {'ρ median':>10} "
          f"{'ρ mean':>9} {'ρ>0':>7}")
    for name, df in zip(subtitles, cols):
        rhos = df['spearman_rho'].dropna()
        print(f"{name:<12} {len(df):>6} "
              f"{(df['top_sss_rank_in_x']==1).mean():>8.1%} "
              f"{rhos.median():>10.3f} {rhos.mean():>9.3f} "
              f"{(rhos>0).mean():>7.1%}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
