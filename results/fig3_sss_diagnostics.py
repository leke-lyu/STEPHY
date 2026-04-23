#!/usr/bin/env python3
"""
SSS diagnostics — what does the true Source/Sink Score actually track, and
does our GNN beat the simulation-parameter oracles?

Four predictors, ranked per graph against true_reg_sss:
  A  true reg_r0          (sim param; from --r0-predictions CSV)
  B  MigIdx               (sim param; (outflow−inflow)/(outflow+inflow)
                           from {batch}/{sim_id}_parameter.csv;
                           same sign convention as SSS)
  C  Initial_Population   (sim param; from {batch}/{sim_id}_nf.csv)
  D  pred_reg_sss         (the GNN's prediction)

For each predictor we compute, per graph:
  • rank of the true-top-SSS location in the predictor's ordering (1 = match)
  • Spearman ρ between true_SSS and the predictor across the graph's locations

Outputs (under --output-dir, defaults to this script's directory):
  fig3_sss_diagnostics.pdf  — 2 × 4 grid: rank histograms / ρ histograms
  fig3_sss_diagnostics.csv  — one row per graph; both metrics × four predictors
  stdout                    — summary table with predictor value-range

Usage:
    python3 fig3_sss_diagnostics.py
    python3 fig3_sss_diagnostics.py \
        --sss-predictions /path/to/reg_sss/test_predictions.csv \
        --r0-predictions  /path/to/reg_r0/test_predictions.csv \
        --nf-root         /path/to/batches/
"""

import argparse
import re
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
_MIG_RE = re.compile(r'^migration_loc_(\d+)_to_loc_(\d+)$')


# ─── per-graph metrics ─────────────────────────────────────────────────────

def per_graph_stats(df, value_col, target_col='true_reg_sss'):
    """For each graph: rank of top-target location in `value_col` ordering,
    plus Spearman ρ between target and value across the graph's locations."""
    rows = []
    for _, grp in df.groupby(GROUP, sort=False):
        rank = grp[value_col].rank(ascending=False, method='min')
        top_idx = grp[target_col].idxmax()
        if grp[value_col].nunique() < 2 or grp[target_col].nunique() < 2:
            rho = np.nan
        else:
            rho, _ = spearmanr(grp[target_col], grp[value_col])
        rows.append({
            'num_locations': len(grp),
            'top_sss_rank_in_x': int(rank.loc[top_idx]),
            'spearman_rho': rho,
        })
    return pd.DataFrame(rows)


# ─── per-(batch, sim_id) value loaders ─────────────────────────────────────

def _join_per_sim(sss_df, nf_root, filename_tpl, value_col, loader):
    """Attach a per-location value (loaded from one file per sim) onto sss_df.

    `loader(path)` must return a DataFrame with columns ['Location', value_col].
    Returns the joined DataFrame; raises if any location is missing.
    """
    cache, pieces = {}, []
    for (batch, sim_id), grp in sss_df.groupby(['batch', 'sim_id'], sort=False):
        key = (batch, sim_id)
        if key not in cache:
            path = Path(nf_root) / str(batch) / filename_tpl.format(sim_id=sim_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
            cache[key] = loader(path)
        merged = grp.merge(cache[key], left_on='location_idx',
                           right_on='Location', how='left')
        if merged[value_col].isna().any():
            missing = merged.loc[merged[value_col].isna(),
                                 'location_idx'].tolist()
            raise ValueError(f"Missing locations {missing} for ({batch},{sim_id})")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _load_migration_index(parameter_path):
    """Per-location MigIdx = (outflow − inflow) / (outflow + inflow) from
    the migration matrix in a single *_parameter.csv (one row).
    Matches the SSS convention: +1 = pure source, −1 = pure sink."""
    row = pd.read_csv(parameter_path).iloc[0]
    inflow, outflow = {}, {}
    for col, val in row.items():
        m = _MIG_RE.match(col)
        if not m:
            continue
        src, tgt = int(m.group(1)), int(m.group(2))
        outflow[src] = outflow.get(src, 0.0) + float(val)
        inflow[tgt]  = inflow.get(tgt,  0.0) + float(val)
    if not inflow:
        raise ValueError(f"No migration_loc_X_to_loc_Y columns in {parameter_path}")
    locs = sorted(set(inflow) | set(outflow))
    mig = [((outflow.get(a, 0.) - inflow.get(a, 0.)) /
            (outflow.get(a, 0.) + inflow.get(a, 0.)))
           if (outflow.get(a, 0.) + inflow.get(a, 0.)) > 0 else np.nan
           for a in locs]
    return pd.DataFrame({'Location': locs, 'mig_idx': mig})


# ─── predictor columns ─────────────────────────────────────────────────────

def build_population_column(sss_df, nf_root):
    joined = _join_per_sim(
        sss_df, nf_root, '{sim_id}_nf.csv', 'Initial_Population',
        lambda p: pd.read_csv(p)[['Location', 'Initial_Population']])
    return (per_graph_stats(joined, 'Initial_Population'),
            joined['Initial_Population'])


def build_migidx_column(sss_df, nf_root):
    joined = _join_per_sim(
        sss_df, nf_root, '{sim_id}_parameter.csv', 'mig_idx',
        _load_migration_index)
    return per_graph_stats(joined, 'mig_idx'), joined['mig_idx']


def build_r0_column(sss_df, r0_path):
    r0 = pd.read_csv(r0_path)[KEY + ['true_reg_r0']]
    merged = sss_df[KEY + ['true_reg_sss']].merge(
        r0, on=KEY, how='inner', validate='one_to_one')
    if len(merged) != len(sss_df):
        raise ValueError(
            f"Join row-count mismatch: sss={len(sss_df)} joined={len(merged)}")
    return per_graph_stats(merged, 'true_reg_r0'), merged['true_reg_r0']


def build_pred_column(sss_df):
    return per_graph_stats(sss_df, 'pred_reg_sss'), sss_df['pred_reg_sss']


# ─── plotting ──────────────────────────────────────────────────────────────

def plot(columns, titles, subtitles, output_path):
    """Top row: rank-of-true-top-1 histograms. Bottom row: Spearman-ρ histograms."""
    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(4.6 * ncols, 7), sharey='row',
                             squeeze=False)
    palette = ['#4C72B0', '#DD8452', '#55A868', '#8172B2']

    n_loc = int(columns[0]['num_locations'].mode().iloc[0])
    bins_rank = np.arange(0.5, n_loc + 1.5, 1)
    bins_rho = np.linspace(-1, 1, 41)

    for j, (df, title, subtitle) in enumerate(zip(columns, titles, subtitles)):
        color = palette[j % len(palette)]

        ax_rank = axes[0, j]
        ax_rank.hist(df['top_sss_rank_in_x'], bins=bins_rank,
                     edgecolor='black', color=color)
        ax_rank.set_xticks(range(1, n_loc + 1))
        ax_rank.set_xlabel(f'Rank of true-SSS top-1 in\n{subtitle} ordering (1 = match)')
        ax_rank.set_title(title)
        if j == 0:
            ax_rank.set_ylabel('Number of test graphs')
        ax_rank.text(0.97, 0.95,
                     f'Rank-1: {(df["top_sss_rank_in_x"] == 1).mean():.1%}\n'
                     f'N: {len(df)}',
                     transform=ax_rank.transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax_rho = axes[1, j]
        rhos = df['spearman_rho'].dropna()
        ax_rho.hist(rhos, bins=bins_rho, edgecolor='black', color=color)
        ax_rho.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        ax_rho.axvline(rhos.median(), color='red', linestyle='-',
                       linewidth=1.5, label=f'median = {rhos.median():.3f}')
        ax_rho.set_xlim(-1.05, 1.05)
        ax_rho.set_xlabel(f'Spearman ρ (rank_SSS, rank_{subtitle})')
        if j == 0:
            ax_rho.set_ylabel('Number of test graphs')
        ax_rho.legend(loc='upper left')
        ax_rho.text(0.97, 0.95,
                    f'median: {rhos.median():.3f}\n'
                    f'mean:   {rhos.mean():.3f}\n'
                    f'ρ > 0:  {(rhos > 0).mean():.1%}',
                    transform=ax_rho.transAxes, ha='right', va='top',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.suptitle(
        'What does the true Source/Sink Score track?\n'
        'Simulation-parameter oracles (R0, MigIdx, Population) vs GNN prediction',
        y=1.00, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ─── entrypoint ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sss-predictions', type=Path,
                   default=Path('/Users/lukelyu/Desktop/data/simu/conformal_prediction/'
                                '100k_result/stephy/reg_sss/test_predictions.csv'))
    p.add_argument('--r0-predictions', type=Path,
                   default=Path('/Users/lukelyu/Desktop/data/simu/conformal_prediction/'
                                '100k_result/stephy/reg_r0/test_predictions.csv'))
    p.add_argument('--nf-root', type=Path,
                   default=Path('/Users/lukelyu/Desktop/data/simu'),
                   help='Root containing {batch}/{sim_id}_{nf,parameter}.csv')
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sss_df = pd.read_csv(args.sss_predictions)
    for col in KEY + ['true_reg_sss', 'pred_reg_sss']:
        if col not in sss_df.columns:
            raise ValueError(f"reg_sss predictions missing '{col}'")

    builders = [
        ('R0',         lambda: build_r0_column(sss_df, args.r0_predictions)),
        ('MigIdx',     lambda: build_migidx_column(sss_df, args.nf_root)),
        ('Population', lambda: build_population_column(sss_df, args.nf_root)),
        ('pred_SSS',   lambda: build_pred_column(sss_df)),
    ]
    short_names = ['r0', 'migidx', 'pop', 'pred']
    titles = ['A. SSS vs R0', 'B. SSS vs MigIdx',
              'C. SSS vs Population', 'D. SSS vs our prediction']

    cols, value_series = zip(*(build() for _, build in builders))
    subtitles = [name for name, _ in builders]

    fig_path = out_dir / 'fig3_sss_diagnostics.pdf'
    plot(cols, titles, subtitles, fig_path)

    per_graph = sss_df.drop_duplicates(GROUP)[GROUP].reset_index(drop=True)
    for short, df in zip(short_names, cols):
        per_graph[f'rank_in_{short}'] = df['top_sss_rank_in_x'].values
        per_graph[f'rho_{short}']     = df['spearman_rho'].values
    csv_path = out_dir / 'fig3_sss_diagnostics.csv'
    per_graph.to_csv(csv_path, index=False)

    print(f"{'Predictor':<12} {'N':>6} {'range':>26} {'rank-1':>8} "
          f"{'ρ median':>10} {'ρ mean':>9} {'ρ>0':>7}")
    for name, df, vals in zip(subtitles, cols, value_series):
        rhos = df['spearman_rho'].dropna()
        v = vals.dropna()
        rng = f"[{v.min():.4g}, {v.max():.4g}]"
        print(f"{name:<12} {len(df):>6} {rng:>26} "
              f"{(df['top_sss_rank_in_x'] == 1).mean():>8.1%} "
              f"{rhos.median():>10.3f} {rhos.mean():>9.3f} "
              f"{(rhos > 0).mean():>7.1%}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
