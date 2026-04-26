#!/usr/bin/env python3
"""
SSS diagnostics (after training) — what does the true Source/Sink Score
actually track on the GNN's test split, and does the trained model beat
the simulation-parameter oracles?

Four predictors, ranked per graph against true_reg_sss:

  A  R0                 (sim param; per-location, from {batch}/{sim_id}_nf.csv)
  B  MigIdx             (sim param; (outflow−inflow)/(outflow+inflow)
                         from {batch}/{sim_id}_parameter.csv;
                         same sign convention as SSS)
  C  Initial_Population (sim param; from {batch}/{sim_id}_nf.csv)
  D  pred_reg_sss       (the GNN's prediction)

For each predictor we compute, per graph:
  • rank of the true-top-SSS location in the predictor's ordering (1 = match)
  • Spearman ρ between true_SSS and the predictor across the graph's locations

Outputs (next to this script, or under --output-dir):
  sss_diagnostics_after_training.pdf  — 2 × 4 grid: rank / ρ histograms
  sss_diagnostics_after_training.csv  — one row per graph; both metrics × four predictors
  stdout                              — true-SSS summary + per-predictor table

Usage:
    python3 sss_diagnostics_after_training.py \
        --sss-predictions /path/to/reg_sss/test_predictions.csv \
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
    """For each graph: rank of top-target location in `value_col`, plus
    Spearman ρ between target and value across the graph's locations."""
    rows = []
    for _, grp in df.groupby(GROUP, sort=False):
        rank = grp[value_col].rank(ascending=False, method='min')
        top_idx = grp[target_col].idxmax()
        if grp[value_col].nunique() < 2 or grp[target_col].nunique() < 2:
            rho = np.nan
        else:
            rho, _ = spearmanr(grp[target_col], grp[value_col])
        rows.append({
            'batch': grp['batch'].iloc[0],
            'sim_id': grp['sim_id'].iloc[0],
            'tree_idx': grp['tree_idx'].iloc[0],
            'num_locations': len(grp),
            'top_sss_rank_in_x': int(rank.loc[top_idx]),
            'spearman_rho': rho,
        })
    return pd.DataFrame(rows)


# ─── data loading ──────────────────────────────────────────────────────────

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


def _join_per_sim(sss_df, nf_root, filename_tpl, value_cols, loader):
    """Attach per-location values (loaded from one file per sim) onto sss_df.

    `loader(path)` must return a DataFrame with columns ['Location', *value_cols].
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
        for col in value_cols:
            if merged[col].isna().any():
                missing = merged.loc[merged[col].isna(),
                                     'location_idx'].tolist()
                raise ValueError(f"Missing '{col}' for locations {missing} in "
                                 f"({batch},{sim_id})")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def attach_oracles(sss_df, nf_root):
    """Join R0, Initial_Population (from _nf.csv) and MigIdx (from
    _parameter.csv) onto each row of sss_df. Returns the enriched frame."""
    enriched = _join_per_sim(
        sss_df, nf_root, '{sim_id}_nf.csv', ['R0', 'Initial_Population'],
        lambda p: pd.read_csv(p)[['Location', 'R0', 'Initial_Population']])
    enriched = _join_per_sim(
        enriched, nf_root, '{sim_id}_parameter.csv', ['mig_idx'],
        _load_migration_index)
    return enriched.rename(columns={
        'R0': 'r0',
        'Initial_Population': 'init_pop',
    })


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
        'What does the true Source/Sink Score track? (after training)\n'
        'Simulation-parameter oracles (R0, MigIdx, Initial_Population) vs GNN prediction',
        y=1.00, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ─── entrypoint ────────────────────────────────────────────────────────────

def main():
    """
    Build the post-training SSS diagnostics figure on the GNN's test split.

    Loads the trained pipeline's reg_sss/test_predictions.csv, joins
    per-location R0 and Initial_Population (from {sim_id}_nf.csv) and a
    per-location MigIdx (from {sim_id}_parameter.csv), then ranks each of
    R0 / MigIdx / Initial_Population / pred_reg_sss against true_reg_sss
    per (batch, sim_id, tree_idx) graph.

    Output: sss_diagnostics_after_training.{pdf,csv} alongside this script,
    plus a stdout summary table.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sss-predictions', type=Path, required=True,
                   help='reg_sss/test_predictions.csv from a trained pipeline')
    p.add_argument('--nf-root', type=Path, required=True,
                   help='Root containing batch_*/{sim_id}_nf.csv and '
                        '{sim_id}_parameter.csv')
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sss_df = pd.read_csv(args.sss_predictions)
    for col in KEY + ['true_reg_sss', 'pred_reg_sss']:
        if col not in sss_df.columns:
            raise ValueError(f"reg_sss predictions missing '{col}'")

    enriched = attach_oracles(sss_df, args.nf_root)

    builders = [
        ('R0',                'r0',           'A. SSS vs R0'),
        ('MigIdx',            'mig_idx',      'B. SSS vs MigIdx'),
        ('Initial_Population','init_pop',     'C. SSS vs Initial_Population'),
        ('pred_SSS',          'pred_reg_sss', 'D. SSS vs our prediction'),
    ]
    cols = [per_graph_stats(enriched, col) for _, col, _ in builders]
    short_names = ['r0', 'migidx', 'pop', 'pred']
    subtitles = [name for name, _, _ in builders]
    titles = [t for _, _, t in builders]

    fig_path = out_dir / 'sss_diagnostics_after_training.pdf'
    plot(cols, titles, subtitles, fig_path)

    per_graph = cols[0][GROUP].copy()
    for short, df in zip(short_names, cols):
        per_graph[f'rank_in_{short}'] = df['top_sss_rank_in_x'].values
        per_graph[f'rho_{short}']     = df['spearman_rho'].values
    csv_path = out_dir / 'sss_diagnostics_after_training.csv'
    per_graph.to_csv(csv_path, index=False)

    sss_vals = enriched['true_reg_sss'].dropna()
    n_graphs = enriched.groupby(GROUP, sort=False).ngroups
    print(f"True SSS  N_graphs={n_graphs}  N_loc={len(sss_vals)}  "
          f"range=[{sss_vals.min():.4g}, {sss_vals.max():.4g}]  "
          f"mean={sss_vals.mean():.3f}  median={sss_vals.median():.3f}")
    print(f"{'Predictor':<20} {'N_graphs':>9} {'N_loc':>7} {'range':>26} "
          f"{'rank-1':>8} {'ρ median':>10} {'ρ mean':>9} {'ρ>0':>7}")
    for (name, col, _), df in zip(builders, cols):
        rhos = df['spearman_rho'].dropna()
        v = enriched[col].dropna()
        rng = f"[{v.min():.4g}, {v.max():.4g}]"
        print(f"{name:<20} {len(df):>9} {len(v):>7} {rng:>26} "
              f"{(df['top_sss_rank_in_x'] == 1).mean():>8.1%} "
              f"{rhos.median():>10.3f} {rhos.mean():>9.3f} "
              f"{(rhos > 0).mean():>7.1%}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
