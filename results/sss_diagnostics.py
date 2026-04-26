#!/usr/bin/env python3
"""
SSS diagnostics — what does the true Source/Sink Score actually track?

Three simulation-parameter "oracle" predictors, ranked per simulation against
the true Source_Sink_Score (all read straight from the simulation files; no
trained model required):

  A  R0                  (per-location, from {batch}/{sim_id}_nf.csv)
  B  MigIdx              (per-location, (outflow−inflow)/(outflow+inflow)
                          from {batch}/{sim_id}_parameter.csv;
                          same sign convention as SSS)
  C  Initial_Population  (per-location, from {batch}/{sim_id}_nf.csv)

For each predictor we compute, per simulation:
  • rank of the true-top-SSS location in the predictor's ordering (1 = match)
  • Spearman ρ between true SSS and the predictor across the sim's locations

Outputs (next to this script, or under --output-dir):
  sss_diagnostics.pdf  — 2 × 3 grid: rank histograms / ρ histograms
  sss_diagnostics.csv  — one row per simulation; both metrics × three predictors
  stdout               — summary table with predictor value-range

Usage:
    python3 sss_diagnostics.py --nf-root /path/to/batches/
    python3 sss_diagnostics.py --nf-root /path/to/batches/ --output-dir /tmp/out
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

GROUP = ['batch', 'sim_id']
_MIG_RE = re.compile(r'^migration_loc_(\d+)_to_loc_(\d+)$')
_NF_RE = re.compile(r'^(.+)_nf\.csv$')


# ─── per-simulation metrics ────────────────────────────────────────────────

def per_graph_stats(df, value_col, target_col='true_sss'):
    """For each simulation: rank of top-target location in `value_col`, plus
    Spearman ρ between target and value across the sim's locations."""
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


def collect_long_table(nf_root):
    """Walk nf_root/batch_*/, joining each {sim_id}_nf.csv with its
    {sim_id}_parameter.csv. Returns a long DataFrame with one row per
    (batch, sim_id, location)."""
    nf_root = Path(nf_root)
    pieces = []
    nf_paths = sorted(nf_root.glob('batch_*/*_nf.csv'))
    if not nf_paths:
        raise FileNotFoundError(f"No batch_*/*_nf.csv files under {nf_root}")
    for nf_path in nf_paths:
        m = _NF_RE.match(nf_path.name)
        if not m:
            continue
        sim_id = m.group(1)
        batch = nf_path.parent.name
        param_path = nf_path.parent / f'{sim_id}_parameter.csv'
        if not param_path.exists():
            raise FileNotFoundError(f"Missing parameter file: {param_path}")

        nf = pd.read_csv(nf_path)
        for col in ('Location', 'Initial_Population', 'R0', 'Source_Sink_Score'):
            if col not in nf.columns:
                raise ValueError(f"{nf_path} missing column '{col}'")

        mig = _load_migration_index(param_path)
        merged = nf.merge(mig, on='Location', how='left')
        if merged['mig_idx'].isna().any():
            missing = merged.loc[merged['mig_idx'].isna(), 'Location'].tolist()
            raise ValueError(f"Missing MigIdx for locations {missing} in "
                             f"({batch},{sim_id})")
        merged['batch'] = batch
        merged['sim_id'] = sim_id
        merged = merged.rename(columns={
            'Location': 'location_idx',
            'Source_Sink_Score': 'true_sss',
            'Initial_Population': 'init_pop',
            'R0': 'r0',
        })
        pieces.append(merged[['batch', 'sim_id', 'location_idx',
                              'true_sss', 'r0', 'mig_idx', 'init_pop']])
    return pd.concat(pieces, ignore_index=True)


# ─── plotting ──────────────────────────────────────────────────────────────

def plot(columns, titles, subtitles, output_path):
    """Top row: rank-of-true-top-1 histograms. Bottom row: Spearman-ρ histograms."""
    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(4.6 * ncols, 7), sharey='row',
                             squeeze=False)
    palette = ['#4C72B0', '#DD8452', '#55A868']

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
            ax_rank.set_ylabel('Number of simulations')
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
            ax_rho.set_ylabel('Number of simulations')
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
        'Simulation-parameter oracles: R0, MigIdx, Initial_Population',
        y=1.00, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# ─── entrypoint ────────────────────────────────────────────────────────────

def main():
    """
    Build the no-model SSS diagnostics figure straight from simulation files.

    For every simulation under --nf-root, joins per-location R0 and
    Initial_Population (from {sim_id}_nf.csv) with a per-location MigIdx
    derived from the migration matrix in {sim_id}_parameter.csv, then ranks
    each predictor against the true Source/Sink Score (also in _nf.csv).

    Output: sss_diagnostics.{pdf,csv} alongside this script, plus a stdout
    summary table.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nf-root', type=Path, required=True,
                   help='Root containing batch_*/{sim_id}_nf.csv and '
                        '{sim_id}_parameter.csv')
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()

    out_dir = args.output_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = collect_long_table(args.nf_root)

    builders = [
        ('R0',                'r0',       'A. SSS vs R0'),
        ('MigIdx',            'mig_idx',  'B. SSS vs MigIdx'),
        ('Initial_Population','init_pop', 'C. SSS vs Initial_Population'),
    ]
    cols = [per_graph_stats(long_df, col) for _, col, _ in builders]
    short_names = ['r0', 'migidx', 'pop']
    subtitles = [name for name, _, _ in builders]
    titles = [t for _, _, t in builders]

    fig_path = out_dir / 'sss_diagnostics.pdf'
    plot(cols, titles, subtitles, fig_path)

    per_graph = cols[0][GROUP].copy()
    for short, df in zip(short_names, cols):
        per_graph[f'rank_in_{short}'] = df['top_sss_rank_in_x'].values
        per_graph[f'rho_{short}']     = df['spearman_rho'].values
    csv_path = out_dir / 'sss_diagnostics.csv'
    per_graph.to_csv(csv_path, index=False)

    sss_vals = long_df['true_sss'].dropna()
    n_sims = long_df.groupby(GROUP, sort=False).ngroups
    print(f"True SSS  N_sims={n_sims}  N_loc={len(sss_vals)}  "
          f"range=[{sss_vals.min():.4g}, {sss_vals.max():.4g}]  "
          f"mean={sss_vals.mean():.3f}  median={sss_vals.median():.3f}")
    print(f"{'Predictor':<20} {'N_sims':>7} {'N_loc':>7} {'range':>26} "
          f"{'rank-1':>8} {'ρ median':>10} {'ρ mean':>9} {'ρ>0':>7}")
    for (name, col, _), df in zip(builders, cols):
        rhos = df['spearman_rho'].dropna()
        v = long_df[col].dropna()
        rng = f"[{v.min():.4g}, {v.max():.4g}]"
        print(f"{name:<20} {len(df):>7} {len(v):>7} {rng:>26} "
              f"{(df['top_sss_rank_in_x'] == 1).mean():>8.1%} "
              f"{rhos.median():>10.3f} {rhos.mean():>9.3f} "
              f"{(rhos > 0).mean():>7.1%}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
