#!/usr/bin/env python3
"""
Figure 3 — Is Source/Sink Score correlated with Initial_Population?

Uses the enriched test_predictions.csv schema (2026-04-17+) to join each test
graph's per-location SSS back to its source simulation's `{batch}/{sim_id}_nf.csv`
for Initial_Population.

  Panel A: histogram of the population rank held by the top-1 SSS location in
           each test graph (rank 1 = largest population). If SSS tracks
           population, the mass concentrates at rank 1.
  Panel B: distribution of per-graph Spearman rank correlation between SSS and
           Initial_Population across all N locations in each graph.

Usage:
    python3 fig3_sss_vs_population.py \
        --predictions /path/to/reg_sss/test_predictions.csv \
        --nf-root    /path/to/raw/batches/
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
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def detect_sss_column(df, prefix):
    candidates = [f'{prefix}_reg_sss', f'{prefix}_Source_Sink_Score']
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No SSS column found. Looked for: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_nf(nf_root, batch, sim_id, cache):
    key = (batch, sim_id)
    if key in cache:
        return cache[key]
    path = Path(nf_root) / str(batch) / f'{sim_id}_nf.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing nf file: {path}")
    nf = pd.read_csv(path)
    cache[key] = nf
    return nf


def analyze(predictions_path, nf_root, sss_source):
    df = pd.read_csv(predictions_path)
    for col in ('batch', 'sim_id', 'tree_idx', 'location_idx'):
        if col not in df.columns:
            raise ValueError(
                f"test_predictions.csv is missing '{col}'. "
                f"Rebuild graphs + retrain with the post-2026-04-17 pipeline."
            )

    sss_col = detect_sss_column(df, sss_source)
    cache = {}
    rows = []

    for (batch, sim_id, tree_idx), grp in df.groupby(['batch', 'sim_id', 'tree_idx'], sort=False):
        nf = load_nf(nf_root, batch, sim_id, cache)
        merged = grp.merge(
            nf[['Location', 'Initial_Population']],
            left_on='location_idx', right_on='Location', how='left'
        )
        if merged['Initial_Population'].isna().any():
            missing = merged.loc[merged['Initial_Population'].isna(), 'location_idx'].tolist()
            raise ValueError(
                f"Locations {missing} missing from {batch}/{sim_id}_nf.csv"
            )

        n = len(merged)
        # Rank 1 = largest value (method='min' handles ties conservatively)
        pop_rank = merged['Initial_Population'].rank(ascending=False, method='min')
        sss_rank = merged[sss_col].rank(ascending=False, method='min')

        top_sss_idx = merged[sss_col].idxmax()
        top_sss_pop_rank = int(pop_rank.loc[top_sss_idx])

        if merged[sss_col].nunique() < 2 or merged['Initial_Population'].nunique() < 2:
            rho = np.nan
        else:
            rho, _ = spearmanr(merged[sss_col], merged['Initial_Population'])

        rows.append({
            'batch': batch,
            'sim_id': sim_id,
            'tree_idx': tree_idx,
            'num_locations': n,
            'top_sss_location_idx': int(merged.loc[top_sss_idx, 'location_idx']),
            'top_sss_pop_rank': top_sss_pop_rank,
            'spearman_rho': rho,
        })

    return pd.DataFrame(rows), sss_col


def plot(summary, sss_col, output_path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A — histogram of pop rank held by top-1 SSS location
    n_loc_mode = int(summary['num_locations'].mode().iloc[0])
    bins = np.arange(0.5, n_loc_mode + 1.5, 1)
    axA.hist(summary['top_sss_pop_rank'], bins=bins, edgecolor='black', color='#4C72B0')
    axA.set_xticks(range(1, n_loc_mode + 1))
    axA.set_xlabel('Population rank of top-1 SSS location (1 = largest)')
    axA.set_ylabel('Number of test graphs')
    axA.set_title('A. Top-1 SSS location vs population rank')
    frac_rank1 = (summary['top_sss_pop_rank'] == 1).mean()
    axA.text(0.97, 0.95,
             f'Rank-1: {frac_rank1:.1%}\nN graphs: {len(summary)}',
             transform=axA.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B — per-graph Spearman rho distribution
    rhos = summary['spearman_rho'].dropna()
    axB.hist(rhos, bins=30, edgecolor='black', color='#55A868')
    axB.axvline(0, color='grey', linestyle='--', linewidth=1)
    axB.axvline(rhos.median(), color='red', linestyle='-', linewidth=1.5,
                label=f'median = {rhos.median():.3f}')
    axB.set_xlabel('Spearman ρ (SSS vs Initial_Population)')
    axB.set_ylabel('Number of test graphs')
    axB.set_title('B. Per-graph rank correlation')
    axB.set_xlim(-1.05, 1.05)
    axB.legend(loc='upper left')
    frac_pos = (rhos > 0).mean()
    axB.text(0.97, 0.95,
             f'ρ > 0: {frac_pos:.1%}\nmean: {rhos.mean():.3f}',
             transform=axB.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'SSS vs Initial_Population  (source: {sss_col})', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions', required=True, type=Path,
                   help='Path to reg_sss test_predictions.csv (post-2026-04-17 schema).')
    p.add_argument('--nf-root', required=True, type=Path,
                   help='Root folder containing {batch}/{sim_id}_nf.csv files.')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Output directory (default: same folder as --predictions).')
    p.add_argument('--sss-source', choices=['true', 'pred'], default='true',
                   help="Which SSS column to use: 'true' (ground truth) or 'pred' (model).")
    args = p.parse_args()

    out_dir = args.output_dir or args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, sss_col = analyze(args.predictions, args.nf_root, args.sss_source)

    csv_path = out_dir / f'fig3_sss_vs_population_{args.sss_source}.csv'
    fig_path = out_dir / f'fig3_sss_vs_population_{args.sss_source}.pdf'
    summary.to_csv(csv_path, index=False)
    plot(summary, sss_col, fig_path)

    rhos = summary['spearman_rho'].dropna()
    print(f"Analyzed {len(summary)} test graphs using column '{sss_col}'.")
    print(f"  Top-1 SSS == pop rank 1: {(summary['top_sss_pop_rank'] == 1).mean():.1%}")
    print(f"  Spearman ρ: median={rhos.median():.3f}, mean={rhos.mean():.3f}, "
          f"fraction > 0 = {(rhos > 0).mean():.1%}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
