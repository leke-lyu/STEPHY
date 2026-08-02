#!/usr/bin/env python3
"""
fig4.py — Denmark 2021 weekly epidemic summary (4 panels).

  a) Estimated detected cases by variant x region
  b) GISAID sequences by variant x region
  c) Sampling proportion (%) before subsampling
  d) Sampling proportion (%) after subsampling

Usage:
    python3 fig4.py
    python3 fig4.py --data_root /path/to/sars_cov2_290k_denmark-main \
                    --gisaid_db /path/to/metadata.db
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


# ── Publication defaults — unified font style across every figure script ───────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':       8,
    'axes.labelsize':  8,
    'axes.titlesize':  9,
    'axes.linewidth':  0.4,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'legend.frameon':  False,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# ── Constants ────────────────────────────────────────────────────────────────
REGIONS  = ['Hovedstaden', 'Midtjylland', 'Syddanmark', 'Sjaelland', 'Nordjylland']
VARIANTS = ['Alpha', 'Delta', 'Omicron']
DATE_START, DATE_END = '2021-01-04', '2022-01-02'
SSI_REGION_MAP = {'Hovedstaden': 'Hovedstaden', 'Midtjylland': 'Midtjylland',
                  'Syddanmark': 'Syddanmark', 'Sjælland': 'Sjaelland',
                  'Nordjylland': 'Nordjylland'}

VARIANT_COLORS = {'Alpha': '#2196F3', 'Delta': '#FF9800', 'Omicron': '#9C27B0'}
REGION_MARKERS = dict(zip(REGIONS, ['o', 's', 'D', '^', 'v']))

SUBSAMPLE_CONFIG = {
    'Alpha':   {'weeks': ('2021-W01', '2021-W22'), 'baseline': 0.08},
    'Delta':   {'weeks': ('2021-W23', '2021-W35'), 'baseline': 0.12},
    'Omicron': {'weeks': ('2021-W47', '2021-W52'), 'baseline': 0.04},
}
RANDOM_SEED = 42


# ── stdout tee — diagnostic output saved next to the script ────────────────────────────
class _Tee:
    """Mirror writes across multiple streams (tees stdout to fig4.out)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


# ── Visual helpers ───────────────────────────────────────────────────────────
def _make_shades(hex_color, n):
    """n shades from light to full saturation of hex_color."""
    r, g, b = to_rgb(hex_color)
    return [tuple(c * t + (1 - t) for c in (r, g, b))
            for t in np.linspace(0.3, 1.0, n)]

REGION_SHADES = {v: dict(zip(REGIONS, _make_shades(c, len(REGIONS))))
                 for v, c in VARIANT_COLORS.items()}


# ── Data helpers ─────────────────────────────────────────────────────────────
def to_epiweek(dates):
    """Convert dates to ISO epiweek strings (e.g. '2021-W03')."""
    iso = dates.dt.isocalendar() if hasattr(dates, 'dt') else dates.isocalendar()
    return iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)


def region_series(df, variant, region, col, w2021):
    """Extract one (variant, region) column reindexed to w2021 epiweeks."""
    sub = df.query('variant == @variant and region == @region')
    return sub.set_index('epiweek')[col].reindex(w2021.index, fill_value=0)


def compute_variant_counts(records):
    """Per-(epiweek, variant, region) counts with within-region variant proportions."""
    by_region = (records.groupby(['epiweek', 'variant', 'region']).size()
                 .rename('seq_count').reset_index())
    totals = records.groupby(['epiweek', 'region']).size().rename('total').reset_index()
    by_region = by_region.merge(totals, on=['epiweek', 'region'])
    by_region['variant_prop'] = by_region['seq_count'] / by_region['total']
    return by_region


def sampling_prop(seq_df, seq_by_region, w2021, variant, region):
    """Sampling proportion (%) = sequences / estimated detected cases."""
    seqs = region_series(seq_df, variant, region, 'seq_count', w2021)
    detected = (region_series(seq_by_region, variant, region, 'variant_prop', w2021)
                * w2021[f'cases_{region}']).replace(0, np.nan)
    return seqs / detected * 100


# ── Plotting helpers ─────────────────────────────────────────────────────────
def add_month_labels(ax, week_starts):
    """Vertical month dividers + centered month labels on a panel."""
    months, positions = [], []
    for i, dt in enumerate(week_starts):
        m = dt.strftime('%b')
        if not months or m != months[-1]:
            positions.append(i)
            months.append(m)
    for idx, (pos, label) in enumerate(zip(positions, months)):
        ax.axvline(x=pos - 0.5, color='gray', ls='--', alpha=0.3, lw=0.4)
        nxt = positions[idx + 1] if idx + 1 < len(positions) else len(week_starts)
        ax.text((pos + nxt) / 2, ax.get_ylim()[1] * 0.95, label,
                fontsize=6, fontweight='bold', ha='center', va='top', color='gray')


def plot_panel(ax, y_fn, w2021):
    """All variant x region lines on a panel using y_fn(variant, region)."""
    x = np.arange(len(w2021))
    for variant in VARIANTS:
        for region in REGIONS:
            y = np.asarray(y_fn(variant, region), dtype=float)
            mask = np.isfinite(y) & (y > 0)
            if mask.any():
                ax.plot(x[mask], y[mask],
                        color=REGION_SHADES[variant][region],
                        marker=REGION_MARKERS[region], ms=2, lw=0.8, alpha=0.8)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='fig4: Denmark 2021 weekly epidemic summary.')
    parser.add_argument(
        '--data_root', type=str,
        default='/Users/lukelyu/Desktop/denmark_case/paper/'
                'sars_cov2_290k_denmark-main',
        help='Root of sars_cov2_290k_denmark-main (contains growth_rates/data/...).')
    parser.add_argument(
        '--gisaid_db', type=str,
        default='/Users/lukelyu/Desktop/denmark_case/GISAID_db/metadata.db',
        help='Path to the SQLite GISAID metadata DB.')
    args = parser.parse_args()

    out_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    log = open(out_dir / 'fig4.out', 'w')
    sys.stdout = _Tee(sys.__stdout__, log)

    # ── Load SSI cases ───────────────────────────────────────────────
    cases_raw = pd.read_csv(
        f'{args.data_root}/growth_rates/data/'
        '03_bekraeftede_tilfaelde_doede_indlagte_pr_dag_pr_koen.csv',
        sep=';', encoding='latin-1')
    date_col   = cases_raw.columns[2]
    case_col   = cases_raw.columns[4]
    region_col = cases_raw.columns[1]
    cases_raw[date_col] = pd.to_datetime(cases_raw[date_col])
    cases_raw[case_col] = pd.to_numeric(cases_raw[case_col], errors='coerce')
    cases_raw['region'] = cases_raw[region_col].map(SSI_REGION_MAP)

    daily_cases = cases_raw.groupby(date_col)[case_col].sum().rename('cases')
    daily_cases.index.name = 'date'
    cases_2021 = daily_cases.loc[DATE_START:DATE_END].to_frame()
    cases_2021['epiweek'] = to_epiweek(cases_2021.index)

    w2021 = (cases_2021.groupby('epiweek')
             .agg(week_start=('cases', lambda x: x.index.min()),
                  cases=('cases', 'sum'), days=('cases', 'count'))
             .query('days == 7').sort_values('week_start').drop(columns='days'))

    daily_region = (cases_raw.groupby([date_col, 'region'])[case_col]
                    .sum().unstack(fill_value=0))
    region_2021 = daily_region.loc[DATE_START:DATE_END].copy()
    region_2021['epiweek'] = to_epiweek(region_2021.index)
    w2021_region = region_2021.groupby('epiweek').sum().reindex(w2021.index)
    w2021_region.columns = [f'cases_{r}' for r in w2021_region.columns]
    w2021 = w2021.join(w2021_region)

    # ── Load GISAID sequences ────────────────────────────────────────
    VARIANT_SQL = """CASE WHEN Variant LIKE '%Alpha%'   THEN 'Alpha'
                          WHEN Variant LIKE '%Delta%'   THEN 'Delta'
                          WHEN Variant LIKE '%Omicron%' THEN 'Omicron'
                          ELSE 'Other' END AS variant"""
    BASE_WHERE = """WHERE Location LIKE '%Denmark%'
                      AND "Collection date" LIKE '____-__-__'
                      AND "Collection date" BETWEEN '2021-01-01' AND '2022-01-02'"""
    region_clause = ' OR '.join(f'Location LIKE "%{r}%"' for r in REGIONS)

    with sqlite3.connect(args.gisaid_db) as conn:
        seq_records = pd.read_sql_query(f"""
            SELECT "Accession ID" AS accession_id, "Virus name" AS virus_name,
                   "Collection date" AS date, Location AS location,
                   "Pango lineage" AS lineage, {VARIANT_SQL}
            FROM metadata {BASE_WHERE} AND ({region_clause})""", conn)

    seq_records['date'] = pd.to_datetime(seq_records['date'])
    seq_records['epiweek'] = to_epiweek(seq_records['date'])
    seq_records['region'] = seq_records['location'].str.split(' / ').str[2].str.strip()

    seq_by_region = compute_variant_counts(seq_records)

    print(f'Loaded: {len(w2021)} weeks, {len(seq_records):,} sequences')

    # ── Subsampling ──────────────────────────────────────────────────
    valid_weeks = {v: [w for w in w2021.index if cfg['weeks'][0] <= w <= cfg['weeks'][1]]
                   for v, cfg in SUBSAMPLE_CONFIG.items()}

    rng = np.random.default_rng(RANDOM_SEED)
    subsampled_ids = set()

    for variant, cfg in SUBSAMPLE_CONFIG.items():
        for week in valid_weeks[variant]:
            for region_name in REGIONS:
                mask = ((seq_records['variant'] == variant) &
                        (seq_records['epiweek'] == week) &
                        (seq_records['region'] == region_name))
                week_seqs = seq_records.loc[mask]
                n_seqs = len(week_seqs)

                vprop = region_series(seq_by_region, variant, region_name,
                                      'variant_prop', w2021).get(week, 0)
                n_detected = vprop * w2021.loc[week, f'cases_{region_name}']

                if n_detected <= 0 or n_seqs == 0:
                    subsampled_ids.update(week_seqs['accession_id'])
                    continue

                n_target = max(1, int(np.floor(cfg['baseline'] * n_detected)))
                if n_seqs > n_target:
                    chosen = rng.choice(week_seqs.index, size=n_target, replace=False)
                    subsampled_ids.update(week_seqs.loc[chosen, 'accession_id'])
                else:
                    subsampled_ids.update(week_seqs['accession_id'])

    sub_records = seq_records[seq_records['accession_id'].isin(subsampled_ids)].copy()
    sub_by_region = compute_variant_counts(sub_records)

    print(f'Subsampled: {len(subsampled_ids):,} sequences')

    # ── Figure: 4 stacked rows + shared legend below ─────────────────
    fig = plt.figure(figsize=(8.30, 10.6))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.25)

    panel_labels = ['a', 'b', 'c', 'd']
    panels = [
        ('Detected Cases per Week',
         lambda v, r: region_series(seq_by_region, v, r, 'variant_prop', w2021)
                       * w2021[f'cases_{r}']),
        ('Sequences per Week',
         lambda v, r: region_series(seq_by_region, v, r, 'seq_count', w2021)),
        ('Sampling Proportion (%)',
         lambda v, r: sampling_prop(seq_by_region, seq_by_region, w2021, v, r)),
        ('Sampling Proportion (%), Subsampled',
         lambda v, r: sampling_prop(sub_by_region, seq_by_region, w2021, v, r)),
    ]

    for row, (ylabel, y_fn) in enumerate(panels):
        ax = fig.add_subplot(gs[row])
        plot_panel(ax, y_fn, w2021)

        ax.set_xlim(-0.5, len(w2021) - 0.5)
        tick_idx = list(range(0, len(w2021), 4))
        ax.set_xticks(tick_idx)
        if row == 3:
            ax.set_xticklabels([w2021.index[i] for i in tick_idx],
                               rotation=45, ha='right', fontsize=6)
        else:
            ax.set_xticklabels([])

        ax.set_ylabel(ylabel, fontsize=8)
        if 'Proportion' in ylabel:
            ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_axisbelow(True)

        add_month_labels(ax, w2021['week_start'])

        ax.text(-0.04, 1.02, panel_labels[row], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')

    legend_handles = [
        Line2D([0], [0], color=REGION_SHADES[variant][region],
               marker=REGION_MARKERS[region], ms=3, lw=0.8,
               label=f'{variant} – {region}')
        for variant in VARIANTS for region in REGIONS
    ]
    fig.legend(handles=legend_handles, loc='lower center', frameon=False,
               ncol=5, fontsize=6, bbox_to_anchor=(0.5, 0.01))

    out_pdf = out_dir / 'fig4.pdf'
    out_png = out_dir / 'fig4.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    main()
