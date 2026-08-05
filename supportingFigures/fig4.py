#!/usr/bin/env python3
"""
Figure 4 — Denmark 2021 weekly epidemic summary (4 rows x 3 wave columns).

Rows:
  a) Estimated detected cases per region
  b) GISAID sequences per region
  c) Sampling proportion (%) before subsampling
  d) Sampling proportion (%) after subsampling

Columns are the three variant waves (Alpha, Delta, Omicron). Each column spans
exactly the subsampling window that fed the phylodynamic analysis, so every
week shown is a week the analysis used, and column widths are proportional to
window length — a week occupies the same space in all three.

Because the column identifies the variant, colour encodes region alone: five
fixed hues instead of the fifteen variant x region shades one shared axis
needs. Every row shares one y-axis across the three columns; the count rows
are log-scaled because Omicron's December peak is ~15x the Alpha and Delta
windows' and would otherwise flatten them against the baseline.

Restricting to the windows leaves the variants' out-of-window circulation off
the figure — most visibly Delta's autumn wave, since only ~29% of Delta
sequences fall inside its window. Per-column coverage is printed to fig4.out.

Row d is capped at 15%; an open caret above the cap marks a series that runs
off-scale there, and every such point is listed in fig4.out.

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
from matplotlib.ticker import NullFormatter, ScalarFormatter

from _paths import under


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

# Okabe-Ito, ordered so the weakest deuteranope pair (green/reddish-purple)
# is not adjacent in the legend. Markers repeat the same distinction, which
# keeps the series separable in greyscale and under colour-vision deficiency.
REGION_COLORS  = dict(zip(REGIONS, ['#0072B2', '#E69F00', '#009E73',
                                    '#D55E00', '#CC79A7']))
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


# ── Wave columns ─────────────────────────────────────────────────────────────
def window_positions(w2021, window):
    """Positions in `w2021.index` covered by one variant's subsampling window."""
    return [i for i, w in enumerate(w2021.index) if window[0] <= w <= window[1]]


def row_decades(y_fn, wave_pos):
    """Decade-aligned (bottom, top) spanning every positive value in a row.

    Log rows share one scale across the three columns, so the bounds have to
    come from all of them at once rather than from matplotlib's per-axes
    autoscale.
    """
    vals = [np.asarray(y_fn(variant, region), dtype=float)[pos]
            for variant, pos in wave_pos.items() for region in REGIONS]
    positive = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in vals])
    # The 1.4 backs the floor off its decade so a series sitting exactly on a
    # power of ten (Omicron/Sjaelland touches 1) keeps a whole marker.
    return (10.0 ** np.floor(np.log10(positive.min())) / 1.4,
            10.0 ** np.ceil(np.log10(positive.max())))


# ── Plotting helpers ─────────────────────────────────────────────────────────
def set_week_ticks(ax, week_starts, n_target=4):
    """Date ticks at a stride giving roughly `n_target` labels on a facet."""
    step = max(1, int(np.ceil(len(week_starts) / n_target)))
    idx = list(range(0, len(week_starts), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([week_starts[i].strftime('%d %b') for i in idx],
                       rotation=45, ha='right', fontsize=6)


def plot_facet(ax, y_fn, variant, pos):
    """The five region series for one variant over that wave's weeks."""
    x = np.arange(len(pos))
    for region in REGIONS:
        y = np.asarray(y_fn(variant, region), dtype=float)[pos]
        mask = np.isfinite(y) & (y > 0)
        if mask.any():
            ax.plot(x[mask], y[mask], color=REGION_COLORS[region],
                    marker=REGION_MARKERS[region], ms=2.2, lw=1.0, alpha=0.9)


def mark_offscale(ax, y_fn, variant, pos, w2021, ymax):
    """Flag every point an axis cap hides — on the panel and on stdout.

    An open caret sits just above the cap in the series colour, so a line
    leaving the top reads as deliberately off-scale rather than as a series
    that simply stops.
    """
    for region in REGIONS:
        y = np.asarray(y_fn(variant, region), dtype=float)[pos]
        for i in np.flatnonzero(np.isfinite(y) & (y > ymax)):
            ax.plot(i, ymax * 1.05, marker='^', ms=3, mew=0.6, clip_on=False,
                    markerfacecolor='none', markeredgecolor=REGION_COLORS[region])
            print(f'  off-scale {variant}/{region} {w2021.index[pos[i]]}: '
                  f'{y[i]:.1f} above the {ymax:g} axis cap')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='fig4: Denmark 2021 weekly epidemic summary.')
    parser.add_argument(
        '--data_root', type=str,
        default=under('denmark', 'paper', 'sars_cov2_290k_denmark-main'),
        help='Root of sars_cov2_290k_denmark-main (contains growth_rates/data/...).')
    parser.add_argument(
        '--gisaid_db', type=str,
        default=under('denmark', 'GISAID_db', 'metadata.db'),
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

    # ── Wave columns: each spans exactly its subsampling window ──────
    wave_pos = {v: window_positions(w2021, cfg['weeks'])
                for v, cfg in SUBSAMPLE_CONFIG.items()}
    for variant, pos in wave_pos.items():
        weeks = set(w2021.index[pos])
        of_variant = seq_records['variant'] == variant
        shown = int((of_variant & seq_records['epiweek'].isin(weeks)).sum())
        total = int(of_variant.sum())
        print(f'{variant:8s} column {w2021.index[pos[0]]}..{w2021.index[pos[-1]]} '
              f'({len(pos)} wks) shows {shown:,}/{total:,} sequences '
              f'({shown / total:.1%}); the rest circulated outside the window')

    # ── Figure: 4 rows x 3 wave columns + shared legend below ────────
    # Width tuned so the bbox-trimmed PDF clears Nature's 183 x 247 mm cap
    # for submission — the five-digit log tick labels widen the left
    # margin, so this is narrower than the linear-axis version needed.
    fig = plt.figure(figsize=(8.15, 9.40))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.32, wspace=0.12,
                           width_ratios=[len(wave_pos[v]) for v in VARIANTS])

    panel_labels = ['a', 'b', 'c', 'd']
    # Every row shares one y-axis across the three columns. The count rows use
    # a log scale because Omicron's December peak is ~15x the Alpha and Delta
    # windows' and would otherwise flatten them against the baseline.
    # ymax caps row d, where a couple of first-in-window points spike past 30%
    # (the subsampler keeps a minimum of one sequence per region-week) and would
    # otherwise squeeze the 4/8/12% plateaus that panel exists to show.
    panels = [
        ('Detected Cases per Week', True, None,
         lambda v, r: region_series(seq_by_region, v, r, 'variant_prop', w2021)
                       * w2021[f'cases_{r}']),
        ('Sequences per Week', True, None,
         lambda v, r: region_series(seq_by_region, v, r, 'seq_count', w2021)),
        ('Sampling Proportion (%)', False, None,
         lambda v, r: sampling_prop(seq_by_region, seq_by_region, w2021, v, r)),
        ('Subsampled Proportion (%)', False, 15.0,
         lambda v, r: sampling_prop(sub_by_region, seq_by_region, w2021, v, r)),
    ]

    label_axes = []          # the col-0 axes, whose ylabels get aligned below
    for row, (ylabel, log_y, ymax, y_fn) in enumerate(panels):
        bounds = row_decades(y_fn, wave_pos) if log_y else None
        anchor = None
        for col, variant in enumerate(VARIANTS):
            ax = fig.add_subplot(gs[row, col], sharey=anchor)
            if anchor is None:
                anchor = ax
            pos = wave_pos[variant]

            plot_facet(ax, y_fn, variant, pos)

            ax.set_xlim(-0.5, len(pos) - 0.5)
            if row == len(panels) - 1:
                set_week_ticks(ax, list(w2021['week_start'].iloc[pos]))
            else:
                ax.set_xticks([])

            if log_y:
                ax.set_yscale('log')
                ax.set_ylim(*bounds)
                fmt = ScalarFormatter()
                fmt.set_scientific(False)
                ax.yaxis.set_major_formatter(fmt)
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(bottom=0)
            if ymax is not None:
                ax.set_ylim(0, ymax)
                # Integer ticks, so the capped row reads like the one above it
                # rather than in 2.5-point decimals.
                ax.set_yticks(np.arange(0, ymax + 1, 5))
                mark_offscale(ax, y_fn, variant, pos, w2021, ymax)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
                ax.text(-0.20, 1.04, panel_labels[row], transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='bottom', ha='left')
                label_axes.append(ax)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(variant, fontsize=8, fontweight='bold', pad=4)

            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out', length=2, width=0.4)
            ax.tick_params(axis='y', which='minor', length=0)

            ax.grid(True, axis='y', alpha=0.2)
            if log_y:
                ax.grid(True, axis='y', which='minor', alpha=0.08)
            ax.set_axisbelow(True)

    # Tick-label widths differ per row (six-digit log counts vs two-digit
    # percentages), which would otherwise leave the four ylabels at four
    # different x positions.
    fig.align_ylabels(label_axes)

    legend_handles = [
        Line2D([0], [0], color=REGION_COLORS[region],
               marker=REGION_MARKERS[region], ms=3, lw=1.0, label=region)
        for region in REGIONS
    ]
    fig.legend(handles=legend_handles, loc='lower center', frameon=False,
               ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.02))

    out_pdf = out_dir / 'fig4.pdf'
    out_png = out_dir / 'fig4.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    main()
