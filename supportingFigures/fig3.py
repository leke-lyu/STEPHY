#!/usr/bin/env python3
"""
Figure 3 — Denmark 2021 by region: detected cases, sampling and Rt
(3 rows x 3 variant waves, one shared legend).

  Row a  Detected cases per week, on a log axis
  Row b  Genomes per detected case (%): every sequenced genome (dashed, faded)
         and the subsampled dataset behind the phylogenies (solid)
  Row c  Weekly Rt from EpiEstim with its 95% credible interval; weeks flagged
         unreliable (posterior CV > 0.3) are left out

Columns are the three variant waves (Alpha, Delta, Omicron). Each spans exactly
that variant's subsampling window, with widths proportional to window length,
so a week occupies the same space in all three. Every row shares one y-axis
across the columns, and colour and marker encode region throughout.

This merges regional_prevalence.pdf, regional_sampling.pdf and regional_rt.pdf
from stage 1 of the stephy-denmark workflow, and reads the same tables those
were drawn from. No GISAID access is needed.

Outputs next to this script: fig3.pdf, fig3.png, and fig3.out (the weeks that
are not drawn, and why).

Defaults resolve from DENMARK_CASE (see _paths.py):
  --workflow_dir  $DENMARK_CASE/stephy-denmark/workflow

Usage:
    export DENMARK_CASE=/path/to/denmark_case
    python3 fig3.py
    python3 fig3.py --workflow_dir /path/to/stephy-denmark/workflow
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from _paths import under


# ── Publication defaults — unified font style across every figure script ───────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':       7,
    'axes.labelsize':  7,
    'axes.titlesize':  8,
    'axes.linewidth':  0.4,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'legend.frameon':  False,
    'mathtext.default': 'regular',
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# ── Constants ────────────────────────────────────────────────────────────────
REGIONS = ['Hovedstaden', 'Midtjylland', 'Syddanmark', 'Sjaelland', 'Nordjylland']
# Variant -> first and last ISO week of its subsampling window.
WAVES = {
    'Alpha':   ('2021-W01', '2021-W22'),
    'Delta':   ('2021-W23', '2021-W35'),
    'Omicron': ('2021-W47', '2021-W52'),
}
KEYS = ['variant', 'epiweek', 'region']

# Okabe-Ito, ordered so the weakest deuteranope pair (green/reddish-purple)
# is not adjacent in the legend. Markers repeat the same distinction, which
# keeps the series separable in greyscale and under colour-vision deficiency.
REGION_COLORS  = dict(zip(REGIONS, ['#0072B2', '#E69F00', '#009E73',
                                    '#D55E00', '#CC79A7']))
REGION_MARKERS = dict(zip(REGIONS, ['o', 's', 'D', '^', 'v']))

LINE = dict(lw=0.8, ms=2.2)
ALL_GENOMES = dict(ls='--', alpha=0.35)     # row b's unthinned series
HALF_WEEK = pd.Timedelta(days=3.5)


# ── stdout tee — diagnostic output saved next to the script ────────────────────────────
class _Tee:
    """Mirror writes across multiple streams (tees stdout to fig3.out)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


# ── Data ─────────────────────────────────────────────────────────────────────
def load_weekly(workflow_dir):
    """One row per variant x week x region inside the wave windows.

    Joins stage 0's prevalence table with the subsampled genome counts and the
    weekly Rt estimates from stage 1.
    """
    stage0 = Path(workflow_dir) / '00_prevalence'
    stage1 = Path(workflow_dir) / '01_subsampleAndRt'

    df = pd.read_csv(stage0 / 'prevalence_by_week.tsv', sep='\t')
    window = df['variant'].map(WAVES)
    df = df[df['region'].isin(REGIONS) & window.notna()
            & (df['epiweek'] >= window.str[0]) & (df['epiweek'] <= window.str[1])].copy()
    df['week_start'] = pd.to_datetime(df['epiweek'] + '-1', format='%G-W%V-%u')

    # From the counts, not the stored ratio, as the workflow's own figures do.
    df['detected'] = np.where(df['total_seq'] > 0,
                              df['n_seq'] / df['total_seq'] * df['cases'], 0.0)

    kept = (pd.read_csv(stage1 / 'subsample_assignment.tsv', sep='\t')
            .groupby(KEYS).size().rename('n_kept'))
    df = df.join(kept, on=KEYS).fillna({'n_kept': 0})
    detected = df['detected'].replace(0, np.nan)
    df['pct_all'] = df['n_seq'] / detected * 100
    df['pct_sub'] = df['n_kept'] / detected * 100

    rt = pd.read_csv(stage1 / 'rt_weekly.tsv', sep='\t')
    df = df.merge(rt[KEYS + ['rt_mean', 'rt_q025', 'rt_q975', 'reliable']],
                  on=KEYS, how='left')
    return df.sort_values('week_start')


def report_undrawn(df):
    """List the weeks that leave a gap in a row, since neither shows on the figure."""
    for _, r in df[df['detected'] <= 0].iterrows():
        print(f'  rows a, b: {r.variant}/{r.region} {r.epiweek} has no detected cases')
    for _, r in df[df['reliable'] != True].iterrows():       # noqa: E712 (NaN-safe)
        print(f'  row c: {r.variant}/{r.region} {r.epiweek} Rt unreliable '
              f'({r.detected:.0f} detected cases)')


# ── Plotting helpers ─────────────────────────────────────────────────────────
def plot_regions(ax, wave, col, **style):
    """One line per region for column `col`, skipping weeks without a value."""
    for region in REGIONS:
        d = wave[(wave['region'] == region) & wave[col].notna()]
        ax.plot(d['week_start'], d[col], color=REGION_COLORS[region],
                marker=REGION_MARKERS[region], **{**LINE, **style})


def draw_cases(ax, wave):
    plot_regions(ax, wave[wave['detected'] > 0], 'detected')


def draw_sampling(ax, wave):
    plot_regions(ax, wave, 'pct_all', **ALL_GENOMES)
    plot_regions(ax, wave, 'pct_sub')


def draw_rt(ax, wave):
    wave = wave[wave['reliable'] == True]                    # noqa: E712 (NaN-safe)
    ax.axhline(1, ls='--', lw=0.5, color='0.4', zorder=1)
    for region in REGIONS:
        d = wave[wave['region'] == region]
        ax.fill_between(d['week_start'], d['rt_q025'], d['rt_q975'],
                        color=REGION_COLORS[region], alpha=0.12, lw=0)
    plot_regions(ax, wave, 'rt_mean')


def set_date_ticks(ax, weeks):
    """Month starts on a long wave, every week start on a short one."""
    lo, hi = weeks.min() - HALF_WEEK, weeks.max() + HALF_WEEK
    ax.set_xlim(lo, hi)
    ticks = weeks if len(weeks) <= 8 else pd.date_range(lo.normalize(), hi, freq='MS')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Figure 3: Denmark 2021 regional detected cases, sampling and Rt.')
    parser.add_argument(
        '--workflow_dir', type=str,
        default=under('denmark', 'stephy-denmark', 'workflow'),
        help='stephy-denmark workflow/ directory (contains 00_prevalence/ and '
             '01_subsampleAndRt/).')
    args = parser.parse_args()

    out_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    log = open(out_dir / 'fig3.out', 'w')
    sys.stdout = _Tee(sys.__stdout__, log)

    df = load_weekly(args.workflow_dir)
    print(f'Loaded: {len(df)} variant x week x region rows, '
          f'{int(df["n_kept"].sum()):,} subsampled genomes')
    print('Not drawn:')
    report_undrawn(df)

    # ── Figure: 3 rows x 3 wave columns + shared legend below ────────
    waves = {v: df[df['variant'] == v] for v in WAVES}
    fig = plt.figure(figsize=(7.05, 6.3))
    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.16, wspace=0.07,
        left=0.085, right=0.995, top=0.955, bottom=0.145,
        width_ratios=[w['epiweek'].nunique() for w in waves.values()])

    panels = [
        ('Detected cases per week', draw_cases),
        ('Genomes per detected case (%)', draw_sampling),
        ('$R_t$', draw_rt),
    ]

    label_axes = []          # the col-0 axes, whose ylabels get aligned below
    col_axes = {}            # column -> top axes, shared x down the column
    for row, (ylabel, draw) in enumerate(panels):
        anchor = None
        for col, (variant, wave) in enumerate(waves.items()):
            ax = fig.add_subplot(gs[row, col], sharey=anchor, sharex=col_axes.get(col))
            anchor = anchor or ax
            col_axes.setdefault(col, ax)

            draw(ax, wave)

            if row == len(panels) - 1:
                set_date_ticks(ax, wave['week_start'].drop_duplicates())
            else:
                ax.tick_params(labelbottom=False)
            if row == 0:
                ax.set_title(variant, fontsize=8, fontweight='bold', pad=4)

            # Every panel keeps its y ticks, but only the first draws the axis
            # line and the labels.
            if col == 0:
                ax.set_ylabel(ylabel)
                ax.text(-0.155, 1.03, 'abc'[row], transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='bottom', ha='left')
                label_axes.append(ax)
            else:
                ax.tick_params(labelleft=False)
                ax.spines['left'].set_visible(False)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(which='both', direction='out', width=0.4)
            ax.tick_params(which='major', length=2)
            ax.tick_params(which='minor', length=1)

        # Row-wide y scale, set once all three columns hold their data.
        if draw is draw_cases:
            anchor.set_yscale('log')
            anchor.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        elif draw is draw_sampling:
            anchor.set_ylim(bottom=0)

    fig.align_ylabels(label_axes)
    fig.text(0.54, 0.072, 'Week starting', ha='center', va='center', fontsize=7)

    grey = dict(color='0.3', lw=LINE['lw'])
    legend_handles = [
        Line2D([0], [0], label='All sequenced genomes', **grey, **ALL_GENOMES),
        Line2D([0], [0], label='Subsampled dataset', **grey),
    ] + [
        Line2D([0], [0], color=REGION_COLORS[region], marker=REGION_MARKERS[region],
               label=region, **LINE)
        for region in REGIONS
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               bbox_to_anchor=(0.54, 0.0), fontsize=6, handlelength=2.2,
               columnspacing=1.0, handletextpad=0.4)

    stem = out_dir / 'fig3'
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=600)
    print(f'Saved: {stem}.pdf')
    print(f'Saved: {stem}.png')
    plt.close()


if __name__ == '__main__':
    main()
