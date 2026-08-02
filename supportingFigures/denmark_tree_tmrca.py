#!/usr/bin/env python3
"""
denmark_tree_tmrca.py — TMRCA (time to most recent common ancestor) of the five
Danish SARS-CoV-2 lineages, with per-lineage bootstrap uncertainty.

For each Nextclade clade (20I Alpha; 21I / 21J Delta; 21K / 21L Omicron) the
STEPHY bootstrap-uncertainty pipeline produces one ML point estimate (source
`ml_boot`) plus 40 nonparametric bootstrap replicate trees. This script reads
the per-lineage `summary/mrca.tsv` files, draws the bootstrap TMRCA distribution
for each clade on a shared calendar-date axis, and highlights where the ML point
estimate falls within that distribution.

Usage:
    python3 denmark_tree_tmrca.py
    python3 denmark_tree_tmrca.py --bootstrap_root /path/to/bootstrap_uncertainty
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Shared font config — unified across every figure script
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# ---------------------------------------------------------------------------
# Clade / lineage bookkeeping
# ---------------------------------------------------------------------------
# Order top-to-bottom by emergence: Alpha -> Delta -> Omicron.
CLADE_ORDER = ['20I', '21I', '21J', '21K', '21L']
LINEAGE_OF  = {'20I': 'Alpha', '21I': 'Delta', '21J': 'Delta',
               '21K': 'Omicron', '21L': 'Omicron'}
CLADE_LABEL = {c: f'{LINEAGE_OF[c]} ({c})' for c in CLADE_ORDER}

# One colour per lineage.
LINEAGE_COLOR = {
    'Alpha':   '#4C72B0',   # blue
    'Delta':   '#C44E52',   # red
    'Omicron': '#55A868',   # green
}
ML_COLOR = '#1a1a1a'        # near-black highlight for the ML point estimate

ML_SOURCE = 'ml_boot'       # ML point estimate reported in the manuscript


def _parse_date(s):
    return datetime.strptime(s.strip(), '%Y-%m-%d')


def load_mrca(bootstrap_root):
    """Return {clade: {'ml': datetime, 'reps': [datetime, ...]}} from all
    per-lineage summary/mrca.tsv files under bootstrap_root."""
    per_clade = defaultdict(lambda: {'ml': None, 'reps': []})
    found = False
    for lineage in ('Alpha', 'Delta', 'Omicron'):
        path = os.path.join(bootstrap_root, lineage, 'summary', 'mrca.tsv')
        if not os.path.exists(path):
            continue
        found = True
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                clade = row['clade']
                d = _parse_date(row['date'])
                if row['source'] == ML_SOURCE:
                    per_clade[clade]['ml'] = d
                elif row['source'].startswith('replicate_'):
                    per_clade[clade]['reps'].append(d)
    if not found:
        sys.exit(f'ERROR: no summary/mrca.tsv found under {bootstrap_root}')
    return per_clade


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


def main():
    default_root = ('/Users/lukelyu/Desktop/denmark_case/nextstrain/'
                    'bootstrap_uncertainty')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bootstrap_root', default=default_root,
                    help='root holding {Alpha,Delta,Omicron}/summary/mrca.tsv')
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    _log = open(os.path.join(out_dir, 'denmark_tree_tmrca.out'), 'w')
    sys.stdout = _Tee(sys.__stdout__, _log)

    data = load_mrca(args.bootstrap_root)

    # ---- diagnostic summary ------------------------------------------------
    print(f'TMRCA bootstrap summary  (ML source = {ML_SOURCE}, '
          f'root = {args.bootstrap_root})\n')
    hdr = f'{"clade":<6}{"lineage":<9}{"n_rep":>6}  {"ML":>12}  {"boot_min":>12}  {"boot_max":>12}  {"span_days":>9}'
    print(hdr)
    print('-' * len(hdr))
    for c in CLADE_ORDER:
        reps = sorted(data[c]['reps'])
        ml = data[c]['ml']
        span = (reps[-1] - reps[0]).days if reps else 0
        print(f'{c:<6}{LINEAGE_OF[c]:<9}{len(reps):>6}  '
              f'{ml.strftime("%Y-%m-%d"):>12}  {reps[0].strftime("%Y-%m-%d"):>12}  '
              f'{reps[-1].strftime("%Y-%m-%d"):>12}  {span:>9}')
    print()

    # ---- figure ------------------------------------------------------------
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))

    n = len(CLADE_ORDER)
    # Row 0 at top: reverse so earliest lineage sits highest.
    y_of = {c: (n - 1 - i) for i, c in enumerate(CLADE_ORDER)}
    half_w = 0.36   # violin half-width in row units

    for c in CLADE_ORDER:
        y = y_of[c]
        lin = LINEAGE_OF[c]
        col = LINEAGE_COLOR[lin]
        reps = data[c]['reps']
        ml = data[c]['ml']
        xr = mdates.date2num(reps)
        xmin, xmax = xr.min(), xr.max()

        # violin of the bootstrap distribution (horizontal, centred on the row)
        if len(reps) > 1 and xmax > xmin:
            vp = ax.violinplot([xr], positions=[y], vert=False,
                               widths=half_w * 2, showextrema=False)
            for body in vp['bodies']:
                body.set_facecolor(col)
                body.set_edgecolor(col)
                body.set_alpha(0.28)
                body.set_zorder(1)

        # full bootstrap range as a thin whisker
        ax.plot([xmin, xmax], [y, y], color=col, lw=1.0, alpha=0.7, zorder=2)
        for xe in (xmin, xmax):
            ax.plot([xe, xe], [y - 0.06, y + 0.06], color=col, lw=1.0,
                    alpha=0.7, zorder=2)

        # jittered replicate points
        jit = rng.uniform(-0.12, 0.12, size=len(xr))
        ax.scatter(xr, y + jit, s=9, color=col, alpha=0.55,
                   edgecolors='none', zorder=3)

        # ML point estimate — highlighted diamond + short vertical stem
        xm = mdates.date2num(ml)
        ax.plot([xm, xm], [y - half_w, y + half_w], color=ML_COLOR,
                lw=1.2, zorder=4)
        ax.scatter([xm], [y], marker='D', s=52, color=ML_COLOR,
                   edgecolors='white', linewidths=0.8, zorder=5)
        ax.annotate(f'{ml:%b %d, %Y}', xy=(xm, y + half_w + 0.02),
                    ha='center', va='bottom', fontsize=6.5, color=ML_COLOR)

    # y axis: clade labels
    ax.set_yticks([y_of[c] for c in CLADE_ORDER])
    ax.set_yticklabels([CLADE_LABEL[c] for c in CLADE_ORDER], fontsize=9)
    ax.set_ylim(-0.7, n - 1 + 0.9)

    # x axis: calendar dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.set_xlabel('TMRCA (calendar date)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7.5)

    ax.grid(axis='x', which='major', color='0.85', lw=0.5, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # legend
    handles = [
        Patch(facecolor='0.6', alpha=0.28, edgecolor='0.6',
              label='Bootstrap distribution (40 replicates)'),
        Line2D([0], [0], color='0.6', lw=1.0, marker='|', markersize=7,
               label='Full bootstrap range'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=ML_COLOR,
               markeredgecolor='white', markersize=7,
               label='ML point estimate'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              frameon=False, handletextpad=0.5, borderaxespad=0.4)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'denmark_tree_tmrca.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
