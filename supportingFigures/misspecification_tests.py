#!/usr/bin/env python3
"""
Model misspecification tests (4-row, 4-column layout).

The 100k model is trained on outbreaks whose sampling rate delta is shared by
all 12 locations and constant in time, on a fixed population scale, seeded
from a single location. Each row evaluates that model, unchanged, on a test
set that violates one of those assumptions, at three levels of severity. Same
layout and metric as fig3.py Row 3 (c-f): per-task headline score (R2 for
regression, accuracy for classification), one line per pipeline.

  Row 1 (a-d): delta varies across regions — location i samples at
               x * U(1-h, 1+h), h = 0.1 / 0.3 / 0.5.
  Row 2 (e-h): delta varies over time — every location steps
               x(1-h) -> x -> x(1+h) over equal thirds of the outbreak.
  Row 3 (i-l): population shift — every location's population scaled by
               1x / 2x / 3x (the same data as fig3.py Row 3).
  Row 4 (m-o): multiple introductions — k = 2 / 3 / 4 locations each seeded
               with one infected at t = 0. Regression targets only: the
               ancestral state (MRCA of all sampled tips) lies outside the
               modelled locations, so that cell carries the legend instead.

Per-level scores are printed as a table and teed to misspecification_tests.out.

Usage:
    python3 misspecification_tests.py
    python3 misspecification_tests.py --gen_root /path/to/dir_holding_5k_result_dirs
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

from _paths import under

# -- Shared style (consistent with fig3.py) ---------------------------------

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# -- Pipelines ---------------------------------------------------------------

PIPELINES = ['CBLV-CNN', 'stephy']
PIPELINE_STYLE = {
    'CBLV-CNN': ('#5B7E9E', 'o'),   # steel-blue circle
    'stephy':   ('#7A6FAC', 'D'),   # purple diamond
}
# Display labels (keys stay lowercase to match directory paths)
PIPELINE_LABELS = {
    'CBLV-CNN': 'CBLV-CNN',
    'stephy':   'STEPHY',
}
# Value labels go above STEPHY's markers and below CBLV-CNN's, so the two
# never collide where the scores coincide (every classification panel).
LABEL_OFFSET = {'stephy': (5, 'bottom'), 'CBLV-CNN': (-6, 'top')}

# -- Targets -----------------------------------------------------------------

TARGETS = ['reg_r0', 'reg_rr', 'reg_sss', 'cls_as']
TARGET_LABELS = {
    'reg_r0':  r'$R_0$ (Reg.)',
    'reg_rr':  r'$\gamma$ (Reg.)',
    'reg_sss': 'SSS (Reg.)',
    'cls_as':  'Ancestral state (Cls.)',
}
IS_CLASSIFICATION = {
    'reg_r0': False, 'reg_rr': False,
    'reg_sss': False, 'cls_as': True,
}
PRED_COLS = {
    'reg_r0':  ('true_reg_r0',  'pred_reg_r0'),
    'reg_rr':  ('true_reg_rr',  'pred_reg_rr'),
    'reg_sss': ('true_reg_sss', 'pred_reg_sss'),
    'cls_as':  ('true_ancestor', 'pred_ancestor'),
}

# -- Misspecification tests --------------------------------------------------
# One entry per row: (row label, x-axis label, levels, dir template, targets).
# Levels are strings so they can be both formatted into the directory name
# and used verbatim as tick labels. `targets` lists the columns a row can be
# scored on; a column missing from it is left blank (no axes, no letter).

HETERO_LEVELS = ['0.1', '0.3', '0.5']
SHIFT_LEVELS = ['X1', 'X2', 'X3']
SEED_LEVELS = ['2', '3', '4']
REGRESSION_TARGETS = [t for t in TARGETS if not IS_CLASSIFICATION[t]]

TESTS = [
    ('Sampling rate varies\nacross regions',
     r'Heterogeneity $h$', HETERO_LEVELS,
     '5k_diverse_population_heterogeneous_sampling_a_{level}_result', TARGETS),
    ('Sampling rate varies\nover time',
     r'Heterogeneity $h$', HETERO_LEVELS,
     '5k_diverse_population_heterogeneous_sampling_b_{level}_result', TARGETS),
    ('Population shift',
     'Population scale', SHIFT_LEVELS,
     '5k_diverse_population_shift_{level}_result', TARGETS),
    # Under multiple introductions the MRCA of the sampled tips is the
    # unsampled external origin, not a modelled location, so the
    # ancestral-state label is undefined and cls_as is not scored.
    ('Multiple\nintroductions',
     r'Seeded locations $k$', SEED_LEVELS,
     '5k_diverse_population_index_loc_{level}_result', REGRESSION_TARGETS),
]
# Text shown in the one cell that has no defined score (row 4, cls_as).
UNDEFINED_NOTE = 'Ancestral state not defined\nunder multiple introductions'

# One y-range for every panel (same as fig3.py Row 3), so a drop reads the
# same in every row and against fig3. The lowest score across all twelve
# test sets is 0.63 (CBLV-CNN SSS at X3), so nothing is clipped.
YLIM = (0.5, 1.0)


# -- stdout tee — score table saved next to the script -----------------------

class _Tee:
    """Mirror writes across multiple streams (tees stdout to misspecification_tests.out)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


# -- Helpers (mirror fig3.py) ------------------------------------------------

def load_predictions(target, pipeline_dir):
    """Return (true, pred) arrays from test_predictions.csv, or (None, None)."""
    path = os.path.join(pipeline_dir, target, 'test_predictions.csv')
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    true_col, pred_col = PRED_COLS[target]
    return df[true_col].values, df[pred_col].values


def compute_score(target, true_vals, pred_vals):
    """Return R2 (regression) or accuracy (classification), or None."""
    if true_vals is None:
        return None
    mask = ~(np.isnan(true_vals.astype(float)) |
             np.isnan(pred_vals.astype(float)))
    t, p = true_vals[mask], pred_vals[mask]
    if len(t) == 0:
        return None
    if IS_CLASSIFICATION[target]:
        return float(np.mean(t.astype(int) == p.astype(int)))
    return float(r2_score(t, p))


def load_test_scores(gen_root, levels, dir_tpl, targets):
    """For one test: {target: {pipeline: [score per level]}}.

    Only `targets` are scored. Missing (level, pipeline) entries are stored
    as None and reported.
    """
    out = {t: {p: [None] * len(levels) for p in PIPELINES} for t in targets}
    for i, level in enumerate(levels):
        level_root = os.path.join(gen_root, dir_tpl.format(level=level))
        for pipeline in PIPELINES:
            pdir = os.path.join(level_root, pipeline)
            for target in targets:
                score = compute_score(target, *load_predictions(target, pdir))
                if score is None:
                    print(f'  missing: {pdir}/{target}/test_predictions.csv')
                out[target][pipeline][i] = score
    return out


def make_pipeline_handles():
    """Build legend handles for the two pipelines."""
    return [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1],
               color=PIPELINE_STYLE[p][0], lw=1.2, label=PIPELINE_LABELS[p],
               markerfacecolor=PIPELINE_STYLE[p][0],
               markeredgecolor='white', markeredgewidth=0.4, markersize=5)
        for p in PIPELINES
    ]


def print_score_table(row_label, levels, scores):
    """One block per test: target x pipeline rows, one column per level."""
    print(f'\n{row_label.replace(chr(10), " ")}')
    header = f'  {"target":<8} {"pipeline":<9}' + ''.join(
        f'{lvl:>8}' for lvl in levels)
    print(header)
    for target in scores:
        for pipeline in PIPELINES:
            vals = scores[target][pipeline]
            cells = ''.join('     n/a' if v is None else f'{v:8.3f}'
                            for v in vals)
            print(f'  {target:<8} {PIPELINE_LABELS[pipeline]:<9}{cells}')


# -- Main --------------------------------------------------------------------

def main():
    """
    Build the misspecification figure — per-task headline score of both
    pipelines under four violated training assumptions, each at three
    severities. Output: misspecification_tests.{pdf,png,out} alongside this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_root', type=str,
                        default=under('models', 'simu'),
                        help='Parent dir holding the twelve 5k_diverse_population_'
                             '{heterogeneous_sampling_{a,b}_{0.1,0.3,0.5},'
                             'shift_{X1,X2,X3},index_loc_{2,3,4}}_result/ dirs')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    _log = open(os.path.join(out_dir, 'misspecification_tests.out'), 'w')
    sys.stdout = _Tee(sys.__stdout__, _log)

    print(f'gen_root = {args.gen_root}')
    all_scores = [load_test_scores(args.gen_root, levels, dir_tpl, targets)
                  for _, _, levels, dir_tpl, targets in TESTS]

    # -- Figure layout: one row per test, one column per target --------------
    n_rows, n_cols = len(TESTS), len(TARGETS)
    # Width sized so the bbox-trimmed PDF clears Nature's 183 mm cap (the
    # rotated row labels at left add to the trim); height keeps the panels
    # square and the trimmed page under the 247 mm cap.
    fig = plt.figure(figsize=(8.15, 9.55))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.40, wspace=0.35)
    pipeline_handles = make_pipeline_handles()

    panel_idx = 0
    for row, ((row_label, xlabel, levels, _, targets), scores) in enumerate(
            zip(TESTS, all_scores)):
        level_x = np.arange(len(levels))
        for col, target in enumerate(TARGETS):
            if target not in targets:
                # Undefined cell: no axes, no letter; carries the pipeline
                # legend and a one-line reason instead.
                ax = fig.add_subplot(gs[row, col])
                ax.set_box_aspect(1)
                ax.axis('off')
                ax.legend(handles=pipeline_handles, loc='center',
                          bbox_to_anchor=(0.5, 0.6), frameon=False,
                          fontsize=7, handlelength=2.0)
                ax.text(0.5, 0.3, UNDEFINED_NOTE, transform=ax.transAxes,
                        fontsize=6.5, color='0.45', ha='center', va='center',
                        linespacing=1.4)
                continue
            ax = fig.add_subplot(gs[row, col])
            is_cls = IS_CLASSIFICATION[target]

            for pipeline in PIPELINES:
                color, marker = PIPELINE_STYLE[pipeline]
                ys = scores[target][pipeline]
                if all(v is None for v in ys):
                    continue
                ax.plot(level_x, ys, marker=marker, color=color, lw=1.2,
                        markersize=5, markeredgecolor='white',
                        markeredgewidth=0.4, zorder=3)
                dy, va = LABEL_OFFSET[pipeline]
                for xi, v in zip(level_x, ys):
                    if v is None:
                        continue
                    ax.annotate(f'{v:.2f}', (xi, v), textcoords='offset points',
                                xytext=(0, dy), ha='center', va=va,
                                fontsize=6, color=color)

            ax.set_xticks(level_x)
            ax.set_xticklabels(levels)
            ax.set_xlim(-0.3, len(levels) - 0.7)
            ax.set_ylim(YLIM)
            ax.set_box_aspect(1)
            ax.set_xlabel(xlabel, fontsize=8)
            # The y-range is shared, so label it once per metric: R2 on the
            # first column, accuracy on the classification column.
            if col == 0 or is_cls:
                ax.set_ylabel('Accuracy' if is_cls else r'R$^2$', fontsize=8)
            if row == 0:
                ax.set_title(TARGET_LABELS[target], fontsize=9,
                             fontweight='bold', pad=10)
            if col == 0:
                ax.text(-0.45, 0.5, row_label, transform=ax.transAxes,
                        fontsize=8, fontweight='bold', rotation=90,
                        va='center', ha='center')
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.text(-0.18, 1.04, chr(ord('a') + panel_idx),
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    va='bottom', ha='left')
            panel_idx += 1

    for (row_label, _, levels, _, _), scores in zip(TESTS, all_scores):
        print_score_table(row_label, levels, scores)

    out_pdf = os.path.join(out_dir, 'misspecification_tests.pdf')
    out_png = os.path.join(out_dir, 'misspecification_tests.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    main()
