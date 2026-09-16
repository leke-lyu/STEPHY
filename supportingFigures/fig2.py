#!/usr/bin/env python3
"""
Figure 2 — Prediction accuracy in distribution and under model misspecification
(6 rows x 4 targets: R_0, gamma, SSS, ancestral state).

  Row 1 (a-d)  STEPHY on the 100k diverse-population test split: true vs
               predicted for each regression target, per-state accuracy for
               the ancestral state.
  Row 2 (e-h)  Both pipelines: score (R2 or accuracy) against conformal
               uncertainty (mean interval width or mean prediction-set size).
  Rows 3-6     The same 100k diverse-population models, both pipelines, on
               5k test sets that each violate one training assumption, scored
               per task at increasing severity:
    Row 3 (i-l)  sampling rate varies across regions, heterogeneity 0.1 / 0.2 / 0.4
    Row 4 (m-p)  sampling rate varies over time, heterogeneity 0.1 / 0.2 / 0.4
    Row 5 (q-t)  population shift, 1x / 2x / 3x
    Row 6 (u-w)  multiple introductions, k = 2 / 3 / 4 seeded locations;
                 regression only, because the ancestral state is undefined

Outputs next to this script: fig2.pdf, fig2.png, and fig2.out (row 1-2
metrics and the per-level misspecification scores).

Defaults resolve from STEPHY_MODELS (see _paths.py):
  --result_dir    $STEPHY_MODELS/simulation_benchmark/100k_diverse_population_result
  --misspec_root  $STEPHY_MODELS/simulation_benchmark/misspecification

Usage:
    export STEPHY_MODELS=/path/to/trained_model
    python3 fig2.py
    python3 fig2.py --result_dir /path/to/100k_result --misspec_root /path/to/misspecification
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from _paths import under

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# -- Pipelines and targets ---------------------------------------------------

PIPELINES = ['CBLV-CNN', 'stephy']
PIPELINE_STYLE = {'CBLV-CNN': ('#5B7E9E', 'o'), 'stephy': ('#7A6FAC', 'D')}
PIPELINE_LABELS = {'CBLV-CNN': 'CBLV-CNN', 'stephy': 'STEPHY'}

TARGETS = ['reg_r0', 'reg_rr', 'reg_sss', 'cls_as']
TARGET_LABELS = {
    'reg_r0':  r'$R_0$ (Reg.)',
    'reg_rr':  r'$\gamma$ (Reg.)',
    'reg_sss': 'SSS (Reg.)',
    'cls_as':  'Ancestral state (Cls.)',
}
TARGET_COLORS = {'reg_r0': '#4878A8', 'reg_rr': '#6AAB6A',
                 'reg_sss': '#E8963E', 'cls_as': '#C25B5B'}
PRED_COLS = {
    'reg_r0':  ('true_reg_r0',  'pred_reg_r0'),
    'reg_rr':  ('true_reg_rr',  'pred_reg_rr'),
    'reg_sss': ('true_reg_sss', 'pred_reg_sss'),
    'cls_as':  ('true_ancestor', 'pred_ancestor'),
}
REGRESSION_TARGETS = [t for t in TARGETS if not t.startswith('cls_')]   # label prefix sets the task type

# Row 1 axis limits and ticks.
AXIS_CFG = {
    'reg_r0':  {'lims': (2, 8),       'ticks': [2, 4, 6, 8]},
    'reg_rr':  {'lims': (0.05, 0.25), 'ticks': [0.05, 0.10, 0.15, 0.20, 0.25]},
    'reg_sss': {'lims': (-1, 1),      'ticks': [-1, -0.5, 0, 0.5, 1]},
    'cls_as':  {'lims': (-0.5, 11.5), 'ticks': list(range(0, 12, 2))},
}
# Row 2 x-limits for mean interval width / mean set size.
CP_XLIMS = {'reg_r0': (0, 6), 'reg_rr': (0, 0.2), 'reg_sss': (0, 2), 'cls_as': (0, 12)}

# -- Misspecification tests (rows 3-6) ---------------------------------------
# (row label, x label, levels, result-dir template, scored targets). Levels
# are strings so they serve as both directory suffix and tick label; a
# target missing from the row's list leaves that cell blank.

HETERO_LEVELS = ['0.1', '0.2', '0.4']
TESTS = [
    ('Sampling rate varies\nacross regions', 'Heterogeneity scale', HETERO_LEVELS,
     '5k_diverse_population_heterogeneous_sampling_a_{level}_result', TARGETS),
    ('Sampling rate varies\nover time', 'Heterogeneity scale', HETERO_LEVELS,
     '5k_diverse_population_heterogeneous_sampling_b_{level}_result', TARGETS),
    ('Population shift', 'Population scale', ['X1', 'X2', 'X3'],
     '5k_diverse_population_shift_{level}_result', TARGETS),
    # With several introductions the MRCA of the sampled tips is the unsampled
    # external origin, not a modelled location, so cls_as is not scored.
    ('Multiple\nintroductions', r'Seeded locations $k$', ['2', '3', '4'],
     '5k_diverse_population_index_loc_{level}_result', REGRESSION_TARGETS),
]
UNDEFINED_NOTE = 'Ancestral state not defined\nunder multiple introductions'
# Shared score range for rows 2-6. The lowest plotted score is 0.62 (CBLV-CNN
# SSS at 3x population); the top runs past 1.0 so labels on scores near 1 clear
# the top spine.
SCORE_YLIM = (0.5, 1.07)
SCORE_YTICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Value labels: at each level the higher score is labelled above its marker
# and the lower below, so labels never collide where lines meet or cross.
# STEPHY wins ties.
LABEL_ABOVE, LABEL_BELOW = (4, 'bottom'), (-5, 'top')
INFO_BOX = dict(boxstyle='round', facecolor='white', alpha=0.8)


class _Tee:
    """Mirror writes across streams (tees stdout to fig2.out)."""
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams: s.write(x)
    def flush(self):
        for s in self.streams: s.flush()


# -- Loading and scoring -----------------------------------------------------

def is_classification(target):
    """Return True for a classification target (cls_ prefix)."""
    return target.startswith('cls_')


def score_label(target):
    """Return the axis label of a target's headline score: accuracy or R2."""
    return 'Accuracy' if is_classification(target) else r'R$^2$'


def load_predictions(pipeline_dir, target):
    """Return (true, pred) arrays from {pipeline_dir}/{target}/test_predictions.csv, or (None, None)."""
    path = os.path.join(pipeline_dir, target, 'test_predictions.csv')
    if not os.path.exists(path):
        return None, None
    true_col, pred_col = PRED_COLS[target]
    df = pd.read_csv(path, usecols=[true_col, pred_col])
    return df[true_col].values, df[pred_col].values


def regression_metrics(true_vals, pred_vals):
    """Return {R2, r, MSE} over pairs where neither value is NaN."""
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[mask], pred_vals[mask]
    return {'R2': r2_score(t, p), 'r': pearsonr(t, p)[0], 'MSE': mean_squared_error(t, p)}


def compute_score(target, true_vals, pred_vals):
    """Return R2 (regression) or top-1 accuracy (classification), or None without data."""
    if true_vals is None:
        return None
    if is_classification(target):
        return float(np.mean(true_vals.astype(int) == pred_vals.astype(int)))
    return float(regression_metrics(true_vals, pred_vals)['R2'])


def load_cp_uncertainty(pipeline_dir, target):
    """Return mean prediction-set size (classification) or mean interval width (regression), or None."""
    path = os.path.join(pipeline_dir, target, 'cp_metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        cp = json.load(fh)
    return cp['mean_set_size' if is_classification(target) else 'mean_interval_width']


def load_test_scores(root, levels, dir_tpl, targets):
    """
    Score every (target, pipeline, level) of one misspecification test.

    Returns {target: {pipeline: [score per level]}}; missing prediction files
    are reported and stored as None.
    """
    out = {t: {p: [] for p in PIPELINES} for t in targets}
    for level in levels:
        for pipeline in PIPELINES:
            pdir = os.path.join(root, dir_tpl.format(level=level), pipeline)
            for target in targets:
                score = compute_score(target, *load_predictions(pdir, target))
                if score is None:
                    print(f'  missing: {pdir}/{target}/test_predictions.csv')
                out[target][pipeline].append(score)
    return out


# -- Panel drawing -----------------------------------------------------------

def plot_true_vs_pred(ax, true_vals, pred_vals, target):
    """Scatter true against predicted on equal axes with a y = x line and an R2 / r / MSE box."""
    cfg = AXIS_CFG[target]
    ax.scatter(true_vals, pred_vals, alpha=0.1, s=3, color=TARGET_COLORS[target],
               linewidths=0, rasterized=True)
    ax.plot(cfg['lims'], cfg['lims'], 'k--', lw=0.8)
    ax.set_xlim(cfg['lims'])
    ax.set_ylim(cfg['lims'])
    ax.set_xticks(cfg['ticks'])
    ax.set_yticks(cfg['ticks'])
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    m = regression_metrics(true_vals, pred_vals)
    ax.text(0.95, 0.05, f"R² = {m['R2']:.3f}\nr = {m['r']:.3f}\nMSE = {m['MSE']:.3g}",
            transform=ax.transAxes, fontsize=5, va='bottom', ha='right', bbox=INFO_BOX)
    return m


def plot_state_accuracy(ax, true_vals, pred_vals, target):
    """Plot accuracy per true state with a dashed line at overall accuracy."""
    t, p = true_vals.astype(int), pred_vals.astype(int)
    states = sorted(set(t))
    overall = float(np.mean(t == p))
    ax.plot(states, [np.mean(p[t == s] == s) for s in states], 'D', ms=3,
            color=TARGET_COLORS[target], markeredgecolor='#333333', markeredgewidth=0.4)
    ax.axhline(overall, color='k', ls='--', lw=0.7)
    cfg = AXIS_CFG[target]
    ax.set_xlim(cfg['lims'])
    ax.set_xticks(cfg['ticks'])
    ax.set_ylim(0, 1)
    ax.set_xlabel('State')
    ax.set_ylabel('Accuracy')
    ax.text(0.95, 0.05, f'Overall acc. = {overall:.3f}', transform=ax.transAxes,
            fontsize=5, va='bottom', ha='right', bbox=INFO_BOX)
    return overall


def plot_score_vs_uncertainty(ax, result_dir, target):
    """Place one marker per pipeline at (conformal uncertainty, score); return the plotted values."""
    values = {}
    for pipeline in PIPELINES:
        pdir = os.path.join(result_dir, pipeline)
        score = compute_score(target, *load_predictions(pdir, target))
        width = load_cp_uncertainty(pdir, target)
        if score is not None and width is not None:
            values[pipeline] = (width, score)
    # The higher-scoring pipeline is labelled above its marker, the other below,
    # so the labels stay apart when the two markers nearly coincide.
    top = max(values, key=lambda p: values[p][1]) if values else None
    for pipeline, (width, score) in values.items():
        color, marker = PIPELINE_STYLE[pipeline]
        ax.scatter(width, score, color=color, marker=marker, s=22,
                   edgecolors='white', linewidths=0.4, zorder=3)
        ax.annotate(PIPELINE_LABELS[pipeline], (width, score), textcoords='offset points',
                    xytext=(4, 4) if pipeline == top else (4, -4),
                    va='bottom' if pipeline == top else 'top', fontsize=5, color=color)
    ax.set_xlim(CP_XLIMS[target])
    ax.set_ylim(SCORE_YLIM)
    ax.set_yticks(SCORE_YTICKS)
    ax.set_xlabel('Mean prediction-set size' if is_classification(target) else 'Mean interval width')
    ax.set_ylabel(score_label(target))
    return values


def plot_score_lines(ax, levels, scores):
    """
    Draw one score-by-level line per pipeline with value labels.

    `scores` maps pipeline -> score per level (None where missing).
    """
    x = np.arange(len(levels))
    for pipeline in PIPELINES:
        ys = scores[pipeline]
        if all(v is None for v in ys):
            continue
        color, marker = PIPELINE_STYLE[pipeline]
        ax.plot(x, [np.nan if v is None else v for v in ys], marker=marker, color=color,
                lw=1.0, markersize=4, markeredgecolor='white', markeredgewidth=0.4, zorder=3)
        other = scores['CBLV-CNN' if pipeline == 'stephy' else 'stephy']
        for xi, v, o in zip(x, ys, other):
            if v is None:
                continue
            above = o is None or v > o or (v == o and pipeline == 'stephy')
            dy, va = LABEL_ABOVE if above else LABEL_BELOW
            ax.annotate(f'{v:.2f}', (xi, v), textcoords='offset points',
                        xytext=(0, dy), ha='center', va=va, fontsize=5, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlim(-0.35, len(levels) - 0.65)
    ax.set_ylim(SCORE_YLIM)
    ax.set_yticks(SCORE_YTICKS)


def draw_legend_cell(ax):
    """Turn an unscored cell into the pipeline legend plus the note on why it is empty."""
    ax.set_box_aspect(1)
    ax.axis('off')
    handles = [Line2D([0], [0], marker=PIPELINE_STYLE[p][1], color=PIPELINE_STYLE[p][0], lw=1.0,
                      label=PIPELINE_LABELS[p], markerfacecolor=PIPELINE_STYLE[p][0],
                      markeredgecolor='white', markeredgewidth=0.4, markersize=4)
               for p in PIPELINES]
    ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.62),
              frameon=False, handlelength=2.0)
    ax.text(0.5, 0.30, UNDEFINED_NOTE, transform=ax.transAxes, fontsize=5.5,
            color='0.45', ha='center', va='center', linespacing=1.4)


def print_score_table(row_label, levels, scores):
    """Print one test's scores: a row per target and pipeline, a column per level."""
    print(f'\n{row_label.replace(chr(10), " ")}')
    print(f'  {"target":<8} {"pipeline":<9}' + ''.join(f'{lvl:>8}' for lvl in levels))
    for target in scores:
        for pipeline in PIPELINES:
            cells = ''.join('     n/a' if v is None else f'{v:8.3f}'
                            for v in scores[target][pipeline])
            print(f'  {target:<8} {PIPELINE_LABELS[pipeline]:<9}{cells}')


# -- Main --------------------------------------------------------------------

def main():
    """Build Figure 2 and write fig2.pdf, fig2.png and fig2.out next to this script."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result_dir', default=under(
        'models', 'simulation_benchmark', '100k_diverse_population_result'),
        help='100k result root holding {pipeline}/{target}/test_predictions.csv and cp_metrics.json')
    parser.add_argument('--misspec_root', default=under(
        'models', 'simulation_benchmark', 'misspecification'),
        help='Directory holding the 5k_diverse_population_*_result test sets')
    args = parser.parse_args()

    stem = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig2')
    log = open(stem + '.out', 'w')
    sys.stdout = _Tee(sys.__stdout__, log)
    print(f'result_dir   = {args.result_dir}')
    print(f'misspec_root = {args.misspec_root}')

    n_rows = 2 + len(TESTS)
    fig = plt.figure(figsize=(7.45, 11.0))
    gs = gridspec.GridSpec(n_rows, len(TARGETS), figure=fig, hspace=0.62, wspace=0.42)
    letters = iter('abcdefghijklmnopqrstuvwxyz')
    last_col_axes = {}   # row -> axes in the last column, for the group brackets

    def panel(row, col, row_label=None):
        """Create a square, gridded panel with its letter, and the row label in column 0."""
        ax = fig.add_subplot(gs[row, col])
        last_col_axes[row] = ax
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.26, 1.06, next(letters), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')
        if col == 0 and row_label:
            ax.text(-0.62, 0.5, row_label, transform=ax.transAxes, fontsize=6,
                    fontweight='bold', rotation=90, va='center', ha='center')
        return ax

    # Row 1: STEPHY on the in-distribution test split.
    print('\nRow 1 — STEPHY, 100k diverse-population test split')
    stephy_dir = os.path.join(args.result_dir, 'stephy')
    for col, target in enumerate(TARGETS):
        ax = panel(0, col, 'Scatter plot')
        ax.set_title(TARGET_LABELS[target], fontweight='bold', pad=22)
        true_vals, pred_vals = load_predictions(stephy_dir, target)
        if true_vals is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            print(f'  {target:<8} missing')
        elif is_classification(target):
            acc = plot_state_accuracy(ax, true_vals, pred_vals, target)
            print(f'  {target:<8} accuracy = {acc:.4f}')
        else:
            m = plot_true_vs_pred(ax, true_vals, pred_vals, target)
            print(f'  {target:<8} R2 = {m["R2"]:.4f}  r = {m["r"]:.4f}  MSE = {m["MSE"]:.4g}')

    # Row 2: both pipelines' score against conformal uncertainty.
    print('\nRow 2 — score vs conformal uncertainty (width or set size, score)')
    for col, target in enumerate(TARGETS):
        ax = panel(1, col, 'Performance space')
        for pipeline, (width, score) in plot_score_vs_uncertainty(ax, args.result_dir, target).items():
            print(f'  {target:<8} {PIPELINE_LABELS[pipeline]:<9} {width:8.4f} {score:8.4f}')

    # Rows 3-6: misspecification tests.
    for row, (row_label, xlabel, levels, dir_tpl, targets) in enumerate(TESTS, start=2):
        scores = load_test_scores(args.misspec_root, levels, dir_tpl, targets)
        print_score_table(row_label, levels, scores)
        for col, target in enumerate(TARGETS):
            if target not in targets:
                last_col_axes[row] = ax = fig.add_subplot(gs[row, col])
                draw_legend_cell(ax)
                continue
            ax = panel(row, col, row_label)
            plot_score_lines(ax, levels, scores[target])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(score_label(target))

    # Group brackets right of the last column: rows 1-2 and rows 3-6. Draw
    # first so the square box aspects are applied and positions are final.
    fig.canvas.draw()
    bracket_x = max(ax.get_position().x1 for ax in last_col_axes.values()) + 0.03
    for first, last, title in [(0, 1, 'Training performance'),
                               (2, n_rows - 1, 'Misspecification tests')]:
        top = last_col_axes[first].get_position().y1
        bottom = last_col_axes[last].get_position().y0
        fig.add_artist(Line2D([bracket_x - 0.008, bracket_x, bracket_x, bracket_x - 0.008],
                              [top, top, bottom, bottom], transform=fig.transFigure,
                              color='#333', lw=0.8))
        fig.text(bracket_x + 0.012, (top + bottom) / 2, title, rotation=270,
                 ha='left', va='center', fontsize=8, fontweight='bold')

    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', bbox_inches='tight', dpi=600)
    print(f'\nSaved: {stem}.pdf')
    print(f'Saved: {stem}.png')
    print(f'Saved: {stem}.out')
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == '__main__':
    main()
