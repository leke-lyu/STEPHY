#!/usr/bin/env python3
"""
Denmark performance — point-estimate model diagnostics on the Denmark-
calibrated dataset (single row, 4 columns). Based on fig2.py Row 1,
retargeted to the 5-region Denmark result and its parameter ranges.

Reads the `pe_old` point-estimate run, whose target directories predate the
reg_*/cls_* convention used by the simulation runs: they are named r0 / rr /
sss / as rather than reg_r0 / reg_rr / reg_sss / cls_as. The pipelines are
stephy, CBLV-CNN and CBLV-GAT. That run has no conformal artifacts, so there
is no coverage row.

  a  R_0   (Reg.)  — true vs predicted, with R2 / Pearson r / MSE
  b  gamma (Reg.)  — true vs predicted, with R2 / Pearson r / MSE
  c  SSS   (Reg.)  — true vs predicted, with R2 / Pearson r / MSE
  d  Index Location (Cls.) — per-state accuracy + overall-accuracy line

Usage:
    python3 denmark_performance.py
    python3 denmark_performance.py --pipeline CBLV-CNN
    python3 denmark_performance.py --base_dir /path/to/pe_old
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from _paths import under

# -- Shared style (consistent with fig2.py / fig3.py) -----------------------

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6,
    'axes.labelsize': 6,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# -- Targets (pe_old naming) -------------------------------------------------

TARGETS = ['r0', 'rr', 'sss', 'as']
TARGET_LABELS = {
    'r0':  r'$R_0$ (Reg.)',
    'rr':  r'$\gamma$ (Reg.)',
    'sss': 'SSS (Reg.)',
    'as':  'Index Location (Cls.)',
}
TARGET_COLORS = {
    'r0':  '#4878A8',
    'rr':  '#6AAB6A',
    'sss': '#E8963E',
    'as':  '#C25B5B',
}
IS_CLASSIFICATION = {'r0': False, 'rr': False, 'sss': False, 'as': True}

# Column names in test_predictions.csv (pe_old spells targets out in full)
PRED_COLS = {
    'r0':  ('true_R0',                'pred_R0'),
    'rr':  ('true_Recovery_Rate',     'pred_Recovery_Rate'),
    'sss': ('true_Source_Sink_Score', 'pred_Source_Sink_Score'),
    'as':  ('true_ancestor',          'pred_ancestor'),
}

# Axis limits and ticks per target — Denmark parameter ranges:
#   R0 ~ Beta on [0.5, 4], gamma ~ U(0.07, 0.23), 5 regions (states 0-4).
#   gamma's axis runs past its prior so over-shooting predictions stay visible.
AXIS_CFG = {
    'r0':  {'lims': (0.5, 4),     'ticks': [1, 2, 3, 4]},
    'rr':  {'lims': (0.05, 0.27), 'ticks': [0.10, 0.15, 0.20, 0.25]},
    'sss': {'lims': (-1, 1),      'ticks': [-1, -0.5, 0, 0.5, 1]},
    'as':  {'lims': (-0.5, 4.5),  'ticks': list(range(5)),
            'ylims': (0, 1),      'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
}

INFO_BOX = dict(boxstyle='round', facecolor='white', alpha=0.8)


# -- Helpers -----------------------------------------------------------------

def load_predictions(target, pipeline_dir):
    """Return (true, pred) arrays from test_predictions.csv, or (None, None)."""
    path = os.path.join(pipeline_dir, target, 'test_predictions.csv')
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    true_col, pred_col = PRED_COLS[target]
    return df[true_col].values, df[pred_col].values


def regression_metrics(true_vals, pred_vals):
    """Return {R2, Pearson_r, MSE} for valid pairs, or None if empty."""
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[mask], pred_vals[mask]
    if len(t) == 0:
        return None
    return {
        'R2':        r2_score(t, p),
        'Pearson_r': pearsonr(t, p)[0],
        'MSE':       mean_squared_error(t, p),
    }


# -- Plot functions ----------------------------------------------------------

def plot_regression(ax, true_vals, pred_vals, target):
    """Scatter plot of true vs predicted with R2/r/MSE annotation."""
    cfg = AXIS_CFG[target]
    ax.scatter(true_vals, pred_vals, alpha=0.1, s=5, color=TARGET_COLORS[target])
    ax.plot(cfg['lims'], cfg['lims'], 'k--', lw=1)
    ax.set_xlim(cfg['lims']);  ax.set_xticks(cfg['ticks'])
    ax.set_ylim(cfg['lims']);  ax.set_yticks(cfg['ticks'])
    ax.set_xlabel('True', fontsize=6)
    ax.set_ylabel('Predicted', fontsize=6)

    m = regression_metrics(true_vals, pred_vals)
    if m:
        info = f"R² = {m['R2']:.4f}\nr  = {m['Pearson_r']:.4f}\nMSE = {m['MSE']:.4f}"
        ax.text(0.95, 0.05, info, transform=ax.transAxes, fontsize=5,
                va='bottom', ha='right', bbox=INFO_BOX)


def plot_classification(ax, true_vals, pred_vals, target):
    """Per-state accuracy with overall accuracy reference line."""
    true_int, pred_int = true_vals.astype(int), pred_vals.astype(int)
    states = sorted(set(true_int))
    accuracies = [np.mean(pred_int[true_int == s] == s) for s in states]
    overall_acc = np.mean(true_int == pred_int)

    ax.plot(states, accuracies, 'D', color=TARGET_COLORS[target], ms=4,
            markeredgecolor='#333333', markeredgewidth=0.5)
    ax.axhline(y=overall_acc, color='k', ls='--', lw=0.7)

    cfg = AXIS_CFG[target]
    ax.set_xlim(cfg['lims']);  ax.set_xticks(cfg['ticks'])
    ax.set_ylim(cfg['ylims']);  ax.set_yticks(cfg['yticks'])
    ax.set_xlabel('State', fontsize=6)
    ax.set_ylabel('Accuracy', fontsize=6)

    ax.text(0.95, 0.05, f"Overall Acc = {overall_acc:.4f}",
            transform=ax.transAxes, fontsize=5, va='bottom', ha='right',
            bbox=INFO_BOX)


# -- Main --------------------------------------------------------------------

def main():
    """
    Build the Denmark performance figure — the selected pipeline's per-target
    point-estimate diagnostics on the 5-region Denmark dataset, one panel per
    target in a single row.

    Output: denmark_performance.{pdf,png} alongside this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default=under('models', 'denmark', '100k_result', 'pe_old'))
    parser.add_argument('--pipeline', type=str, default='stephy',
                        help='Pipeline subdir to read predictions from '
                             '(stephy, CBLV-CNN, CBLV-GAT)')
    args = parser.parse_args()

    pipeline_dir = os.path.join(args.base_dir, args.pipeline)
    if not os.path.isdir(pipeline_dir):
        sys.exit(f'error: no such pipeline directory: {pipeline_dir}')

    n_targets = len(TARGETS)
    fig, axes = plt.subplots(1, n_targets, figsize=(8.55, 2.55))
    fig.subplots_adjust(wspace=0.35)

    for col, target in enumerate(TARGETS):
        ax = axes[col]
        true_vals, pred_vals = load_predictions(target, pipeline_dir)

        if true_vals is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        elif IS_CLASSIFICATION[target]:
            plot_classification(ax, true_vals, pred_vals, target)
        else:
            plot_regression(ax, true_vals, pred_vals, target)

        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(TARGET_LABELS[target], fontsize=7, fontweight='bold')
        ax.text(-0.18, 1.04, chr(ord('a') + col), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_pdf = os.path.join(out_dir, 'denmark_performance.pdf')
    out_png = os.path.join(out_dir, 'denmark_performance.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    main()
