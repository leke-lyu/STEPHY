#!/usr/bin/env python3
"""
Population-shift scatter (3 x 4) — per-pipeline diagnostics across X1/X2/X3.

Mirrors fig1 Row 1 (true-vs-pred scatter for regression, per-state accuracy
for classification), but extended to three rows — one per population scale —
so visual degradation under population shift is directly comparable. One
figure is produced per pipeline (STEPHY and CBLV-CNN).

  Rows:    X1 (1x), X2 (2x), X3 (3x) population scale
  Cols:    reg_r0, reg_rr, reg_sss, cls_as
  Source:  {gen_root}/5k_diverse_population_{X}_result/{pipeline}/{label}/test_predictions.csv
  Output:  population_shift_scatter_{stephy,cblv-cnn}.{pdf,png}

Usage:
    python3 population_shift_scatter.py
    python3 population_shift_scatter.py --gen_root /path/to/5k_parent_dir
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# -- Shared style (consistent with fig1.py / fig2.py) -----------------------

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

# -- Targets (mirrors fig1.py) ----------------------------------------------

TARGETS = ['reg_r0', 'reg_rr', 'reg_sss', 'cls_as']
TARGET_LABELS = {
    'reg_r0':  'Reproduction Number (Reg.)',
    'reg_rr':  'Recovery Rate (Reg.)',
    'reg_sss': 'Source-Sink Score (Reg.)',
    'cls_as':  'Ancestral State (Cls.)',
}
TARGET_COLORS = {
    'reg_r0':  '#4878A8',
    'reg_rr':  '#6AAB6A',
    'reg_sss': '#E8963E',
    'cls_as':  '#C25B5B',
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

AXIS_CFG = {
    'reg_r0':  {'lims': (2, 8),       'ticks': [2, 3, 4, 5, 6, 7, 8]},
    'reg_rr':  {'lims': (0.05, 0.25), 'ticks': [0.05, 0.10, 0.15, 0.20, 0.25]},
    'reg_sss': {'lims': (-1, 1),      'ticks': [-1, -0.5, 0, 0.5, 1]},
    'cls_as':  {'lims': (-0.5, 11.5), 'ticks': list(range(12)),
                'ylims': (0, 1),      'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
}

INFO_BOX = dict(boxstyle='round', facecolor='white', alpha=0.8)

SCALES = ['X1', 'X2', 'X3']
SCALE_LABELS = {'X1': '1x population',
                'X2': '2x population',
                'X3': '3x population'}
SCALE_DIR_TPL = '5k_diverse_population_{scale}_result'

# Pipelines to render (subdir name -> output-filename suffix)
PIPELINES = ['stephy', 'CBLV-CNN']


# -- Helpers (mirrors fig1.py) ----------------------------------------------

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


def plot_regression(ax, true_vals, pred_vals, target):
    """Scatter plot of true vs predicted with R2/r/MSE annotation."""
    cfg = AXIS_CFG[target]
    ax.scatter(true_vals, pred_vals, alpha=0.1, s=2, color=TARGET_COLORS[target])
    ax.plot(cfg['lims'], cfg['lims'], 'k--', lw=0.5)
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
            markeredgecolor='#333333', markeredgewidth=0.4)
    ax.axhline(y=overall_acc, color='k', ls='--', lw=0.5)

    cfg = AXIS_CFG[target]
    ax.set_xlim(cfg['lims']);  ax.set_xticks(cfg['ticks'])
    ax.set_ylim(cfg['ylims']);  ax.set_yticks(cfg['yticks'])
    ax.set_xlabel('State', fontsize=6)
    ax.set_ylabel('Accuracy', fontsize=6)

    ax.text(0.95, 0.05, f"Overall Acc = {overall_acc:.4f}",
            transform=ax.transAxes, fontsize=5, va='bottom', ha='right',
            bbox=INFO_BOX)


# -- Main --------------------------------------------------------------------

def build_figure(gen_root, pipeline):
    """Build the 3 x 4 population-shift scatter grid for one pipeline."""
    n_cols = len(TARGETS)
    n_rows = len(SCALES)
    fig = plt.figure(figsize=(2.1 * n_cols, 2.1 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.35)

    panel_idx = 0
    for row, scale in enumerate(SCALES):
        pipeline_dir = os.path.join(gen_root, SCALE_DIR_TPL.format(scale=scale),
                                    pipeline)
        for col, target in enumerate(TARGETS):
            ax = fig.add_subplot(gs[row, col])
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

            if row == 0:
                ax.set_title(TARGET_LABELS[target], fontsize=7,
                             fontweight='bold')
            if col == 0:
                ax.text(-0.32, 0.5, SCALE_LABELS[scale],
                        transform=ax.transAxes, fontsize=7, fontweight='bold',
                        rotation=90, va='center', ha='center')

            ax.text(-0.15, 1.02, chr(ord('a') + panel_idx),
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    va='bottom', ha='left')
            panel_idx += 1

    return fig


def main():
    """
    Build the population-shift scatter grid (3 x 4) for each pipeline.

    Each row is a population scale (X1/X2/X3), each column is a task
    (reg_r0, reg_rr, reg_sss, cls_as). Regression panels show true-vs-pred
    scatter; the classification panel shows per-state accuracy. One figure
    per pipeline is written next to this script as both PDF and PNG:
    population_shift_scatter_{pipeline}.{pdf,png}.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_root', type=str,
                        default='/Users/lukelyu/Desktop/trained_model/simu',
                        help='Parent dir holding 5k_diverse_population_{X1,X2,X3}_result/')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for pipeline in PIPELINES:
        fig = build_figure(args.gen_root, pipeline)
        suffix = pipeline.lower()
        out_pdf = os.path.join(out_dir, f'population_shift_scatter_{suffix}.pdf')
        out_png = os.path.join(out_dir, f'population_shift_scatter_{suffix}.png')
        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, bbox_inches='tight', dpi=600)
        print(f'Saved: {out_pdf}')
        print(f'Saved: {out_png}')
        plt.close(fig)


if __name__ == '__main__':
    main()
