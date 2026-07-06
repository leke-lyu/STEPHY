#!/usr/bin/env python3
"""
Denmark performance — model diagnostics on the Denmark-calibrated dataset
(2-row, 4-column layout). Based on fig2.py, retargeted to the 5-region
Denmark result and its parameter ranges.

  Row 1 (a-d): STEPHY scatter (regression) and per-state accuracy (classification)
  Row 2 (e-h): R2 / accuracy vs conformal interval width / set size per pipeline

Usage:
    python3 denmark_performance.py
    python3 denmark_performance.py --base_dir /path/to/denmark_result
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

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

# -- Targets -----------------------------------------------------------------

TARGETS = ['reg_r0', 'reg_rr', 'reg_sss', 'cls_as']
TARGET_LABELS = {
    'reg_r0':  r'$R_0$ (Reg.)',
    'reg_rr':  r'$\gamma$ (Reg.)',
    'reg_sss': 'SSS (Reg.)',
    'cls_as':  'Index Location (Cls.)',
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

# Column names in test_predictions.csv
PRED_COLS = {
    'reg_r0':  ('true_reg_r0',  'pred_reg_r0'),
    'reg_rr':  ('true_reg_rr',  'pred_reg_rr'),
    'reg_sss': ('true_reg_sss', 'pred_reg_sss'),
    'cls_as':  ('true_ancestor', 'pred_ancestor'),
}

# Axis limits and ticks per target — Denmark parameter ranges:
#   R0 ~ Beta on [0.5, 4], gamma ~ U(0.07, 0.23), 5 regions (states 0-4).
AXIS_CFG = {
    'reg_r0':  {'lims': (0.5, 4),     'ticks': [1, 2, 3, 4]},
    'reg_rr':  {'lims': (0.07, 0.23), 'ticks': [0.10, 0.15, 0.20]},
    'reg_sss': {'lims': (-1, 1),      'ticks': [-1, -0.5, 0, 0.5, 1]},
    'cls_as':  {'lims': (-0.5, 4.5),  'ticks': list(range(5)),
                'ylims': (0, 1),      'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
}

# Row 2 x-axis (CP interval width / set size) per target.
ROW2_XLIMS = {
    'reg_r0':  (0, 1.5),
    'reg_rr':  (0, 0.2),
    'reg_sss': (0, 1.0),
    'cls_as':  (0, 5),
}
# Row 2 y-axis (R2 / accuracy) — floor below 0.5 so lower-R2 points fit.
ROW2_YLIM = (0.4, 1.0)

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


def compute_score(target, true_vals, pred_vals):
    """Return R2 (regression) or accuracy (classification), or None."""
    if true_vals is None:
        return None
    if IS_CLASSIFICATION[target]:
        return np.mean(true_vals.astype(int) == pred_vals.astype(int))
    m = regression_metrics(true_vals, pred_vals)
    return m['R2'] if m else None


def load_cp_metrics(target, pipeline_dir):
    """Return cp_metrics dict from cp_metrics.json, or None."""
    path = os.path.join(pipeline_dir, target, 'cp_metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# -- Row 1 plot functions ----------------------------------------------------

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
    Build the Denmark performance figure — model diagnostics across pipelines
    and targets on the 5-region Denmark-calibrated dataset.

    Row 1 shows STEPHY's per-target scatter (regression) or per-state accuracy
    (classification). Row 2 shows R2/accuracy vs. conformal interval width or
    set size for each pipeline. Output: denmark_performance.{pdf,png} alongside
    this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default='/Users/lukelyu/Desktop/trained_model/denmark/100k_result/cp')
    args = parser.parse_args()
    base_dir = args.base_dir

    n_targets = len(TARGETS)
    fig = plt.figure(figsize=(8.55, 4.76))
    gs = gridspec.GridSpec(2, n_targets, figure=fig,
                           height_ratios=[1, 1], hspace=0.05, wspace=0.35)

    # -- Row 1 (a-d): stephy scatter / classification -----------------------
    stephy_dir = os.path.join(base_dir, 'stephy')
    for col, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, col])
        true_vals, pred_vals = load_predictions(target, stephy_dir)

        if true_vals is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        elif IS_CLASSIFICATION[target]:
            plot_classification(ax, true_vals, pred_vals, target)
        else:
            plot_regression(ax, true_vals, pred_vals, target)

        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)
        ax.set_title(TARGET_LABELS[target], fontsize=7, fontweight='bold')
        ax.text(-0.18, 1.04, chr(ord('a') + col), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')

    # -- Row 2 (e-h): R2/accuracy vs interval_width/set_size ----------------
    for col, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[1, col])
        is_cls = IS_CLASSIFICATION[target]

        for pipeline in PIPELINES:
            color, marker = PIPELINE_STYLE[pipeline]
            pdir = os.path.join(base_dir, pipeline)
            score = compute_score(target, *load_predictions(target, pdir))
            cp = load_cp_metrics(target, pdir)
            if score is None or cp is None:
                continue
            x_val = cp['mean_set_size'] if is_cls else cp['mean_interval_width']
            ax.scatter(x_val, score, color=color, marker=marker,
                       s=30, edgecolors='white', linewidths=0.4, zorder=3)
            ax.annotate(PIPELINE_LABELS[pipeline], (x_val, score), textcoords='offset points',
                        xytext=(4, -3), fontsize=5, color=color)

        ax.set_xlim(ROW2_XLIMS[target])
        ax.set_ylim(ROW2_YLIM)
        ax.set_box_aspect(1)
        ax.set_xlabel('Prediction uncertainty (set size)' if is_cls
                      else 'Prediction uncertainty (interval width)',
                      fontsize=6)
        ax.set_ylabel('Accuracy' if is_cls else r'R$^2$', fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.18, 1.04, chr(ord('e') + col), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='left')

    # -- Shared legend ------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1], color='w', label=PIPELINE_LABELS[p],
               markerfacecolor=PIPELINE_STYLE[p][0], markeredgecolor='white',
               markeredgewidth=0.4, markersize=5)
        for p in PIPELINES
    ]
    fig.legend(handles=legend_handles, loc='lower center', frameon=False,
               ncol=len(PIPELINES), fontsize=6, bbox_to_anchor=(0.5, -0.02))

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
