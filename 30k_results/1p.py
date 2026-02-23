#!/usr/bin/env python3
"""
Scatter plots for a single ML pipeline (2x2 grid).

Top row:    R0, Recovery Rate          (regression: true vs predicted)
Bottom row: Source-Sink Score, Ancestral State (regression + classification)

Outputs scatter_<pipeline>.pdf into the pipeline directory.

Usage:
    python3 1p.py <pipeline_dir>
    python3 1p.py /Users/lukelyu/Desktop/epidata/results/stephy2
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# ── Configuration ────────────────────────────────────────────────────────────

TARGETS = ['r0', 'rr', 'sss', 'as']

TARGET_LABELS = {
    'r0':  'R0',
    'rr':  'Recovery Rate',
    'sss': 'Source-Sink Score',
    'as':  'Ancestral State',
}

TARGET_COLORS = {
    'r0':  '#4878A8',  # steel blue
    'rr':  '#6AAB6A',  # sage green
    'sss': '#E8963E',  # warm amber
    'as':  '#C25B5B',  # dusty rose
}

PRED_COLS = {
    'r0':  ('true_R0',               'pred_R0'),
    'rr':  ('true_Recovery_Rate',    'pred_Recovery_Rate'),
    'sss': ('true_Source_Sink_Score', 'pred_Source_Sink_Score'),
    'as':  ('true_ancestor',         'pred_ancestor'),
}

AXIS_CFG = {
    'r0':  {'lims': (2, 8),       'ticks': [2, 3, 4, 5, 6, 7, 8]},
    'rr':  {'lims': (0.01, 0.05), 'ticks': [0.01, 0.02, 0.03, 0.04, 0.05]},
    'sss': {'lims': (-1, 1),      'ticks': [-1, -0.5, 0, 0.5, 1]},
    'as':  {'lims': (0, 1),       'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
}

INFO_BOX = dict(boxstyle='round', facecolor='white', alpha=0.8)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_predictions(target, base_dir):
    """Load test_predictions.csv, return (true, pred) arrays or (None, None)."""
    path = os.path.join(base_dir, target, 'test_predictions.csv')
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    true_col, pred_col = PRED_COLS[target]
    return df[true_col].values, df[pred_col].values


def compute_regression_metrics(true_vals, pred_vals):
    """Return dict with R2, Pearson_r, MSE (NaN-safe), or None."""
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[mask], pred_vals[mask]
    if len(t) == 0:
        return None
    return {
        'R2':        r2_score(t, p),
        'Pearson_r': pearsonr(t, p)[0],
        'MSE':       mean_squared_error(t, p),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_regression(ax, true_vals, pred_vals, target):
    """Scatter plot with diagonal reference line and metric annotations."""
    cfg = AXIS_CFG[target]
    ax.scatter(true_vals, pred_vals, alpha=0.1, s=5, color=TARGET_COLORS[target])
    ax.plot(cfg['lims'], cfg['lims'], 'k--', lw=1)
    ax.set_xlim(cfg['lims'])
    ax.set_xticks(cfg['ticks'])
    ax.set_xlabel('True', fontsize=10)
    ax.set_ylabel('Predicted', fontsize=10)

    m = compute_regression_metrics(true_vals, pred_vals)
    if m:
        info = f"R² = {m['R2']:.4f}\nr  = {m['Pearson_r']:.4f}\nMSE = {m['MSE']:.4f}"
        ax.text(0.95, 0.05, info, transform=ax.transAxes, fontsize=9,
                va='bottom', ha='right', bbox=INFO_BOX)


def plot_classification(ax, true_vals, pred_vals, target):
    """Per-state accuracy diamond plot with overall accuracy reference line."""
    true_int, pred_int = true_vals.astype(int), pred_vals.astype(int)
    states = sorted(set(true_int))
    accuracies = [np.mean(pred_int[true_int == s] == s) for s in states]
    overall_acc = np.mean(true_int == pred_int)

    ax.plot(states, accuracies, 'D', color=TARGET_COLORS[target], ms=8,
            markeredgecolor='#8B3A3A', markeredgewidth=0.8)
    ax.axhline(y=overall_acc, color='k', ls='--', lw=1)
    ax.set_xlim(-0.5, 9.5)
    ax.set_xticks(range(10))
    ax.set_xlabel('State', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)

    ax.text(0.95, 0.05, f"Overall Acc = {overall_acc:.4f}", transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right', bbox=INFO_BOX)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <pipeline_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    pipeline_name = os.path.basename(base_dir)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    for ax, target in zip(axes.flat, TARGETS):
        true_vals, pred_vals = load_predictions(target, base_dir)

        if true_vals is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        elif target == 'as':
            plot_classification(ax, true_vals, pred_vals, target)
        else:
            plot_regression(ax, true_vals, pred_vals, target)

        # Shared styling
        cfg = AXIS_CFG[target]
        ax.set_ylim(cfg['lims'])
        ax.set_yticks(cfg['ticks'])
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)
        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight='bold')

    fig.align_ylabels(axes[:, 0])

    out_path = os.path.join(base_dir, f'scatter_{pipeline_name}.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
