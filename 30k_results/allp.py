#!/usr/bin/env python3
"""
Compare 3 pipelines across 4 prediction targets (2-row figure).

Row 1: R² (regression) / Accuracy (Ancestral State)
Row 2: Best epoch (min val_loss for all targets)

Shared x-axis, independent y-axes.
Outputs comparison_all_pipelines.pdf into the base results directory.

Usage:
    python3 allp.py
    python3 allp.py --base_dir /path/to/results
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

# ── Configuration ────────────────────────────────────────────────────────────

PIPELINES = ['CBLV-CNN2', 'CBLV-GAT2', 'stephy2']

TARGETS = ['r0', 'rr', 'sss', 'as']

TARGET_LABELS = {
    'r0':  'R0',
    'rr':  'Recovery Rate',
    'sss': 'Source-Sink Score',
    'as':  'Ancestral State',
}

PRED_COLS = {
    'r0':  ('true_R0',               'pred_R0'),
    'rr':  ('true_Recovery_Rate',    'pred_Recovery_Rate'),
    'sss': ('true_Source_Sink_Score', 'pred_Source_Sink_Score'),
    'as':  ('true_ancestor',         'pred_ancestor'),
}

PIPELINE_STYLE = {
    'CBLV-CNN2': ('#5B7E9E', 'o'),  # slate
    'CBLV-GAT2': ('#D4845A', 's'),  # terracotta
    'stephy2':   ('#7A6FAC', 'D'),  # lavender
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_predictions(target, pipeline_dir):
    """Load test_predictions.csv, return (true, pred) arrays or (None, None)."""
    path = os.path.join(pipeline_dir, target, 'test_predictions.csv')
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    true_col, pred_col = PRED_COLS[target]
    return df[true_col].values, df[pred_col].values


def compute_score(target, true_vals, pred_vals):
    """R² for regression targets, accuracy for Ancestral State."""
    if true_vals is None:
        return None
    if target == 'as':
        return np.mean(true_vals.astype(int) == pred_vals.astype(int))
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[mask], pred_vals[mask]
    return r2_score(t, p) if len(t) > 0 else None


def get_best_epoch(target, pipeline_dir):
    """1-indexed epoch with minimum val_loss."""
    path = os.path.join(pipeline_dir, target, 'training_history.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return int(np.argmin(df['val_loss'].values)) + 1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Compare pipelines across targets')
    parser.add_argument('--base_dir', type=str,
                        default='/Users/lukelyu/Desktop/epidata/results')
    args = parser.parse_args()

    # Collect scores and best epochs
    scores, best_epochs = {}, {}
    for pipeline in PIPELINES:
        pipeline_dir = os.path.join(args.base_dir, pipeline)
        p_scores, p_epochs = [], []
        for target in TARGETS:
            true_vals, pred_vals = load_predictions(target, pipeline_dir)
            s = compute_score(target, true_vals, pred_vals)
            e = get_best_epoch(target, pipeline_dir)
            p_scores.append(s)
            p_epochs.append(e)

            metric = "Acc" if target == 'as' else "R²"
            print(f"  {pipeline:12s}  {TARGET_LABELS[target]:20s}  {metric} = {s:.4f}  best_epoch = {e}"
                  if s and e else f"  {pipeline:12s}  {TARGET_LABELS[target]:20s}  No data")
        scores[pipeline] = p_scores
        best_epochs[pipeline] = p_epochs

    # ── Plot ─────────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    x_pos = np.arange(len(TARGETS))
    offsets = np.linspace(-0.15, 0.15, len(PIPELINES))

    for i, pipeline in enumerate(PIPELINES):
        color, marker = PIPELINE_STYLE[pipeline]
        kw = dict(color=color, marker=marker, s=120,
                  edgecolors='white', linewidths=0.5, zorder=3)
        ax1.scatter(x_pos + offsets[i], scores[pipeline], **kw)
        ax2.scatter(x_pos + offsets[i], best_epochs[pipeline], **kw)

    # Row 1: Performance
    ax1.set_ylabel(r'R$^2$ / Accuracy', fontsize=14)
    ax1.set_title('Performance', fontsize=16, fontweight='bold')

    # Row 2: Convergence
    ax2.set_ylabel('Best Epoch', fontsize=14)
    ax2.set_title('Convergence', fontsize=16, fontweight='bold')

    # Shared x-axis
    ax2.set_xlim(-0.5, len(TARGETS) - 0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], fontsize=12)

    # Legend: single row above figure, no box
    legend_handles = [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1], color='w', label=p,
               markerfacecolor=PIPELINE_STYLE[p][0], markeredgecolor='white',
               markeredgewidth=0.5, markersize=9)
        for p in PIPELINES
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(PIPELINES),
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.04))

    # Shared styling
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)

    plt.tight_layout()

    out_path = os.path.join(args.base_dir, 'comparison_all_pipelines.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
