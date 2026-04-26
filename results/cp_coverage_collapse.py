#!/usr/bin/env python3
"""
CP coverage collapse — what conformal prediction promises vs delivers under
population-scale distribution shift.

Two rows × four tasks (reg_r0, reg_rr, reg_sss, cls_as):

  Row 1 (a-d): Performance-space view. x = CP interval width (regression) or
               mean prediction-set size (classification); y = R2 (regression)
               or accuracy (classification). 6 points per panel
               (2 pipelines × 3 population scales). Shows that the CP
               *uncertainty* stays roughly stable across scales while the
               actual *quality* (R2 / accuracy) drops under shift.

  Row 2 (e-h): Empirical CP coverage vs population scale. The X1 coverage
               is shown as a dashed reference (calibration is in-distribution
               there). Reveals the failure that Row 1 hides — interval widths
               look fine, but coverage collapses on X2/X3, especially for the
               more confident pipeline.

Usage:
    python3 cp_coverage_collapse.py
    python3 cp_coverage_collapse.py --gen_root /path/to/5k_parent_dir
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

# -- Shared style (consistent with fig1.py / fig2.py) -----------------------

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

PIPELINES = ['CBLV-CNN', 'stephy']
PIPELINE_STYLE = {
    'CBLV-CNN': ('#5B7E9E', 'o'),
    'stephy':   ('#7A6FAC', 'D'),
}

TARGETS = ['reg_r0', 'reg_rr', 'reg_sss', 'cls_as']
TARGET_LABELS = {
    'reg_r0':  'Reproduction Number (Reg.)',
    'reg_rr':  'Recovery Rate (Reg.)',
    'reg_sss': 'Source-Sink Score (Reg.)',
    'cls_as':  'Ancestral State (Cls.)',
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

SCALES = ['X1', 'X2', 'X3']
SCALE_DIR_TPL = '5k_diverse_population_{scale}_result'

ROW1_XLIMS = {
    'reg_r0':  (0, 6),
    'reg_rr':  (0, 0.2),
    'reg_sss': (0, 2),
    'cls_as':  (0, 12),
}


def load_predictions(target, pipeline_dir):
    """Return (true, pred) arrays from test_predictions.csv, or (None, None)."""
    path = os.path.join(pipeline_dir, target, 'test_predictions.csv')
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    true_col, pred_col = PRED_COLS[target]
    return df[true_col].values, df[pred_col].values


def compute_score(target, true_vals, pred_vals):
    """R2 (regression) or accuracy (classification); None if no data."""
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


def load_cp_metrics(target, pipeline_dir):
    """Return cp_metrics dict from cp_metrics.json, or None."""
    path = os.path.join(pipeline_dir, target, 'cp_metrics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_gen_points(gen_root):
    """For each (scale, pipeline, target): (scale, pipeline, x, y, coverage).

    x = CP interval width (regression) / mean set size (classification).
    y = R2 / accuracy on that scale's test set.
    coverage = empirical CP coverage on the same set.
    """
    out = {t: [] for t in TARGETS}
    for scale in SCALES:
        scale_root = os.path.join(gen_root, SCALE_DIR_TPL.format(scale=scale))
        for pipeline in PIPELINES:
            pdir = os.path.join(scale_root, pipeline)
            for target in TARGETS:
                score = compute_score(target, *load_predictions(target, pdir))
                cp = load_cp_metrics(target, pdir)
                if score is None or cp is None:
                    continue
                x_val = (cp['mean_set_size'] if IS_CLASSIFICATION[target]
                         else cp['mean_interval_width'])
                cov = cp.get('empirical_coverage')
                out[target].append((scale, pipeline, x_val, score, cov))
    return out


def make_pipeline_handles():
    """Build legend handles for the two pipelines."""
    return [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1],
               color=PIPELINE_STYLE[p][0], lw=2, label=p,
               markerfacecolor=PIPELINE_STYLE[p][0],
               markeredgecolor='white', markeredgewidth=0.8, markersize=8)
        for p in PIPELINES
    ]


def main():
    """
    Build the CP-coverage-collapse figure across X1/X2/X3 population scales.

    Row 1 (a-d): performance-space scatter — CP interval width / set size on
    x-axis, R2 / accuracy on y-axis, 6 points per panel (2 pipelines × 3
    scales).
    Row 2 (e-h): empirical CP coverage vs scale, with X1's coverage as a
    dashed reference line.

    Output: cp_coverage_collapse.pdf alongside this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_root', type=str,
                        default='/Users/lukelyu/Desktop/data/simu',
                        help='Parent dir holding 5k_diverse_population_{X1,X2,X3}_result/')
    args = parser.parse_args()

    gen_points = load_gen_points(args.gen_root)
    pipeline_handles = make_pipeline_handles()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    # -- Row 1 (a-d): performance-space view --------------------------------
    for col, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, col])
        is_cls = IS_CLASSIFICATION[target]

        for scale, pipeline, x_val, y_val, _cov in gen_points[target]:
            color, marker = PIPELINE_STYLE[pipeline]
            ax.scatter(x_val, y_val, color=color, marker=marker,
                       s=120, edgecolors='white', linewidths=0.5, zorder=3)
            ax.annotate(scale, (x_val, y_val), textcoords='offset points',
                        xytext=(6, -4), fontsize=8, color=color)

        ax.set_xlim(ROW1_XLIMS[target])
        ax.set_ylim(0.5, 1.0)
        ax.set_box_aspect(1)
        ax.set_xlabel('Prediction uncertainty (set size)' if is_cls
                      else 'Prediction uncertainty (interval width)',
                      fontsize=10)
        ax.set_ylabel('Accuracy' if is_cls else r'R$^2$', fontsize=12)
        ax.set_title(TARGET_LABELS[target], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.15, 1.02, chr(ord('a') + col), transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')

    # -- Row 2 (e-h): empirical CP coverage across population scales --------
    scale_x = np.arange(len(SCALES))
    for col, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[1, col])

        by_pipeline = {p: [None] * len(SCALES) for p in PIPELINES}
        for scale, pipeline, _x, _y, cov in gen_points[target]:
            if cov is not None and pipeline in by_pipeline:
                by_pipeline[pipeline][SCALES.index(scale)] = cov

        nominal = None
        for pipeline in PIPELINES:
            color, marker = PIPELINE_STYLE[pipeline]
            ys = by_pipeline[pipeline]
            if all(v is None for v in ys):
                continue
            ax.plot(scale_x, ys, marker=marker, color=color, lw=2,
                    markersize=9, markeredgecolor='white', markeredgewidth=0.8,
                    zorder=3)
            for xi, v in zip(scale_x, ys):
                if v is None:
                    continue
                ax.annotate(f'{v:.2f}', (xi, v), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color=color)
            if nominal is None and ys[0] is not None:
                nominal = ys[0]

        if nominal is not None:
            ax.axhline(nominal, color='#888888', ls='--', lw=1, zorder=1,
                       label=f'X1 nominal ({nominal:.2f})')
            ax.legend(loc='lower left', fontsize=8, frameon=False)

        ax.set_xticks(scale_x)
        ax.set_xticklabels(SCALES)
        ax.set_xlim(-0.3, len(SCALES) - 0.7)
        ax.set_ylim(0.0, 1.05)
        ax.set_box_aspect(1)
        ax.set_xlabel('Population scale', fontsize=10)
        ax.set_ylabel('Empirical CP coverage', fontsize=12)
        ax.set_title(TARGET_LABELS[target], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.15, 1.02, chr(ord('e') + col), transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')

    fig.legend(handles=pipeline_handles, loc='lower center', frameon=False,
               ncol=len(PIPELINES), fontsize=11, bbox_to_anchor=(0.5, -0.01))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cp_coverage_collapse.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
