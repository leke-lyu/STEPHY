#!/usr/bin/env python3
"""
Figure 3 — Performance analysis (3-row layout).

  Row 1 (a):   Top-k identification accuracy (100k dataset).
  Row 2 (b):   R0 distribution by rank with gap violins (100k dataset).
  Row 3 (c-f): Per-task headline metric (R2 for regression, accuracy for
               classification) across X1/X2/X3 population scales. One line
               per pipeline. Shows the raw performance shift under
               population-scale change; the CP / coverage angle lives in
               cp_coverage_collapse.py.

Usage:
    python3 fig3.py
    python3 fig3.py --result_dir /path/to/100k_result --gen_root /path/to/5k_parent_dir
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

# -- Shared style (consistent with fig2.py) ---------------------------------

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

# -- Targets -----------------------------------------------------------------

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

# -- Layout constants --------------------------------------------------------

NUM_LOCATIONS = 12
VIOLIN_R0_COLOR = '#6baed6'
GAP_COLOR = '#fc8d59'

SCALES = ['X1', 'X2', 'X3']
SCALE_DIR_TPL = '5k_diverse_population_{scale}_result'


# -- Helpers -----------------------------------------------------------------

def style_violin(parts, color, alpha=0.7):
    """Apply face color and edge style to violin plot parts."""
    for body in parts['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor('none')
        body.set_alpha(alpha)
    for key in ('cmins', 'cmaxes', 'cbars', 'cmedians'):
        if key in parts:
            parts[key].set_edgecolor('#333333')
            parts[key].set_linewidth(0.8)


def draw_iqr(ax, data_list, positions, linewidth=1.5):
    """Overlay IQR bars on a violin plot."""
    for i, d in enumerate(data_list):
        q25, q75 = np.percentile(d, [25, 75])
        ax.vlines(positions[i], q25, q75, color='#333333',
                  linewidth=linewidth, zorder=4)


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


def load_scale_scores(gen_root):
    """For each (target, pipeline): list of scores aligned with SCALES.

    Returns: {target: {pipeline: [score_X1, score_X2, score_X3]}}.
    Missing (scale, pipeline) entries are stored as None.
    """
    out = {t: {p: [None] * len(SCALES) for p in PIPELINES} for t in TARGETS}
    for i, scale in enumerate(SCALES):
        scale_root = os.path.join(gen_root, SCALE_DIR_TPL.format(scale=scale))
        for pipeline in PIPELINES:
            pdir = os.path.join(scale_root, pipeline)
            for target in TARGETS:
                score = compute_score(target, *load_predictions(target, pdir))
                out[target][pipeline][i] = score
    return out


def make_pipeline_handles():
    """Build legend handles for the two pipelines."""
    return [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1],
               color=PIPELINE_STYLE[p][0], lw=1.2, label=p,
               markerfacecolor=PIPELINE_STYLE[p][0],
               markeredgecolor='white', markeredgewidth=0.4, markersize=5)
        for p in PIPELINES
    ]


# -- Main --------------------------------------------------------------------

def main():
    """
    Build Figure 3 — top-k identification, R0 rank structure, and per-task
    headline metric across population scales.

    Row 1 (a): top-k accuracy across pipelines on the 100k dataset.
    Row 2 (b): R0 distribution per rank with gap violins (stephy split —
    true values are identical across pipelines).
    Row 3 (c-f): R2 (regression) / accuracy (classification) vs population
    scale (X1/X2/X3), one line per pipeline.

    Output: fig3.pdf alongside this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str,
                        default='/Users/lukelyu/Desktop/trained_model/simu/100k_diverse_population_result',
                        help='100k result root containing {pipeline}/{label}/test_predictions.csv')
    parser.add_argument('--gen_root', type=str,
                        default='/Users/lukelyu/Desktop/trained_model/simu',
                        help='Parent dir holding 5k_diverse_population_{X1,X2,X3}_result/')
    args = parser.parse_args()
    result_dir = args.result_dir
    gen_root = args.gen_root

    # -- Load R0 predictions (100k) and reshape to (n_outbreaks, 12) --------
    data = {}
    for pipeline in PIPELINES:
        path = os.path.join(result_dir, pipeline, 'reg_r0',
                            'test_predictions.csv')
        df = pd.read_csv(path)
        true, pred = df['true_reg_r0'].values, df['pred_reg_r0'].values
        n = len(true) // NUM_LOCATIONS
        data[pipeline] = {
            'true': true[:n * NUM_LOCATIONS].reshape(n, NUM_LOCATIONS),
            'pred': pred[:n * NUM_LOCATIONS].reshape(n, NUM_LOCATIONS),
        }

    n_outbreaks = data['stephy']['true'].shape[0]
    sorted_desc = np.sort(data['stephy']['true'], axis=1)[:, ::-1]
    gap_data = [sorted_desc[:, i] - sorted_desc[:, i + 1]
                for i in range(NUM_LOCATIONS - 1)]

    # -- Compute top-k accuracies -------------------------------------------
    ks = [1, 2, 3, 4, 5]
    exact_accs = {p: [] for p in PIPELINES}
    miss1_accs = {p: [] for p in PIPELINES}
    for k in ks:
        for p in PIPELINES:
            tk = np.argsort(data[p]['true'], axis=1)[:, -k:]
            pk = np.argsort(data[p]['pred'], axis=1)[:, -k:]
            overlaps = np.array([len(set(tk[i]) & set(pk[i]))
                                 for i in range(n_outbreaks)])
            exact_accs[p].append((overlaps == k).mean())
            miss1_accs[p].append((overlaps >= max(k - 1, 1)).mean())

    # -- Load per-scale R2 / accuracy ---------------------------------------
    scale_scores = load_scale_scores(gen_root)

    # -- Figure layout -------------------------------------------------------
    fig = plt.figure(figsize=(8.57, 8.03))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1],
                           hspace=0.35, wspace=0.35)
    pipeline_handles = make_pipeline_handles()

    # -- Row 1 (a): top-k accuracy ------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    style_handles = [
        Line2D([0], [0], color='#555555', lw=1.2, linestyle='-',
               label='Allow 1 miss'),
        Line2D([0], [0], color='#555555', lw=1.2, linestyle='--', alpha=0.4,
               label='Exact match'),
    ]

    x = np.array(ks)
    for pipeline in PIPELINES:
        color, marker = PIPELINE_STYLE[pipeline]
        ax_a.plot(x, exact_accs[pipeline], marker=marker, markersize=5,
                  color=color, linestyle='--', lw=1.2, alpha=0.4, zorder=3,
                  markeredgecolor='white', markeredgewidth=0.4)
        ax_a.plot(x, miss1_accs[pipeline], marker=marker, markersize=5,
                  color=color, linestyle='-', lw=1.2, zorder=4,
                  markeredgecolor='white', markeredgewidth=0.4)
        for xi, val in zip(x, miss1_accs[pipeline]):
            ax_a.annotate(f'{val:.0%}', (xi, val), textcoords='offset points',
                          xytext=(0, 6), ha='center', fontsize=5, color=color)

    ax_a.set_xlabel('Top-k')
    ax_a.set_ylabel('Accuracy')
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels([f'Top-{k}' for k in ks])
    ax_a.set_ylim(-0.02, 1.08)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_axisbelow(True)
    ax_a.legend(handles=pipeline_handles + style_handles,
                loc='upper right', ncol=4, fontsize=6)
    ax_a.text(-0.04, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=9, fontweight='bold', va='bottom', ha='left')

    # -- Row 2 (b): R0 violins + gap violins --------------------------------
    ax_b = fig.add_subplot(gs[1, :])
    ranks = np.arange(1, NUM_LOCATIONS + 1)
    violin_data = [sorted_desc[:, i] for i in range(NUM_LOCATIONS)]

    vp = ax_b.violinplot(violin_data, positions=ranks,
                         showmedians=True, showextrema=False, widths=0.55)
    style_violin(vp, VIOLIN_R0_COLOR)
    draw_iqr(ax_b, violin_data, ranks)

    gap_x = [i + 1.5 for i in range(NUM_LOCATIONS - 1)]
    vp2 = ax_b.violinplot(gap_data, positions=gap_x,
                          showmedians=True, showextrema=False, widths=0.35)
    style_violin(vp2, GAP_COLOR, alpha=0.6)
    draw_iqr(ax_b, gap_data, gap_x, linewidth=1.2)

    for i, d in enumerate(gap_data):
        ax_b.text(gap_x[i], 1.05, f'{np.median(d):.2f}', ha='center',
                  va='bottom', fontsize=5, color=GAP_COLOR, fontweight='bold')

    ax_b.axhline(1.7, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)
    ax_b.legend(handles=[
        mpatches.Patch(color=VIOLIN_R0_COLOR, alpha=0.7, label='R0 per rank'),
        mpatches.Patch(color=GAP_COLOR, alpha=0.6,
                       label='ΔR0 between consecutive ranks'),
    ], loc='upper right')
    ax_b.set_xlabel('Location rank (by true R0)')
    ax_b.set_ylabel('R0  /  ΔR0')
    ax_b.set_xticks(ranks)
    ax_b.set_ylim(-0.5, 9)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_axisbelow(True)
    ax_b.text(-0.04, 1.02, 'b', transform=ax_b.transAxes,
              fontsize=9, fontweight='bold', va='bottom', ha='left')

    # -- Row 3 (c-f): per-task R2 / accuracy across population scales -------
    scale_x = np.arange(len(SCALES))
    for col, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[2, col])
        is_cls = IS_CLASSIFICATION[target]

        for pipeline in PIPELINES:
            color, marker = PIPELINE_STYLE[pipeline]
            ys = scale_scores[target][pipeline]
            if all(v is None for v in ys):
                continue
            ax.plot(scale_x, ys, marker=marker, color=color, lw=1.2,
                    markersize=5, markeredgecolor='white', markeredgewidth=0.4,
                    zorder=3)
            for xi, v in zip(scale_x, ys):
                if v is None:
                    continue
                ax.annotate(f'{v:.2f}', (xi, v), textcoords='offset points',
                            xytext=(0, 5), ha='center', fontsize=5, color=color)

        ax.set_xticks(scale_x)
        ax.set_xticklabels(SCALES)
        ax.set_xlim(-0.3, len(SCALES) - 0.7)
        ax.set_ylim(0.5, 1.0)
        ax.set_box_aspect(1)
        ax.set_xlabel('Population scale', fontsize=6)
        ax.set_ylabel('Accuracy' if is_cls else r'R$^2$', fontsize=6)
        ax.set_title(TARGET_LABELS[target], fontsize=7, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.18, 1.04, chr(ord('c') + col), transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='bottom', ha='left')

    fig.legend(handles=pipeline_handles, loc='lower center', frameon=False,
               ncol=len(PIPELINES), fontsize=6, bbox_to_anchor=(0.5, 0.04))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig3.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
