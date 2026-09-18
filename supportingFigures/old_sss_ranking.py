#!/usr/bin/env python3
"""
SSS ranking — source/sink identification analysis (2-row layout).

How well each pipeline ranks locations by Source-Sink Score (SSS) on the 100k
test split.

  Row 1 (a): Top-k source identification accuracy (100k dataset). How often
             each pipeline recovers the true top-k highest-SSS locations
             (the strongest sources).
  Row 2 (b): SSS distribution by rank with gap violins (100k dataset). Per
             outbreak, locations are sorted by true SSS (descending);
             violins show the SSS spread at each rank and the gap (ΔSSS)
             between consecutive ranks.

Usage:
    python3 sss_ranking.py
    python3 sss_ranking.py --result_dir /path/to/100k_result
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from _paths import under

# -- Shared style (consistent with fig2.py) ---------------------------------

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

# -- Layout constants --------------------------------------------------------

NUM_LOCATIONS = 12
VIOLIN_SSS_COLOR = '#6baed6'
GAP_COLOR = '#fc8d59'

# SSS lives in [-1, 1] on the left axis; ΔSSS gaps (~0.04-0.08) get their own
# secondary right axis so they aren't crushed against zero.
ROW2_YLIM = (-1.1, 1.1)
ROW2_GAP_YLIM = (0, 2.0)


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


def make_pipeline_handles():
    """Build legend handles for the two pipelines."""
    return [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1],
               color=PIPELINE_STYLE[p][0], lw=1.2, label=PIPELINE_LABELS[p],
               markerfacecolor=PIPELINE_STYLE[p][0],
               markeredgecolor='white', markeredgewidth=0.4, markersize=5)
        for p in PIPELINES
    ]


# -- Main --------------------------------------------------------------------

def main():
    """
    Build the SSS-ranking figure — top-k source identification and the SSS
    rank structure that governs its difficulty.

    Row 1 (a): top-k accuracy across pipelines on the 100k dataset.
    Row 2 (b): SSS distribution per rank with gap violins (stephy split —
    true values are identical across pipelines).

    Output: sss_ranking.{pdf,png} alongside this script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str,
                        default=under('models', 'simulation_benchmark', '100k_diverse_population_result'),
                        help='100k result root containing {pipeline}/reg_sss/test_predictions.csv')
    args = parser.parse_args()
    result_dir = args.result_dir

    # -- Load SSS predictions (100k) and reshape to (n_outbreaks, 12) -------
    data = {}
    for pipeline in PIPELINES:
        path = os.path.join(result_dir, pipeline, 'reg_sss',
                            'test_predictions.csv')
        df = pd.read_csv(path)
        true, pred = df['true_reg_sss'].values, df['pred_reg_sss'].values
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

    # -- Figure layout -------------------------------------------------------
    fig = plt.figure(figsize=(7.9, 5.05))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1],
                           hspace=0.30, wspace=0.35)
    pipeline_handles = make_pipeline_handles()

    # -- Row 1 (a): top-k accuracy ------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    style_handles = [
        Line2D([0], [0], color='#555555', lw=1.2, linestyle='-',
               label='Allow 1 miss'),
        Line2D([0], [0], color='#555555', lw=1.2, linestyle='--', alpha=0.4,
               label='Exact match'),
    ]

    # Offset each pipeline's labels in opposite directions so they never stack.
    label_offset = {'stephy': (0, 7, 'bottom'), 'CBLV-CNN': (0, -9, 'top')}

    x = np.array(ks)
    for pipeline in PIPELINES:
        color, marker = PIPELINE_STYLE[pipeline]
        ax_a.plot(x, exact_accs[pipeline], marker=marker, markersize=5,
                  color=color, linestyle='--', lw=1.2, alpha=0.4, zorder=3,
                  markeredgecolor='white', markeredgewidth=0.4)
        ax_a.plot(x, miss1_accs[pipeline], marker=marker, markersize=5,
                  color=color, linestyle='-', lw=1.2, zorder=4,
                  markeredgecolor='white', markeredgewidth=0.4)
        dx, dy, va = label_offset[pipeline]
        for xi, val in zip(x, miss1_accs[pipeline]):
            ax_a.annotate(f'{val:.0%}', (xi, val), textcoords='offset points',
                          xytext=(dx, dy), ha='center', va=va, fontsize=6,
                          color=color)

    ax_a.set_xlabel('Top-k')
    ax_a.set_ylabel('Accuracy')
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels([f'Top-{k}' for k in ks])
    ax_a.set_ylim(-0.02, 1.08)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_axisbelow(True)
    # Lower right is the only corner clear of both the solid lines (top)
    # and the dashed exact-match lines (0.2-0.35 on the right).
    ax_a.legend(handles=pipeline_handles + style_handles,
                loc='lower right', ncol=4, fontsize=7)
    ax_a.text(-0.04, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=14, fontweight='bold', va='bottom', ha='left')

    # -- Row 2 (b): SSS violins + gap violins -------------------------------
    ax_b = fig.add_subplot(gs[1, :])
    ranks = np.arange(1, NUM_LOCATIONS + 1)
    violin_data = [sorted_desc[:, i] for i in range(NUM_LOCATIONS)]

    vp = ax_b.violinplot(violin_data, positions=ranks,
                         showmedians=True, showextrema=False, widths=0.55)
    style_violin(vp, VIOLIN_SSS_COLOR)
    draw_iqr(ax_b, violin_data, ranks)

    # ΔSSS gaps on a secondary right axis (own scale) so they stay visible.
    ax_b2 = ax_b.twinx()
    gap_x = [i + 1.5 for i in range(NUM_LOCATIONS - 1)]
    vp2 = ax_b2.violinplot(gap_data, positions=gap_x,
                           showmedians=True, showextrema=False, widths=0.35)
    style_violin(vp2, GAP_COLOR, alpha=0.6)
    draw_iqr(ax_b2, gap_data, gap_x, linewidth=1.2)

    ax_b2.set_ylim(ROW2_GAP_YLIM)
    ax_b2.set_ylabel('ΔSSS between consecutive ranks', color=GAP_COLOR)
    ax_b2.tick_params(axis='y', colors=GAP_COLOR)
    ax_b2.spines['right'].set_color(GAP_COLOR)

    ax_b.axhline(0, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)
    ax_b.legend(handles=[
        mpatches.Patch(color=VIOLIN_SSS_COLOR, alpha=0.7, label='SSS per rank'),
        mpatches.Patch(color=GAP_COLOR, alpha=0.6,
                       label='ΔSSS between consecutive ranks'),
    ], loc='lower left')
    ax_b.set_xlabel('Location rank (by true SSS)')
    ax_b.set_ylabel('SSS per rank', color=VIOLIN_SSS_COLOR)
    ax_b.tick_params(axis='y', colors=VIOLIN_SSS_COLOR)
    ax_b.set_xticks(ranks)
    ax_b.set_ylim(ROW2_YLIM)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_axisbelow(True)
    ax_b.text(-0.04, 1.02, 'b', transform=ax_b.transAxes,
              fontsize=14, fontweight='bold', va='bottom', ha='left')

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_pdf = os.path.join(out_dir, 'sss_ranking.pdf')
    out_png = os.path.join(out_dir, 'sss_ranking.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    main()
