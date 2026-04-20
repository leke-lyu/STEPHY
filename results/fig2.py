#!/usr/bin/env python3
"""
Figure 2 — Performance analysis (3-row layout).

  Row 1 (a):   Top-k identification accuracy (100k dataset)
  Row 2 (b):   R0 distribution by rank with gap violins (100k dataset)
  Row 3 (c-f): Generalization across population scales (5k datasets)

Usage:
    python3 fig2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# -- Shared style (consistent with fig1.py) ---------------------------------

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

# -- Pipelines ---------------------------------------------------------------

BASE_DIR = '/Users/lukelyu/Desktop/data/simu'
RESULT_DIR = os.path.join(BASE_DIR, '100k_result')

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

# -- Layout constants --------------------------------------------------------

NUM_LOCATIONS = 12
VIOLIN_R0_COLOR = '#6baed6'
GAP_COLOR = '#fc8d59'

DATASETS = [
    ('5k_sp_result', '\u00d70.33'),   # small population
    ('5k_mp_result', '\u00d71'),      # medium population
    ('5k_lp_result', '\u00d73'),      # large population
]


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


def load_summary_metrics():
    """Load R2/accuracy from each population-scale summary.csv."""
    metrics = {p: {t: [] for t in TARGETS} for p in PIPELINES}
    for ds_dir, _ in DATASETS:
        df = pd.read_csv(os.path.join(BASE_DIR, ds_dir, 'summary.csv'))
        for _, row in df.iterrows():
            pipeline, label = row['pipeline'], row['label']
            if label in TARGETS and pipeline in PIPELINES:
                if row['metric_name'] in ('r2', 'accuracy'):
                    metrics[pipeline][label].append(row['metric_value'])
    return metrics


def make_pipeline_handles():
    """Build legend handles for the two pipelines."""
    return [
        Line2D([0], [0], marker=PIPELINE_STYLE[p][1],
               color=PIPELINE_STYLE[p][0], lw=2, label=p,
               markerfacecolor=PIPELINE_STYLE[p][0],
               markeredgecolor='white', markeredgewidth=0.8, markersize=8)
        for p in PIPELINES
    ]


# -- Main --------------------------------------------------------------------

def main():
    """
    Build Figure 2 — top-k identification, R0 rank structure, and 5k generalization.

    Row 1 (a): top-k accuracy across pipelines on the 100k dataset.
    Row 2 (b): R0 distribution per rank with gap violins (stephy only — single-
    pipeline view of rank structure, not a cross-pipeline comparison).
    Row 3 (c-f): pipeline metrics across three population-scale datasets.
    Output: fig2.pdf at the top of BASE_DIR.
    """
    # -- Load R0 predictions (100k) and reshape to (n_outbreaks, 12) --------
    data = {}
    for pipeline in PIPELINES:
        path = os.path.join(RESULT_DIR, pipeline, 'reg_r0',
                            'test_predictions.csv')
        df = pd.read_csv(path)
        true, pred = df['true_reg_r0'].values, df['pred_reg_r0'].values
        n = len(true) // NUM_LOCATIONS
        data[pipeline] = {
            'true': true[:n * NUM_LOCATIONS].reshape(n, NUM_LOCATIONS),
            'pred': pred[:n * NUM_LOCATIONS].reshape(n, NUM_LOCATIONS),
        }

    n_outbreaks = data['stephy']['true'].shape[0]
    # Row 2 visualizes true-R0 rank structure on the stephy split only; true
    # values are identical across pipelines since they share the same test set.
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

    # -- Load 5k generalization data ----------------------------------------
    gen_metrics = load_summary_metrics()

    # -- Figure layout -------------------------------------------------------
    fig = plt.figure(figsize=(16, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1],
                           hspace=0.35, wspace=0.35)
    pipeline_handles = make_pipeline_handles()

    # -- Row 1 (a): top-k accuracy ------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    style_handles = [
        Line2D([0], [0], color='#555555', lw=2, linestyle='-',
               label='Allow 1 miss'),
        Line2D([0], [0], color='#555555', lw=2, linestyle='--', alpha=0.4,
               label='Exact match'),
    ]

    x = np.array(ks)
    for pipeline in PIPELINES:
        color, marker = PIPELINE_STYLE[pipeline]
        ax_a.plot(x, exact_accs[pipeline], marker=marker, markersize=8,
                  color=color, linestyle='--', lw=2, alpha=0.4, zorder=3,
                  markeredgecolor='white', markeredgewidth=0.8)
        ax_a.plot(x, miss1_accs[pipeline], marker=marker, markersize=8,
                  color=color, linestyle='-', lw=2, zorder=4,
                  markeredgecolor='white', markeredgewidth=0.8)
        for xi, val in zip(x, miss1_accs[pipeline]):
            ax_a.annotate(f'{val:.0%}', (xi, val), textcoords='offset points',
                          xytext=(0, 10), ha='center', fontsize=8, color=color)

    ax_a.set_xlabel('Top-k')
    ax_a.set_ylabel('Accuracy')
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels([f'Top-{k}' for k in ks])
    ax_a.set_ylim(-0.02, 1.08)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_axisbelow(True)
    ax_a.legend(handles=pipeline_handles + style_handles,
                loc='upper right', ncol=4, fontsize=11)
    ax_a.text(-0.04, 1.02, 'a', transform=ax_a.transAxes,
              fontsize=18, fontweight='bold', va='bottom', ha='left')

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
                  va='bottom', fontsize=8, color=GAP_COLOR, fontweight='bold')

    ax_b.axhline(1.7, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)
    ax_b.legend(handles=[
        mpatches.Patch(color=VIOLIN_R0_COLOR, alpha=0.7, label='R0 per rank'),
        mpatches.Patch(color=GAP_COLOR, alpha=0.6,
                       label='\u0394R0 between consecutive ranks'),
    ], loc='upper right')
    ax_b.set_xlabel('Location rank (by true R0)')
    ax_b.set_ylabel('R0  /  \u0394R0')
    ax_b.set_xticks(ranks)
    ax_b.set_ylim(-0.5, 9)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_axisbelow(True)
    ax_b.text(-0.04, 1.02, 'b', transform=ax_b.transAxes,
              fontsize=18, fontweight='bold', va='bottom', ha='left')

    # -- Row 3 (c-f): 5k generalization across population scales ------------
    x_ds = np.arange(len(DATASETS))
    ds_labels = [lab for _, lab in DATASETS]

    for col, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[2, col])
        for pipeline in PIPELINES:
            color, marker = PIPELINE_STYLE[pipeline]
            vals = gen_metrics[pipeline][target]
            if not vals:
                continue
            ax.plot(x_ds, vals, marker=marker, color=color, lw=2,
                    markersize=8, markeredgecolor='white', markeredgewidth=0.8)
            for xi, v in zip(x_ds, vals):
                ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                            xytext=(0, 10), ha='center', fontsize=8,
                            color=color)

        ax.set_ylim(0.5, 1.0)
        ax.set_title(TARGET_LABELS[target], fontsize=12, fontweight='bold')
        ax.set_xticks(x_ds)
        ax.set_xticklabels(ds_labels, fontsize=10)
        ax.set_ylabel('Accuracy' if target == 'cls_as' else r'R$^2$',
                       fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.15, 1.02, chr(ord('c') + col), transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')

    fig.legend(handles=pipeline_handles, loc='lower center', frameon=False,
               ncol=len(PIPELINES), fontsize=11, bbox_to_anchor=(0.5, 0.06))

    out_path = os.path.join(BASE_DIR, 'fig2.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
