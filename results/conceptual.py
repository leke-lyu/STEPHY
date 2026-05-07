#!/usr/bin/env python3
"""Three-panel conceptual figure (publication-quality).

Panel (a): the SIR + migration simulation engine (4 populations,
K4 migration network, index case at population a).

Panel (b): an example outbreak tree produced by simulating the engine
forward in time. The event-type legend lives directly under this panel.

Panel (c): the K4 population network with per-node GNN prediction
annotations — the *graph structure* a GNN operates on, with each
node tagged by the parameters being regressed (R_e, mu, SSS) and the
classification target (which node is the index case).

Two arrows connect the panels:
  (a) -> (b)   stochastic simulation / Gillespie algorithm
  (b) -> (c)   graph encoding / GNN training

Saves conceptual.pdf next to this script.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------
# Publication-quality matplotlib defaults. These apply to imports below
# (draw_engine, draw_tree) too, so fonts and line weights are uniform.
# ---------------------------------------------------------------------
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset':  'stixsans',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.linewidth':    0.8,
})

from simulation_engine_4loc import draw_engine, I_POS        # noqa: E402
from simulated_tree_concept import draw_tree                 # noqa: E402


# ---------------------------------------------------------------------
# Publication font hierarchy (matches simulation_engine_4loc.py and
# simulated_tree_concept.py constants).
# ---------------------------------------------------------------------
FS_ARROW_TOP    = 20   # pipeline-stage caption above each arrow
FS_ARROW_BOT    = 17   # algorithm subtitle below each arrow (italic)
FS_PANEL_C_PARM = 18   # per-node "R_e, mu, SSS ?" regression caption
FS_PANEL_C_QUES = 19   # "Which location is the index case?" caption


def _draw_arrow_block(ax, top_text, bottom_text):
    """Draw a left-to-right arrow on `ax` with two stacked captions."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')

    ax.add_patch(FancyArrowPatch(
        (0.06, 0.5), (0.94, 0.5),
        arrowstyle='-|>,head_width=0.5,head_length=0.8',
        mutation_scale=36, lw=3.6,
        color='#1f1f1f', zorder=2,
        capstyle='round',
    ))
    # clip_on=False: matplotlib's default subplot padding shrinks the
    # arrow axes inside its GridSpec slot, so a long caption (e.g.
    # "stochastic simulation") can be wider than the inner axes box
    # even when it fits inside the slot. Disabling clipping lets the
    # caption render in full instead of cutting the trailing letter.
    ax.text(0.5, 0.585, top_text,
            ha='center', va='bottom', clip_on=False,
            fontsize=FS_ARROW_TOP, fontweight='bold', color='#111111')
    ax.text(0.5, 0.415, bottom_text,
            ha='center', va='top', clip_on=False,
            fontsize=FS_ARROW_BOT, style='italic', color='#444444')


def main():
    """Render the three-panel conceptual figure and save as PDF."""
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "conceptual.pdf")

    # Each main panel renders at ~12 in wide so it matches the standalone
    # simulation_engine_4loc.pdf and simulated_tree_concept.pdf sizes.
    # Arrow columns are intentionally narrow (~2.4 in) — the captions
    # are wider than this and overhang into the adjacent panel slots,
    # which is fine because clip_on=False is set on the arrow text and
    # the adjacent panel labels sit well inside their xlim margins.
    fig = plt.figure(figsize=(40, 12))
    gs = GridSpec(1, 5, figure=fig,
                  width_ratios=[1.0, 0.20, 1.0, 0.20, 1.0],
                  wspace=0.0)

    # ---- Panel (a): full simulation engine --------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    draw_engine(ax_a, show_compartments=True)

    # ---- Arrow 1: simulation ---------------------------------------
    ax_arr1 = fig.add_subplot(gs[0, 1])
    _draw_arrow_block(ax_arr1,
                      top_text='stochastic simulation',
                      bottom_text='Gillespie algorithm')

    # ---- Panel (b): simulated tree (legend lives under this panel) --
    ax_b = fig.add_subplot(gs[0, 2])
    draw_tree(ax_b, with_legend=True)

    # ---- Arrow 2: graph encoding + GNN training --------------------
    ax_arr2 = fig.add_subplot(gs[0, 3])
    _draw_arrow_block(ax_arr2,
                      top_text='graph encoding',
                      bottom_text='GNN training')

    # ---- Panel (c): K4 network with GNN prediction tasks ------------
    # Treat each population as a graph node — relabel I_x as Loc_x and
    # drop the surrounding "Location x" captions (the node labels carry
    # that info now). Uses the same xlim/ylim as panel (a) so the node
    # visual size matches.
    #
    # Each node carries an annotation showing the GNN's per-node
    # outputs:
    #   - regression  : predict per-location R0, RR, SSS
    #   - classification: identify the index case (★ on Loc_a)
    ax_c = fig.add_subplot(gs[0, 4])
    draw_engine(ax_c, show_compartments=False,
                node_prefix='Loc', show_pop_labels=False)

    # Per-node parameter caption (placed radially outward from each node
    # so it doesn't overlap with the migration K4 edges).
    param_text = r'$R_e,\ \mu,\ \mathrm{SSS}\ ?$'
    param_offset = {
        'a': (0,  2.2),   # above the top node
        'b': (2.2, 0),    # right of the right node
        'c': (0, -2.2),   # below the bottom node
        'd': (-2.2, 0),   # left of the left node
    }
    param_align = {
        'a': dict(ha='center', va='bottom'),
        'b': dict(ha='left',   va='center'),
        'c': dict(ha='center', va='top'),
        'd': dict(ha='right',  va='center'),
    }
    for loc in 'abcd':
        cx, cy = I_POS[loc]
        dx, dy = param_offset[loc]
        ax_c.text(cx + dx, cy + dy, param_text,
                  fontsize=FS_PANEL_C_PARM, fontweight='bold',
                  color='#333333', **param_align[loc])

    # Classification question: which node is the index case?
    # The star is drawn with matplotlib's '*' marker (separate from the
    # text) so it shares the exact same glyph as the index-case stars
    # in panels (a) and (b).
    a_cx, a_cy = I_POS['a']
    ax_c.text(a_cx - 0.5, a_cy + 3.7,
              'Which location is the index case?',
              ha='right', va='center',
              fontsize=FS_PANEL_C_QUES, fontweight='bold', style='italic',
              color='#8A6500')
    ax_c.plot(a_cx + 0.6, a_cy + 3.7, marker='*', markersize=34,
              color='#FFC107', markeredgecolor='#8A6500',
              markeredgewidth=1.6, zorder=4)

    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
