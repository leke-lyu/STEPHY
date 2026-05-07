#!/usr/bin/env python3
"""Conceptual figure: 4-location SIR + migration model (graph view).

Each location is a self-contained mini-SIR system with its own S, I, R,
and Sampled compartments stacked vertically (S above I; R and Sampled
below I). The four locations are arranged as the four nodes of a graph
in a diamond layout (North, East, South, West); migration events form
the edges of K4 (4 nodes, 6 edges) — 4 perimeter edges around the
diamond plus 2 cross diagonals.

Color scheme matches simulation_engine.pdf:

    blue   - S compartment / infection arrow
    red    - I compartment / infectious lineage
    grey   - R compartment / removal arrow
    green  - Sampled compartment / sampling arrow
    purple - migration edge (dashed, bidirectional)

Saves simulation_engine_4loc.pdf next to this script.
"""
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse


# ---------------------------------------------------------------------
# Diamond layout: 4 locations placed at N, E, S, W of a diamond. Each
# location's I compartment is the "graph node"; the per-location stack
# (S above I; R and Sampled below I) hangs off that node.
# ---------------------------------------------------------------------

LOC_R = 5.0  # diamond radius (centre-to-I distance)
ANGLES = {'a': 90, 'b': 0, 'c': 270, 'd': 180}

DY_S  =  2.5  # S above I (extra room for the larger I node)
DY_RS = -2.5  # R/Sampled below I
DX_RS =  1.10  # R left, Sampled right of I (wide enough that the
               # a-c cross edge can pass between R_a and Sampled_a)

I_POS, S_POS, R_POS, SAMP_POS = {}, {}, {}, {}
for _loc, _angle in ANGLES.items():
    _rad = math.radians(_angle)
    _cx, _cy = LOC_R * math.cos(_rad), LOC_R * math.sin(_rad)
    I_POS[_loc]    = (_cx,            _cy)
    S_POS[_loc]    = (_cx,            _cy + DY_S)
    R_POS[_loc]    = (_cx - DX_RS,    _cy + DY_RS)
    SAMP_POS[_loc] = (_cx + DX_RS,    _cy + DY_RS)

# Location labels: radially outward from the centre of the figure.
# Gap chosen so the publication-weight (fontsize=20) "Location x"
# caption doesn't visually clip into the I-node circle outline.
LOC_LABEL_POS = {
    'a': (0,  LOC_R + DY_S + 2.0),                   # above N location
    'b': (LOC_R + 3.2, 0),                           # right of E location
    'c': (0, -(LOC_R + abs(DY_RS) + 2.0)),           # below S location
    'd': (-(LOC_R + 3.2), 0),                        # left of W location
}

# K4 migration edges: 4 perimeter (diamond sides) + 2 cross diagonals.
# Each edge is rendered as two overlapping single-headed arrows so both
# heads share the same glyph (see the migration drawing block below).
MIG_PAIRS = [
    ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a'),  # perimeter
    ('a', 'c'),                                        # vertical cross
    ('b', 'd'),                                        # horizontal cross
]

COMP_RADIUS = 0.60   # S, R, Sampled compartments — small (visual weight
                     # is on the I graph nodes, not the per-location SIR)
I_RADIUS    = 1.15   # I compartments are the graph nodes — bigger

# ---------------------------------------------------------------------
# Publication font hierarchy. Centralized so panel (a)/(c) text reads
# at consistent weight when this engine is composed into conceptual.py.
# ---------------------------------------------------------------------
FS_NODE_I    = 26   # I_a / Loc_a labels inside the big graph nodes
FS_NODE_LOC  = 22   # same node when relabeled Loc_a (slightly smaller —
                    #   "Loc" + subscript is wider than "I" alone)
FS_COMP      = 16   # S_a, R_a labels
FS_COMP_SMP  = 13   # Smp_a (longer string, smaller to fit)
FS_EVENT     = 17   # infection / removal / sampling italic labels
FS_MIG       = 19   # "migration" label at the K4 cross-diagonal
FS_LOC_LABEL = 20   # "Location a/b/c/d" captions
FS_INDEX_CAP = 18   # "index case" caption next to the gold star


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------

def draw_compartment(ax, pos, color, label, fontsize=FS_COMP,
                     radius=COMP_RADIUS, linewidth=2.8):
    """Draw a labeled circular compartment with a colored outline."""
    x, y = pos
    ax.add_patch(Circle((x, y), radius, facecolor='white',
                        edgecolor=color, linewidth=linewidth, zorder=3))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold', zorder=4)


def edge_endpoints(p1, p2, r1=COMP_RADIUS, r2=COMP_RADIUS):
    """Shrink the segment p1->p2 so arrows touch the circle outlines."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    return ((p1[0] + ux * r1, p1[1] + uy * r1),
            (p2[0] - ux * r2, p2[1] - uy * r2))


def draw_arrow(ax, p1, p2, color, lw=2.2):
    """Single straight directed arrow from p1 to p2."""
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle='-|>', mutation_scale=22,
        color=color, lw=lw, zorder=2,
    ))


def draw_engine(ax, show_compartments=True,
                node_prefix='I', show_pop_labels=True):
    """Draw the diamond-layout 4-location migration model onto `ax`.

    `show_compartments=True` (default): full engine — location bubbles,
    S/R/Sampled compartments, within-location infection/removal/
    sampling arrows, and event-type labels.

    `show_compartments=False`: stripped network view — only the four
    nodes and migration K4 edges. Used as the "graph structure" panel
    that a GNN would operate on.

    `node_prefix`: text used for each node label. 'I' renders as $I_a$
    etc. (default, full SIR engine). 'Loc' renders as Loc_a etc.
    (cleaner when the panel is treated purely as a graph).

    `show_pop_labels`: if True, draw the surrounding "Location a/b/c/d"
    captions. Set False when the node labels (e.g. Loc_a) already
    identify each population.

    Self-contained: sets its own xlim/ylim/aspect/axis-off so it works
    inside any Axes (standalone figure or subplot panel).
    """
    # Colors match simulation_engine.pdf
    C_S     = '#2E7DBF'
    C_I     = '#D04F4F'
    C_R     = '#888888'
    C_SAMP  = '#3CB371'
    C_MIG   = '#7B4FB4'
    C_INDEX = '#FFC107'   # gold star marking the index case

    # ---- Location bubbles (dashed grey ellipses around each stack) ---
    if show_compartments:
        for loc in 'abcd':
            cx, cy = I_POS[loc]
            ax.add_patch(Ellipse((cx, cy), 3.8, 6.8,
                                 facecolor='none', edgecolor='#aaaaaa',
                                 linewidth=1.0, linestyle=':', zorder=1))

    # ---- Per-location compartments and within-location arrows --------
    # I compartments (the graph nodes) are always drawn. S/R/Sampled
    # and their arrows are only drawn in full mode.
    use_I_label = (node_prefix == 'I')
    node_fontsize = FS_NODE_I if use_I_label else FS_NODE_LOC
    for loc in 'abcd':
        if use_I_label:
            node_label = f'$I_{loc}$'
        else:
            node_label = f'$\\mathrm{{{node_prefix}}}_{loc}$'
        draw_compartment(ax, I_POS[loc], C_I, node_label,
                         fontsize=node_fontsize, radius=I_RADIUS,
                         linewidth=4.5)

        if show_compartments:
            draw_compartment(ax, S_POS[loc],    C_S,    f'$S_{loc}$',
                             fontsize=FS_COMP)
            draw_compartment(ax, R_POS[loc],    C_R,    f'$R_{loc}$',
                             fontsize=FS_COMP)
            draw_compartment(ax, SAMP_POS[loc], C_SAMP,
                             f'$\\mathrm{{Smp}}_{loc}$',
                             fontsize=FS_COMP_SMP)

            # Infection (S -> I): I edge is at the bigger I_RADIUS
            s_end, i_top = edge_endpoints(S_POS[loc], I_POS[loc],
                                          r1=COMP_RADIUS, r2=I_RADIUS)
            draw_arrow(ax, s_end, i_top, C_S, lw=2.4)

            # Removal (I -> R)
            i_end_r, r_in = edge_endpoints(I_POS[loc], R_POS[loc],
                                           r1=I_RADIUS, r2=COMP_RADIUS)
            draw_arrow(ax, i_end_r, r_in, C_R, lw=2.0)

            # Sampling (I -> Sampled)
            i_end_s, s_in = edge_endpoints(I_POS[loc], SAMP_POS[loc],
                                           r1=I_RADIUS, r2=COMP_RADIUS)
            draw_arrow(ax, i_end_s, s_in, C_SAMP, lw=2.0)

    # One italic event-type label apiece (using location a as the example).
    # White bbox so the label stays readable when migration lines cross it.
    if show_compartments:
        label_bbox = dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.95)

        s_end, i_top = edge_endpoints(S_POS['a'], I_POS['a'],
                                      r1=COMP_RADIUS, r2=I_RADIUS)
        ax.text(s_end[0] + 0.40, 0.5 * (s_end[1] + i_top[1]),
                'infection', ha='left', va='center',
                fontsize=FS_EVENT, color=C_S, style='italic',
                bbox=label_bbox, zorder=5)

        i_end, r_in = edge_endpoints(I_POS['a'], R_POS['a'],
                                     r1=I_RADIUS, r2=COMP_RADIUS)
        ax.text(0.5 * (i_end[0] + r_in[0]) - 0.30,
                0.5 * (i_end[1] + r_in[1]),
                'removal', ha='right', va='center',
                fontsize=FS_EVENT, color=C_R, style='italic',
                bbox=label_bbox, zorder=5)

        i_end, s_in = edge_endpoints(I_POS['a'], SAMP_POS['a'],
                                     r1=I_RADIUS, r2=COMP_RADIUS)
        ax.text(0.5 * (i_end[0] + s_in[0]) + 0.30,
                0.5 * (i_end[1] + s_in[1]),
                'sampling', ha='left', va='center',
                fontsize=FS_EVENT, color=C_SAMP, style='italic',
                bbox=label_bbox, zorder=5)

    # ---- Migration K4 graph (straight bidirectional edges) -----------
    # Each edge is rendered as TWO overlapping single-headed `-|>` arrows
    # (forward + reverse) instead of one `<|-|>` arrow. Reason: `<|-|>`
    # uses different glyph code paths for the two heads, which can
    # produce visibly inconsistent arrowhead shapes in some renderers.
    # Two `-|>` arrows guarantee both ends are the same filled triangle.
    # Drawn at zorder=2 (above the dotted bubble at zorder=1 but BELOW
    # compartments at zorder=3), so when the a-c cross edge passes
    # through S_c the node draws on top of the line — standard graph
    # drawing convention for edge/node crossings.
    for loc1, loc2 in MIG_PAIRS:
        p1, p2 = edge_endpoints(I_POS[loc1], I_POS[loc2],
                                r1=I_RADIUS, r2=I_RADIUS)
        for start, end in [(p1, p2), (p2, p1)]:
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle='-|>', mutation_scale=32,
                connectionstyle='arc3,rad=0',
                color=C_MIG, lw=5.0, linestyle='-', zorder=2,
            ))

    # "migration" italic label, placed at the crossing of the K4
    # diagonals — needs a white bbox so the lines that cross under it
    # don't visually run through the text.
    ax.text(0.0, 0.65, 'migration', ha='center', va='bottom',
            fontsize=FS_MIG, color=C_MIG, style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                      edgecolor='none', alpha=0.95),
            zorder=5)

    # ---- Location labels ---------------------------------------------
    if show_pop_labels:
        for loc in 'abcd':
            lx, ly = LOC_LABEL_POS[loc]
            ax.text(lx, ly, f'Location {loc}', ha='center', va='center',
                    fontsize=FS_LOC_LABEL, fontweight='bold', color='#444444')

        # Index-case marker: gold star + "index case" caption next to
        # the Location a label. Marks Loc a as the true seed of the
        # simulation; matches the classification star used in panel (c).
        a_lx, a_ly = LOC_LABEL_POS['a']
        ax.plot(a_lx - 3.1, a_ly, marker='*', markersize=42,
                color=C_INDEX, markeredgecolor='#8A6500',
                markeredgewidth=1.6, zorder=4)
        ax.text(a_lx + 2.2, a_ly, 'index case',
                ha='left', va='center',
                fontsize=FS_INDEX_CAP, fontweight='bold', style='italic',
                color='#8A6500')

    # ---- Cosmetics ---------------------------------------------------
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    """Render the engine figure as a standalone PDF."""
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "simulation_engine_4loc.pdf")
    fig, ax = plt.subplots(figsize=(12, 12))
    draw_engine(ax)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
