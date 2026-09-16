#!/usr/bin/env python3
"""
Figure 1 — STEPHY end-to-end overview (conceptual; reads no data).

  Panel a  Data generation: 4-location SIR + migration engine, a
           'stochastic simulation / Gillespie algorithm' arrow, and a
           hand-laid 12-tip outbreak tree.
  Panel b  Encoder pipeline (2 x 2):
             subtree extraction  generated 50-tip tree; Loc_a's subtree red,
                                 one neighbouring clade collapsed to a dashed
                                 triangle
             per-node encoder    CBLV (W x 4) -> plain / stride / dilate CNN
                                 -> (1 x 96); Aux (1 x 5) -> MLP -> (1 x 32);
                                 concat -> h_a (1 x 128)
             per-edge encoder    tip-time densities rho_a, rho_b -> DTW ->
                                 e_{b->a} (distance / mean lag / std lag)
             graph assemble      directed K4 carrying node and edge features
  Panel c  Inference: message-passing layer -> MLP head (256 -> 128 -> 64 ->
           32) -> per-location R_0 / gamma / SSS with 95% CP intervals, plus
           the predicted ancestral state and its CP set.

Only the panel-b tree is random (--seed); everything else is fixed. Saves
fig1.pdf and fig1.png next to this script.

Usage:
    python3 fig1.py
    python3 fig1.py --seed 7
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle, ConnectionPatch, Ellipse, FancyArrowPatch, Polygon, Rectangle,
)

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.linewidth':   0.8,
})

# -- Palette and shared fonts ------------------------------------------------

C_INF       = '#2E7DBF'   # infection, susceptible
C_HL        = '#D04F4F'   # infected, Loc_a highlight
C_REM       = '#888888'   # recovery / removal
C_SAMP      = '#3CB371'   # sampling
C_MIG       = '#7B4FB4'   # migration
C_DIM       = '#BFBFBF'   # background lineages and nodes
C_GOLD      = '#FFC107'   # seed star
C_GOLD_DARK = '#8A6500'   # seed star edge, ancestral-state captions
C_NODE_FEAT = '#3F76C0'
C_EDGE_FEAT = '#7E57C2'
C_CONV      = '#F2A93B'   # CNN and MLP blocks

# Sized for Nature print (5-8 pt at the 7.2-inch final width).
FS_TITLE    = 8   # sub-panel titles
FS_QUESTION = 8   # ancestral-state caption (a) and readout (c)
FS_TIME     = 7
FS_CAPTION  = 7

LOCS = ['a', 'b', 'c', 'd']


def _blank(ax):
    """Hide an axes' spines and ticks, keeping its background patch."""
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _title(ax, text, y=0.97, va='top'):
    """Write a bold sub-panel title centred at axes-fraction height `y`."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va=va,
            fontsize=FS_TITLE, fontweight='bold', color='#222')


# -- Panel a: simulation engine ----------------------------------------------

# Locations sit on a diamond: a north, d east, c south, b west. Loc_b is west
# so panel b's Loc_a-Loc_b edge strip lands beside the per-edge encoder.
# Each location's S sits above its I node, R and Sampled below.
LOC_R = 5.0
ANGLES = {'a': 90, 'b': 180, 'c': 270, 'd': 0}
DY_S, DY_RS, DX_RS = 3.4, -3.4, 1.30

I_POS = {l: (LOC_R * math.cos(math.radians(ang)),
             LOC_R * math.sin(math.radians(ang)))
         for l, ang in ANGLES.items()}
S_POS    = {l: (x,         y + DY_S)  for l, (x, y) in I_POS.items()}
R_POS    = {l: (x - DX_RS, y + DY_RS) for l, (x, y) in I_POS.items()}
SAMP_POS = {l: (x + DX_RS, y + DY_RS) for l, (x, y) in I_POS.items()}

MIG_PAIRS = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a'),
             ('a', 'c'), ('b', 'd')]

COMP_RADIUS = 0.75
I_RADIUS    = 1.15
FS_NODE_I   = 8
FS_COMP     = 6
FS_COMP_SMP = 5
FS_EVENT    = 6


def draw_compartment(ax, pos, color, label, fontsize=FS_COMP,
                     radius=COMP_RADIUS, linewidth=0.7):
    """Draw a labelled circular compartment with a coloured outline."""
    x, y = pos
    ax.add_patch(Circle((x, y), radius, facecolor='white',
                        edgecolor=color, linewidth=linewidth, zorder=3))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold', zorder=4)


def edge_endpoints(p1, p2, r1=COMP_RADIUS, r2=COMP_RADIUS):
    """Shrink the segment p1 -> p2 so it starts and ends on the circle outlines."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    return ((p1[0] + ux * r1, p1[1] + uy * r1),
            (p2[0] - ux * r2, p2[1] - uy * r2))


def draw_arrow(ax, p1, p2, color, lw=0.55):
    """Draw a straight directed arrow from p1 to p2."""
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=4,
                                 color=color, lw=lw, zorder=2))


def draw_engine(ax):
    """
    Draw the 4-location SIR engine onto `ax`.

    Each location has a dotted bubble holding S, I, R and Sampled
    compartments with infection, recovery and sampling arrows; the I nodes
    are joined by bidirectional migration edges.
    """
    for loc in LOCS:
        ax.add_patch(Ellipse(I_POS[loc], 4.2, 8.4, facecolor='none',
                             edgecolor='#aaaaaa', linewidth=0.6,
                             linestyle=':', zorder=1))

    for loc in LOCS:
        draw_compartment(ax, I_POS[loc], C_HL, f'$I_{loc}$',
                         fontsize=FS_NODE_I, radius=I_RADIUS, linewidth=1.1)
        draw_compartment(ax, S_POS[loc], C_INF, f'$S_{loc}$')
        draw_compartment(ax, R_POS[loc], C_REM, f'$R_{loc}$')
        draw_compartment(ax, SAMP_POS[loc], C_SAMP,
                         f'$\\mathrm{{Smp}}_{loc}$', fontsize=FS_COMP_SMP)
        draw_arrow(ax, *edge_endpoints(S_POS[loc], I_POS[loc], r2=I_RADIUS),
                   C_INF, lw=0.6)
        draw_arrow(ax, *edge_endpoints(I_POS[loc], R_POS[loc], r1=I_RADIUS),
                   C_REM, lw=0.5)
        draw_arrow(ax, *edge_endpoints(I_POS[loc], SAMP_POS[loc], r1=I_RADIUS),
                   C_SAMP, lw=0.5)

    # Two opposing single-headed arrows per edge so both heads match.
    for loc1, loc2 in MIG_PAIRS:
        p1, p2 = edge_endpoints(I_POS[loc1], I_POS[loc2],
                                r1=I_RADIUS, r2=I_RADIUS)
        for start, end in [(p1, p2), (p2, p1)]:
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle='-|>', mutation_scale=8,
                connectionstyle='arc3,rad=0', color=C_MIG, lw=1.2,
                linestyle='-', zorder=2))

    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-9.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')


# -- Panel a: hand-laid 12-tip tree ------------------------------------------

TIP_Y = [0.4 + 0.6 * i for i in range(12)]
TIP_END_X = [6.0, 8.0, 7.0, 5.0, 8.0, 7.5, 6.0, 8.0, 7.0, 5.0, 8.0, 6.5]
TIP_TYPES = ['removed', 'sampled', 'removed', 'removed',
             'sampled', 'removed', 'removed', 'sampled',
             'removed', 'removed', 'sampled', 'removed']
TIP_LOCATIONS = ['b', 'b', 'c', 'c', 'a', 'a', 'c', 'c', 'd', 'd', 'd', 'd']

NODES = {
    'root_inf': (0.3, 3.10),
    'L3_join':  (1.3, 4.90),
    'SC1':      (2.2, 1.30),
    'SC2':      (2.5, 3.70),
    'SC3':      (2.2, 6.10),
    'pair01':   (3.3, 0.70),
    'pair23':   (3.6, 1.90),
    'pair45':   (3.7, 3.10),
    'pair67':   (4.0, 4.30),
    'pair89':   (3.3, 5.50),
    'pair1011': (4.1, 6.70),
}
# Children as (node name or tip index, y).
CHILDREN = {
    'root_inf': [('SC1', 1.30),     ('L3_join', 4.90)],
    'L3_join':  [('SC2', 3.70),     ('SC3', 6.10)],
    'SC1':      [('pair01', 0.70),  ('pair23', 1.90)],
    'SC2':      [('pair45', 3.10),  ('pair67', 4.30)],
    'SC3':      [('pair89', 5.50),  ('pair1011', 6.70)],
    'pair01':   [(0, TIP_Y[0]),     (1, TIP_Y[1])],
    'pair23':   [(2, TIP_Y[2]),     (3, TIP_Y[3])],
    'pair45':   [(4, TIP_Y[4]),     (5, TIP_Y[5])],
    'pair67':   [(6, TIP_Y[6]),     (7, TIP_Y[7])],
    'pair89':   [(8, TIP_Y[8]),     (9, TIP_Y[9])],
    'pair1011': [(10, TIP_Y[10]),   (11, TIP_Y[11])],
}
NODE_LOCATIONS = {
    'root_inf': 'a', 'L3_join': 'a', 'SC1': 'b', 'SC2': 'a', 'SC3': 'd',
    'pair01': 'b', 'pair23': 'c', 'pair45': 'a', 'pair67': 'c',
    'pair89': 'd', 'pair1011': 'd',
}
# (x, y, from, to): each migration sits midway along its branch.
MIGRATIONS_INFO = [
    (0.5 * (NODES['root_inf'][0] + NODES['SC1'][0]),    NODES['SC1'][1],    'a', 'b'),
    (0.5 * (NODES['SC1'][0]      + NODES['pair23'][0]), NODES['pair23'][1], 'b', 'c'),
    (0.5 * (NODES['L3_join'][0]  + NODES['SC3'][0]),    NODES['SC3'][1],    'a', 'd'),
    (0.5 * (NODES['SC2'][0]      + NODES['pair67'][0]), NODES['pair67'][1], 'a', 'c'),
]

SEED_X_T12   = -0.3
FS_EVENT_TAG = 5
FS_MIG_TAG   = 5
FS_T12_TIME  = 6
LW_BRANCH    = 1.2
MS_INF, MS_SAMP, MS_REM, MS_MIG, MS_INDEX = 4.5, 5.0, 4.7, 4.0, 8.0


def draw_tree(ax):
    """
    Draw the hand-laid 12-tip outbreak tree onto `ax`.

    Infection nodes are blue, sampled tips green dots, removed tips grey
    crosses, migrations purple diamonds tagged 'x -> y'; the seed at Loc_a is
    a gold star. Sets its own limits and draws the time arrow.
    """
    root_x, root_y = NODES['root_inf']
    ax.plot([SEED_X_T12, root_x], [root_y, root_y],
            color=C_HL, lw=LW_BRANCH, solid_capstyle='round', zorder=1)

    for parent, kids in CHILDREN.items():
        px = NODES[parent][0]
        ax.plot([px, px], [kids[0][1], kids[1][1]],
                color=C_HL, lw=LW_BRANCH, zorder=1)
        for child, child_y in kids:
            child_x = TIP_END_X[child] if isinstance(child, int) else NODES[child][0]
            ax.plot([px, child_x], [child_y, child_y], color=C_HL,
                    lw=LW_BRANCH, solid_capstyle='round', zorder=1)

    for label, (x, y) in NODES.items():
        ax.plot(x, y, 'o', color=C_INF, markersize=MS_INF, zorder=3,
                markeredgecolor='white', markeredgewidth=0.4)
        ax.text(x + 0.24, y, NODE_LOCATIONS[label], fontsize=FS_EVENT_TAG,
                fontweight='bold', style='italic', color=C_INF,
                ha='left', va='center', zorder=4)

    for i in range(12):
        x, y = TIP_END_X[i], TIP_Y[i]
        if TIP_TYPES[i] == 'sampled':
            ax.plot(x, y, 'o', color=C_SAMP, markersize=MS_SAMP, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.45)
            label_color = C_SAMP
        else:
            ax.plot(x, y, 'X', color=C_REM, markersize=MS_REM, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.4)
            label_color = C_REM
        ax.text(x + 0.26, y, TIP_LOCATIONS[i], fontsize=FS_EVENT_TAG,
                fontweight='bold', style='italic', color=label_color,
                ha='left', va='center', zorder=4)

    for mx, my, from_loc, to_loc in MIGRATIONS_INFO:
        ax.plot([mx, mx], [my - 0.26, my + 0.26],
                color=C_MIG, lw=0.6, linestyle='--', zorder=2)
        ax.plot(mx, my, marker='D', color=C_MIG, markersize=MS_MIG,
                zorder=3, markeredgecolor='white', markeredgewidth=0.4)
        ax.text(mx, my + 0.48, f'{from_loc} $\\rightarrow$ {to_loc}',
                fontsize=FS_MIG_TAG, fontweight='bold', style='italic',
                color=C_MIG, ha='center', va='bottom', zorder=4)

    ax.plot(SEED_X_T12, root_y, marker='*', markersize=MS_INDEX,
            color=C_GOLD, markeredgecolor=C_GOLD_DARK,
            markeredgewidth=0.4, zorder=3)

    ax.set_xlim(-1.6, 9.2)
    ax.set_ylim(-1.35, 9.2)
    ax.set_aspect('equal')
    _blank(ax)
    ax.annotate('', xy=(8.7, -0.55), xytext=(SEED_X_T12, -0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
    ax.text(0.5 * (SEED_X_T12 + 8.7), -1.05, 'Time',
            ha='center', fontsize=FS_T12_TIME, fontweight='bold')


def _draw_arrow_block(ax, top_text=None, bottom_text=None):
    """Fill `ax` with a left-to-right arrow and optional captions above and below it."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.add_patch(FancyArrowPatch(
        (0.06, 0.5), (0.94, 0.5),
        arrowstyle='-|>,head_width=0.5,head_length=0.8',
        mutation_scale=9, lw=0.9, color='#1f1f1f', zorder=2,
        capstyle='round'))
    # clip_on=False lets captions overhang into the neighbouring sub-panels.
    if top_text:
        ax.text(0.5, 0.585, top_text, ha='center', va='bottom', clip_on=False,
                fontsize=7, fontweight='bold', color='#111111')
    if bottom_text:
        ax.text(0.5, 0.415, bottom_text, ha='center', va='top', clip_on=False,
                fontsize=6, style='italic', color='#444444')


def _draw_panel_a(fig, spec):
    """Draw panel a (engine, arrow, 12-tip tree) and return the engine axes."""
    gs = spec.subgridspec(1, 3, width_ratios=[6.0, 1.5, 6.0], wspace=0.04)

    ax_engine = fig.add_subplot(gs[0, 0])
    draw_engine(ax_engine)
    # Right of I_a at its height; bubble a ends at x = 2.1.
    ax_engine.text(2.5, 5.0, 'Loc$_a$, ancestral state',
                   ha='left', va='center', clip_on=False,
                   fontsize=FS_QUESTION, fontweight='bold', style='italic',
                   color=C_GOLD_DARK)
    ax_engine.legend(handles=[
        Line2D([0], [0], color=C_INF, lw=1.2, label='infection'),
        Line2D([0], [0], color=C_MIG, lw=1.4, label='migration'),
        Line2D([0], [0], color=C_REM, lw=1.2, label='recovery'),
        Line2D([0], [0], color=C_SAMP, lw=1.2, label='sampling'),
    ], loc='lower right', frameon=False, fontsize=FS_EVENT,
        handlelength=1.4, handletextpad=0.5, labelspacing=0.35,
        borderaxespad=0.2)

    _draw_arrow_block(fig.add_subplot(gs[0, 1]),
                      'stochastic simulation', 'Gillespie algorithm')
    draw_tree(fig.add_subplot(gs[0, 2]))
    return ax_engine


# -- Panel b: generated 50-tip tree ------------------------------------------

TARGET_LOC  = 'a'    # location whose subtree is extracted
INITIAL_LOC = 'b'    # root location; must differ so Loc_a's MRCA is not the root
OTHER_LOC   = 'b'    # partner location for the per-edge encoder
N_TIPS      = 50
TREE_T      = 8.0


class TipNode:
    """Leaf lineage from start_x to end_x."""
    __slots__ = ('end_x', 'sampled', 'start_x', 'y', 'location',
                 'in_induced', 'in_dashed')

    def __init__(self, end_x, sampled, start_x):
        self.end_x = end_x
        self.sampled = sampled
        self.start_x = start_x
        self.y = None
        self.location = None
        self.in_induced = False
        self.in_dashed = False


class IntNode:
    """Internal node (branching event) at time x."""
    __slots__ = ('x', 'children', 'y', 'location', 'is_migration',
                 'in_induced', 'in_dashed')

    def __init__(self, x, children):
        self.x = x
        self.children = children
        self.y = None
        self.location = None
        self.is_migration = False
        self.in_induced = False
        self.in_dashed = False


def iter_nodes(node):
    """Yield every node in `node`'s subtree, pre-order, children in list order."""
    yield node
    if isinstance(node, IntNode):
        for c in node.children:
            yield from iter_nodes(c)


def _tips(node):
    """Return the tips under `node`, top to bottom."""
    return [n for n in iter_nodes(node) if isinstance(n, TipNode)]


def gen_tree(n_tips, T, rng, sample_frac=0.55):
    """
    Build a random binary tree with `n_tips` leaves over the time span [0, T].

    Each split falls 10-45% of the way into its remaining time. Sampled tips
    end in the last half of their remaining time, unsampled ones earlier.
    """
    def _build(start_x, n_leaves):
        if n_leaves == 1:
            sampled = rng.random() < sample_frac
            lo, hi = (0.50, 1.00) if sampled else (0.15, 0.65)
            return TipNode(start_x + rng.uniform(lo, hi) * (T - start_x),
                           sampled, start_x)
        split_x = start_x + rng.uniform(0.10, 0.45) * (T - start_x)
        m = int(rng.integers(1, n_leaves))
        left = _build(split_x, m)
        right = _build(split_x, n_leaves - m)
        return IntNode(split_x, [left, right])

    return _build(0.0, n_tips)


def _count_tips(node):
    """Return the number of tips under `node`."""
    return sum(isinstance(n, TipNode) for n in iter_nodes(node))


def ladderize(root):
    """Sort every internal node's children by tip count, smallest first."""
    for n in iter_nodes(root):
        if isinstance(n, IntNode):
            n.children.sort(key=_count_tips)


def assign_y(root):
    """Give tips consecutive integer y and each internal node the midpoint of its outer children."""
    next_y = [0]

    def _walk(node):
        if isinstance(node, TipNode):
            node.y = float(next_y[0])
            next_y[0] += 1
            return
        for c in node.children:
            _walk(c)
        node.y = 0.5 * (node.children[0].y + node.children[-1].y)

    _walk(root)


def assign_locations(root, rng, target_loc, initial_loc, a_subclade_target,
                     a_exit_migrations, n_other_migrations):
    """
    Assign locations so that target_loc is present but not monophyletic.

    The non-root internal node whose tip count is closest to
    `a_subclade_target` (random tie-break) becomes the migration into
    target_loc. Inside that clade, `a_exit_migrations` small internals
    (2-5 tips when enough exist) migrate out again, and `n_other_migrations`
    internals elsewhere migrate too; each migration picks a random location
    other than its parent's and target_loc. Everything else inherits its
    parent's location, starting from `initial_loc` at the root.
    """
    non_root = [n for n in iter_nodes(root) if isinstance(n, IntNode)][1:]

    deviations = [abs(_count_tips(n) - a_subclade_target) for n in non_root]
    candidates = [n for n, d in zip(non_root, deviations) if d == min(deviations)]
    a_clade_root = candidates[int(rng.integers(len(candidates)))]
    a_clade_root.is_migration = True

    a_internals = [n for c in a_clade_root.children for n in iter_nodes(c)
                   if isinstance(n, IntNode)]
    preferred = [n for n in a_internals if 2 <= _count_tips(n) <= 5]
    if len(preferred) < a_exit_migrations:
        preferred = a_internals
    n_exit = min(a_exit_migrations, len(preferred))
    if n_exit > 0:
        for idx in rng.choice(len(preferred), size=n_exit, replace=False):
            preferred[int(idx)].is_migration = True

    in_a_clade = {id(n) for n in iter_nodes(a_clade_root)}
    others = [n for n in non_root if id(n) not in in_a_clade]
    n_other = min(n_other_migrations, len(others))
    if n_other > 0:
        for idx in rng.choice(len(others), size=n_other, replace=False):
            others[int(idx)].is_migration = True

    def _assign(node, parent_loc):
        if isinstance(node, TipNode):
            node.location = parent_loc
            return
        if node is a_clade_root:
            node.location = target_loc
        elif node.is_migration:
            choices = [l for l in LOCS if l not in (parent_loc, target_loc)]
            node.location = choices[int(rng.integers(len(choices)))]
        else:
            node.location = parent_loc
        for c in node.children:
            _assign(c, node.location)

    _assign(root, initial_loc)


def find_mrca(root, target_loc):
    """Return the deepest node whose subtree holds every sampled target_loc tip."""
    def _has_target(n):
        return any(t.location == target_loc and t.sampled for t in _tips(n))

    node = root
    while isinstance(node, IntNode):
        kids = [c for c in node.children if _has_target(c)]
        if len(kids) != 1:
            break
        node = kids[0]
    return node


def rebias_loc_a_tip_times(root, target_loc, T, rng,
                           mu_frac=0.55, sigma_frac=0.18, tries=40):
    """
    Redraw sampled target_loc tip times from a truncated Gaussian.

    Gives Loc_a's tip-time density in panel b a bell shape. Draws must fall
    in (start_x + 0.05, T); a tip with no valid draw after `tries` attempts
    is clamped to the mean within that range.
    """
    t_mu, t_sigma = T * mu_frac, T * sigma_frac
    for tip in _tips(root):
        if tip.location != target_loc or not tip.sampled:
            continue
        low = tip.start_x + 0.05
        if low >= T:
            continue
        for _ in range(tries):
            candidate = rng.normal(t_mu, t_sigma)
            if low < candidate < T:
                tip.end_x = candidate
                break
        else:
            tip.end_x = min(T - 0.01, max(low, t_mu))


def mark_induced_subtree(root, mrca, target_loc):
    """Flag the nodes under `mrca` that lie on a path to a sampled target_loc tip."""
    for n in iter_nodes(root):
        n.in_induced = False

    def _walk(node):
        if isinstance(node, TipNode):
            node.in_induced = node.location == target_loc and node.sampled
        else:
            node.in_induced = any([_walk(c) for c in node.children])
        return node.in_induced

    _walk(mrca)


def find_below_loca_clade(root, mrca):
    """
    Return the clade drawn collapsed: the largest subtree whose tips all sit
    just below the MRCA's lowest tip (highest max y wins, then more tips).
    """
    y_min_a = min(t.y for t in _tips(mrca))
    candidates = []

    def _walk(n):
        if isinstance(n, IntNode):
            sub_tips = _tips(n)
            sub_max_y = max(t.y for t in sub_tips)
            if sub_max_y < y_min_a:
                candidates.append((sub_max_y, len(sub_tips), n))
            else:
                for c in n.children:
                    _walk(c)

    _walk(root)
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    return candidates[0][2]


def mark_dashed_clade(dashed_root):
    """Flag every node under `dashed_root`, if any, for collapsed dashed drawing."""
    if dashed_root is not None:
        for n in iter_nodes(dashed_root):
            n.in_dashed = True


def draw_branches(ax, root, seed_x, lw_hl=1.1, lw_dim=0.7):
    """
    Draw the generated tree's branches.

    Branches inside the induced subtree are red, the rest light grey. The
    dashed clade gets a dashed stem and a dashed triangle spanning its tips
    instead of its internal branches.
    """
    dashed_kw = dict(color=C_DIM, lw=lw_dim + 0.3, linestyle=(0, (4, 3)),
                     zorder=1, solid_capstyle='butt')
    ax.plot([seed_x, root.x], [root.y, root.y],
            color=C_DIM, lw=lw_dim, zorder=1, solid_capstyle='round')

    def _triangle(clade_root):
        tips = _tips(clade_root)
        y_min, y_max = min(t.y for t in tips), max(t.y for t in tips)
        base_x = max(t.end_x for t in tips)
        ax.plot([clade_root.x, base_x], [clade_root.y, y_max], **dashed_kw)
        ax.plot([clade_root.x, base_x], [clade_root.y, y_min], **dashed_kw)
        ax.plot([base_x, base_x], [y_min, y_max], **dashed_kw)

    def _walk(node):
        if isinstance(node, TipNode):
            return
        for c in node.children:
            child_x = c.end_x if isinstance(c, TipNode) else c.x
            if c.in_dashed:
                if not node.in_dashed:
                    ax.plot([node.x, node.x], [node.y, c.y], **dashed_kw)
                    ax.plot([node.x, child_x], [c.y, c.y], **dashed_kw)
                    if isinstance(c, IntNode):
                        _triangle(c)
                continue
            hl = node.in_induced and c.in_induced
            kw = dict(color=C_HL if hl else C_DIM, lw=lw_hl if hl else lw_dim,
                      zorder=3 if hl else 1, solid_capstyle='round')
            ax.plot([node.x, node.x], [node.y, c.y], **kw)
            ax.plot([node.x, child_x], [c.y, c.y], **kw)
            _walk(c)

    _walk(root)


def draw_tips(ax, root, ms_hl=4, ms_dim=3):
    """Mark tips: green with a red edge in the induced subtree, small grey otherwise, none in the dashed clade."""
    for tip in _tips(root):
        if tip.in_dashed:
            continue
        if tip.in_induced:
            ax.plot(tip.end_x, tip.y, 'o', color=C_SAMP, markersize=ms_hl,
                    zorder=5, markeredgecolor=C_HL, markeredgewidth=0.35)
        elif tip.sampled:
            ax.plot(tip.end_x, tip.y, 'o', color=C_DIM, markersize=ms_dim,
                    zorder=2, markeredgecolor='white', markeredgewidth=0.5,
                    alpha=0.85)


def annotate_mrca(ax, mrca, target_loc):
    """Mark the MRCA with a red dot and caption it as target_loc's subtree."""
    ax.plot(mrca.x, mrca.y, marker='o', markersize=5, markerfacecolor=C_HL,
            markeredgecolor=C_HL, markeredgewidth=0.8, zorder=6)
    ax.text(mrca.x, mrca.y + 1.4, f'subtree of Loc$_{target_loc}$',
            ha='center', va='bottom', fontsize=FS_CAPTION,
            fontweight='bold', color=C_HL, zorder=7)


def gauss_kde(times, x_grid, bandwidth=0.40):
    """Return a Gaussian kernel density estimate of `times` on `x_grid`."""
    if len(times) == 0:
        return np.zeros_like(x_grid)
    arr = np.asarray(times)
    diff = (x_grid[:, None] - arr[None, :]) / bandwidth
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1)
    dens /= len(arr) * np.sqrt(2 * np.pi) * bandwidth
    return dens


def draw_tip_time_kde(ax, root, target_loc, other_loc, seed_x, x_max,
                      bandwidth=0.40):
    """
    Draw the per-edge encoder's two tip-time densities onto `ax`.

    Filled KDEs of sampled tip times for other_loc (grey) and target_loc
    (red), a rug of target_loc times, and a two-line label for each curve
    to the right of the time range.
    """
    tips = _tips(root)
    a_times = [t.end_x for t in tips if t.location == target_loc and t.sampled]
    b_times = [t.end_x for t in tips if t.location == other_loc and t.sampled]
    x_grid = np.linspace(seed_x, x_max, 400)
    kde_a = gauss_kde(a_times, x_grid, bandwidth=bandwidth)
    kde_b = gauss_kde(b_times, x_grid, bandwidth=bandwidth)
    y_max = max(kde_a.max(), kde_b.max(), 1e-6) * 1.25

    ax.fill_between(x_grid, 0, kde_b, color='#9a9a9a', alpha=0.35, zorder=1)
    ax.plot(x_grid, kde_b, color='#555', lw=0.8, zorder=2)
    ax.fill_between(x_grid, 0, kde_a, color=C_HL, alpha=0.30, zorder=3)
    ax.plot(x_grid, kde_a, color=C_HL, lw=0.9, zorder=4)
    ax.scatter(a_times, [-0.08 * y_max] * len(a_times),
               marker='|', color=C_HL, s=10, lw=0.65, zorder=5)

    # Labels sit past the curves' x range so none is hidden behind a curve.
    peak_b = float(kde_b.max())
    for loc, y, color in [(other_loc, peak_b * 0.55, '#555'),
                          (target_loc, peak_b * 0.10, C_HL)]:
        ax.text(x_max + 0.10, y,
                r'$\rho_{' + loc + r'}$' + ': density curve of\ntip time from '
                + r'$\mathrm{Loc}_{' + loc + r'}$',
                ha='left', va='center', fontsize=5, color=color,
                style='italic', clip_on=False, zorder=6)

    ax.set_ylim(-0.22 * y_max, y_max)
    _blank(ax)


# -- Panel b: encoder and graph drawing helpers ------------------------------

def draw_cblv_grid(ax, x0, y0, width, height, n_rows, n_cols, n_filled, seed=7):
    """Draw a stylised CBLV matrix: `n_filled` random viridis rows above zero-padding rows marked '0'."""
    cmap = plt.get_cmap('viridis')
    values = np.random.default_rng(seed).random((n_rows, n_cols))
    cell_w, cell_h = width / n_cols, height / n_rows
    for r in range(n_rows):
        padded = r >= n_filled
        for c in range(n_cols):
            xy = (x0 + c * cell_w, y0 + (n_rows - 1 - r) * cell_h)
            ax.add_patch(Rectangle(
                xy, cell_w, cell_h,
                facecolor='#e6e6e6' if padded else cmap(values[r, c]),
                edgecolor='white', lw=0.4, zorder=3))
            if padded:
                ax.text(xy[0] + cell_w / 2, xy[1] + cell_h / 2, '0',
                        ha='center', va='center', fontsize=3, color='#888',
                        zorder=4)
    ax.add_patch(Rectangle((x0, y0), width, height, facecolor='none',
                           edgecolor='#333', lw=1.2, zorder=4))


def draw_conv_block(ax, x_left, y_center, width, height, depth,
                    color=C_CONV, edgecolor='#333', alpha=0.92):
    """Draw a pseudo-3D block: a front face plus lighter top and right faces receding by `depth`."""
    ax.add_patch(Rectangle((x_left, y_center - height / 2), width, height,
                           facecolor=color, edgecolor=edgecolor,
                           lw=0.4, alpha=alpha, zorder=3))
    top_pts = [
        (x_left,                 y_center + height / 2),
        (x_left + depth,         y_center + height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,         y_center + height / 2),
    ]
    right_pts = [
        (x_left + width,         y_center - height / 2),
        (x_left + width + depth, y_center - height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,         y_center + height / 2),
    ]
    for pts, fade in [(top_pts, 0.75), (right_pts, 0.55)]:
        ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edgecolor,
                             lw=0.3, alpha=alpha * fade, zorder=3))


def draw_feature_strip(ax, x_left, y_bottom, n_boxes, box_w, box_h, color):
    """Draw a feature-vector strip of adjacent boxes, each with a fixed pseudo-random alpha."""
    for i in range(n_boxes):
        ax.add_patch(Rectangle(
            (x_left + i * box_w, y_bottom), box_w, box_h, facecolor=color,
            alpha=0.40 + 0.55 * (((i * 37 + 5) % 11) / 10.0),
            edgecolor='#333', lw=0.8, zorder=4))


# Shared by the graph-assemble (b) and predictions (c) K4s so they align.
K4_POSITIONS = {'a': (6.0, 7.6), 'b': (3.4, 5.0), 'c': (6.0, 2.4), 'd': (8.6, 5.0)}
# (src, dst) -> weight, drawn as arrow width. Asymmetric on purpose: a GAT
# learns alpha_ij != alpha_ji. The a<->b pair is moderately high to match the
# edge feature strip drawn on that edge.
K4_ATTENTION = {
    ('a', 'b'): 0.45, ('b', 'a'): 0.70,
    ('a', 'c'): 0.30, ('c', 'a'): 0.55,
    ('a', 'd'): 0.20, ('d', 'a'): 0.50,
    ('b', 'c'): 0.55, ('c', 'b'): 0.35,
    ('b', 'd'): 0.40, ('d', 'b'): 0.65,
    ('c', 'd'): 0.25, ('d', 'c'): 0.45,
}


def draw_directed_edges(ax, pos, attention, offset, shrink, lw_scale,
                        mutation_scale, zorder):
    """
    Draw one arrow per ordered location pair, width 0.25 + lw_scale * weight.

    Each arrow is shifted `offset` data units perpendicular to its edge so a
    pair's two opposing arrows sit side by side; `shrink` (points) keeps the
    arrowheads outside the node markers.
    """
    for src in LOCS:
        for dst in LOCS:
            if src == dst:
                continue
            (x0, y0), (x1, y1) = pos[src], pos[dst]
            dx, dy = x1 - x0, y1 - y0
            length = (dx * dx + dy * dy) ** 0.5
            px, py = -dy / length, dx / length
            ax.annotate('', xy=(x1 + offset * px, y1 + offset * py),
                        xytext=(x0 + offset * px, y0 + offset * py),
                        arrowprops=dict(
                            arrowstyle='-|>', color='#444',
                            lw=0.25 + lw_scale * attention[(src, dst)],
                            mutation_scale=mutation_scale,
                            shrinkA=shrink, shrinkB=shrink, zorder=zorder))


def draw_k4_graph(ax, target_loc, node_size=18):
    """Draw the directed location K4 at K4_POSITIONS; `target_loc` is red, the rest grey ('' for none)."""
    draw_directed_edges(ax, K4_POSITIONS, K4_ATTENTION, offset=0.26,
                        shrink=max(7, int(node_size * 0.55)), lw_scale=0.6,
                        mutation_scale=5, zorder=2)
    for l in LOCS:
        x, y = K4_POSITIONS[l]
        hl = l == target_loc
        ax.plot(x, y, 'o', markersize=node_size, color=C_HL if hl else C_DIM,
                markeredgecolor='#7a1f1f' if hl else '#5a5a5a',
                markeredgewidth=0.6, zorder=5)
        ax.text(x, y, f'Loc$_{l}$', ha='center', va='center',
                fontsize=max(5, int(0.42 * node_size)), fontweight='bold',
                color='white', zorder=6)


def draw_dtw_outputs(ax, target_loc, other_loc):
    """
    Draw the DTW edge feature e_{b->a} (1 x 3) with its three caption lines.

    Returns the strip's right-middle point in `ax` data coordinates, the
    start of the arrow into the graph-assemble cell.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _blank(ax)
    ax.patch.set_alpha(0.0)

    cell_w, cell_h, n_cells, cell_y = 0.109, 0.156, 3, 0.55
    start_x = (1.0 - n_cells * cell_w) / 2
    draw_feature_strip(ax, start_x, cell_y, n_cells, cell_w, cell_h, C_EDGE_FEAT)

    for dy, text, style in [
        (0.10, r'$e_{' + other_loc + r'\to ' + target_loc + r'}\ (1 \times 3)$', 'italic'),
        (0.25, r'$\mathrm{DTW}(\rho_{' + other_loc + r'},\, \rho_{' + target_loc + r'})$', 'normal'),
        (0.40, 'distance / mean lag / std lag', 'italic'),
    ]:
        ax.text(0.5, cell_y - dy, text, ha='center', va='top', fontsize=6,
                color=C_EDGE_FEAT, style=style)
    return start_x + n_cells * cell_w, cell_y + cell_h / 2


def _arrow(ax, start, end, color='#555', rad=None):
    """Draw a thin arrow inside the per-node encoder cell, curved when `rad` is given."""
    props = dict(arrowstyle='->', mutation_scale=6, lw=0.45, color=color, zorder=4)
    if rad is not None:
        props['connectionstyle'] = f'arc3,rad={rad:.2f}'
    ax.annotate('', xy=end, xytext=start, arrowprops=props)


def _draw_panel_b(fig, spec, seed):
    """
    Draw panel b, the encoder pipeline, and return its top-left axes.

    Cells: subtree extraction (top-left), per-node encoder (top-right),
    per-edge encoder (bottom-left), graph assemble (bottom-right). Two
    cross-cell arrows carry the node feature to Loc_a's strip and the DTW
    edge feature to the Loc_a-Loc_b edge strip.
    """
    rng = np.random.default_rng(seed)
    T = TREE_T
    root = gen_tree(N_TIPS, T, rng, sample_frac=1.0)
    ladderize(root)
    assign_y(root)
    assign_locations(root, rng, TARGET_LOC, INITIAL_LOC, a_subclade_target=32,
                     a_exit_migrations=4, n_other_migrations=3)
    rebias_loc_a_tip_times(root, TARGET_LOC, T, rng)
    mrca = find_mrca(root, TARGET_LOC)
    mark_induced_subtree(root, mrca, TARGET_LOC)
    mark_dashed_clade(find_below_loca_clade(root, mrca))

    seed_x = -0.6
    x_axis_max = T + 1.6
    time_x_end = T + 0.15

    gs = spec.subgridspec(2, 2, width_ratios=[1.0, 1.0],
                          height_ratios=[1.0, 1.0], hspace=0.04, wspace=0.08)
    ax_tree  = fig.add_subplot(gs[0, 0])
    ax_kde   = fig.add_subplot(gs[1, 0], sharex=ax_tree)
    ax_conv  = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[1, 1])

    # ---- subtree extraction ---------------------------------------------
    draw_branches(ax_tree, root, seed_x)
    draw_tips(ax_tree, root)
    annotate_mrca(ax_tree, mrca, TARGET_LOC)
    ax_tree.plot(seed_x, root.y, marker='*', markersize=10, color=C_GOLD,
                 markeredgecolor=C_GOLD_DARK, markeredgewidth=0.5, zorder=6)
    ax_tree.set_xlim(seed_x - 1.4, x_axis_max + 0.1)
    ax_tree.set_ylim(-1.5, N_TIPS + 1.5)
    _blank(ax_tree)
    _title(ax_tree, 'subtree extraction')

    # ---- per-edge encoder -----------------------------------------------
    draw_tip_time_kde(ax_kde, root, TARGET_LOC, OTHER_LOC, seed_x=seed_x,
                      x_max=time_x_end, bandwidth=0.45)
    _title(ax_kde, 'per-edge encoder')
    # Inset sized so the DTW cells match the Aux (1 x 5) cells.
    ax_dtw = ax_kde.inset_axes([0.24, 0.51, 0.32, 0.32])
    dtw_right_x, dtw_mid_y = draw_dtw_outputs(ax_dtw, TARGET_LOC, OTHER_LOC)

    y_lo = ax_kde.get_ylim()[0]
    ax_kde.annotate('', xy=(time_x_end, y_lo * 0.65), xytext=(seed_x, y_lo * 0.65),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.75,
                                    mutation_scale=6))
    ax_kde.text(0.5 * (seed_x + time_x_end), y_lo * 0.95, 'Time',
                ha='center', va='top', fontsize=FS_TIME, fontweight='bold')

    # ---- per-node encoder -----------------------------------------------
    ax_conv.set_xlim(0, 12)
    ax_conv.set_ylim(0, 10)
    _blank(ax_conv)
    _title(ax_conv, 'per-node encoder')

    CELL = 0.42

    # CBLV (W x 4): m filled rows, W - m zero-padding rows.
    cblv_rows, cblv_cols, cblv_n_filled = 12, 4, 10
    cblv_w, cblv_h = cblv_cols * CELL, cblv_rows * CELL
    cblv_x0, cblv_y0 = 0.55, 2.85
    draw_cblv_grid(ax_conv, cblv_x0, cblv_y0, cblv_w, cblv_h,
                   cblv_rows, cblv_cols, cblv_n_filled)

    bracket_x = cblv_x0 + cblv_w + 0.06
    bracket_tip = 0.13
    y_top    = cblv_y0 + cblv_h
    y_split  = cblv_y0 + (cblv_rows - cblv_n_filled) * CELL
    y_bottom = cblv_y0
    ax_conv.plot([bracket_x, bracket_x + bracket_tip, bracket_x + bracket_tip, bracket_x],
                 [y_top, y_top, y_split, y_split],
                 color='#444', lw=1.4, solid_capstyle='butt')
    ax_conv.text(bracket_x + bracket_tip + 0.08, (y_top + y_split) / 2, 'm',
                 ha='left', va='center', fontsize=6, fontweight='bold', color='#222')
    ax_conv.plot([bracket_x, bracket_x + bracket_tip, bracket_x + bracket_tip, bracket_x],
                 [y_split, y_split, y_bottom, y_bottom],
                 color='#888', lw=1.4, solid_capstyle='butt')
    ax_conv.text(bracket_x + bracket_tip + 0.08, (y_split + y_bottom) / 2,
                 r'$W\!-\!m$' + '\nzero pad',
                 ha='left', va='center', fontsize=5, color='#666')
    ax_conv.text(cblv_x0 + cblv_w / 2, cblv_y0 - 0.30, r'CBLV $(W \times 4)$',
                 ha='center', va='top', fontsize=6, fontweight='bold', color='#222')
    ax_conv.text(cblv_x0 + cblv_w / 2, cblv_y0 - 0.65,
                 r'$W$ = fixed unified dimension;' ' \n' r'$m$ = subtree tips',
                 ha='center', va='top', fontsize=5, color='#666', style='italic')

    # Three CNN branches.
    branches = [(7.30, 'plain'), (5.45, 'stride'), (3.60, 'dilate')]
    br_x0, br_w, br_h, br_d = 4.10, 2.00, 1.20, 0.38
    cblv_exit = (cblv_x0 + cblv_w + 0.05, cblv_y0 + cblv_h / 2)
    for y_c, name in branches:
        draw_conv_block(ax_conv, br_x0, y_c, br_w, br_h, br_d)
        ax_conv.text(br_x0 + br_w / 2, y_c, name, ha='center', va='center',
                     fontsize=6, fontweight='bold', color='#333', zorder=5)
        _arrow(ax_conv, cblv_exit, (br_x0 - 0.04, y_c),
               rad=(y_c - cblv_exit[1]) / 14)

    # Aux (1 x 5) and its MLP branch.
    aux_n, aux_bw, aux_bh = 5, CELL, CELL
    aux_x0 = cblv_x0 + cblv_w / 2 - (aux_n * aux_bw) / 2
    aux_y0 = 1.10
    draw_feature_strip(ax_conv, aux_x0, aux_y0, aux_n, aux_bw, aux_bh, '#90C695')
    ax_conv.text(aux_x0 + aux_n * aux_bw / 2, aux_y0 - 0.30, r'Aux $(1 \times 5)$',
                 ha='center', va='top', fontsize=6, fontweight='bold', color='#222')
    ax_conv.text(aux_x0 + aux_n * aux_bw / 2, aux_y0 - 0.78,
                 'mrca depth / earliest tip / latest tip / mean tip dist / n_tips',
                 ha='center', va='top', fontsize=5, color='#666', style='italic')

    aux_box_x, aux_box_y = br_x0, aux_y0 + aux_bh / 2
    aux_box_w, aux_box_h, aux_box_d = br_w, 1.30, 0.38
    draw_conv_block(ax_conv, aux_box_x, aux_box_y, aux_box_w, aux_box_h,
                    aux_box_d, color='#90C695')
    ax_conv.text(aux_box_x + aux_box_w / 2, aux_box_y, 'AuxBranch\nMLP',
                 ha='center', va='center', fontsize=5, fontweight='bold',
                 color='#1f5933', zorder=5)
    _arrow(ax_conv, (aux_x0 + aux_n * aux_bw + 0.05, aux_y0 + aux_bh / 2),
           (aux_box_x - 0.04, aux_box_y))

    # Branches converge into the (1 x 128) node feature.
    concat_x, concat_y = 8.20, 5.45
    for y_c, _ in branches:
        _arrow(ax_conv, (br_x0 + br_w + br_d + 0.05, y_c), (concat_x, concat_y),
               color='#444', rad=(concat_y - y_c) / 14)
    ax_conv.text(br_x0 + br_w + br_d + 1.10, 7.70, r'$(1 \times 96)$',
                 ha='center', va='center', fontsize=5, color='#444', fontweight='bold')
    _arrow(ax_conv, (aux_box_x + aux_box_w + aux_box_d + 0.05, aux_box_y),
           (concat_x, concat_y), color='#444', rad=(concat_y - aux_box_y) / 14)
    ax_conv.text(aux_box_x + aux_box_w + aux_box_d + 1.10, 1.80, r'$(1 \times 32)$',
                 ha='center', va='center', fontsize=5, color='#444', fontweight='bold')

    nf_n, nf_bw, nf_bh = 7, CELL, CELL
    nf_x0 = concat_x + 0.70
    nf_y0 = concat_y - nf_bh / 2
    draw_feature_strip(ax_conv, nf_x0, nf_y0, nf_n, nf_bw, nf_bh, C_NODE_FEAT)
    _arrow(ax_conv, (concat_x, nf_y0 + nf_bh / 2), (nf_x0 - 0.04, nf_y0 + nf_bh / 2),
           color='#444')
    ax_conv.text(0.5 * (concat_x + nf_x0), nf_y0 + nf_bh / 2 + 0.22, 'concat',
                 ha='center', va='bottom', fontsize=5, fontweight='bold', color='#555')
    ax_conv.text(nf_x0 + nf_n * nf_bw / 2, nf_y0 - 0.35,
                 r'$h_{' + TARGET_LOC + r'}\ (1 \times 128)$',
                 ha='center', va='top', fontsize=6, fontweight='bold', color=C_NODE_FEAT)

    # ---- graph assemble -------------------------------------------------
    # xlim leaves room for the node-feature strips beside the diamond.
    ax_graph.set_xlim(-2, 14)
    ax_graph.set_ylim(0, 10)
    ax_graph.set_aspect('equal', adjustable='box')
    _blank(ax_graph)
    # aspect='equal' makes this axes shorter than its cell, so place the
    # title in figure coordinates level with the per-edge encoder title.
    kde_pos, graph_pos = ax_kde.get_position(), ax_graph.get_position()
    fig.text(graph_pos.x0 + 0.5 * graph_pos.width,
             kde_pos.y0 + 0.97 * kde_pos.height, 'graph assemble',
             ha='center', va='top', fontsize=FS_TITLE, fontweight='bold', color='#222')

    draw_k4_graph(ax_graph, TARGET_LOC)

    # Strip cells match the Aux and DTW cells in physical size.
    strip_n, strip_bw, strip_bh = 7, 0.592, 0.504
    strip_offsets = {
        'a': (-strip_n * strip_bw / 2,  1.05),
        'b': (-strip_n * strip_bw - 1.05, -strip_bh / 2),
        'c': (-strip_n * strip_bw / 2, -1.05 - strip_bh),
        'd': (1.05, -strip_bh / 2),
    }
    for l, (dx, dy) in strip_offsets.items():
        nx, ny = K4_POSITIONS[l]
        draw_feature_strip(ax_graph, nx + dx, ny + dy, strip_n, strip_bw,
                           strip_bh, C_NODE_FEAT)

    # Edge-feature strip beside the Loc_a-Loc_b edge.
    (ax_x, ax_y), (nb_x, nb_y) = K4_POSITIONS[TARGET_LOC], K4_POSITIONS['b']
    edge_n, edge_bw, edge_bh = 3, 0.592, 0.504
    edge_x0 = 0.5 * (ax_x + nb_x) - edge_n * edge_bw - 0.55
    edge_y0 = 0.5 * (ax_y + nb_y) + 0.30
    draw_feature_strip(ax_graph, edge_x0, edge_y0, edge_n, edge_bw, edge_bh, C_EDGE_FEAT)

    # ---- cross-cell arrows ----------------------------------------------
    a_strip_x = K4_POSITIONS['a'][0] + strip_offsets['a'][0] + strip_n * strip_bw / 2
    a_strip_y = K4_POSITIONS['a'][1] + strip_offsets['a'][1] + strip_bh
    for xy_a, coords_a, xy_b, color, rad in [
        ((nf_x0 + nf_bw * 0.5, nf_y0), ax_conv.transData,
         (a_strip_x, a_strip_y + 0.05), C_NODE_FEAT, -0.15),
        ((dtw_right_x, dtw_mid_y), ax_dtw.transData,
         (edge_x0 + edge_n * edge_bw / 2, edge_y0 + edge_bh + 0.05), C_EDGE_FEAT, -0.30),
    ]:
        fig.add_artist(ConnectionPatch(
            xyA=xy_a, coordsA=coords_a, xyB=xy_b, coordsB=ax_graph.transData,
            arrowstyle='->', mutation_scale=7, lw=0.65, color=color, zorder=20,
            connectionstyle=f'arc3,rad={rad}'))

    return ax_tree


# -- Panel c: inference ------------------------------------------------------

# Mock predictions: Loc_a (the seed) has the highest R0 and exports (SSS > 0);
# the others have moderate R0 and import.
PREDS = {
    'a': dict(R0=2.31, R0_lo=1.95, R0_hi=2.68,
              gamma=0.045, gamma_lo=0.035, gamma_hi=0.058,
              SSS=+0.42, SSS_lo=+0.28, SSS_hi=+0.55),
    'b': dict(R0=1.84, R0_lo=1.51, R0_hi=2.19,
              gamma=0.062, gamma_lo=0.048, gamma_hi=0.078,
              SSS=-0.18, SSS_lo=-0.32, SSS_hi=-0.05),
    'c': dict(R0=1.97, R0_lo=1.62, R0_hi=2.34,
              gamma=0.058, gamma_lo=0.045, gamma_hi=0.073,
              SSS=-0.11, SSS_lo=-0.24, SSS_hi=+0.02),
    'd': dict(R0=1.62, R0_lo=1.30, R0_hi=1.98,
              gamma=0.073, gamma_lo=0.058, gamma_hi=0.089,
              SSS=-0.13, SSS_lo=-0.27, SSS_hi=+0.01),
}
PRED_ANCESTRAL = 'a'
PRED_CP_SET = ['a', 'b']

# Message-passing attention: incoming weights to 'a' spread wide so the
# gather-arrow widths are clearly distinct.
MP_ATTENTION = {
    ('a', 'b'): 0.45, ('b', 'a'): 0.08,
    ('a', 'c'): 0.30, ('c', 'a'): 0.45,
    ('a', 'd'): 0.20, ('d', 'a'): 0.28,
    ('b', 'c'): 0.55, ('c', 'b'): 0.35,
    ('b', 'd'): 0.40, ('d', 'b'): 0.65,
    ('c', 'd'): 0.25, ('d', 'c'): 0.45,
}

C_PRED_VAL    = '#1f3550'   # prediction values
C_PRED_INT    = '#7E57C2'   # CP intervals
C_MLP_EDGE    = '#7a5310'
C_ALPHA       = '#3a1f6b'   # attention gather arrows
C_MSG_EDGE    = '#444'
C_PLANE_TOP   = '#EDF1F6'   # plate face carrying the K4
C_PLANE_FRONT = '#C9D1DC'   # plate top edge
C_PLANE_SIDE  = '#B6BFCD'   # plate front edge
C_PLANE_EDGE  = '#7d8794'

PLATE_THICK, PLATE_HEIGHT, PLATE_DEPTH_X, PLATE_DEPTH_Y = 0.6, 8.0, 2.6, 1.0


def _draw_pred_card(ax, x_center, y_center, loc):
    """Draw a bordered card of R_0, gamma and SSS point estimates with their 95% CP intervals."""
    p = PREDS[loc]
    card_w, card_h = 4.80, 2.00
    ax.add_patch(Rectangle((x_center - card_w / 2, y_center - card_h / 2),
                           card_w, card_h, facecolor='white',
                           edgecolor='#7a7a7a', lw=0.4, zorder=4))
    val_gap = 0.08
    sign = '+' if p['SSS'] > 0 else '−'
    rows = [
        (y_center + 0.60, fr'$R_0\!=\!{p["R0"]:.2f}$',
         fr'$[{p["R0_lo"]:.2f},\,{p["R0_hi"]:.2f}]$'),
        (y_center, fr'$\gamma\!=\!{p["gamma"]:.3f}$',
         fr'$[{p["gamma_lo"]:.3f},\,{p["gamma_hi"]:.3f}]$'),
        (y_center - 0.60, fr'$\mathrm{{SSS}}\!=\!{sign}{abs(p["SSS"]):.2f}$',
         fr'$[{p["SSS_lo"]:+.2f},\,{p["SSS_hi"]:+.2f}]$'),
    ]
    for y, val_str, ci_str in rows:
        ax.text(x_center - val_gap, y, val_str, ha='right', va='center',
                fontsize=6, fontweight='bold', color=C_PRED_VAL, zorder=6)
        ax.text(x_center + val_gap, y, ci_str, ha='left', va='center',
                fontsize=5, style='italic', color=C_PRED_INT, zorder=6)


def _draw_3d_plate(ax, x_left, y_bottom):
    """
    Draw a thin slab seen edge-on, its large face receding to the upper right.

    Returns (origin, u_vec, v_vec), the face's affine basis: face coordinates
    (u, v) in [0, 1]^2 map to origin + u * u_vec + v * v_vec.
    """
    t, h, dx, dy = PLATE_THICK, PLATE_HEIGHT, PLATE_DEPTH_X, PLATE_DEPTH_Y
    top_pts = [(x_left, y_bottom + h), (x_left + t, y_bottom + h),
               (x_left + t + dx, y_bottom + h + dy), (x_left + dx, y_bottom + h + dy)]
    face_pts = [(x_left + t, y_bottom), (x_left + t + dx, y_bottom + dy),
                (x_left + t + dx, y_bottom + dy + h), (x_left + t, y_bottom + h)]
    front_pts = [(x_left, y_bottom), (x_left + t, y_bottom),
                 (x_left + t, y_bottom + h), (x_left, y_bottom + h)]
    for pts, fc, lw, z in [(top_pts, C_PLANE_FRONT, 1.0, 0),
                           (face_pts, C_PLANE_TOP, 1.2, 1),
                           (front_pts, C_PLANE_SIDE, 1.0, 2)]:
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                             edgecolor=C_PLANE_EDGE, lw=lw, zorder=z))
    return (x_left + t, y_bottom), (dx, dy), (0.0, h)


def _draw_layer(ax, x_left, y_bottom, label):
    """
    Draw one message-passing layer: a plate with a K4 of h_a..h_d on its face.

    Nodes are grey ellipses foreshortened to the face; edges use MP_ATTENTION.
    Returns each node's centre in data coordinates.
    """
    origin, u_vec, v_vec = _draw_3d_plate(ax, x_left, y_bottom)
    cu, cv, r = 0.5, 0.5, 0.30
    uv = {'a': (cu, cv + r), 'b': (cu - r, cv), 'c': (cu, cv - r), 'd': (cu + r, cv)}
    pos = {l: (origin[0] + u * u_vec[0] + v * v_vec[0],
               origin[1] + u * u_vec[1] + v * v_vec[1]) for l, (u, v) in uv.items()}

    draw_directed_edges(ax, pos, MP_ATTENTION, offset=0.06, shrink=4,
                        lw_scale=0.55, mutation_scale=3.5, zorder=3)
    node_rx, node_ry = 0.10 * abs(u_vec[0]), 0.10 * abs(v_vec[1])
    for l in LOCS:
        ax.add_patch(Ellipse(pos[l], width=2 * node_rx, height=2 * node_ry,
                             facecolor=C_DIM, edgecolor='#5a5a5a', lw=1.6, zorder=5))
        ax.text(*pos[l], f'$h_{l}$', ha='center', va='center', fontsize=5,
                fontweight='bold', color='white', zorder=6)

    ax.text(x_left + PLATE_THICK / 2, y_bottom + 0.20, label, ha='center',
            va='bottom', fontsize=6, color='#666', style='italic')
    return pos


def _draw_panel_c_mp(ax):
    """
    Draw the message-passing layer: plate l -> (h_a^(l), agg_a) -> concat -> plate l+1.

    Neighbours b, c, d feed agg_a through purple arrows whose widths encode
    their incoming attention; STEPHY's K4 has no self-loop, so the sum runs
    over u != a. Loc_a's own state passes through ungated and is concatenated
    with agg_a (stephy/model.py).
    """
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 12.5)
    ax.set_aspect('equal', adjustable='box')
    _blank(ax)
    _title(ax, 'message passing layer', y=1.02, va='bottom')

    pl_y = 2.25
    pos_l = _draw_layer(ax, 0.4, pl_y, r'$\ell$')

    msg_x, msg_w, msg_h = 5.0, 1.8, 1.5
    plate_mid_y = pl_y + PLATE_HEIGHT / 2
    self_xy = (msg_x, plate_mid_y + 2.50)
    agg_xy = (msg_x, plate_mid_y - 1.75)
    for (cx, cy), label in [(self_xy, r'$h_{a}^{(\ell)}$'),
                            (agg_xy, r'$\mathrm{agg}_{a}$')]:
        ax.add_patch(Ellipse((cx, cy), msg_w, msg_h, facecolor='#FFFFFF',
                             edgecolor=C_MSG_EDGE, lw=0.5, zorder=4))
        ax.text(cx, cy, label, ha='center', va='center', fontsize=8,
                color='#222', zorder=5)
    ax.text(agg_xy[0], agg_xy[1] - msg_h / 2 - 0.30,
            r'$\sum_{u \neq a}\, \alpha_{u \to a}\, h_{u}^{(\ell)}$',
            ha='center', va='top', fontsize=6, color='#444', style='italic')

    agg_left = (agg_xy[0] - msg_w / 2 - 0.02, agg_xy[1])
    self_left = (self_xy[0] - msg_w / 2 - 0.02, self_xy[1])
    for l, rad in [('b', -0.20), ('c', -0.04), ('d', -0.20)]:
        w = MP_ATTENTION[(l, 'a')]
        ax.add_patch(FancyArrowPatch(
            pos_l[l], agg_left, connectionstyle=f'arc3,rad={rad}',
            shrinkA=4.5, shrinkB=1.2, arrowstyle='-|>', color=C_ALPHA,
            lw=0.35 + (w - 0.08) / 0.82 * 2.0, mutation_scale=4, zorder=4))
    ax.add_patch(FancyArrowPatch(
        pos_l['a'], self_left, connectionstyle='arc3,rad=-0.18',
        shrinkA=4.5, shrinkB=1.2, arrowstyle='-|>', color='#444',
        lw=0.45, mutation_scale=3.5, zorder=4))

    pos_r = _draw_layer(ax, 7.0, pl_y, r'$\ell\!+\!1$')

    concat_x, concat_y = self_xy[0], 0.5 * (self_xy[1] + agg_xy[1])
    ax.text(concat_x, concat_y, 'concat', ha='center', va='center',
            fontsize=6, fontweight='bold', color='#333', style='italic', zorder=6)
    for start, end in [((self_xy[0], self_xy[1] - msg_h / 2), (concat_x, concat_y + 0.45)),
                       ((agg_xy[0], agg_xy[1] + msg_h / 2), (concat_x, concat_y - 0.45))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', color='#444',
                                     lw=0.45, mutation_scale=4, zorder=4))
    ax.add_patch(FancyArrowPatch(
        (concat_x + 0.55, concat_y), pos_r['a'], connectionstyle='arc3,rad=-0.18',
        arrowstyle='-|>', color='#444', lw=0.5, mutation_scale=4.5,
        shrinkA=0.5, shrinkB=6, zorder=4))


def _draw_panel_c_mlp(ax):
    """Draw the MLP head: four blocks 256 -> 128 -> 64 -> 32 joined by ReLU arrows."""
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12.5)
    ax.set_aspect('equal', adjustable='box')
    _blank(ax)
    _title(ax, 'mlp layer', y=1.02, va='bottom')

    dims    = [256, 128, 64, 32]
    heights = [5.8, 4.5, 3.4, 2.4]
    blk_w, blk_dep, spacing, centers_y = 0.55, 0.22, 1.20, 6.25
    x_left0 = (6.0 - (len(dims) * blk_w + (len(dims) - 1) * spacing)) / 2

    lefts = [x_left0 + i * (blk_w + spacing) for i in range(len(dims))]
    for x_left, d, h in zip(lefts, dims, heights):
        draw_conv_block(ax, x_left, centers_y, blk_w, h, blk_dep,
                        edgecolor=C_MLP_EDGE)
        ax.text(x_left + blk_w / 2, centers_y, str(d), ha='center', va='center',
                fontsize=6, fontweight='bold', color='#222', zorder=6)

    for i in range(len(dims) - 1):
        x_src = lefts[i] + blk_w + blk_dep + 0.04
        x_dst = lefts[i + 1] - 0.04
        ax.annotate('', xy=(x_dst, centers_y), xytext=(x_src, centers_y),
                    arrowprops=dict(arrowstyle='->', mutation_scale=2.5,
                                    lw=0.4, color='#444'))
        ax.text(0.5 * (x_src + x_dst), centers_y + max(heights) / 2 + 0.22,
                'ReLU', ha='center', va='bottom', fontsize=5, color='#666',
                style='italic')


def _draw_panel_c_predictions(ax):
    """
    Draw the per-location predictions: the panel-b K4 with a card beside each
    node and, below, the predicted ancestral state with its 95% CP set.
    """
    # Same limits as the graph-assemble cell so the two K4s align in x.
    ax.set_xlim(-2, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal', adjustable='box')
    _blank(ax)
    _title(ax, 'per-location predictions', y=1.02, va='bottom')

    draw_k4_graph(ax, target_loc='')
    card_offsets = {'a': (0.0, +1.40), 'b': (-2.90, 0.0),
                    'c': (0.0, -1.40), 'd': (+2.90, 0.0)}
    for loc, (dx, dy) in card_offsets.items():
        nx, ny = K4_POSITIONS[loc]
        _draw_pred_card(ax, nx + dx, ny + dy, loc)

    # Readout styled like panel a's 'Loc_a, ancestral state' caption.
    star_x, star_y = 3.30, -0.6
    ax.plot(star_x, star_y, marker='*', markersize=7, markerfacecolor='#FFD43B',
            markeredgecolor=C_GOLD_DARK, markeredgewidth=0.35, linestyle='None',
            zorder=8, clip_on=False)
    cp_set = ', '.join(f'Loc$_{{{l}}}$' for l in PRED_CP_SET)
    ax.text(star_x + 0.45, star_y,
            f'ancestral state: Loc$_{{{PRED_ANCESTRAL}}}$  [{cp_set}]',
            ha='left', va='center', fontsize=FS_QUESTION, fontweight='bold',
            style='italic', color=C_GOLD_DARK, zorder=8, clip_on=False)


def _draw_panel_c(fig, spec):
    """
    Draw panel c (message passing -> MLP -> predictions) and return the
    message-passing axes.

    Under aspect='equal' the width ratios 11 : 6 : 20 give the three main
    cells equal physical height; the predictions cell spans the right half so
    its K4 aligns with panel b's. The first column is an empty spacer.
    """
    gs = spec.subgridspec(1, 6, width_ratios=[2.5, 11.0, 1.5, 6.0, 1.5, 20.0],
                          wspace=0.04)
    ax_mp = fig.add_subplot(gs[0, 1])
    _draw_panel_c_mp(ax_mp)
    _draw_arrow_block(fig.add_subplot(gs[0, 2]))
    _draw_panel_c_mlp(fig.add_subplot(gs[0, 3]))
    _draw_arrow_block(fig.add_subplot(gs[0, 4]))
    _draw_panel_c_predictions(fig.add_subplot(gs[0, 5]))
    return ax_mp


# -- Main --------------------------------------------------------------------

def main():
    """Render Figure 1 and save fig1.pdf and fig1.png next to this script."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=11,
                        help='RNG seed for the panel-b tree')
    args = parser.parse_args()

    fig = plt.figure(figsize=(8.42, 11.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[13.0, 26.0, 13.5], hspace=0.04)
    ax_a = _draw_panel_a(fig, outer[0])
    ax_b = _draw_panel_b(fig, outer[1], args.seed)
    ax_c = _draw_panel_c(fig, outer[2])

    # All three letters share panel b's left edge, each at its panel's top.
    label_x = ax_b.get_position().x0
    for letter, ax in [('a', ax_a), ('b', ax_b), ('c', ax_c)]:
        fig.text(label_x, ax.get_position().y1, letter,
                 fontsize=14, fontweight='bold', va='top', ha='left')

    stem = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    print(f'Saved: {stem}.pdf')
    fig.savefig(stem + '.png', bbox_inches='tight', dpi=600)
    print(f'Saved: {stem}.png')


if __name__ == '__main__':
    main()
