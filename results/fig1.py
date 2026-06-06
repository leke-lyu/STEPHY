#!/usr/bin/env python3
"""Merged conceptual figure — STEPHY end-to-end overview (self-contained).

Three panels stacked vertically:

Panel (a) — DATA GENERATION
    Stochastic 4-location SIR engine + migration K4 -> simulated 12-tip
    outbreak tree. Two sub-panels separated by a 'stochastic simulation /
    Gillespie algorithm' arrow block.

Panel (b) — STEPHY ENCODER PIPELINE (2x2 GRID)
    Top-left:  subtree extraction — full simulated tree with Loc_a's
               virtual subtree highlighted in red, neighbouring clade
               collapsed into a dashed grey triangle.
    Top-right: per-node encoder — CBLV (W x 4) -> 3-branch CNN (plain /
               stride / dilate) -> (1 x 96), plus Aux (1 x 5) ->
               AuxBranch MLP -> (1 x 32), concat -> (1 x 128) node
               feature.
    Bot-left:  per-edge encoder — two tip-time densities (rho_a red,
               rho_b grey) with a DTW output inset naming the edge
               feature (distance / mean lag / std lag).
    Bot-right: graph assemble — directed K4 with per-node and per-edge
               feature strips painted onto Loc_a and the Loc_a-Loc_b
               edge respectively.

Panel (c) — INFERENCE
    LGAT message-passing layer -> MLP head (256 -> 128 -> 64 -> 32) ->
    per-location predictions K4 carrying (R_e, mu, SSS) cards plus the
    predicted index case + 95% CP set.

This file is fully self-contained — all drawing helpers (engine, 12-tip
tree, 50-tip tree generation + encoder + K4 + plates) live below in
clearly labelled SECTIONs. Saves fig1.pdf next to this script.

Usage:
    python3 fig1.py
    python3 fig1.py --seed 7
"""
import argparse
import math
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import (
    Circle, ConnectionPatch, Ellipse, FancyArrowPatch, Polygon, Rectangle,
)


# Publication-quality matplotlib defaults.
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset':  'stixsans',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.linewidth':    0.8,
})


# =====================================================================
# SECTION A: Engine helpers (4-location SIR + migration K4).
# Originally simulation_engine_4loc.py.
# =====================================================================

# Diamond layout: 4 locations placed at N, E, S, W of a diamond. Each
# location's I compartment is the "graph node"; the per-location stack
# (S above I; R and Sampled below I) hangs off that node.
LOC_R = 5.0
ANGLES = {'a': 90, 'b': 0, 'c': 270, 'd': 180}

DY_S  =  3.4
DY_RS = -3.4
DX_RS =  1.30

I_POS, S_POS, R_POS, SAMP_POS = {}, {}, {}, {}
for _loc, _angle in ANGLES.items():
    _rad = math.radians(_angle)
    _cx, _cy = LOC_R * math.cos(_rad), LOC_R * math.sin(_rad)
    I_POS[_loc]    = (_cx,            _cy)
    S_POS[_loc]    = (_cx,            _cy + DY_S)
    R_POS[_loc]    = (_cx - DX_RS,    _cy + DY_RS)
    SAMP_POS[_loc] = (_cx + DX_RS,    _cy + DY_RS)

MIG_PAIRS = [
    ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a'),
    ('a', 'c'),
    ('b', 'd'),
]

COMP_RADIUS = 0.75
I_RADIUS    = 1.15

# Engine font hierarchy. Sized for Nature print: ~5–8 pt at final
# 7.2"-wide output. See figsize at bottom of file.
FS_NODE_I    = 8
FS_COMP      = 6
FS_COMP_SMP  = 5
FS_EVENT     = 6
FS_MIG       = 7


def draw_compartment(ax, pos, color, label, fontsize=FS_COMP,
                     radius=COMP_RADIUS, linewidth=0.7):
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


def draw_arrow(ax, p1, p2, color, lw=0.55):
    """Single straight directed arrow from p1 to p2."""
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle='-|>', mutation_scale=4,
        color=color, lw=lw, zorder=2,
    ))


def draw_engine(ax):
    """Draw the diamond-layout 4-location SIR engine onto `ax`.

    Renders the full engine: per-location bubbles, S/I/R/Sampled
    compartments, within-location infection / removal / sampling
    arrows, event-type labels, and the K4 migration edges between
    the four I-node "graph nodes".
    """
    C_S    = '#2E7DBF'
    C_I    = '#D04F4F'
    C_R    = '#888888'
    C_SAMP = '#3CB371'
    C_MIG  = '#7B4FB4'

    for loc in 'abcd':
        cx, cy = I_POS[loc]
        ax.add_patch(Ellipse((cx, cy), 4.2, 8.4,
                             facecolor='none', edgecolor='#aaaaaa',
                             linewidth=0.6, linestyle=':', zorder=1))

    for loc in 'abcd':
        draw_compartment(ax, I_POS[loc], C_I, f'$I_{loc}$',
                         fontsize=FS_NODE_I, radius=I_RADIUS,
                         linewidth=1.1)
        draw_compartment(ax, S_POS[loc],    C_S,    f'$S_{loc}$',
                         fontsize=FS_COMP)
        draw_compartment(ax, R_POS[loc],    C_R,    f'$R_{loc}$',
                         fontsize=FS_COMP)
        draw_compartment(ax, SAMP_POS[loc], C_SAMP,
                         f'$\\mathrm{{Smp}}_{loc}$',
                         fontsize=FS_COMP_SMP)

        s_end, i_top = edge_endpoints(S_POS[loc], I_POS[loc],
                                      r1=COMP_RADIUS, r2=I_RADIUS)
        draw_arrow(ax, s_end, i_top, C_S, lw=0.6)

        i_end_r, r_in = edge_endpoints(I_POS[loc], R_POS[loc],
                                       r1=I_RADIUS, r2=COMP_RADIUS)
        draw_arrow(ax, i_end_r, r_in, C_R, lw=0.5)

        i_end_s, s_in = edge_endpoints(I_POS[loc], SAMP_POS[loc],
                                       r1=I_RADIUS, r2=COMP_RADIUS)
        draw_arrow(ax, i_end_s, s_in, C_SAMP, lw=0.5)

    # K4 migration edges (two overlapping single-headed arrows per
    # edge so both heads use the same glyph).
    for loc1, loc2 in MIG_PAIRS:
        p1, p2 = edge_endpoints(I_POS[loc1], I_POS[loc2],
                                r1=I_RADIUS, r2=I_RADIUS)
        for start, end in [(p1, p2), (p2, p1)]:
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle='-|>', mutation_scale=8,
                connectionstyle='arc3,rad=0',
                color=C_MIG, lw=1.2, linestyle='-', zorder=2,
            ))


    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-9.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')


# =====================================================================
# SECTION B: 12-tip conceptual tree helpers.
# Originally simulated_tree_concept.py.
# =====================================================================

# Tip y-coordinates (evenly spaced).
TIP_Y = [0.4 + 0.6 * i for i in range(12)]

TIP_END_X = [6.0, 8.0, 7.0, 5.0, 8.0, 7.5, 6.0, 8.0, 7.0, 5.0, 8.0, 6.5]
TIP_TYPES = ['removed', 'sampled', 'removed', 'removed',
             'sampled', 'removed', 'removed', 'sampled',
             'removed', 'removed', 'sampled', 'removed']

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

SEED_X_T12 = -0.3   # 12-tip tree's seed x (renamed to avoid collision
                    # with the 50-tip helper's local SEED_X).

# 12-tip-tree font hierarchy. Names suffixed `_T12` where the 50-tip
# section (later in this file) defines a same-named constant at a
# different size.
FS_EVENT_TAG     = 5
FS_MIG_TAG       = 5
FS_T12_INDEX_CAP = 6
FS_T12_TIME      = 6

LW_BRANCH    = 1.2
MS_INF       = 4.5
MS_SAMP      = 5.0
MS_REM       = 4.7
MS_MIG       = 4.0
MS_INDEX     = 8.0

NODE_LOCATIONS = {
    'root_inf': 'a',
    'L3_join':  'a',
    'SC1':      'b',
    'SC2':      'a',
    'SC3':      'd',
    'pair01':   'b',
    'pair23':   'c',
    'pair45':   'a',
    'pair67':   'c',
    'pair89':   'd',
    'pair1011': 'd',
}
TIP_LOCATIONS = ['b', 'b', 'c', 'c', 'a', 'a',
                 'c', 'c', 'd', 'd', 'd', 'd']

MIGRATIONS_INFO = [
    (0.5 * (NODES['root_inf'][0] + NODES['SC1'][0]),    NODES['SC1'][1],
     'a', 'b'),
    (0.5 * (NODES['SC1'][0]      + NODES['pair23'][0]), NODES['pair23'][1],
     'b', 'c'),
    (0.5 * (NODES['L3_join'][0]  + NODES['SC3'][0]),    NODES['SC3'][1],
     'a', 'd'),
    (0.5 * (NODES['SC2'][0]      + NODES['pair67'][0]), NODES['pair67'][1],
     'a', 'c'),
]


def draw_tree(ax):
    """Draw the conceptual 12-tip outbreak-tree onto `ax`.

    Hand-laid 12-tip topology with each event color-coded: infection
    (blue), sampling (green tip dot), removal (grey tip X), migration
    (purple diamond, with `Loc x -> Loc y` direction tag). The seed
    sits at Loc_a (gold star). Self-contained: sets its own
    xlim/ylim/aspect/ticks/time-arrow.
    """
    C_I    = "#D04F4F"
    C_INF  = "#2E7DBF"
    C_REM  = "#888888"
    C_SAMP = "#3CB371"
    C_MIG  = "#7B4FB4"

    root_x, root_y = NODES['root_inf']
    ax.plot([SEED_X_T12, root_x], [root_y, root_y],
            color=C_I, lw=LW_BRANCH, solid_capstyle='round', zorder=1)

    for parent, kids in CHILDREN.items():
        px = NODES[parent][0]
        c1_y, c2_y = kids[0][1], kids[1][1]
        ax.plot([px, px], [c1_y, c2_y], color=C_I, lw=LW_BRANCH, zorder=1)
        for child_label, child_y in kids:
            if isinstance(child_label, int):
                child_x = TIP_END_X[child_label]
            else:
                child_x = NODES[child_label][0]
            ax.plot([px, child_x], [child_y, child_y],
                    color=C_I, lw=LW_BRANCH, solid_capstyle='round',
                    zorder=1)

    for label, (x, y) in NODES.items():
        ax.plot(x, y, 'o', color=C_INF, markersize=MS_INF, zorder=3,
                markeredgecolor='white', markeredgewidth=0.4)
        loc = NODE_LOCATIONS[label]
        ax.text(x + 0.24, y, loc, fontsize=FS_EVENT_TAG,
                fontweight='bold', style='italic',
                color=C_INF, ha='left', va='center', zorder=4)

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
        loc = TIP_LOCATIONS[i]
        ax.text(x + 0.26, y, loc, fontsize=FS_EVENT_TAG,
                fontweight='bold', style='italic',
                color=label_color, ha='left', va='center', zorder=4)

    for mx, my, from_loc, to_loc in MIGRATIONS_INFO:
        ax.plot([mx, mx], [my - 0.26, my + 0.26],
                color=C_MIG, lw=0.6, linestyle='--', zorder=2)
        ax.plot(mx, my, marker='D', color=C_MIG, markersize=MS_MIG,
                zorder=3, markeredgecolor='white', markeredgewidth=0.4)
        ax.text(mx, my + 0.48,
                f'{from_loc} $\\rightarrow$ {to_loc}',
                fontsize=FS_MIG_TAG, fontweight='bold', style='italic',
                color=C_MIG, ha='center', va='bottom', zorder=4)

    ax.plot(SEED_X_T12, root_y, marker='*', markersize=MS_INDEX,
            color='#FFC107', markeredgecolor='#8A6500',
            markeredgewidth=0.4, zorder=3)
    ax.text(SEED_X_T12, root_y + 0.65, 'source case',
            ha='center', va='bottom',
            fontsize=FS_T12_INDEX_CAP, fontweight='bold', style='italic',
            color='#8A6500')

    ax.set_xlim(-1.6, 9.2)
    ax.set_ylim(-1.35, 9.2)
    ax.set_aspect('equal')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.annotate('', xy=(8.7, -0.55), xytext=(SEED_X_T12, -0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
    ax.text(0.5 * (SEED_X_T12 + 8.7), -1.05, 'Time',
            ha='center', fontsize=FS_T12_TIME, fontweight='bold')


# =====================================================================
# SECTION C: 50-tip tree generation, KDE, encoder, K4 helpers.
# Originally simulated_tree_concept_50.py.
# =====================================================================

# Palette.
C_HL    = "#D04F4F"
C_DIM   = "#BFBFBF"
C_SAMP  = "#3CB371"
C_GOLD  = "#FFC107"
C_GOLD_DARK = "#8A6500"

# 50-tip-tree font hierarchy (used throughout the bottom panels).
FS_INDEX_CAP = 7
FS_TIME      = 7
FS_CAPTION   = 7
FS_KDE_HDR   = 7


# Tree node classes.
class TipNode:
    __slots__ = ('id', 'end_x', 'sampled', 'y', 'location', 'in_induced',
                 'start_x', 'in_dashed')

    def __init__(self, node_id, end_x, sampled, start_x=0.0):
        self.id = node_id
        self.end_x = end_x
        self.sampled = sampled
        self.start_x = start_x
        self.y = None
        self.location = None
        self.in_induced = False
        self.in_dashed = False


class IntNode:
    __slots__ = ('id', 'x', 'children', 'y', 'location',
                 'is_migration', 'mig_from', 'mig_to', 'in_induced',
                 'in_dashed')

    def __init__(self, node_id, x, children):
        self.id = node_id
        self.x = x
        self.children = children
        self.y = None
        self.location = None
        self.is_migration = False
        self.mig_from = None
        self.mig_to = None
        self.in_induced = False
        self.in_dashed = False


def gen_tree(n_tips, T, rng, sample_frac=0.55):
    """Recursively build a binary tree with `n_tips` leaves over time
    horizon T. Sampled tip end times are spread across the latter half
    of each lineage's lifespan."""
    next_id = [0]

    def _new_id():
        i = next_id[0]
        next_id[0] += 1
        return i

    def _build(start_x, n_leaves):
        if n_leaves == 1:
            sampled = rng.random() < sample_frac
            if sampled:
                end_x = start_x + rng.uniform(0.50, 1.00) * (T - start_x)
            else:
                end_x = start_x + rng.uniform(0.15, 0.65) * (T - start_x)
            return TipNode(_new_id(), end_x, sampled, start_x=start_x)
        delta = rng.uniform(0.10, 0.45) * (T - start_x)
        split_x = start_x + delta
        m = int(rng.integers(1, n_leaves))
        left = _build(split_x, m)
        right = _build(split_x, n_leaves - m)
        return IntNode(_new_id(), split_x, [left, right])

    return _build(0.0, n_tips)


def _count_tips(node):
    """Return the number of TipNodes in `node`'s subtree."""
    if isinstance(node, TipNode):
        return 1
    return sum(_count_tips(c) for c in node.children)


def ladderize(node, smaller_first=True):
    """In-place ladderize: at every internal, sort children by tip count."""
    if isinstance(node, TipNode):
        return
    decorated = sorted(
        ((_count_tips(c), c) for c in node.children),
        key=lambda t: t[0], reverse=not smaller_first,
    )
    node.children = [c for _, c in decorated]
    for c in node.children:
        ladderize(c, smaller_first)


def assign_y(root):
    """Post-order: tips get sequential integer y, internals = midpoint."""
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


def _collect_internals(root):
    """Return every IntNode in `root`'s subtree (pre-order)."""
    out = []
    def _walk(node):
        if isinstance(node, IntNode):
            out.append(node)
            for c in node.children:
                _walk(c)
    _walk(root)
    return out


def _collect_subclade_ids(root):
    """Return `id(...)` for every node in `root`'s subtree."""
    out = set()
    def _walk(node):
        out.add(id(node))
        if isinstance(node, IntNode):
            for c in node.children:
                _walk(c)
    _walk(root)
    return out


def assign_locations(root, rng, target_loc='a', initial_loc='b',
                     a_subclade_target=32,
                     a_exit_migrations=4,
                     n_other_migrations=3):
    """Designed location assignment guaranteeing non-monophyletic
    target_loc with MRCA != root."""
    LOCS = ['a', 'b', 'c', 'd']

    internals = _collect_internals(root)
    non_root = [n for n in internals if n is not root]

    deviations = [abs(_count_tips(n) - a_subclade_target) for n in non_root]
    min_dev = min(deviations)
    candidates = [n for n, d in zip(non_root, deviations) if d == min_dev]
    a_clade_root = candidates[int(rng.integers(len(candidates)))]
    a_clade_root.is_migration = True

    a_subclade_internals = []
    def _collect_a_internals(n):
        if isinstance(n, IntNode):
            a_subclade_internals.append(n)
            for c in n.children:
                _collect_a_internals(c)
    for c in a_clade_root.children:
        _collect_a_internals(c)

    preferred = [n for n in a_subclade_internals
                 if 2 <= _count_tips(n) <= 5]
    if len(preferred) < a_exit_migrations:
        preferred = a_subclade_internals
    n_exit = min(a_exit_migrations, len(preferred))
    if n_exit > 0:
        chosen = rng.choice(len(preferred), size=n_exit, replace=False)
        for idx in chosen:
            preferred[int(idx)].is_migration = True

    a_subclade_ids = _collect_subclade_ids(a_clade_root)
    other_internals = [
        n for n in non_root
        if n is not a_clade_root and id(n) not in a_subclade_ids
    ]
    n_other = min(n_other_migrations, len(other_internals))
    if n_other > 0:
        chosen = rng.choice(len(other_internals), size=n_other,
                            replace=False)
        for idx in chosen:
            other_internals[int(idx)].is_migration = True

    def _assign(node, parent_loc):
        if isinstance(node, TipNode):
            node.location = parent_loc
            return
        if node is a_clade_root:
            node.location = target_loc
            node.mig_from, node.mig_to = parent_loc, target_loc
        elif node.is_migration:
            choices = [l for l in LOCS if l != parent_loc and l != target_loc]
            new_loc = (choices[int(rng.integers(len(choices)))]
                       if choices else parent_loc)
            node.location = new_loc
            node.mig_from, node.mig_to = parent_loc, new_loc
        else:
            node.location = parent_loc
        for c in node.children:
            _assign(c, node.location)

    _assign(root, initial_loc)
    return a_clade_root


def find_mrca(root, target_loc, sampled_only=True):
    """Return the deepest internal node containing every target_loc tip."""
    def _has_target(n):
        if isinstance(n, TipNode):
            return n.location == target_loc and (not sampled_only or n.sampled)
        return any(_has_target(c) for c in n.children)

    node = root
    while isinstance(node, IntNode):
        target_kids = [c for c in node.children if _has_target(c)]
        if len(target_kids) == 1:
            node = target_kids[0]
        else:
            break
    return node


def rebias_loc_a_tip_times(root, target_loc, T, rng,
                           mu_frac=0.55, sigma_frac=0.18, tries=40):
    """Resample target_loc tip end_x from a truncated Gaussian to give
    a bell-shaped tip-time distribution."""
    t_mu = T * mu_frac
    t_sigma = T * sigma_frac

    def _walk(n):
        if isinstance(n, TipNode):
            if n.location == target_loc and n.sampled:
                low = n.start_x + 0.05
                if low >= T:
                    return
                for _ in range(tries):
                    candidate = rng.normal(t_mu, t_sigma)
                    if low < candidate < T:
                        n.end_x = candidate
                        return
                n.end_x = min(T - 0.01, max(low, t_mu))
        elif isinstance(n, IntNode):
            for c in n.children:
                _walk(c)
    _walk(root)


def mark_induced_subtree(root, mrca, target_loc='a', sampled_only=True):
    """Set in_induced=True for nodes in the MRCA-rooted subtree that lie
    on the path to a target_loc tip."""
    def _init(node):
        node.in_induced = False
        if isinstance(node, IntNode):
            for c in node.children:
                _init(c)
    _init(root)

    def _walk(node):
        if isinstance(node, TipNode):
            in_sub = (node.location == target_loc and
                      (not sampled_only or node.sampled))
            node.in_induced = in_sub
            return in_sub
        any_in = False
        for c in node.children:
            if _walk(c):
                any_in = True
        node.in_induced = any_in
        return any_in

    _walk(mrca)


def _subtree_tips(node):
    """Yield every TipNode in `node`'s subtree."""
    if isinstance(node, TipNode):
        yield node
    else:
        for c in node.children:
            yield from _subtree_tips(c)


def find_below_loca_clade(root, mrca):
    """Pick the internal node whose subtree's tips sit immediately
    below the MRCA's lowest tip y."""
    a_tips = list(_subtree_tips(mrca))
    y_min_a = min(t.y for t in a_tips)

    candidates = []

    def _walk(n):
        if isinstance(n, IntNode):
            sub_tips = list(_subtree_tips(n))
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
    """Set in_dashed=True for every node in dashed_root's subtree."""
    if dashed_root is None:
        return
    def _walk(n):
        n.in_dashed = True
        if isinstance(n, IntNode):
            for c in n.children:
                _walk(c)
    _walk(dashed_root)


def draw_branches(ax, root, seed_x, lw_hl=1.1, lw_dim=0.7):
    """Draw branches: red solid for induced subtree, grey solid for
    background, dashed triangle for the in_dashed clade."""
    DASH_STYLE = (0, (4, 3))

    ax.plot([seed_x, root.x], [root.y, root.y],
            color=C_DIM, lw=lw_dim, zorder=1, solid_capstyle='round')

    def _draw_collapsed_triangle(clade_root):
        tips = list(_subtree_tips(clade_root))
        if not tips:
            return
        y_min = min(t.y for t in tips)
        y_max = max(t.y for t in tips)
        base_x = max(t.end_x for t in tips)
        apex_x = clade_root.x
        apex_y = clade_root.y
        kw = dict(color=C_DIM, lw=lw_dim + 0.3, linestyle=DASH_STYLE,
                  zorder=1, solid_capstyle='butt')
        ax.plot([apex_x, base_x], [apex_y, y_max], **kw)
        ax.plot([apex_x, base_x], [apex_y, y_min], **kw)
        ax.plot([base_x, base_x], [y_min, y_max], **kw)

    def _walk(node):
        if isinstance(node, TipNode):
            return
        for c in node.children:
            if c.in_dashed and not node.in_dashed:
                child_x = c.end_x if isinstance(c, TipNode) else c.x
                kw = dict(color=C_DIM, lw=lw_dim + 0.3, zorder=1,
                          linestyle=DASH_STYLE, solid_capstyle='butt')
                ax.plot([node.x, node.x], [node.y, c.y], **kw)
                ax.plot([node.x, child_x], [c.y, c.y], **kw)
                if isinstance(c, IntNode):
                    _draw_collapsed_triangle(c)
                continue
            if c.in_dashed and node.in_dashed:
                continue

            if node.in_induced and c.in_induced:
                color, width, z, ls = C_HL, lw_hl, 3, '-'
                capstyle = 'round'
            else:
                color, width, z, ls = C_DIM, lw_dim, 1, '-'
                capstyle = 'round'
            child_x = c.end_x if isinstance(c, TipNode) else c.x
            ax.plot([node.x, node.x], [node.y, c.y],
                    color=color, lw=width, zorder=z,
                    linestyle=ls, solid_capstyle=capstyle)
            ax.plot([node.x, child_x], [c.y, c.y],
                    color=color, lw=width, zorder=z,
                    linestyle=ls, solid_capstyle=capstyle)
            _walk(c)

    _walk(root)


def draw_tips(ax, root, ms_hl=10, ms_dim=6):
    """Draw tip markers: green-on-red for induced, small grey for
    background, none for in_dashed clades."""
    def _walk(node):
        if isinstance(node, TipNode):
            if node.in_dashed:
                pass
            elif node.in_induced:
                ax.plot(node.end_x, node.y, 'o',
                        color=C_SAMP, markersize=ms_hl, zorder=5,
                        markeredgecolor=C_HL, markeredgewidth=0.35)
            elif node.sampled:
                ax.plot(node.end_x, node.y, 'o',
                        color=C_DIM, markersize=ms_dim, zorder=2,
                        markeredgecolor='white', markeredgewidth=0.5,
                        alpha=0.85)
        else:
            for c in node.children:
                _walk(c)
    _walk(root)


def collect_tips(root):
    """Return a flat list of every TipNode in the tree (post-order)."""
    out = []
    def _walk(node):
        if isinstance(node, TipNode):
            out.append(node)
        else:
            for c in node.children:
                _walk(c)
    _walk(root)
    return out


def gauss_kde(times, x_grid, bandwidth=0.40):
    """Sum-of-Gaussians density estimate."""
    if len(times) == 0:
        return np.zeros_like(x_grid)
    arr = np.asarray(times)
    diff = (x_grid[:, None] - arr[None, :]) / bandwidth
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1)
    dens /= len(arr) * np.sqrt(2 * np.pi) * bandwidth
    return dens


def draw_tip_time_kde(ax_kde, root, target_loc, other_loc, seed_x,
                      x_axis_max, bandwidth=0.40):
    """KDE sub-panel showing two per-location tip-time densities."""
    tips = collect_tips(root)
    a_times = [t.end_x for t in tips
               if t.location == target_loc and t.sampled]
    b_times = [t.end_x for t in tips
               if t.location == other_loc and t.sampled]

    x_grid = np.linspace(seed_x, x_axis_max, 400)
    kde_a = gauss_kde(a_times, x_grid, bandwidth=bandwidth)
    kde_b = gauss_kde(b_times, x_grid, bandwidth=bandwidth)

    y_max = max(kde_a.max(), kde_b.max(), 1e-6) * 1.25

    OTHER_COLOR = '#555'
    OTHER_FILL  = '#9a9a9a'

    ax_kde.fill_between(x_grid, 0, kde_b,
                        color=OTHER_FILL, alpha=0.35, zorder=1)
    ax_kde.plot(x_grid, kde_b, color=OTHER_COLOR, lw=0.8, zorder=2)

    ax_kde.fill_between(x_grid, 0, kde_a,
                        color=C_HL, alpha=0.30, zorder=3)
    ax_kde.plot(x_grid, kde_a, color=C_HL, lw=0.9, zorder=4)

    rug_y_a = -0.08 * y_max
    ax_kde.scatter(a_times, [rug_y_a] * len(a_times),
                   marker='|', color=C_HL, s=10, lw=0.65, zorder=5)

    # Density-curve labels — anchored past the curves' x range so no
    # label sits behind a curve. Grey label sits mid-low; red label
    # near the bottom of the curve mass area. Two-line wrap.
    peak_b_y = float(kde_b.max())
    label_x = x_axis_max + 0.10
    y_grey  = peak_b_y * 0.55
    y_red   = peak_b_y * 0.10
    ax_kde.text(label_x, y_grey,
                r'$\rho_{' + other_loc + r'}$' +
                ': density curve of\ntip time from '
                + r'$\mathrm{Loc}_{' + other_loc + r'}$',
                ha='left', va='center',
                fontsize=5, color=OTHER_COLOR, style='italic',
                clip_on=False, zorder=6)
    ax_kde.text(label_x, y_red,
                r'$\rho_{' + target_loc + r'}$' +
                ': density curve of\ntip time from '
                + r'$\mathrm{Loc}_{' + target_loc + r'}$',
                ha='left', va='center',
                fontsize=5, color=C_HL, style='italic',
                clip_on=False, zorder=6)

    ax_kde.set_ylim(-0.22 * y_max, y_max)
    for sp in ('top', 'right', 'left'):
        ax_kde.spines[sp].set_visible(False)
    ax_kde.spines['bottom'].set_visible(False)
    ax_kde.set_xticks([])
    ax_kde.set_yticks([])

    return x_grid, kde_a, kde_b


def annotate_mrca(ax, mrca, target_loc):
    """Mark MRCA + caption."""
    ax.plot(mrca.x, mrca.y, marker='o', markersize=5,
            markerfacecolor=C_HL, markeredgecolor=C_HL,
            markeredgewidth=0.8, zorder=6)
    ax.text(mrca.x, mrca.y + 1.4, f'subtree of Loc$_{target_loc}$',
            ha='center', va='bottom',
            fontsize=FS_CAPTION, fontweight='bold',
            color=C_HL, zorder=7)


# Encoder / K4 colors.
C_NODE_FEAT = "#3F76C0"
C_EDGE_FEAT = "#7E57C2"
C_CONV      = "#F2A93B"


def draw_cblv_grid(ax, x0, y0, width, height,
                   n_rows=4, n_cols=8, n_filled=None,
                   cmap_name='viridis', pad_facecolor='#e6e6e6',
                   pad_text_color='#888',
                   seed=7, border_color='#333', border_lw=1.2):
    """Stylized CBLV (W x 4) heatmap."""
    cmap = plt.get_cmap(cmap_name)
    rng = np.random.default_rng(seed)
    values = rng.random((n_rows, n_cols))
    if n_filled is None:
        n_filled = n_rows
    cell_w = width / n_cols
    cell_h = height / n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            xy = (x0 + c * cell_w, y0 + (n_rows - 1 - r) * cell_h)
            if r < n_filled:
                fc = cmap(values[r, c])
            else:
                fc = pad_facecolor
            ax.add_patch(Rectangle(
                xy, cell_w, cell_h,
                facecolor=fc, edgecolor='white', lw=0.4, zorder=3,
            ))
            if r >= n_filled:
                ax.text(xy[0] + cell_w / 2, xy[1] + cell_h / 2, '0',
                        ha='center', va='center',
                        fontsize=3, color=pad_text_color, zorder=4)
    ax.add_patch(Rectangle(
        (x0, y0), width, height,
        facecolor='none', edgecolor=border_color, lw=border_lw,
        zorder=4,
    ))


def draw_conv_block(ax, x_left, y_center, width, height, depth,
                    color=C_CONV, edgecolor='#333', alpha=0.92,
                    label=None, label_fontsize=5, label_dy=0.32):
    """Render a 3D-ish conv feature-map block."""
    front = Rectangle((x_left, y_center - height / 2), width, height,
                      facecolor=color, edgecolor=edgecolor,
                      lw=0.4, alpha=alpha, zorder=3)
    ax.add_patch(front)
    top_pts = [
        (x_left,             y_center + height / 2),
        (x_left + depth,     y_center + height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,     y_center + height / 2),
    ]
    ax.add_patch(Polygon(top_pts, facecolor=color, edgecolor=edgecolor,
                         lw=0.3, alpha=alpha * 0.75, zorder=3))
    right_pts = [
        (x_left + width,         y_center - height / 2),
        (x_left + width + depth, y_center - height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,         y_center + height / 2),
    ]
    ax.add_patch(Polygon(right_pts, facecolor=color, edgecolor=edgecolor,
                         lw=0.3, alpha=alpha * 0.55, zorder=3))
    if label is not None:
        ax.text(x_left + width / 2, y_center - height / 2 - label_dy,
                label, ha='center', va='top',
                fontsize=label_fontsize, color='#333')


def draw_feature_strip(ax, x_left, y_bottom, n_boxes,
                       box_w, box_h, color,
                       alphas=None, edgecolor='#333', edgewidth=0.8,
                       zorder=4):
    """Render a feature-vector strip — n_boxes adjacent filled
    rectangles with deterministic per-box alpha."""
    if alphas is None:
        alphas = 0.40 + 0.55 * np.array([
            ((i * 37 + 5) % 11) / 10.0 for i in range(n_boxes)
        ])
    for i in range(n_boxes):
        ax.add_patch(Rectangle(
            (x_left + i * box_w, y_bottom), box_w, box_h,
            facecolor=color, alpha=alphas[i],
            edgecolor=edgecolor, lw=edgewidth, zorder=zorder,
        ))


def draw_k4_graph(ax, target_loc='a',
                  positions=None, node_size=46,
                  edge_color='#9a9a9a', edge_lw=0.8,
                  hl_color=C_HL, dim_color=C_DIM,
                  hl_edge='#7a1f1f', dim_edge='#5a5a5a',
                  label_fontsize=None, draw_edges=True,
                  edge_attention=None):
    """Diamond-layout K4 with 4 location nodes.

    If `edge_attention` is provided (a dict mapping ordered (src, dst)
    tuples to weights in [0, 1]), the K4 is rendered as 12 directed
    arrows (2 per node-pair, slightly offset perpendicular so they
    don't overlap). Arrow line widths encode the per-direction
    weights — visualizing the asymmetric edge attention a GAT learns
    (alpha_ij != alpha_ji). When None, falls back to 6 plain
    undirected edges (controlled by `draw_edges`)."""
    if positions is None:
        positions = {
            'a': (6.0, 7.8),
            'b': (9.4, 5.0),
            'c': (6.0, 2.2),
            'd': (2.6, 5.0),
        }
    if label_fontsize is None:
        label_fontsize = max(5, int(0.42 * node_size))
    locs = ['a', 'b', 'c', 'd']
    if edge_attention is not None:
        # Directed arrows — one per ordered pair, offset perpendicular
        # to the edge centerline so the two opposing arrows for each
        # node pair don't overlap. Arrow LW encodes attention weight.
        # shrink must exceed the node's display radius (~node_size/2
        # in points) so the arrowhead sits OUTSIDE the node circle
        # and stays visible. mutation_scale tuned so heads are
        # readable at publication scale without overwhelming the K4.
        directed_color = '#444'
        shrink = max(7, int(node_size * 0.55))    # > node radius
        offset_d = 0.26           # perpendicular offset in data units
        for src in locs:
            for dst in locs:
                if src == dst:
                    continue
                weight = edge_attention.get((src, dst), 0.5)
                lw = 0.25 + 0.6 * weight    # ~[0.25, 0.9] for w in [0, 1]
                x0, y0 = positions[src]
                x1, y1 = positions[dst]
                dx, dy = x1 - x0, y1 - y0
                length = (dx * dx + dy * dy) ** 0.5
                if length < 1e-6:
                    continue
                # Perpendicular unit vector, rotated +90° from edge.
                px, py = -dy / length, dx / length
                xs0 = x0 + offset_d * px
                ys0 = y0 + offset_d * py
                xs1 = x1 + offset_d * px
                ys1 = y1 + offset_d * py
                ax.annotate(
                    '',
                    xy=(xs1, ys1), xytext=(xs0, ys0),
                    arrowprops=dict(
                        arrowstyle='-|>', color=directed_color,
                        lw=lw, mutation_scale=5, alpha=1.0,
                        shrinkA=shrink, shrinkB=shrink, zorder=2,
                    ),
                )
    elif draw_edges:
        for i in range(len(locs)):
            for j in range(i + 1, len(locs)):
                x0, y0 = positions[locs[i]]
                x1, y1 = positions[locs[j]]
                ax.plot([x0, x1], [y0, y1],
                        color=edge_color, lw=edge_lw,
                        zorder=1, solid_capstyle='round')
    for l in locs:
        x, y = positions[l]
        fc = hl_color if l == target_loc else dim_color
        ec = hl_edge if l == target_loc else dim_edge
        ax.plot(x, y, 'o', markersize=node_size,
                color=fc, markeredgecolor=ec, markeredgewidth=0.6,
                zorder=5)
        ax.text(x, y, f'Loc$_{l}$',
                ha='center', va='center',
                fontsize=label_fontsize, fontweight='bold',
                color='white', zorder=6)
    return positions


def _colored_inline_text(ax, x, y, parts,
                         fontsize=5, fontweight='bold'):
    """Render horizontally-laid text with per-part colors, centered on
    (x, y) in axes-data coords."""
    boxes = [TextArea(t, textprops=dict(color=c, fontsize=fontsize,
                                        fontweight=fontweight,
                                        family='sans-serif'))
             for t, c in parts]
    pack = HPacker(children=boxes, align='center', pad=0, sep=0)
    ax.add_artist(AnchoredOffsetbox(
        loc='center', child=pack, pad=0, frameon=False,
        bbox_to_anchor=(x, y), bbox_transform=ax.transData,
    ))


def draw_dtw_outputs(ax, target_loc='a', other_loc='b'):
    """Right-side sub-panel for the per-edge encoder — DTW output,
    styled to match the Aux (1x5) strip in the per-node encoder."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)

    cell_w = 0.109
    cell_h = 0.156
    n_cells = 3
    total_w = n_cells * cell_w
    start_x = (1.0 - total_w) / 2
    cell_y = 0.55

    draw_feature_strip(ax, x_left=start_x, y_bottom=cell_y,
                       n_boxes=n_cells, box_w=cell_w, box_h=cell_h,
                       color=C_EDGE_FEAT)

    # Line 1: e_{b->a} (1x3) — subscript b->a in mathtext renders
    # smaller and tucks against 'e'.
    ax.text(0.5, cell_y - 0.10,
            r'$e_{' + other_loc + r'\to ' + target_loc
            + r'}\ (1 \times 3)$',
            ha='center', va='top',
            fontsize=6, color=C_EDGE_FEAT, style='italic')

    # Line 2: DTW(ρ_b, ρ_a) — ρ is the standard math symbol for a
    # density function. ρ_b and ρ_a denote the per-location
    # tip-time density curves shown in the KDE panel above.
    ax.text(0.5, cell_y - 0.25,
            r'$\mathrm{DTW}(\rho_{' + other_loc
            + r'},\, \rho_{' + target_loc + r'})$',
            ha='center', va='top',
            fontsize=6, color=C_EDGE_FEAT)

    # Line 3: distance / mean lag / std lag — describes the 3 cells.
    ax.text(0.5, cell_y - 0.40,
            'distance / mean lag / std lag',
            ha='center', va='top',
            fontsize=6, color=C_EDGE_FEAT, style='italic')

    return (start_x + total_w, cell_y + cell_h / 2)


# =====================================================================
# SECTION D: Engine position swap.
# Putting Loc_b on the LEFT and Loc_d on the RIGHT makes the
# graph-assemble edge-feature strip (Loc_a-Loc_b edge) live on the LEFT
# side of the K4 — closer to the per-edge encoder cell (bottom-left of
# panel (b)), so the per-edge -> graph-assemble arrow is short and
# clean. All four position dicts need the swap because each is computed
# independently from ANGLES above.
# =====================================================================
for _d in (I_POS, S_POS, R_POS, SAMP_POS):
    _d['b'], _d['d'] = _d['d'], _d['b']


# =====================================================================
# SECTION E: Panel (a) composition (top row).
# =====================================================================

FS_ARROW_TOP    = 7
FS_ARROW_BOT    = 6
FS_PANEL_A_QUES = 8


def _draw_arrow_block(ax, top_text, bottom_text):
    """Left-to-right arrow with two stacked captions between the three
    sub-panels of the top row (panel a)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')

    ax.add_patch(FancyArrowPatch(
        (0.06, 0.5), (0.94, 0.5),
        arrowstyle='-|>,head_width=0.5,head_length=0.8',
        mutation_scale=9, lw=0.9,
        color='#1f1f1f', zorder=2,
        capstyle='round',
    ))
    # clip_on=False so captions can overhang into adjacent sub-panels.
    ax.text(0.5, 0.585, top_text,
            ha='center', va='bottom', clip_on=False,
            fontsize=FS_ARROW_TOP, fontweight='bold', color='#111111')
    ax.text(0.5, 0.415, bottom_text,
            ha='center', va='top', clip_on=False,
            fontsize=FS_ARROW_BOT, style='italic', color='#444444')


def _draw_panel_a(fig, subplotspec):
    """Render panel (a): engine + arrow + tree (1x2 main sub-panels).
    Right-hand K4-GNN sub-panel was removed; its prediction-task
    questions migrated to the top of panel (c)."""
    gs_a = subplotspec.subgridspec(
        1, 3,
        width_ratios=[6.0, 1.5, 6.0],
        wspace=0.04,
    )

    # Sub-panel a1: simulation engine (full SIR + migration view).
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    draw_engine(ax_a1)

    # Top caption: Loc_a is the seed (centered above the diamond).
    # Caption sits to the right of I_a, single line, at I_a's
    # vertical height (y=5). Bubble a ends at x=2.1 so x=2.5 clears it.
    ax_a1.text(2.5, 5.0, 'Loc$_a$, the source of the outbreak',
               ha='left', va='center', clip_on=False,
               fontsize=FS_PANEL_A_QUES, fontweight='bold', style='italic',
               color='#8A6500')

    # Shared legend for the engine — replaces the inline 'infection /
    # removal / sampling / migration' arrow labels. Placed in the
    # lower-right empty corner of the engine cell (below bubble b).
    legend_handles = [
        Line2D([0], [0], color='#2E7DBF', lw=1.2, label='infection'),
        Line2D([0], [0], color='#7B4FB4', lw=1.4, label='migration'),
        Line2D([0], [0], color='#888888', lw=1.2, label='recovery'),
        Line2D([0], [0], color='#3CB371', lw=1.2, label='sampling'),
    ]
    ax_a1.legend(handles=legend_handles, loc='lower right',
                 frameon=False, fontsize=FS_EVENT,
                 handlelength=1.4, handletextpad=0.5,
                 labelspacing=0.35, borderaxespad=0.2)

    ax_arr1 = fig.add_subplot(gs_a[0, 1])
    _draw_arrow_block(ax_arr1,
                      top_text='stochastic simulation',
                      bottom_text='Gillespie algorithm')

    # Sub-panel a2: simulated 12-tip tree.
    ax_a2 = fig.add_subplot(gs_a[0, 2])
    draw_tree(ax_a2)

    return ax_a1, ax_a2


# =====================================================================
# SECTION F: Panel (b) composition (2x2 grid: subtree extraction,
# per-node encoder, per-edge encoder, graph assemble).
# =====================================================================

def _draw_panel_b(fig, subplotspec, args):
    """Render panel (b): STEPHY encoder pipeline as a 2x2 grid.

    Top-left  : subtree extraction (full simulated tree with Loc_a's
                induced subtree highlighted, one neighbor clade dashed).
    Top-right : per-node encoder (CBLV CNN + Aux MLP -> 128-d).
    Bottom-left: per-edge encoder (tip-time KDE + DTW inset).
    Bottom-right: graph assemble (directed K4 with feature strips).
    Cross-axes arrows link the encoder outputs to the graph-assemble
    K4: node feature -> Loc_a strip, edge feature -> Loc_a-Loc_b strip.
    """
    # ---- Tree generation -----------------------------------------
    rng = np.random.default_rng(args.seed)
    T = 8.0
    root = gen_tree(args.n_tips, T, rng, sample_frac=1.0)
    ladderize(root, smaller_first=True)
    assign_y(root)
    assign_locations(
        root, rng,
        target_loc=args.target_loc,
        initial_loc=args.initial_loc,
        a_subclade_target=args.a_subclade_target,
        a_exit_migrations=args.a_exit_migrations,
        n_other_migrations=args.n_other_migrations,
    )
    rebias_loc_a_tip_times(root, args.target_loc, T, rng,
                           mu_frac=0.55, sigma_frac=0.18)
    mrca = find_mrca(root, args.target_loc, sampled_only=True)
    mark_induced_subtree(root, mrca,
                         target_loc=args.target_loc, sampled_only=True)
    dashed_root = find_below_loca_clade(root, mrca)
    mark_dashed_clade(dashed_root)

    SEED_X = -0.6
    x_axis_max = T + 1.6
    time_x_end = T + 0.15

    # 2x2 layout for panel (b)'s 4 standardized sub-panels:
    # top-left subtree extraction, top-right per-node encoder,
    # bottom-left per-edge encoder, bottom-right graph assemble.
    # Equal width and height ratios so all 4 cells have identical
    # dimensions.
    gs_bcde = subplotspec.subgridspec(
        2, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.04, wspace=0.08,
    )
    ax_tree  = fig.add_subplot(gs_bcde[0, 0])
    ax_kde   = fig.add_subplot(gs_bcde[1, 0], sharex=ax_tree)
    ax_conv  = fig.add_subplot(gs_bcde[0, 1])
    ax_graph = fig.add_subplot(gs_bcde[1, 1])

    # ---- subtree extraction (top-left) -----------------------------
    draw_branches(ax_tree, root, SEED_X, lw_hl=1.1, lw_dim=0.7)
    draw_tips(ax_tree, root, ms_hl=4, ms_dim=3)
    annotate_mrca(ax_tree, mrca, args.target_loc)

    ax_tree.plot(SEED_X, root.y, marker='*', markersize=10,
                 color=C_GOLD, markeredgecolor=C_GOLD_DARK,
                 markeredgewidth=0.5, zorder=6)
    ax_tree.text(SEED_X, root.y + 1.7, 'source case',
                 ha='center', va='bottom',
                 fontsize=FS_INDEX_CAP, fontweight='bold', style='italic',
                 color=C_GOLD_DARK)

    ax_tree.set_xlim(SEED_X - 1.4, x_axis_max + 0.1)
    ax_tree.set_ylim(-1.5, args.n_tips + 1.5)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_tree.spines[sp].set_visible(False)
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    ax_tree.text(0.5, 0.97, 'subtree extraction',
                 transform=ax_tree.transAxes, ha='center', va='top',
                 fontsize=FS_KDE_HDR + 1, fontweight='bold', color='#222')

    # ---- per-edge encoder (bottom-left): KDE + DTW inset -----------
    other_loc = 'b' if args.target_loc != 'b' else 'c'
    draw_tip_time_kde(
        ax_kde, root, args.target_loc, other_loc=other_loc,
        seed_x=SEED_X, x_axis_max=time_x_end, bandwidth=0.45,
    )
    ax_kde.text(0.5, 0.97, 'per-edge encoder',
                transform=ax_kde.transAxes, ha='center', va='top',
                fontsize=FS_KDE_HDR + 1, fontweight='bold', color='#222')

    # Inset height 0.32 so DTW cells render at the same physical size
    # as the Aux (1x5) strip in panel (c).
    ax_dtw = ax_kde.inset_axes([0.24, 0.51, 0.32, 0.32])
    dtw_right_x, dtw_mid_y = draw_dtw_outputs(
        ax_dtw, target_loc=args.target_loc, other_loc=other_loc,
    )

    y_lo = ax_kde.get_ylim()[0]
    arrow_y = y_lo * 0.65
    ax_kde.annotate('', xy=(time_x_end, arrow_y),
                    xytext=(SEED_X, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.75,
                                    mutation_scale=6))
    ax_kde.text(0.5 * (SEED_X + time_x_end), y_lo * 0.95, 'Time',
                ha='center', va='top',
                fontsize=FS_TIME, fontweight='bold')

    # ---- per-node encoder (top-right) ------------------------------
    ax_conv.set_xlim(0, 12)
    ax_conv.set_ylim(0, 10)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_conv.spines[sp].set_visible(False)
    ax_conv.set_xticks([])
    ax_conv.set_yticks([])

    ax_conv.text(0.5, 0.97, 'per-node encoder',
                 transform=ax_conv.transAxes, ha='center', va='top',
                 fontsize=FS_KDE_HDR + 1, fontweight='bold', color='#222')

    CELL = 0.42

    # CBLV (W x 4)
    cblv_rows, cblv_cols = 12, 4
    cblv_n_filled = 10
    cblv_w = cblv_cols * CELL
    cblv_h = cblv_rows * CELL
    cblv_x0 = 0.55
    cblv_y0 = 2.85
    draw_cblv_grid(ax_conv, x0=cblv_x0, y0=cblv_y0,
                   width=cblv_w, height=cblv_h,
                   n_rows=cblv_rows, n_cols=cblv_cols,
                   n_filled=cblv_n_filled, seed=7)

    bracket_x = cblv_x0 + cblv_w + 0.06
    bracket_tip = 0.13
    y_top    = cblv_y0 + cblv_h
    y_split  = cblv_y0 + (cblv_rows - cblv_n_filled) * CELL
    y_bottom = cblv_y0
    ax_conv.plot(
        [bracket_x, bracket_x + bracket_tip,
         bracket_x + bracket_tip, bracket_x],
        [y_top, y_top, y_split, y_split],
        color='#444', lw=1.4, solid_capstyle='butt',
    )
    ax_conv.text(bracket_x + bracket_tip + 0.08,
                 (y_top + y_split) / 2, 'm',
                 ha='left', va='center',
                 fontsize=6, fontweight='bold', color='#222')
    ax_conv.plot(
        [bracket_x, bracket_x + bracket_tip,
         bracket_x + bracket_tip, bracket_x],
        [y_split, y_split, y_bottom, y_bottom],
        color='#888', lw=1.4, solid_capstyle='butt',
    )
    ax_conv.text(bracket_x + bracket_tip + 0.08,
                 (y_split + y_bottom) / 2,
                 r'$W\!-\!m$' + '\nzero pad',
                 ha='left', va='center',
                 fontsize=5, color='#666')

    ax_conv.text(cblv_x0 + cblv_w / 2, cblv_y0 - 0.30,
                 r'CBLV $(W \times 4)$',
                 ha='center', va='top',
                 fontsize=6, fontweight='bold', color='#222')
    ax_conv.text(cblv_x0 + cblv_w / 2, cblv_y0 - 0.65,
                 r'$W$ = fixed unified dimension;'
                 ' \n'
                 r'$m$ = subtree tips',
                 ha='center', va='top',
                 fontsize=5, color='#666', style='italic')

    # 3-branch conv encoder
    branches = [
        (7.30, 'plain'),
        (5.45, 'stride'),
        (3.60, 'dilate'),
    ]
    br_x0, br_w, br_h, br_d = 4.10, 2.00, 1.20, 0.38
    cblv_exit = (cblv_x0 + cblv_w + 0.05,
                 cblv_y0 + cblv_h / 2)
    for y_c, name in branches:
        draw_conv_block(ax_conv, x_left=br_x0, y_center=y_c,
                        width=br_w, height=br_h, depth=br_d,
                        color=C_CONV, label=None)
        ax_conv.text(br_x0 + br_w / 2, y_c, name,
                     ha='center', va='center',
                     fontsize=6, fontweight='bold', color='#333',
                     zorder=5)
        ax_conv.annotate(
            '', xy=(br_x0 - 0.04, y_c),
            xytext=cblv_exit,
            arrowprops=dict(arrowstyle='->', mutation_scale=6, lw=0.45,
                            color='#555', zorder=4,
                            connectionstyle=f'arc3,rad={(y_c - cblv_exit[1]) / 14:.2f}'),
        )

    # Aux (1 x 5)
    aux_n, aux_bw, aux_bh = 5, CELL, CELL
    aux_x0 = cblv_x0 + cblv_w / 2 - (aux_n * aux_bw) / 2
    aux_y0 = 1.10
    draw_feature_strip(ax_conv, x_left=aux_x0, y_bottom=aux_y0,
                       n_boxes=aux_n, box_w=aux_bw, box_h=aux_bh,
                       color='#90C695')
    ax_conv.text(aux_x0 + aux_n * aux_bw / 2, aux_y0 - 0.30,
                 r'Aux $(1 \times 5)$',
                 ha='center', va='top',
                 fontsize=6, fontweight='bold', color='#222')
    ax_conv.text(aux_x0 + aux_n * aux_bw / 2, aux_y0 - 0.78,
                 'mrca depth / earliest tip / latest tip / avg BL / n_tips',
                 ha='center', va='top',
                 fontsize=5, color='#666', style='italic')

    # AuxBranch MLP
    aux_box_x = br_x0
    aux_box_y = aux_y0 + aux_bh / 2
    aux_box_w, aux_box_h, aux_box_d = br_w, 1.30, 0.38
    draw_conv_block(ax_conv, x_left=aux_box_x, y_center=aux_box_y,
                    width=aux_box_w, height=aux_box_h, depth=aux_box_d,
                    color='#90C695', label=None)
    ax_conv.text(aux_box_x + aux_box_w / 2, aux_box_y, 'AuxBranch\nMLP',
                 ha='center', va='center',
                 fontsize=5, fontweight='bold', color='#1f5933',
                 zorder=5)
    ax_conv.annotate(
        '', xy=(aux_box_x - 0.04, aux_box_y),
        xytext=(aux_x0 + aux_n * aux_bw + 0.05, aux_y0 + aux_bh / 2),
        arrowprops=dict(arrowstyle='->', mutation_scale=6, lw=0.45,
                        color='#555', zorder=4),
    )

    # Convergence + (1 x 128) node feature
    concat_x, concat_y = 8.20, 5.45
    for y_c, _ in branches:
        ax_conv.annotate(
            '', xy=(concat_x, concat_y),
            xytext=(br_x0 + br_w + br_d + 0.05, y_c),
            arrowprops=dict(arrowstyle='->', mutation_scale=6, lw=0.45,
                            color='#444', zorder=4,
                            connectionstyle=f'arc3,rad={(concat_y - y_c) / 14:.2f}'),
        )
    ax_conv.text(br_x0 + br_w + br_d + 1.10, 7.70, r'$(1 \times 96)$',
                 ha='center', va='center',
                 fontsize=5, color='#444', fontweight='bold')

    ax_conv.annotate(
        '', xy=(concat_x, concat_y),
        xytext=(aux_box_x + aux_box_w + aux_box_d + 0.05, aux_box_y),
        arrowprops=dict(arrowstyle='->', mutation_scale=6, lw=0.45,
                        color='#444', zorder=4,
                        connectionstyle=f'arc3,rad={(concat_y - aux_box_y) / 14:.2f}'),
    )
    ax_conv.text(aux_box_x + aux_box_w + aux_box_d + 1.10, 1.80,
                 r'$(1 \times 32)$',
                 ha='center', va='center',
                 fontsize=5, color='#444', fontweight='bold')

    nf_n, nf_bw, nf_bh = 7, CELL, CELL
    nf_x0 = concat_x + 0.70
    nf_y0 = concat_y - nf_bh / 2
    draw_feature_strip(ax_conv, x_left=nf_x0, y_bottom=nf_y0,
                       n_boxes=nf_n, box_w=nf_bw, box_h=nf_bh,
                       color=C_NODE_FEAT)
    ax_conv.annotate(
        '', xy=(nf_x0 - 0.04, nf_y0 + nf_bh / 2),
        xytext=(concat_x, nf_y0 + nf_bh / 2),
        arrowprops=dict(arrowstyle='->', mutation_scale=6, lw=0.45,
                        color='#444', zorder=4),
    )
    ax_conv.text(0.5 * (concat_x + nf_x0),
                 nf_y0 + nf_bh / 2 + 0.22,
                 'concat',
                 ha='center', va='bottom',
                 fontsize=5, fontweight='bold', color='#555')
    ax_conv.text(nf_x0 + nf_n * nf_bw / 2, nf_y0 - 0.35,
                 r'$h_{' + args.target_loc + r'}\ (1 \times 128)$',
                 ha='center', va='top',
                 fontsize=6, fontweight='bold', color=C_NODE_FEAT)

    # ---- graph assemble (bottom-right): directed K4 ----------------
    # xlim extended so the wide Aux-matched node-feature strips fit
    # alongside the K4 diamond without overflowing.
    ax_graph.set_xlim(-2, 14)
    ax_graph.set_ylim(0, 10)
    ax_graph.set_aspect('equal', adjustable='box')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_graph.spines[sp].set_visible(False)
    ax_graph.set_xticks([])
    ax_graph.set_yticks([])
    # Title positioned in figure-fraction so it aligns vertically
    # with the 'per-edge encoder' title — ax_graph is shorter than
    # its cell (aspect='equal'), so its axes-frac 0.97 would otherwise
    # sit lower than ax_kde's at the same fraction.
    _kde_pos   = ax_kde.get_position()
    _graph_pos = ax_graph.get_position()
    _title_y   = _kde_pos.y0 + 0.97 * _kde_pos.height
    _title_x   = _graph_pos.x0 + 0.5 * _graph_pos.width
    fig.text(_title_x, _title_y, 'graph assemble',
             ha='center', va='top',
             fontsize=FS_KDE_HDR + 1, fontweight='bold', color='#222')

    # Asymmetric (directed) K4: 12 directional arrows with varying
    # widths convey that the GAT learns asymmetric edge attention
    # (alpha_ij != alpha_ji). Weights chosen to look distinct, with
    # the a<->b pair given moderate-high values to match the visible
    # edge feature strip on that edge.
    graph_positions = draw_k4_graph(
        ax_graph, target_loc=args.target_loc,
        positions={
            'a': (6.0, 7.6),
            'b': (3.4, 5.0),
            'c': (6.0, 2.4),
            'd': (8.6, 5.0),
        },
        node_size=18,
        edge_attention={
            ('a', 'b'): 0.45, ('b', 'a'): 0.70,
            ('a', 'c'): 0.30, ('c', 'a'): 0.55,
            ('a', 'd'): 0.20, ('d', 'a'): 0.50,
            ('b', 'c'): 0.55, ('c', 'b'): 0.35,
            ('b', 'd'): 0.40, ('d', 'b'): 0.65,
            ('c', 'd'): 0.25, ('d', 'c'): 0.45,
        },
    )

    # Strip count 7 matches the Loc_a node-feature strip in panel (c).
    # Cell size 0.592 x 0.504 data units = 0.70 x 0.596 in physical,
    # matching Aux/DTW.
    strip_n  = 7
    strip_bw = 0.592
    strip_bh = 0.504
    strip_offsets = {
        'a': (-strip_n * strip_bw / 2,  1.05),
        'b': (-strip_n * strip_bw - 1.05,  -strip_bh / 2),
        'c': (-strip_n * strip_bw / 2, -1.05 - strip_bh),
        'd': ( 1.05, -strip_bh / 2),
    }
    for l, (dx, dy) in strip_offsets.items():
        nx, ny = graph_positions[l]
        draw_feature_strip(
            ax_graph, x_left=nx + dx, y_bottom=ny + dy,
            n_boxes=strip_n, box_w=strip_bw, box_h=strip_bh,
            color=C_NODE_FEAT,
        )

    # Edge-feature strip on the Loc_a-Loc_b edge (Loc_b on LEFT after
    # the b/d swap, so the strip is on the LEFT side of the diamond).
    ax_pos = graph_positions[args.target_loc]
    nb_pos = graph_positions['b']
    edge_mid_x = 0.5 * (ax_pos[0] + nb_pos[0])
    edge_mid_y = 0.5 * (ax_pos[1] + nb_pos[1])
    edge_n, edge_bw, edge_bh = 3, 0.592, 0.504
    edge_x0 = edge_mid_x - edge_n * edge_bw - 0.55
    edge_y0 = edge_mid_y + 0.30
    draw_feature_strip(
        ax_graph, x_left=edge_x0, y_bottom=edge_y0,
        n_boxes=edge_n, box_w=edge_bw, box_h=edge_bh,
        color=C_EDGE_FEAT,
    )

    # ---- Cross-cell data-flow arrows within panel (b) --------------
    # per-node encoder feature  ->  graph-assemble Loc_a strip.
    a_strip_x = graph_positions['a'][0] + strip_offsets['a'][0] \
                + strip_n * strip_bw / 2
    a_strip_y = graph_positions['a'][1] + strip_offsets['a'][1] \
                + strip_bh
    con_c_e = ConnectionPatch(
        xyA=(nf_x0 + nf_bw * 0.5, nf_y0),
        coordsA=ax_conv.transData,
        xyB=(a_strip_x, a_strip_y + 0.05),
        coordsB=ax_graph.transData,
        arrowstyle='->', mutation_scale=7, lw=0.65,
        color=C_NODE_FEAT, zorder=20,
        connectionstyle='arc3,rad=-0.15',
    )
    fig.add_artist(con_c_e)

    # per-edge encoder DTW inset  ->  graph-assemble Loc_a-Loc_b edge.
    con_d_e = ConnectionPatch(
        xyA=(dtw_right_x, dtw_mid_y),
        coordsA=ax_dtw.transData,
        xyB=(edge_x0 + edge_n * edge_bw / 2,
             edge_y0 + edge_bh + 0.05),
        coordsB=ax_graph.transData,
        arrowstyle='->', mutation_scale=7, lw=0.65,
        color=C_EDGE_FEAT, zorder=20,
        connectionstyle='arc3,rad=-0.30',
    )
    fig.add_artist(con_d_e)

    return ax_tree, ax_kde, ax_conv, ax_graph


# =====================================================================
# SECTION G: Panel (c) composition.
# Inference pipeline: message-passing layer -> MLP head -> per-location
# predictions K4. The predictions K4 reuses panel (b)'s graph-assemble
# layout, but each Loc node now carries (R_e, mu, SSS) cards plus the
# predicted index-case location with its 95% CP set. Internal names
# (PANEL_F_*, _draw_panel_f_*) keep the legacy "f" prefix to avoid
# churn — the rendered output is the figure's panel (c).
# =====================================================================

# Mock predictions for each location. Designed so Loc_a (the seed)
# has the highest R_e and a positive Source-Sink Score (it exports),
# and the others have moderate R_e with negative SSS (they import).
PANEL_F_PREDS = {
    'a': dict(R_e=2.31, R_e_lo=1.95, R_e_hi=2.68,
              mu=0.045, mu_lo=0.035, mu_hi=0.058,
              SSS=+0.42, SSS_lo=+0.28, SSS_hi=+0.55),
    'b': dict(R_e=1.84, R_e_lo=1.51, R_e_hi=2.19,
              mu=0.062, mu_lo=0.048, mu_hi=0.078,
              SSS=-0.18, SSS_lo=-0.32, SSS_hi=-0.05),
    'c': dict(R_e=1.97, R_e_lo=1.62, R_e_hi=2.34,
              mu=0.058, mu_lo=0.045, mu_hi=0.073,
              SSS=-0.11, SSS_lo=-0.24, SSS_hi=+0.02),
    'd': dict(R_e=1.62, R_e_lo=1.30, R_e_hi=1.98,
              mu=0.073, mu_lo=0.058, mu_hi=0.089,
              SSS=-0.13, SSS_lo=-0.27, SSS_hi=+0.01),
}

# Predicted index case (classification answer) and its CP set at 95%.
PANEL_F_PRED_INDEX  = 'a'
PANEL_F_PRED_CP_SET = ['a', 'b']

C_PRED_VAL  = '#1f3550'   # ink-blue for prediction values
C_PRED_INT  = '#7E57C2'   # purple for CP intervals
C_INDEX_GLD = '#8A6500'   # matches panel-(a) question color

# Colors / fontsize used by panel (f): combined LGAT message-passing
# + MLP-head panel and per-location predictions panel.
C_MLP         = "#F2A93B"
C_MLP_EDGE    = "#7a5310"
C_ALPHA       = "#3a1f6b"        # purple gather arrows (alpha_au)
C_MSG_EDGE    = "#444"
C_PLANE_TOP   = "#EDF1F6"        # K4-plate top face
C_PLANE_FRONT = "#C9D1DC"        # K4-plate front (thickness)
C_PLANE_SIDE  = "#B6BFCD"        # K4-plate side (thickness)
C_PLANE_EDGE  = "#7d8794"
FS_F_HDR      = FS_KDE_HDR + 1   # 8 pt — column titles


def _draw_pred_card(ax, x_center, y_center, loc, is_target):
    """Draw a bordered card carrying R_e, mu, SSS predictions for one
    Loc node. Each row shows the point estimate plus its 95% CP
    interval. All cards use the same neutral border — the predicted
    index case is communicated by the star + readout above, not by
    card highlighting."""
    p = PANEL_F_PREDS[loc]
    card_w, card_h = 4.80, 2.00
    ax.add_patch(Rectangle(
        (x_center - card_w / 2, y_center - card_h / 2),
        card_w, card_h,
        facecolor='white', edgecolor='#7a7a7a', lw=0.4,
        zorder=4,
    ))
    # 3 rows: each row carries value (left, dark-blue) and CP
    # interval (right, purple). Unified colors — no red/blue sign
    # distinction for SSS. Minimum font size matches the 'mrca
    # depth / earliest tip / ...' caption in panel (b).
    row_ys = [y_center + 0.60, y_center, y_center - 0.60]
    val_gap = 0.08   # gap (in data units) between value and interval
    sign = '+' if p['SSS'] > 0 else '−'
    rows = [
        (row_ys[0], fr'$R_0\!=\!{p["R_e"]:.2f}$',
         fr'$[{p["R_e_lo"]:.2f},\,{p["R_e_hi"]:.2f}]$'),
        (row_ys[1], fr'$\gamma\!=\!{p["mu"]:.3f}$',
         fr'$[{p["mu_lo"]:.3f},\,{p["mu_hi"]:.3f}]$'),
        (row_ys[2], fr'$\mathrm{{SSS}}\!=\!{sign}{abs(p["SSS"]):.2f}$',
         fr'$[{p["SSS_lo"]:+.2f},\,{p["SSS_hi"]:+.2f}]$'),
    ]
    for y, val_str, ci_str in rows:
        ax.text(x_center - val_gap, y, val_str,
                ha='right', va='center',
                fontsize=6, fontweight='bold',
                color=C_PRED_VAL, zorder=6)
        ax.text(x_center + val_gap, y, ci_str,
                ha='left', va='center',
                fontsize=5, style='italic',
                color=C_PRED_INT, zorder=6)


def _draw_3d_plate(ax, x_left, y_bottom, thick, height, depth_x, depth_y,
                   side='right'):
    """3D cuboid 'plate' (thin slab) onto which a K4 is painted. The
    thin front face is what the viewer sees head-on; the larger K4
    face recedes into the picture along the depth axis. Returns
    (origin, u_vec, v_vec) — the affine basis on the K4 face so a UV
    point (u, v) in [0, 1]^2 maps to origin + u*u_vec + v*v_vec."""
    if side == 'right':
        front_pts = [
            (x_left,                   y_bottom),
            (x_left + thick,           y_bottom),
            (x_left + thick,           y_bottom + height),
            (x_left,                   y_bottom + height),
        ]
        face_pts = [
            (x_left + thick,           y_bottom),
            (x_left + thick + depth_x, y_bottom + depth_y),
            (x_left + thick + depth_x, y_bottom + depth_y + height),
            (x_left + thick,           y_bottom + height),
        ]
        top_pts = [
            (x_left,                   y_bottom + height),
            (x_left + thick,           y_bottom + height),
            (x_left + thick + depth_x, y_bottom + height + depth_y),
            (x_left + depth_x,         y_bottom + height + depth_y),
        ]
        origin = (x_left + thick, y_bottom)
        u_vec  = (depth_x, depth_y)
        v_vec  = (0.0, height)
    else:
        front_x = x_left + depth_x
        front_pts = [
            (front_x,                   y_bottom),
            (front_x + thick,           y_bottom),
            (front_x + thick,           y_bottom + height),
            (front_x,                   y_bottom + height),
        ]
        face_pts = [
            (front_x,                   y_bottom),
            (front_x - depth_x,         y_bottom + depth_y),
            (front_x - depth_x,         y_bottom + depth_y + height),
            (front_x,                   y_bottom + height),
        ]
        top_pts = [
            (front_x - depth_x,         y_bottom + height + depth_y),
            (front_x,                   y_bottom + height),
            (front_x + thick,           y_bottom + height),
            (front_x + thick - depth_x, y_bottom + height + depth_y),
        ]
        origin = (front_x, y_bottom)
        u_vec  = (-depth_x, depth_y)
        v_vec  = (0.0, height)

    ax.add_patch(Polygon(top_pts, closed=True,
                         facecolor=C_PLANE_FRONT, edgecolor=C_PLANE_EDGE,
                         lw=1.0, zorder=0))
    ax.add_patch(Polygon(face_pts, closed=True,
                         facecolor=C_PLANE_TOP, edgecolor=C_PLANE_EDGE,
                         lw=1.2, zorder=1))
    ax.add_patch(Polygon(front_pts, closed=True,
                         facecolor=C_PLANE_SIDE, edgecolor=C_PLANE_EDGE,
                         lw=1.0, zorder=2))
    return origin, u_vec, v_vec


def _draw_k4_on_face(ax, origin, u_vec, v_vec,
                     self_loc='a', label_template='h',
                     edge_attention=None,
                     k4_uv_radius=0.30, node_uv_radius=0.10,
                     label_fontsize=5):
    """Diamond K4 painted on a 3D face's UV coords (origin, u_vec,
    v_vec from `_draw_3d_plate`). Nodes are flat tilted ellipses
    (perspective-matched to the face) with no sphere shading. If
    `edge_attention` is given, the K4 is rendered as 12 directed
    arrows whose line widths encode per-direction weights."""
    def _uv_to_xy(u, v):
        return (origin[0] + u * u_vec[0] + v * v_vec[0],
                origin[1] + u * u_vec[1] + v * v_vec[1])
    cu, cv = 0.5, 0.5
    r = k4_uv_radius
    uv = {
        'a': (cu,     cv + r),
        'b': (cu - r, cv),
        'c': (cu,     cv - r),
        'd': (cu + r, cv),
    }
    pos = {l: _uv_to_xy(*uv[l]) for l in uv}
    locs = ['a', 'b', 'c', 'd']

    node_ry = node_uv_radius * abs(v_vec[1])
    node_rx = node_uv_radius * abs(u_vec[0])

    if edge_attention is not None:
        shrink = 4
        offset_d = 0.06
        for src in locs:
            for dst in locs:
                if src == dst:
                    continue
                w = edge_attention.get((src, dst), 0.5)
                lw = 0.25 + 0.55 * w
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                dx, dy = x1 - x0, y1 - y0
                L = (dx * dx + dy * dy) ** 0.5
                if L < 1e-6:
                    continue
                px, py = -dy / L, dx / L
                xs0, ys0 = x0 + offset_d * px, y0 + offset_d * py
                xs1, ys1 = x1 + offset_d * px, y1 + offset_d * py
                ax.annotate(
                    '', xy=(xs1, ys1), xytext=(xs0, ys0),
                    arrowprops=dict(arrowstyle='-|>', color='#444',
                                    lw=lw, mutation_scale=3.5,
                                    shrinkA=shrink, shrinkB=shrink,
                                    zorder=3),
                )

    for l in locs:
        x, y = pos[l]
        fc = C_HL if l == self_loc else C_DIM
        ec = '#7a1f1f' if l == self_loc else '#5a5a5a'
        ax.add_patch(Ellipse((x, y),
                             width=2 * node_rx, height=2 * node_ry,
                             facecolor=fc, edgecolor=ec,
                             lw=1.6, zorder=5))
        ax.text(x, y, f'${label_template}_{l}$',
                ha='center', va='center',
                fontsize=label_fontsize, fontweight='bold',
                color='white', zorder=6)
    return pos


def _draw_panel_f_mp(ax, args):
    """LEFT sub-panel of panel (f): LGAT message-passing layer.

    Layer ℓ K4 plate → (h_a^(ℓ), agg_a) → concat → layer ℓ+1 K4
    plate. Self node = Loc_a (red); neighbors b/c/d gather into
    agg_a via purple attention arrows whose widths encode incoming
    α_au weights. STEPHY's K4 has no self-loop (stephy/data.py), so
    the sum runs over u ≠ a — three terms for K4.
    """
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 12.5)
    ax.set_aspect('equal', adjustable='box')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title sits ABOVE the axes (axes-frac 1.02, matching mlp + pred)
    # so all three panel-(f) sub-panel titles land at the same fig-y.
    ax.text(0.5, 1.02, 'message passing layer',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=FS_F_HDR, fontweight='bold', color='#222')

    # Incoming-to-'a' attention spread wide so per-arrow widths are
    # unambiguously visible in the gather arrows.
    eatt = {
        ('a', 'b'): 0.45, ('b', 'a'): 0.08,
        ('a', 'c'): 0.30, ('c', 'a'): 0.45,
        ('a', 'd'): 0.20, ('d', 'a'): 0.28,
        ('b', 'c'): 0.55, ('c', 'b'): 0.35,
        ('b', 'd'): 0.40, ('d', 'b'): 0.65,
        ('c', 'd'): 0.25, ('d', 'c'): 0.45,
    }

    plate_thick   = 0.6
    plate_height  = 8.0
    plate_depth_x = 2.6
    plate_depth_y = 1.0

    # ----- Layer ℓ (LEFT plate) -------------------------------------
    pl_l_x, pl_l_y = 0.4, 2.25
    origin_l, u_l, v_l = _draw_3d_plate(
        ax, pl_l_x, pl_l_y,
        thick=plate_thick, height=plate_height,
        depth_x=plate_depth_x, depth_y=plate_depth_y,
        side='right',
    )
    pos_l = _draw_k4_on_face(
        ax, origin_l, u_l, v_l,
        self_loc='', label_template='h',
        edge_attention=eatt,
        label_fontsize=5,
    )
    ax.text(pl_l_x + plate_thick / 2, pl_l_y + 0.20,
            r'$\ell$',
            ha='center', va='bottom', fontsize=6,
            color='#666', style='italic')

    # ----- Aggregation + self pass-through ellipses (MIDDLE) --------
    # STEPHY GAT (stephy/model.py): agg_v = sum_u alpha_uv * h_u over
    # neighbors u != v; then h_v^(l+1) = concat(h_v^(l), agg_v). The
    # "self vs neighbor" split lives in the CONCAT, not in two
    # separate alpha-gated messages.
    msg_x = 5.0
    msg_w, msg_h = 1.8, 1.5
    plate_mid_y = pl_l_y + plate_height / 2
    self_xy = (msg_x, plate_mid_y + 2.50)
    agg_xy  = (msg_x, plate_mid_y - 1.75)
    for cx_e, cy_e, label in [
        (self_xy[0], self_xy[1], r'$h_{a}^{(\ell)}$'),
        (agg_xy[0],  agg_xy[1],  r'$\mathrm{agg}_{a}$'),
    ]:
        ax.add_patch(Ellipse((cx_e, cy_e), msg_w, msg_h,
                             facecolor='#FFFFFF', edgecolor=C_MSG_EDGE,
                             lw=0.5, zorder=4))
        ax.text(cx_e, cy_e, label, ha='center', va='center',
                fontsize=8, color='#222', zorder=5)
    ax.text(agg_xy[0], agg_xy[1] - msg_h / 2 - 0.30,
            r'$\sum_{u \neq a}\, \alpha_{au}\, h_{u}^{(\ell)}$',
            ha='center', va='top',
            fontsize=6, color='#444', style='italic')

    agg_left  = (agg_xy[0] - msg_w / 2 - 0.02, agg_xy[1])
    self_left = (self_xy[0] - msg_w / 2 - 0.02, self_xy[1])

    # Gather arrows from b, c, d -> agg_a. Solid purple line whose
    # width encodes alpha_au (= incoming attention weight on edge
    # u -> a).
    rad_for = {'b': -0.20, 'c': -0.04, 'd': -0.20}
    incoming_alpha = {l: eatt[(l, 'a')] for l in ('b', 'c', 'd')}
    for l in ('b', 'c', 'd'):
        sx, sy = pos_l[l]
        w = incoming_alpha[l]
        lw = 0.35 + (w - 0.08) / 0.82 * 2.0
        ax.add_patch(FancyArrowPatch(
            (sx, sy), agg_left,
            connectionstyle=f'arc3,rad={rad_for[l]}',
            shrinkA=4.5, shrinkB=1.2,
            arrowstyle='-|>', color=C_ALPHA,
            lw=lw, mutation_scale=4, zorder=4,
        ))
    # Self pass-through: a -> h_a^(l). Solid grey (no alpha gating).
    sx, sy = pos_l['a']
    ax.add_patch(FancyArrowPatch(
        (sx, sy), self_left,
        connectionstyle='arc3,rad=-0.18',
        shrinkA=4.5, shrinkB=1.2,
        arrowstyle='-|>', color='#444',
        lw=0.45, mutation_scale=3.5, zorder=4,
    ))

    # ----- Layer ℓ+1 (RIGHT plate) ----------------------------------
    pl_r_x, pl_r_y = 7.0, pl_l_y
    origin_r, u_r, v_r = _draw_3d_plate(
        ax, pl_r_x, pl_r_y,
        thick=plate_thick, height=plate_height,
        depth_x=plate_depth_x, depth_y=plate_depth_y,
        side='right',
    )
    pos_r = _draw_k4_on_face(
        ax, origin_r, u_r, v_r,
        self_loc='', label_template='h',
        edge_attention=eatt,
        label_fontsize=5,
    )
    ax.text(pl_r_x + plate_thick / 2, pl_r_y + 0.20,
            r'$\ell\!+\!1$',
            ha='center', va='bottom', fontsize=6,
            color='#666', style='italic')

    # concat label + arrows from h_a^(ℓ) and agg_a into concat, then
    # one arrow from concat into h_a in layer ℓ+1.
    tx, ty = pos_r['a']
    concat_x = self_xy[0]
    concat_y = 0.5 * (self_xy[1] + agg_xy[1])
    ax.text(concat_x, concat_y, 'concat',
            ha='center', va='center',
            fontsize=6, fontweight='bold',
            color='#333', style='italic', zorder=6)
    # h_a^(ℓ) -> concat (downward)
    ax.add_patch(FancyArrowPatch(
        (self_xy[0], self_xy[1] - msg_h / 2),
        (concat_x, concat_y + 0.45),
        arrowstyle='-|>', color='#444',
        lw=0.45, mutation_scale=4, zorder=4,
    ))
    # agg_a -> concat (upward)
    ax.add_patch(FancyArrowPatch(
        (agg_xy[0], agg_xy[1] + msg_h / 2),
        (concat_x, concat_y - 0.45),
        arrowstyle='-|>', color='#444',
        lw=0.45, mutation_scale=4, zorder=4,
    ))
    # concat -> h_a in layer ℓ+1
    ax.add_patch(FancyArrowPatch(
        (concat_x + 0.55, concat_y), (tx, ty),
        connectionstyle='arc3,rad=-0.18',
        arrowstyle='-|>', color='#444',
        lw=0.5, mutation_scale=4.5,
        shrinkA=0.5, shrinkB=6, zorder=4,
    ))


def _draw_panel_f_mlp(ax):
    """MIDDLE sub-panel of panel (f): MLP head 256 → 128 → 64 → 32.
    Four amber 3D blocks in series with ReLU labels on the inter-
    block arrows. Output captions sit just below the block stack."""
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12.5)
    ax.set_aspect('equal', adjustable='box')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.5, 1.02, 'mlp layer',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=FS_F_HDR, fontweight='bold', color='#222')

    dims    = [256, 128, 64,  32]
    heights = [5.8, 4.5, 3.4, 2.4]
    blk_w   = 0.55
    blk_dep = 0.22
    spacing = 1.20
    centers_y = 6.25
    n_blocks = len(dims)
    total_blocks_w = n_blocks * blk_w + (n_blocks - 1) * spacing
    mlp_x_left0 = (6.0 - total_blocks_w) / 2

    left_edges, right_edges = [], []
    for i, (d, h) in enumerate(zip(dims, heights)):
        x_left = mlp_x_left0 + i * (blk_w + spacing)
        draw_conv_block(ax, x_left=x_left, y_center=centers_y,
                        width=blk_w, height=h, depth=blk_dep,
                        color=C_MLP, edgecolor=C_MLP_EDGE,
                        label=None)
        ax.text(x_left + blk_w / 2, centers_y, str(d),
                ha='center', va='center',
                fontsize=6, fontweight='bold', color='#222',
                zorder=6)
        left_edges.append(x_left)
        right_edges.append(x_left + blk_w)

    for src, dst in [(0, 1), (1, 2), (2, 3)]:
        x_src = right_edges[src] + blk_dep + 0.04
        x_dst = left_edges[dst] - 0.04
        ax.annotate('',
                    xy=(x_dst, centers_y),
                    xytext=(x_src, centers_y),
                    arrowprops=dict(arrowstyle='->', mutation_scale=2.5,
                                    lw=0.4, color='#444'))
        ax.text(0.5 * (x_src + x_dst),
                centers_y + max(heights) / 2 + 0.22,
                'ReLU', ha='center', va='bottom',
                fontsize=5, color='#666', style='italic')



def _draw_panel_f_predictions(ax):
    """RIGHT column of panel (f): per-location predictions on the K4.
    K4 layout matches panel (e) — same diamond positions, same node
    size — so panels (e) and (f) are visually paired. Each Loc node
    carries its predicted (R_e [CP_lo, CP_hi], µ, SSS) on a bordered
    card. Predicted index case (Loc_a) gets a gold border + gold star
    + CP set caption (classification head)."""
    # xlim and ylim match panel (e)'s K4 coordinate system EXACTLY,
    # so the K4 here sits at the same physical x-coords as panel (e)
    # above (both subplots occupy the right half of the figure under
    # aspect='equal').
    ax.set_xlim(-2, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal', adjustable='box')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title sits ABOVE the axes (axes-frac 1.02) so it doesn't
    # overlap the a-card (whose top edge sits very close to ylim=10).
    ax.text(0.5, 1.02, 'per-location predictions',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=FS_F_HDR, fontweight='bold', color='#222')

    # K4 — same diamond layout, node size, AND directed-arrow style
    # as panel (e). No red highlight on Loc_a — the predicted index
    # case is already marked by the gold-bordered prediction card.
    k4_positions = {
        'a': (6.0, 7.6),
        'b': (3.4, 5.0),
        'c': (6.0, 2.4),
        'd': (8.6, 5.0),
    }
    draw_k4_graph(
        ax, target_loc='',
        positions=k4_positions,
        node_size=18,
        edge_attention={
            ('a', 'b'): 0.45, ('b', 'a'): 0.70,
            ('a', 'c'): 0.30, ('c', 'a'): 0.55,
            ('a', 'd'): 0.20, ('d', 'a'): 0.50,
            ('b', 'c'): 0.55, ('c', 'b'): 0.35,
            ('b', 'd'): 0.40, ('d', 'b'): 0.65,
            ('c', 'd'): 0.25, ('d', 'c'): 0.45,
        },
    )

    # Per-node prediction cards (scaled for the panel-(e)-sized K4).
    # a/c offsets keep cards just inside ylim(0, 10). b/d offsets
    # widened so the larger card_w (4.40) doesn't overlap nodes.
    card_offsets = {
        'a': (0.0,  +1.40),
        'b': (-2.90, 0.0),
        'c': (0.0,  -1.40),
        'd': (+2.90, 0.0),
    }
    for loc, (dx, dy) in card_offsets.items():
        nx, ny = k4_positions[loc]
        _draw_pred_card(ax, nx + dx, ny + dy, loc,
                        is_target=(loc == PANEL_F_PRED_INDEX))

    # Single-line classification readout: ★ + index-case prediction +
    # CP set inline. Font / color match the panel-(a) caption
    # ("Loc_a, the index case") so the question-answer pair reads
    # consistently across panels. Placed below the K4 (mirrors the
    # title that sits above at axes-frac 1.02) so the upper-right
    # area stays empty and the figure trims tighter horizontally.
    star_x, star_y = 3.30, -0.6
    ax.plot(star_x, star_y, marker='*', markersize=7,
            markerfacecolor='#FFD43B', markeredgecolor=C_INDEX_GLD,
            markeredgewidth=0.35, linestyle='None', zorder=8,
            clip_on=False)
    cp_set_str = ', '.join(f'Loc$_{{{l}}}$' for l in PANEL_F_PRED_CP_SET)
    ax.text(star_x + 0.45, star_y,
            f'source case: Loc$_{{{PANEL_F_PRED_INDEX}}}$  '
            f'[{cp_set_str}]',
            ha='left', va='center',
            fontsize=FS_PANEL_A_QUES, fontweight='bold', style='italic',
            color=C_INDEX_GLD, zorder=8, clip_on=False)

    # Inlet anchor for inter-axes arrow from MLP. Pointed at the
    # left edge of Loc_b's card (the leftmost card).
    in_x = card_offsets['b'][0] + k4_positions['b'][0] - 1.20 - 0.10
    in_y = k4_positions['b'][1]
    return in_x, in_y


def _draw_panel_f(fig, subplotspec, args):
    """Render panel (c): inference pipeline.

    LEFT    — message-passing layer (LGAT)
    MIDDLE  — MLP head (256 -> 128 -> 64 -> 32)
    RIGHT   — per-location predictions K4
    Two arrow blocks between, carrying h_a^(L) (post-MP node embedding)
    and y_hat_a (per-location prediction). Width ratios chosen so that,
    under aspect='equal', the three main sub-panels render at the same
    physical height (so titles and bottoms align):
        W_mp/H = 11/12.5, W_mlp/H = 6/12.5, W_pred/H = 16/10
        => ratios 0.88 : 0.48 : 1.6 ≈ 11 : 6 : 20.
    Pred takes ~50% of row width so its K4 sits in the right half of
    the figure — same physical x-range as panel (b)'s graph-assemble
    K4, so the two K4 diamonds align in x-coordinate when stacked.
    """
    # First column is a left spacer (no subplot) so mp+mlp group sits
    # shifted right within the panel.
    gs_f = subplotspec.subgridspec(
        1, 6,
        width_ratios=[2.5, 11.0, 1.5, 6.0, 1.5, 20.0],
        wspace=0.04,
    )
    ax_mp = fig.add_subplot(gs_f[0, 1])
    _draw_panel_f_mp(ax_mp, args)

    ax_arr1 = fig.add_subplot(gs_f[0, 2])
    _draw_arrow_block(ax_arr1, top_text='', bottom_text='')

    ax_mlp = fig.add_subplot(gs_f[0, 3])
    _draw_panel_f_mlp(ax_mlp)

    ax_arr2 = fig.add_subplot(gs_f[0, 4])
    _draw_arrow_block(ax_arr2, top_text='', bottom_text='')

    ax_pred = fig.add_subplot(gs_f[0, 5])
    _draw_panel_f_predictions(ax_pred)

    return ax_mp, ax_mlp, ax_pred
# =====================================================================
# SECTION H: Main entry point.
# =====================================================================

def main():
    """Render the merged conceptual figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_pdf',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'fig1.pdf'),
        help='Output PDF path (default: alongside this script)',
    )
    parser.add_argument('--seed', type=int, default=11,
                        help='RNG seed for tree topology / event types')
    parser.add_argument('--n_tips', type=int, default=50,
                        help='Number of tips in the generated tree')
    parser.add_argument('--a_subclade_target', type=int, default=32)
    parser.add_argument('--a_exit_migrations', type=int, default=4)
    parser.add_argument('--n_other_migrations', type=int, default=3)
    parser.add_argument('--target_loc', type=str, default='a',
                        choices=['a', 'b', 'c', 'd'])
    parser.add_argument('--initial_loc', type=str, default='b',
                        choices=['a', 'b', 'c', 'd'])
    args = parser.parse_args()

    if args.initial_loc == args.target_loc:
        parser.error("--initial_loc must differ from --target_loc so "
                     "the target location's MRCA is not the root")

    # Top-row height = each top sub-panel's width = 11.11" so the
    # engine / tree / K4 slots are SQUARE. aspect='equal' inside
    # draw_engine/draw_tree then fills the entire slot.
    # Third row (panel f) holds the combined LGAT message-passing +
    # MLP-head panel alongside the per-location predictions K4.
    fig = plt.figure(figsize=(8.42, 11.6))
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[13.0, 26.0, 13.5],
        hspace=0.04,
    )

    ax_a1, ax_a2 = _draw_panel_a(fig, outer[0])
    ax_tree, ax_kde, ax_conv, ax_graph = _draw_panel_b(
        fig, outer[1], args,
    )
    ax_mp, ax_mlp, ax_pred = _draw_panel_f(fig, outer[2], args)

    # Panel letters — all three at the SAME fig-x (left edge of the
    # outer gridspec, taken from ax_tree which sits flush with the
    # column edge in panel b). Each label's fig-y is the top of its
    # respective panel so they sit at the title row of each panel.
    panel_label_kw = dict(fontsize=14, fontweight='bold',
                          va='top', ha='left')
    bb_tree = ax_tree.get_position()
    bb_a1   = ax_a1.get_position()
    bb_mp   = ax_mp.get_position()
    label_x = bb_tree.x0
    fig.text(label_x, bb_a1.y1,   'a', **panel_label_kw)
    fig.text(label_x, bb_tree.y1, 'b', **panel_label_kw)
    fig.text(label_x, bb_mp.y1,   'c', **panel_label_kw)

    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f'Saved: {args.out_pdf}')


if __name__ == '__main__':
    main()
