#!/usr/bin/env python3
"""Conceptual figure: simulated outbreak tree with events colored by type.

Mirrors the color palette of simulation_engine.pdf:

    red    - infectious lineage (I state); branch segments
    blue   - infection event (S+I -> 2I; lineage bifurcates)
    green  - sampling event (I -> R + sample; becomes a tip)
    grey   - removal event (I -> R; lineage dies)
    purple - migration event (I[i] -> I[j]; lineage changes location)

The figure shows the *full* transmission tree (12 lineages). Every
branch is drawn as an infectious lineage (red); the event type at the
terminus is carried by the tip marker (green dot for sampled, grey X
for removed).

Each event also carries a location tag ("Loc a", "Loc b", ...) so the
tree's location bookkeeping stays consistent with simulation_engine_4loc:
the seed is at Location a (index case), each lineage inherits its
parent's location through infection events, and flips at migration
events (e.g. "Loc a -> Loc b").

Saves simulated_tree_concept.pdf next to this script.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Tree layout: 12 tips arranged as 3 subclades of 4. Hierarchy:
#
#   root_inf
#   ├── SC1 ──┬── pair01 → tip0, tip1
#   │        └── pair23 → tip2, tip3
#   └── L3_join
#       ├── SC2 ──┬── pair45 → tip4, tip5
#       │        └── pair67 → tip6, tip7
#       └── SC3 ──┬── pair89 → tip8, tip9
#                └── pair1011 → tip10, tip11
# ---------------------------------------------------------------------

# Tip y-coordinates (evenly spaced).
TIP_Y = [0.4 + 0.6 * i for i in range(12)]

# Tip terminal x and event type (4 sampled at the "present" x = 8.0,
# one in each location a/b/c/d; 8 removed at varied earlier times).
TIP_END_X = [6.0, 8.0, 7.0, 5.0, 8.0, 7.5, 6.0, 8.0, 7.0, 5.0, 8.0, 6.5]
TIP_TYPES = ['removed', 'sampled', 'removed', 'removed',
             'sampled', 'removed', 'removed', 'sampled',
             'removed', 'removed', 'sampled', 'removed']

# Internal infection nodes: label -> (x, y).
# y is the midpoint of the node's two children's y-coordinates.
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

# Bifurcation children: parent -> [(child_label, child_y), (child_label, child_y)].
# child_label is an int for tips, str for internal nodes.
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

SEED_X = -0.3   # where the seed lineage starts on the time axis

# ---------------------------------------------------------------------
# Publication font hierarchy. Mirrors the constants in
# simulation_engine_4loc.py so the composite (conceptual.py) reads at
# uniform weight across panels (a), (b), and (c).
# ---------------------------------------------------------------------
FS_EVENT_TAG = 14   # per-event "Loc a" tags next to markers
FS_MIG_TAG   = 14   # "Loc a -> Loc b" migration direction
FS_INDEX_CAP = 16   # "index case" caption above the gold star
FS_TIME      = 16   # "Time" axis label
FS_LEGEND    = 16   # legend below the tree

LW_BRANCH    = 2.8   # infectious-lineage branch width
MS_INF       = 18    # internal infection (blue circle)
MS_SAMP      = 20    # sampled tip (green circle)
MS_REM       = 19    # removed tip (grey X)
MS_MIG       = 17    # migration event (purple diamond)
MS_INDEX     = 34    # index-case star at the seed

# ---------------------------------------------------------------------
# Per-event location tracking — keeps the conceptual tree consistent
# with simulation_engine_4loc.pdf. The seed is at Location a (index
# case); each lineage inherits its parent's location through infection
# events and flips at migration events.
#
# Migrations on inbound branches (chosen so the four locations all
# appear in the tree):
#   mig1 (root_inf -> SC1)  : a -> b
#   mig2 (SC1     -> pair23): b -> c
#   mig3 (L3_join -> SC3)   : a -> d
#   mig4 (SC2     -> pair67): a -> c
# ---------------------------------------------------------------------

NODE_LOCATIONS = {
    'root_inf': 'a',     # index case
    'L3_join':  'a',     # inherits from root_inf
    'SC1':      'b',     # mig1: a -> b
    'SC2':      'a',     # inherits from L3_join
    'SC3':      'd',     # mig3: a -> d
    'pair01':   'b',     # inherits from SC1
    'pair23':   'c',     # mig2: b -> c
    'pair45':   'a',     # inherits from SC2
    'pair67':   'c',     # mig4: a -> c
    'pair89':   'd',     # inherits from SC3
    'pair1011': 'd',     # inherits from SC3
}
TIP_LOCATIONS = ['b', 'b', 'c', 'c', 'a', 'a',
                 'c', 'c', 'd', 'd', 'd', 'd']

# (x, y, from_loc, to_loc) for each migration event
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


def draw_tree(ax, with_legend=True):
    """Draw the conceptual outbreak-tree onto `ax`.

    Self-contained: sets its own xlim/ylim/aspect/spines/ticks/time-axis
    and (optionally) legend. Works inside any Axes.
    """
    # Colors match simulation_engine.pdf
    C_I    = "#D04F4F"   # I state / infectious lineage
    C_INF  = "#2E7DBF"   # infection event
    C_REM  = "#888888"   # removal event / pruned branch
    C_SAMP = "#3CB371"   # sampling event
    C_MIG  = "#7B4FB4"   # migration event

    # ---- Seed segment (root lineage before first infection) --------
    root_x, root_y = NODES['root_inf']
    ax.plot([SEED_X, root_x], [root_y, root_y],
            color=C_I, lw=LW_BRANCH, solid_capstyle='round', zorder=1)

    # ---- Bifurcation branches (vertical at parent + horizontal to each child)
    for parent, kids in CHILDREN.items():
        px = NODES[parent][0]
        c1_y, c2_y = kids[0][1], kids[1][1]

        # Vertical at parent's x, connecting children's y's
        ax.plot([px, px], [c1_y, c2_y], color=C_I, lw=LW_BRANCH, zorder=1)

        # Horizontal segment to each child
        for child_label, child_y in kids:
            if isinstance(child_label, int):
                child_x = TIP_END_X[child_label]
            else:
                child_x = NODES[child_label][0]
            ax.plot([px, child_x], [child_y, child_y],
                    color=C_I, lw=LW_BRANCH, solid_capstyle='round',
                    zorder=1)

    # ---- Internal infection markers (blue dots) + location tags -----
    # Location label sits to the right of the marker, matching the
    # placement used for sampled/removed tips. Label color matches the
    # event type (blue for infection).
    for label, (x, y) in NODES.items():
        ax.plot(x, y, 'o', color=C_INF, markersize=MS_INF, zorder=3,
                markeredgecolor='white', markeredgewidth=1.6)
        loc = NODE_LOCATIONS[label]
        ax.text(x + 0.24, y, f'Loc {loc}', fontsize=FS_EVENT_TAG,
                fontweight='bold', style='italic',
                color=C_INF, ha='left', va='center', zorder=4)

    # ---- Tip markers + location tags --------------------------------
    # Label color matches the event type (green = sampling, grey = removal).
    for i in range(12):
        x, y = TIP_END_X[i], TIP_Y[i]
        if TIP_TYPES[i] == 'sampled':
            ax.plot(x, y, 'o', color=C_SAMP, markersize=MS_SAMP, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.8)
            label_color = C_SAMP
        else:
            ax.plot(x, y, 'X', color=C_REM, markersize=MS_REM, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.6)
            label_color = C_REM
        loc = TIP_LOCATIONS[i]
        ax.text(x + 0.26, y, f'Loc {loc}', fontsize=FS_EVENT_TAG,
                fontweight='bold', style='italic',
                color=label_color, ha='left', va='center', zorder=4)

    # ---- Migration events (purple) + direction tags -----------------
    for mx, my, from_loc, to_loc in MIGRATIONS_INFO:
        ax.plot([mx, mx], [my - 0.26, my + 0.26],
                color=C_MIG, lw=2.4, linestyle='--', zorder=2)
        ax.plot(mx, my, marker='D', color=C_MIG, markersize=MS_MIG, zorder=3,
                markeredgecolor='white', markeredgewidth=1.6)
        # Show the direction (e.g. "Loc a -> Loc b") above the diamond.
        ax.text(mx, my + 0.48,
                f'Loc {from_loc} $\\rightarrow$ Loc {to_loc}',
                fontsize=FS_MIG_TAG, fontweight='bold', style='italic',
                color=C_MIG, ha='center', va='bottom', zorder=4)

    # ---- Index-case marker at the root (gold star) ------------------
    # Matches the gold star used in panels (a) and (c): identifies
    # Location a as the true seed of the outbreak.
    ax.plot(SEED_X, root_y, marker='*', markersize=MS_INDEX,
            color='#FFC107', markeredgecolor='#8A6500',
            markeredgewidth=1.6, zorder=3)
    ax.text(SEED_X, root_y + 0.65, 'index case',
            ha='center', va='bottom',
            fontsize=FS_INDEX_CAP, fontweight='bold', style='italic',
            color='#8A6500')

    # ---- Cosmetics ---------------------------------------------------
    # ylim_bottom is tight against the "Time" label below the time
    # arrow (no extra whitespace) so the legend can dock right under
    # the timescale via bbox_to_anchor=(_, -0.005).
    ax.set_xlim(-1.6, 9.2)
    ax.set_ylim(-1.35, 9.2)
    ax.set_aspect('equal')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Time arrow at the bottom of the tree content (just below tip 0)
    ax.annotate('', xy=(8.7, -0.55), xytext=(SEED_X, -0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
    ax.text(0.5 * (SEED_X + 8.7), -1.05, 'Time',
            ha='center', fontsize=FS_TIME, fontweight='bold')

    # ---- Legend ------------------------------------------------------
    if with_legend:
        handles = [
            Line2D([0], [0], color=C_I, lw=3.6,
                   label='Infectious lineage (I)'),
            Line2D([0], [0], marker='o', color=C_INF, markersize=18,
                   linestyle='none', label='Infection event'),
            Line2D([0], [0], marker='D', color=C_MIG, markersize=17,
                   linestyle='none', label='Migration event'),
            Line2D([0], [0], marker='o', color=C_SAMP, markersize=20,
                   linestyle='none', label='Sampling event'),
            Line2D([0], [0], marker='X', color=C_REM, markersize=20,
                   linestyle='none', label='Removal event'),
        ]
        # Center the legend horizontally on the time arrow's midpoint
        # (not the axes midpoint), so the legend column lines up with
        # the timescale beneath the tree.
        x_lo, x_hi = ax.get_xlim()
        time_mid_x = 0.5 * (SEED_X + 8.7)
        legend_x_axes = (time_mid_x - x_lo) / (x_hi - x_lo)
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(legend_x_axes, -0.005),
                  frameon=False, fontsize=FS_LEGEND, ncol=3,
                  handletextpad=0.9, handlelength=2.5,
                  columnspacing=2.0, labelspacing=0.9)


def main():
    """Render the tree figure as a standalone PDF."""
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "simulated_tree_concept.pdf")
    fig, ax = plt.subplots(figsize=(12, 12))
    draw_tree(ax, with_legend=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
