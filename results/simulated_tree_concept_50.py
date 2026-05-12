#!/usr/bin/env python3
"""Conceptual figure: STEPHY's per-location virtual subtree extraction
and how it feeds the graph neural network.

2x2 layout:

  LEFT COLUMN — data view
    (a) Full simulated tree. The index case sits at a non-target
        location (gold star at the seed). A single migration on an
        interior branch establishes the Loc_a clade; a few exit
        migrations scatter Loc_a tips so they're non-monophyletic.
        The induced subtree of Loc_a sampled tips — exactly what the
        CBLV encoder sees — is drawn solid red. Other lineages are
        solid grey, EXCEPT one immediately-adjacent existing clade
        that's collapsed into a dashed grey triangle to convey
        'rest of tree, not shown in detail'.
    (b) Loc_a's bell-shaped tip-time KDE (with rug strip) and a
        synthetic Loc_b density side by side — the per-location pair
        DTW compares to produce edge features.

  RIGHT COLUMN — model view
    (c) Per-node encoder. The Loc_a virtual subtree is encoded as a
        CBLV (M x 4) tensor — M is a fixed pre-set width, the top N
        rows hold the actual subtree tips, the bottom M - N rows are
        zero-padded — and fed through a 3-branch conv encoder
        (plain / stride / dilate, concatenated -> 96-d). In parallel,
        a 1 x 5 aux summary (mrca depth, earliest tip time, latest
        tip time, avg branch length, n_tips) goes through AuxBranch
        MLP -> 32-d.
        The concatenation gives the 128-d node feature for Loc_a,
        which is exactly what ``stephy/model.py`` consumes.
    (d) The K4 location graph (Loc_a/b/c/d in a diamond) the GNN
        operates on. Loc_a is highlighted in red. Its node feature
        comes from (c); the Loc_a-Loc_b edge feature comes from (b)
        via DTW.

Cross-figure arrows make the data flow explicit:
    (a) MRCA region -> (c) mini-tree input
    (c) node feature -> (d) Loc_a node feature
    (b) KDE pair    -> DTW glyph -> (d) Loc_a-Loc_b edge feature

Visual language:

    red, lw=4.6      - branches of the Loc_a induced subtree
    grey, lw=2.8     - regular background lineages
    grey dashed △    - one adjacent clade collapsed into a triangle
    green circle     - sampling event on a Loc_a tip
    small grey dot   - other sampled tip
    gold star        - index case at the seed
    solid red dot    - MRCA of Loc_a with "subtree of Loc_a" caption
    amber 3D blocks  - conv layer feature maps (panel c)
    blue strip       - node feature vector
    purple strip     - edge feature vector (DTW-derived)

Defaults: 50-tip tree with ~24 Loc_a tips (entry-clade target = 32,
trimmed by 4 exit migrations). ``--seed`` controls the random
topology so the figure is reproducible. ``--n_tips``,
``--a_subclade_target``, ``--a_exit_migrations``,
``--n_other_migrations``, ``--target_loc``, ``--initial_loc`` are
exposed for quick variants.

Usage:
    python3 simulated_tree_concept_50.py
    python3 simulated_tree_concept_50.py --seed 7 --n_tips 50
"""
import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Polygon, Rectangle


# Publication-quality matplotlib defaults (mirrors conceptual.py).
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset':  'stixsans',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.linewidth':    1.0,
})


# ---------------------------------------------------------------------
# Palette + font hierarchy.
# ---------------------------------------------------------------------
C_HL    = "#D04F4F"   # red — Loc_a induced subtree
C_DIM   = "#BFBFBF"   # grey — background lineages
C_SAMP  = "#3CB371"   # green — sampling event on a highlighted tip
C_GOLD  = "#FFC107"
C_GOLD_DARK = "#8A6500"

FS_INDEX_CAP = 22
FS_TIME      = 24
FS_LEGEND    = 18
FS_CAPTION   = 22
FS_KDE_HDR   = 22


# ---------------------------------------------------------------------
# Tree node classes.
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Tree generation, ladderization.
# ---------------------------------------------------------------------
def gen_tree(n_tips, T, rng, sample_frac=0.55):
    """Recursively build a binary tree with `n_tips` leaves over time
    horizon T.

    Sampled tip end times are spread across the latter half of each
    lineage's lifespan (serial sampling), not pinned to T — otherwise
    every sampled tip would line up on a single vertical at the
    present, which looks unrealistic for an outbreak with sequencing
    spread over time."""
    next_id = [0]

    def _new_id():
        i = next_id[0]
        next_id[0] += 1
        return i

    def _build(start_x, n_leaves):
        if n_leaves == 1:
            sampled = rng.random() < sample_frac
            if sampled:
                # Latter 50% of this lineage's lifespan; capped at T.
                end_x = start_x + rng.uniform(0.50, 1.00) * (T - start_x)
            else:
                # Removed earlier in the lineage's lifespan.
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


# ---------------------------------------------------------------------
# Designed location assignment that guarantees Loc_a's MRCA is not the
# whole-tree root.
# ---------------------------------------------------------------------
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
    """Return `id(...)` for every node (internal + tip) in `root`'s subtree."""
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
    """Designed location assignment.

    1. Root = ``initial_loc`` (default 'b').
    2. Pick the non-root internal node whose subtree size is closest
       to ``a_subclade_target`` — call this ``a_clade_root`` — and
       place a single ``initial_loc -> target_loc`` entry migration
       there.
    3. Within ``a_clade_root``'s subtree, place
       ``a_exit_migrations`` *exit* migrations on random sub-branches.
       Each exit flips a sub-branch back to a non-target location, so
       Loc_a ends up scattered (non-monophyletic) within the focus
       tree rather than forming a clean monophyletic group.
    4. Outside that subtree place ``n_other_migrations`` migrations
       randomly, also drawing from non-target locations only.

    All non-root migration nodes pick a non-target, non-parent
    location — so target_loc only appears at the single entry point
    (``a_clade_root``).

    Returns ``a_clade_root`` (the entry point of the Loc_a clade,
    which is one ancestor of the dynamic MRCA computed by
    ``find_mrca``)."""
    LOCS = ['a', 'b', 'c', 'd']

    internals = _collect_internals(root)
    non_root = [n for n in internals if n is not root]

    # ---- Pick the entry node for the a-clade ----------------------
    deviations = [abs(_count_tips(n) - a_subclade_target) for n in non_root]
    min_dev = min(deviations)
    candidates = [n for n, d in zip(non_root, deviations) if d == min_dev]
    a_clade_root = candidates[int(rng.integers(len(candidates)))]
    a_clade_root.is_migration = True

    # ---- Exit migrations within the a-clade -----------------------
    # We pick small-to-medium sub-clades (2-5 tips each) so each exit
    # carves a visible gap in the Loc_a region without removing the
    # bulk of Loc_a tips.
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
        preferred = a_subclade_internals  # fall back if not enough
    n_exit = min(a_exit_migrations, len(preferred))
    if n_exit > 0:
        chosen = rng.choice(len(preferred), size=n_exit, replace=False)
        for idx in chosen:
            preferred[int(idx)].is_migration = True

    # ---- Migrations outside the a-clade ---------------------------
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

    # ---- Top-down assignment --------------------------------------
    def _assign(node, parent_loc):
        if isinstance(node, TipNode):
            node.location = parent_loc
            return
        if node is a_clade_root:
            # Single entry migration into target_loc.
            node.location = target_loc
            node.mig_from, node.mig_to = parent_loc, target_loc
        elif node.is_migration:
            # Always pick non-target, non-parent — this serves as both
            # within-a-clade EXIT (target -> non-target) and as a
            # regular outside migration (non-target -> non-target).
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
    """Return the deepest internal node whose subtree contains every
    target_loc (sampled) tip. With non-monophyletic Loc_a this may sit
    deeper than ``a_clade_root``."""
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
    """Resample target_loc sampled-tip ``end_x`` from a truncated
    Gaussian centered at ``mu_frac * T`` with std ``sigma_frac * T``,
    clipped to ``(start_x + small_offset, T)`` — produces a bell-shaped
    tip-time distribution in absolute time."""
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
    """Set ``in_induced=True`` for nodes in the subtree rooted at MRCA
    that lie on the path to a target_loc (sampled) tip. All other nodes
    keep ``in_induced=False``. This is what the CBLV encoder consumes."""
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
    """Pick the existing internal node whose subtree's tips all sit
    immediately below the MRCA's lowest tip y — i.e., the largest
    real clade visually adjacent to (and below) the Loc_a clade. We
    render that clade with dashed lines to convey 'rest of tree, not
    shown in detail'. Returns None if no such clade exists."""
    a_tips = list(_subtree_tips(mrca))
    y_min_a = min(t.y for t in a_tips)

    # Find every internal node whose subtree's tips are all below y_min_a;
    # among those, pick the one whose max-y is closest to y_min_a (the
    # largest immediately-adjacent clade).
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
    # Closest max-y first, ties broken by larger subtree.
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    return candidates[0][2]


def mark_dashed_clade(dashed_root):
    """Set ``in_dashed=True`` for every node in `dashed_root`'s subtree."""
    if dashed_root is None:
        return
    def _walk(n):
        n.in_dashed = True
        if isinstance(n, IntNode):
            for c in n.children:
                _walk(c)
    _walk(dashed_root)


# ---------------------------------------------------------------------
# Drawing.
# ---------------------------------------------------------------------
def draw_branches(ax, root, seed_x, lw_hl=4.6, lw_dim=2.8):
    """Draw branches:
      - red solid     iff both endpoints are in the Loc_a induced subtree
      - grey solid    regular background lineages
      - collapsed     when entering an `in_dashed` clade, the stem edge
                      is drawn dashed and the clade itself is rendered
                      as a dashed triangle (apex at the clade root,
                      base spanning the clade tips' y range); no
                      internal structure is shown."""
    DASH_STYLE = (0, (4, 3))

    # Seed segment: regular solid grey (root is, by construction, not the MRCA).
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
        # Apex -> top corner
        ax.plot([apex_x, base_x], [apex_y, y_max], **kw)
        # Apex -> bottom corner
        ax.plot([apex_x, base_x], [apex_y, y_min], **kw)
        # Vertical base
        ax.plot([base_x, base_x], [y_min, y_max], **kw)

    def _walk(node):
        if isinstance(node, TipNode):
            return
        for c in node.children:
            if c.in_dashed and not node.in_dashed:
                # Entry into the dashed clade: draw the stem edge dashed
                # then collapse the rest into a triangle.
                child_x = c.end_x if isinstance(c, TipNode) else c.x
                kw = dict(color=C_DIM, lw=lw_dim + 0.3, zorder=1,
                          linestyle=DASH_STYLE, solid_capstyle='butt')
                ax.plot([node.x, node.x], [node.y, c.y], **kw)
                ax.plot([node.x, child_x], [c.y, c.y], **kw)
                if isinstance(c, IntNode):
                    _draw_collapsed_triangle(c)
                continue  # don't recurse — internal structure suppressed
            if c.in_dashed and node.in_dashed:
                continue  # inside an already-collapsed clade

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
    """Highlighted tips (Loc_a sampled) keep the green-circle event
    encoding with a red outline. Regular background sampled tips get a
    small dim grey dot. Tips inside an in_dashed (abstracted) clade
    have NO marker — leaving only the dashed branch silhouette."""
    def _walk(node):
        if isinstance(node, TipNode):
            if node.in_dashed:
                pass  # no tip marker — clade is rendered abstractly
            elif node.in_induced:
                ax.plot(node.end_x, node.y, 'o',
                        color=C_SAMP, markersize=ms_hl, zorder=5,
                        markeredgecolor=C_HL, markeredgewidth=1.4)
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
    """Sum-of-Gaussians density estimate. Returns zero array if `times`
    is empty."""
    if len(times) == 0:
        return np.zeros_like(x_grid)
    arr = np.asarray(times)
    diff = (x_grid[:, None] - arr[None, :]) / bandwidth
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1)
    dens /= len(arr) * np.sqrt(2 * np.pi) * bandwidth
    return dens


def draw_tip_time_kde(ax_kde, root, target_loc, seed_x, x_axis_max,
                      bandwidth=0.40, T=8.0):
    """Bottom-panel KDE:
      - red curve: Gaussian KDE over the target_loc's sampled tip
        times, with a rug strip of actual tip positions
      - grey curve: a SYNTHETIC Loc_b tip-time density (no rug
        ticks because Loc_b tips aren't drawn on the tree). The
        grey curve illustrates that DTW edge features compare the
        target location's tip-time density against the
        densities of other locations."""
    tips = collect_tips(root)
    a_times = [t.end_x for t in tips
               if t.location == target_loc and t.sampled]

    x_grid = np.linspace(seed_x, x_axis_max, 400)
    kde_a = gauss_kde(a_times, x_grid, bandwidth=bandwidth)

    # Synthetic Loc_b density — different peak/spread from Loc_a so the
    # two curves are visibly distinct (which is the whole point of DTW
    # as an edge feature). Peak sits just left of Loc_a's so the two
    # bells are clearly distinct but neighboring.
    b_mu = T * 0.78
    b_sigma = T * 0.18
    kde_b_raw = np.exp(-0.5 * ((x_grid - b_mu) / b_sigma) ** 2)
    kde_b_raw /= (b_sigma * np.sqrt(2 * np.pi))
    # Match scale to kde_a's peak so both read as "densities" not "noise"
    kde_b = kde_b_raw * (kde_a.max() / max(kde_b_raw.max(), 1e-6)) * 0.85

    y_max = max(kde_a.max(), kde_b.max(), 1e-6) * 1.20

    # Loc_b first (background grey)
    ax_kde.fill_between(x_grid, 0, kde_b,
                        color=C_DIM, alpha=0.25, zorder=1)
    ax_kde.plot(x_grid, kde_b, color=C_DIM, lw=3.0, zorder=2)

    # Loc_a on top (highlighted red)
    ax_kde.fill_between(x_grid, 0, kde_a,
                        color=C_HL, alpha=0.30, zorder=3)
    ax_kde.plot(x_grid, kde_a, color=C_HL, lw=3.4, zorder=4)

    # Rug strip — Loc_a only.
    rug_y = -0.06 * y_max
    ax_kde.scatter(a_times, [rug_y] * len(a_times),
                   marker='|', color=C_HL, s=180, lw=2.4, zorder=5)

    # Panel captions.
    ax_kde.text(seed_x + 0.10, y_max * 0.93,
                f'Loc$_{target_loc}$ tip-time density',
                fontsize=FS_KDE_HDR, color=C_HL, fontweight='bold',
                ha='left', va='top')
    ax_kde.text(seed_x + 0.10, y_max * 0.65,
                r'Loc$_b$ tip-time density',
                fontsize=FS_KDE_HDR - 2, color=C_DIM, fontweight='bold',
                ha='left', va='top')

    ax_kde.set_ylim(-0.18 * y_max, y_max)
    for sp in ('top', 'right', 'left'):
        ax_kde.spines[sp].set_visible(False)
    ax_kde.spines['bottom'].set_visible(False)
    ax_kde.set_xticks([])
    ax_kde.set_yticks([])


def annotate_mrca(ax, mrca, target_loc):
    """Mark the MRCA node with a solid red dot + a "subtree of Loc_x"
    caption just above the dot."""
    ax.plot(mrca.x, mrca.y, marker='o', markersize=22,
            markerfacecolor=C_HL, markeredgecolor=C_HL,
            markeredgewidth=0.8, zorder=6)
    ax.text(mrca.x, mrca.y + 1.4, f'subtree of Loc$_{target_loc}$',
            ha='center', va='bottom',
            fontsize=FS_CAPTION, fontweight='bold',
            color=C_HL, zorder=7)


# ---------------------------------------------------------------------
# Right-column helpers — panel (c) conv decoder, panel (d) K4 graph.
# ---------------------------------------------------------------------

C_NODE_FEAT = "#3F76C0"   # blue   — node feature vector
C_EDGE_FEAT = "#7E57C2"   # purple — edge feature vector (DTW-derived,
                          #          matches migration purple in the
                          #          conceptual figures)
C_CONV      = "#F2A93B"   # amber  — conv-layer feature-map fill


def draw_mini_subtree(ax, cx, cy, w, h, color=C_HL, lw=3.0, n_tips=5):
    """Stylized comb subtree icon for panel (c) — n_tips horizontal twigs
    hanging off a single vertical spine, with green tip dots. Represents
    the Loc_a virtual subtree that the conv decoder consumes."""
    xf = lambda f: cx - w / 2 + f * w
    yf = lambda f: cy - h / 2 + f * h
    tip_yfs = np.linspace(0.08, 0.92, n_tips)
    ax.plot([xf(0.12), xf(0.12)], [yf(tip_yfs[0]), yf(tip_yfs[-1])],
            color=color, lw=lw, solid_capstyle='round', zorder=3)
    for tyf in tip_yfs:
        ax.plot([xf(0.12), xf(0.92)], [yf(tyf), yf(tyf)],
                color=color, lw=lw, solid_capstyle='round', zorder=3)
        ax.plot(xf(0.92), yf(tyf), 'o',
                color=C_SAMP, markersize=9,
                markeredgecolor=color, markeredgewidth=1.3, zorder=5)


def draw_cblv_grid(ax, x0, y0, width, height,
                   n_rows=4, n_cols=8, n_filled=None,
                   cmap_name='viridis', pad_facecolor='#e6e6e6',
                   pad_text_color='#888',
                   seed=7, border_color='#333', border_lw=1.2):
    """Stylized CBLV (M x 4) heatmap.

    The first ``n_filled`` rows (top of the heatmap, corresponding to
    actual subtree tips) get viridis colors. Remaining rows are
    rendered as light-grey cells with a small "0" mark — they convey
    the zero-padding applied when the subtree has fewer tips than the
    fixed CBLV width M. If ``n_filled`` is None, every row is filled
    (no padding visualized)."""
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
                        fontsize=7, color=pad_text_color, zorder=4)
    ax.add_patch(Rectangle(
        (x0, y0), width, height,
        facecolor='none', edgecolor=border_color, lw=border_lw,
        zorder=4,
    ))


def draw_conv_block(ax, x_left, y_center, width, height, depth,
                    color=C_CONV, edgecolor='#333', alpha=0.92,
                    label=None, label_fontsize=14, label_dy=0.32):
    """Render a 3D-ish conv feature-map block (front face + top + right
    parallelograms suggesting perspective)."""
    front = Rectangle((x_left, y_center - height / 2), width, height,
                      facecolor=color, edgecolor=edgecolor,
                      lw=1.4, alpha=alpha, zorder=3)
    ax.add_patch(front)
    top_pts = [
        (x_left,             y_center + height / 2),
        (x_left + depth,     y_center + height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,     y_center + height / 2),
    ]
    ax.add_patch(Polygon(top_pts, facecolor=color, edgecolor=edgecolor,
                         lw=1.0, alpha=alpha * 0.75, zorder=3))
    right_pts = [
        (x_left + width,         y_center - height / 2),
        (x_left + width + depth, y_center - height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,         y_center + height / 2),
    ]
    ax.add_patch(Polygon(right_pts, facecolor=color, edgecolor=edgecolor,
                         lw=1.0, alpha=alpha * 0.55, zorder=3))
    if label is not None:
        ax.text(x_left + width / 2, y_center - height / 2 - label_dy,
                label, ha='center', va='top',
                fontsize=label_fontsize, color='#333')


def draw_feature_strip(ax, x_left, y_bottom, n_boxes,
                       box_w, box_h, color,
                       alphas=None, edgecolor='#333', edgewidth=0.8,
                       zorder=4):
    """Render a feature-vector strip — n_boxes adjacent filled rectangles.
    Deterministic per-box alpha so the figure is reproducible."""
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
                  edge_color='#9a9a9a', edge_lw=2.6,
                  hl_color=C_HL, dim_color=C_DIM,
                  hl_edge='#7a1f1f', dim_edge='#5a5a5a'):
    """Diamond-layout K4 with 4 location nodes. Loc_a (or whatever
    ``target_loc`` is) is rendered in red so it visually receives the
    node feature derived in panel (c)."""
    if positions is None:
        positions = {
            'a': (6.0, 7.8),  # N
            'b': (9.4, 5.0),  # E
            'c': (6.0, 2.2),  # S
            'd': (2.6, 5.0),  # W
        }
    locs = ['a', 'b', 'c', 'd']
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
                color=fc, markeredgecolor=ec, markeredgewidth=2.0,
                zorder=5)
        ax.text(x, y, l,
                ha='center', va='center',
                fontsize=22, fontweight='bold',
                color='white', zorder=6)
    return positions


def draw_dtw_glyph(ax, cx, cy, size=1.2, color='#444', lw=1.4,
                   path_color=None, label='DTW', label_fontsize=13):
    """Small DTW cost-matrix glyph: a square grid with a near-diagonal
    warping path drawn through it. Sits in the (b) -> (d) edge-feature
    pipeline."""
    if path_color is None:
        path_color = C_EDGE_FEAT
    s2 = size / 2
    ax.add_patch(Rectangle((cx - s2, cy - s2), size, size,
                           facecolor='white', edgecolor=color,
                           lw=lw, zorder=3))
    n = 6
    for k in range(1, n):
        f = k / n
        ax.plot([cx - s2 + f * size, cx - s2 + f * size],
                [cy - s2, cy + s2],
                color=color, lw=0.5, alpha=0.35, zorder=3)
        ax.plot([cx - s2, cx + s2],
                [cy - s2 + f * size, cy - s2 + f * size],
                color=color, lw=0.5, alpha=0.35, zorder=3)
    path = [(0.06, 0.08), (0.22, 0.20), (0.38, 0.42),
            (0.55, 0.55), (0.72, 0.78), (0.92, 0.94)]
    px = [cx - s2 + p[0] * size for p in path]
    py = [cy - s2 + p[1] * size for p in path]
    ax.plot(px, py, color=path_color, lw=2.6,
            solid_capstyle='round', zorder=4)
    if label is not None:
        ax.text(cx, cy - s2 - 0.22, label,
                ha='center', va='top',
                fontsize=label_fontsize, fontweight='bold',
                color='#333')


def main():
    """Render the 50-tip 'MRCA != root' concept figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_pdf',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'simulated_tree_concept_50.pdf'),
        help='Output PDF path (default: alongside this script)',
    )
    parser.add_argument('--seed', type=int, default=11,
                        help='RNG seed for tree topology / event types')
    parser.add_argument('--n_tips', type=int, default=50,
                        help='Number of tips in the generated tree')
    parser.add_argument('--a_subclade_target', type=int, default=32,
                        help='Initial tip count for the entry-clade '
                             '(the figure picks the non-root internal '
                             'node whose subtree size is closest to '
                             'this value). Exit migrations later prune '
                             'some sub-branches from Loc_a.')
    parser.add_argument('--a_exit_migrations', type=int, default=4,
                        help='Number of exit migrations placed within '
                             'the a-clade — each flips a sub-branch '
                             'back to non-target, making Loc_a '
                             'non-monophyletic / sparse')
    parser.add_argument('--n_other_migrations', type=int, default=3,
                        help='Migrations placed outside the Loc_a clade '
                             '(for visual variety)')
    parser.add_argument('--target_loc', type=str, default='a',
                        choices=['a', 'b', 'c', 'd'],
                        help='Which location to highlight as the '
                             'extracted virtual subtree')
    parser.add_argument('--initial_loc', type=str, default='b',
                        choices=['a', 'b', 'c', 'd'],
                        help='Location of the index case at the seed '
                             '(must differ from target_loc to satisfy '
                             'the "MRCA != root" scenario)')
    args = parser.parse_args()

    if args.initial_loc == args.target_loc:
        parser.error("--initial_loc must differ from --target_loc so "
                     "the target location's MRCA is not the root")

    rng = np.random.default_rng(args.seed)
    T = 8.0
    # sample_frac=1.0 → every tip is sampled, so the figure shows only
    # the tree STEPHY/BEAST2 actually sees (no removal X markers).
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
    # Bell-shape the Loc_a tip-time distribution (otherwise sampling
    # bunches near the present and the KDE is left-skewed).
    rebias_loc_a_tip_times(root, args.target_loc, T, rng,
                           mu_frac=0.55, sigma_frac=0.18)
    # With non-monophyletic Loc_a the MRCA may sit deeper than the
    # entry-clade root — recompute it dynamically.
    mrca = find_mrca(root, args.target_loc, sampled_only=True)
    mark_induced_subtree(root, mrca,
                         target_loc=args.target_loc, sampled_only=True)

    # Mark the existing clade immediately below the Loc_a clade as
    # 'in_dashed' — to be rendered as a dashed silhouette signalling
    # 'rest of tree, not shown in detail'.
    dashed_root = find_below_loca_clade(root, mrca)
    mark_dashed_clade(dashed_root)

    SEED_X = -0.6
    x_axis_max = T + 1.6
    time_x_end = T + 0.15

    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[10, 3.5],
        hspace=0.06, wspace=0.06,
    )
    ax_tree  = fig.add_subplot(gs[0, 0])
    ax_kde   = fig.add_subplot(gs[1, 0], sharex=ax_tree)
    ax_conv  = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[1, 1])

    # ---- Panel (a): tree axes --------------------------------------
    draw_branches(ax_tree, root, SEED_X, lw_hl=4.6, lw_dim=2.8)
    draw_tips(ax_tree, root, ms_hl=15, ms_dim=10)
    annotate_mrca(ax_tree, mrca, args.target_loc)

    # Index-case star at the seed.
    ax_tree.plot(SEED_X, root.y, marker='*', markersize=42,
                 color=C_GOLD, markeredgecolor=C_GOLD_DARK,
                 markeredgewidth=2.0, zorder=6)
    ax_tree.text(SEED_X, root.y + 1.7, 'index case',
                 ha='center', va='bottom',
                 fontsize=FS_INDEX_CAP, fontweight='bold', style='italic',
                 color=C_GOLD_DARK)

    ax_tree.set_xlim(SEED_X - 1.4, x_axis_max + 0.1)
    ax_tree.set_ylim(-1.5, args.n_tips + 1.5)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_tree.spines[sp].set_visible(False)
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])

    # ---- Panel (b): KDE axes ---------------------------------------
    draw_tip_time_kde(ax_kde, root, args.target_loc, SEED_X, time_x_end,
                      bandwidth=0.45, T=T)

    # Time arrow + label under the KDE panel (shared x-axis with tree).
    y_lo = ax_kde.get_ylim()[0]
    arrow_y = y_lo * 0.65
    ax_kde.annotate('', xy=(time_x_end, arrow_y),
                    xytext=(SEED_X, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=3.0,
                                    mutation_scale=24))
    ax_kde.text(0.5 * (SEED_X + time_x_end), y_lo * 0.95, 'Time',
                ha='center', va='top',
                fontsize=FS_TIME, fontweight='bold')

    # ---- Panel (c): CBLV encoder + Aux MLP -> node feature ---------
    # Mirrors stephy/model.py at a conceptual level:
    #   CBLV (M x 4) -> 3-branch CNN (plain / stride / dilate) -> 1 x 96
    #   Aux  (1 x 5) -> AuxBranch MLP                          -> 1 x 32
    #   concat -> (1 x 128) node feature
    ax_conv.set_xlim(0, 12)
    ax_conv.set_ylim(0, 10)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_conv.spines[sp].set_visible(False)
    ax_conv.set_xticks([])
    ax_conv.set_yticks([])

    # Panel header.
    ax_conv.text(6.0, 9.65, 'Per-node encoder',
                 ha='center', va='center',
                 fontsize=FS_KDE_HDR + 4, fontweight='bold', color='#222')

    # --- Unified tensor-cell size (used for CBLV, Aux, node feature) -
    CELL = 0.42

    # --- CBLV (M x 4) input (top-left) ------------------------------
    # CBLV width is a *fixed* M >= N (N = # tips in this subtree).
    # The top N rows hold the actual encoding; the remaining M - N
    # rows are zero-padded.
    cblv_rows, cblv_cols = 12, 4   # M = 12 (visual), N = cblv_n_filled
    cblv_n_filled = 10             # illustrative N (so M - N = 2)
    cblv_w = cblv_cols * CELL
    cblv_h = cblv_rows * CELL
    cblv_x0 = 0.55
    cblv_y0 = 2.85
    draw_cblv_grid(ax_conv, x0=cblv_x0, y0=cblv_y0,
                   width=cblv_w, height=cblv_h,
                   n_rows=cblv_rows, n_cols=cblv_cols,
                   n_filled=cblv_n_filled, seed=7)

    # Right-side brackets labelling the N (real) and M-N (pad) regions.
    bracket_x = cblv_x0 + cblv_w + 0.06
    bracket_tip = 0.13
    y_top    = cblv_y0 + cblv_h
    y_split  = cblv_y0 + (cblv_rows - cblv_n_filled) * CELL
    y_bottom = cblv_y0
    # N bracket
    ax_conv.plot(
        [bracket_x, bracket_x + bracket_tip,
         bracket_x + bracket_tip, bracket_x],
        [y_top, y_top, y_split, y_split],
        color='#444', lw=1.4, solid_capstyle='butt',
    )
    ax_conv.text(bracket_x + bracket_tip + 0.08,
                 (y_top + y_split) / 2, 'N',
                 ha='left', va='center',
                 fontsize=16, fontweight='bold', color='#222')
    # M - N (zero pad) bracket
    ax_conv.plot(
        [bracket_x, bracket_x + bracket_tip,
         bracket_x + bracket_tip, bracket_x],
        [y_split, y_split, y_bottom, y_bottom],
        color='#888', lw=1.4, solid_capstyle='butt',
    )
    ax_conv.text(bracket_x + bracket_tip + 0.08,
                 (y_split + y_bottom) / 2,
                 r'$M\!-\!N$' + '\nzero pad',
                 ha='left', va='center',
                 fontsize=11, color='#666')

    ax_conv.text(cblv_x0 + cblv_w / 2, cblv_y0 - 0.30,
                 r'CBLV $(M \times 4)$',
                 ha='center', va='top',
                 fontsize=17, fontweight='bold', color='#222')
    ax_conv.text(cblv_x0 + cblv_w / 2, cblv_y0 - 0.88,
                 r'$M$ = fixed unified dimension;'
                 ' \n'
                 r'$N$ = subtree tips',
                 ha='center', va='top',
                 fontsize=12, color='#666', style='italic')

    # --- 3-branch conv encoder (top-middle) -------------------------
    # Single labeled box per branch — conceptual, no kernel / stride /
    # dilation hyperparameters spelled out.
    branches = [
        (7.30, 'plain'),
        (5.45, 'stride'),
        (3.60, 'dilate'),
    ]
    br_x0, br_w, br_h, br_d = 4.10, 2.00, 1.20, 0.38
    cblv_exit = (cblv_x0 + cblv_w + 0.05,
                 cblv_y0 + cblv_h / 2)  # right edge midpoint
    for y_c, name in branches:
        draw_conv_block(ax_conv, x_left=br_x0, y_center=y_c,
                        width=br_w, height=br_h, depth=br_d,
                        color=C_CONV, label=None)
        ax_conv.text(br_x0 + br_w / 2, y_c, name,
                     ha='center', va='center',
                     fontsize=17, fontweight='bold', color='#333',
                     zorder=5)
        # CBLV -> branch
        ax_conv.annotate(
            '', xy=(br_x0 - 0.04, y_c),
            xytext=cblv_exit,
            arrowprops=dict(arrowstyle='->', mutation_scale=22, lw=1.8,
                            color='#555', zorder=4,
                            connectionstyle=f'arc3,rad={(y_c - cblv_exit[1]) / 14:.2f}'),
        )

    # --- Aux (1 x 5) input (bottom-left) ----------------------------
    aux_n, aux_bw, aux_bh = 5, CELL, CELL
    aux_x0 = cblv_x0 + cblv_w / 2 - (aux_n * aux_bw) / 2
    aux_y0 = 1.10
    draw_feature_strip(ax_conv, x_left=aux_x0, y_bottom=aux_y0,
                       n_boxes=aux_n, box_w=aux_bw, box_h=aux_bh,
                       color='#90C695')
    ax_conv.text(aux_x0 + aux_n * aux_bw / 2, aux_y0 - 0.30,
                 r'Aux $(1 \times 5)$',
                 ha='center', va='top',
                 fontsize=17, fontweight='bold', color='#222')
    ax_conv.text(aux_x0 + aux_n * aux_bw / 2, aux_y0 - 0.78,
                 'mrca depth / earliest tip / latest tip / avg BL / n_tips',
                 ha='center', va='top',
                 fontsize=11, color='#666', style='italic')

    # --- AuxBranch MLP (bottom-middle) ------------------------------
    aux_box_x = br_x0
    aux_box_y = aux_y0 + aux_bh / 2
    aux_box_w, aux_box_h, aux_box_d = br_w, 1.30, 0.38
    draw_conv_block(ax_conv, x_left=aux_box_x, y_center=aux_box_y,
                    width=aux_box_w, height=aux_box_h, depth=aux_box_d,
                    color='#90C695', label=None)
    ax_conv.text(aux_box_x + aux_box_w / 2, aux_box_y, 'AuxBranch\nMLP',
                 ha='center', va='center',
                 fontsize=15, fontweight='bold', color='#1f5933',
                 zorder=5)
    # Aux -> MLP
    ax_conv.annotate(
        '', xy=(aux_box_x - 0.04, aux_box_y),
        xytext=(aux_x0 + aux_n * aux_bw + 0.05, aux_y0 + aux_bh / 2),
        arrowprops=dict(arrowstyle='->', mutation_scale=22, lw=1.8,
                        color='#555', zorder=4),
    )

    # --- Convergence point + (1 x 128) node feature (right) ---------
    concat_x, concat_y = 8.20, 5.45

    # CNN branches -> convergence point   (with (1 x 96) caption near
    # the middle arrow).
    for y_c, _ in branches:
        ax_conv.annotate(
            '', xy=(concat_x, concat_y),
            xytext=(br_x0 + br_w + br_d + 0.05, y_c),
            arrowprops=dict(arrowstyle='->', mutation_scale=22, lw=1.8,
                            color='#444', zorder=4,
                            connectionstyle=f'arc3,rad={(concat_y - y_c) / 14:.2f}'),
        )
    ax_conv.text(br_x0 + br_w + br_d + 1.10, 7.70, r'$(1 \times 96)$',
                 ha='center', va='center',
                 fontsize=14, color='#444', fontweight='bold')

    # AuxBranch -> convergence point   (with (1 x 32) caption).
    ax_conv.annotate(
        '', xy=(concat_x, concat_y),
        xytext=(aux_box_x + aux_box_w + aux_box_d + 0.05, aux_box_y),
        arrowprops=dict(arrowstyle='->', mutation_scale=22, lw=1.8,
                        color='#444', zorder=4,
                        connectionstyle=f'arc3,rad={(concat_y - aux_box_y) / 14:.2f}'),
    )
    ax_conv.text(aux_box_x + aux_box_w + aux_box_d + 1.10, 1.80,
                 r'$(1 \times 32)$',
                 ha='center', va='center',
                 fontsize=14, color='#444', fontweight='bold')

    # Convergence -> node feature strip, with "concat" caption above.
    nf_n, nf_bw, nf_bh = 7, CELL, CELL
    nf_x0 = concat_x + 0.70
    nf_y0 = concat_y - nf_bh / 2
    draw_feature_strip(ax_conv, x_left=nf_x0, y_bottom=nf_y0,
                       n_boxes=nf_n, box_w=nf_bw, box_h=nf_bh,
                       color=C_NODE_FEAT)
    ax_conv.annotate(
        '', xy=(nf_x0 - 0.04, nf_y0 + nf_bh / 2),
        xytext=(concat_x, nf_y0 + nf_bh / 2),
        arrowprops=dict(arrowstyle='->', mutation_scale=22, lw=1.8,
                        color='#444', zorder=4),
    )
    ax_conv.text(0.5 * (concat_x + nf_x0),
                 nf_y0 + nf_bh / 2 + 0.22,
                 'concat',
                 ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color='#555')
    ax_conv.text(nf_x0 + nf_n * nf_bw / 2, nf_y0 - 0.35,
                 f'Loc$_{args.target_loc}$ node feature '
                 r'$(1 \times 128)$',
                 ha='center', va='top',
                 fontsize=16, fontweight='bold', color=C_NODE_FEAT)

    # ---- Panel (d): K4 graph ---------------------------------------
    ax_graph.set_xlim(0, 12)
    ax_graph.set_ylim(0, 10)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_graph.spines[sp].set_visible(False)
    ax_graph.set_xticks([])
    ax_graph.set_yticks([])

    graph_positions = draw_k4_graph(ax_graph, target_loc=args.target_loc)

    # Per-node feature strips (small) — Loc_a's strip is the destination
    # of the cross-panel "node feature" arrow from (c).
    strip_n  = 5
    strip_bw = 0.30
    strip_bh = 0.40
    strip_offsets = {
        'a': (-strip_n * strip_bw / 2,  0.95),   # above
        'b': ( 0.85, -strip_bh / 2),             # right
        'c': (-strip_n * strip_bw / 2, -1.20),   # below
        'd': (-strip_n * strip_bw - 0.85,
              -strip_bh / 2),                    # left
    }
    for l, (dx, dy) in strip_offsets.items():
        nx, ny = graph_positions[l]
        draw_feature_strip(
            ax_graph, x_left=nx + dx, y_bottom=ny + dy,
            n_boxes=strip_n, box_w=strip_bw, box_h=strip_bh,
            color=C_NODE_FEAT,
        )

    # Edge feature strip on the Loc_a — Loc_b edge.
    ax_pos = graph_positions[args.target_loc]
    nb_pos = graph_positions['b']
    edge_mid_x = 0.5 * (ax_pos[0] + nb_pos[0])
    edge_mid_y = 0.5 * (ax_pos[1] + nb_pos[1])
    edge_n, edge_bw, edge_bh = 3, 0.32, 0.42
    edge_x0 = edge_mid_x + 0.35
    edge_y0 = edge_mid_y + 0.15
    draw_feature_strip(
        ax_graph, x_left=edge_x0, y_bottom=edge_y0,
        n_boxes=edge_n, box_w=edge_bw, box_h=edge_bh,
        color=C_EDGE_FEAT,
    )
    ax_graph.text(edge_x0 + edge_n * edge_bw / 2,
                  edge_y0 + edge_bh + 0.12,
                  'edge feature',
                  ha='center', va='bottom',
                  fontsize=12, fontweight='bold', color=C_EDGE_FEAT)

    # DTW glyph above the graph — mediates (b) -> edge feature.
    dtw_cx, dtw_cy, dtw_size = 1.55, 8.4, 1.4
    draw_dtw_glyph(ax_graph, cx=dtw_cx, cy=dtw_cy,
                   size=dtw_size, color='#444',
                   path_color=C_EDGE_FEAT, label='DTW')

    # Internal arrow: DTW glyph -> edge feature strip.
    ax_graph.annotate(
        '', xy=(edge_x0 - 0.10, edge_y0 + edge_bh / 2),
        xytext=(dtw_cx + dtw_size / 2 + 0.05, dtw_cy),
        arrowprops=dict(arrowstyle='->', mutation_scale=22, lw=2.0,
                        color=C_EDGE_FEAT, zorder=6,
                        connectionstyle='arc3,rad=-0.25'),
    )

    # ---- Panel letters --------------------------------------------
    panel_label_kw = dict(fontsize=FS_KDE_HDR + 4, fontweight='bold',
                          va='top', ha='left', color='#222')
    ax_tree.text(0.015, 0.99, '(a)',
                 transform=ax_tree.transAxes, **panel_label_kw)
    ax_kde.text(0.015, 0.99, '(b)',
                transform=ax_kde.transAxes, **panel_label_kw)
    ax_conv.text(0.015, 0.99, '(c)',
                 transform=ax_conv.transAxes, **panel_label_kw)
    ax_graph.text(0.015, 0.99, '(d)',
                  transform=ax_graph.transAxes, **panel_label_kw)

    # ---- Cross-figure data-flow arrows ----------------------------
    # (a) MRCA region  ->  (c) CBLV (4 x N) heatmap (left edge midpoint).
    # The CBLV tensor is the actual input the conv branches consume;
    # routing the cross-figure arrow here makes the "subtree -> CBLV"
    # encoding step explicit.
    con_a_c = ConnectionPatch(
        xyA=(mrca.x + 0.6, mrca.y),
        coordsA=ax_tree.transData,
        xyB=(cblv_x0 - 0.10, cblv_y0 + cblv_h / 2),
        coordsB=ax_conv.transData,
        arrowstyle='->', mutation_scale=28, lw=2.6,
        color=C_HL, zorder=20,
        connectionstyle='arc3,rad=-0.18',
    )
    fig.add_artist(con_a_c)

    # (c) node feature  ->  (d) Loc_a node-feature strip.
    # Origin: left-bottom corner of the node-feature strip in (c) (so
    # the arrow leaves on the side that faces panel (d)'s Loc_a node).
    # Target: top-center of the Loc_a node-feature strip in (d). A
    # mild negative ``rad`` curves the arrow left, keeping it clear of
    # the strip label and the K4 edge feature.
    a_strip_x = graph_positions['a'][0] + strip_offsets['a'][0] \
                + strip_n * strip_bw / 2
    a_strip_y = graph_positions['a'][1] + strip_offsets['a'][1] \
                + strip_bh
    con_c_d = ConnectionPatch(
        xyA=(nf_x0 + nf_bw * 0.5, nf_y0),
        coordsA=ax_conv.transData,
        xyB=(a_strip_x, a_strip_y + 0.05),
        coordsB=ax_graph.transData,
        arrowstyle='->', mutation_scale=28, lw=2.6,
        color=C_NODE_FEAT, zorder=20,
        connectionstyle='arc3,rad=-0.15',
    )
    fig.add_artist(con_c_d)

    # (b) KDE right edge  ->  DTW glyph in (d)
    con_b_dtw = ConnectionPatch(
        xyA=(time_x_end - 0.3, ax_kde.get_ylim()[1] * 0.55),
        coordsA=ax_kde.transData,
        xyB=(dtw_cx - dtw_size / 2 - 0.05, dtw_cy),
        coordsB=ax_graph.transData,
        arrowstyle='->', mutation_scale=28, lw=2.6,
        color='#444', zorder=20,
        connectionstyle='arc3,rad=0.25',
    )
    fig.add_artist(con_b_dtw)

    # ---- Legend (anchored below the whole figure) -----------------
    handles = [
        Line2D([0], [0], color=C_HL, lw=4.4,
               label=f'Loc$_{args.target_loc}$ virtual subtree'),
        Line2D([0], [0], color=C_DIM, lw=3.4,
               label='Background lineages'),
        Line2D([0], [0], color=C_DIM, lw=3.4, linestyle=(0, (4, 3)),
               label='Rest of tree (not shown in detail)'),
        Line2D([0], [0], marker='o', color=C_SAMP, markersize=15,
               linestyle='none',
               markeredgecolor=C_HL, markeredgewidth=1.8,
               label=f'Loc$_{args.target_loc}$ sampled tip'),
        Line2D([0], [0], marker='o', color=C_DIM, markersize=11,
               linestyle='none', label='Other sampled tip'),
        Line2D([0], [0], marker='s', color=C_NODE_FEAT, markersize=14,
               linestyle='none', label='Node feature'),
        Line2D([0], [0], marker='s', color=C_EDGE_FEAT, markersize=14,
               linestyle='none', label='Edge feature'),
    ]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.02),
               frameon=False, fontsize=FS_LEGEND, ncol=4,
               handletextpad=0.9, handlelength=2.4,
               columnspacing=2.4, labelspacing=0.9)

    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f'Saved: {args.out_pdf}')


if __name__ == '__main__':
    main()
