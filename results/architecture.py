#!/usr/bin/env python3
"""Detailed STEPHY architecture figure.

Two-row composite:

  ROW 1 — Input representations
    (a) Per-NODE features:  subtree -> CBLV (4 x W) heatmap + 5-d aux vector.
    (b) Per-EDGE features:  two location KDE curves, DTW cost matrix with
        optimal warping path, and the 3-d edge feature vector
        (distance, lag_mean*dt, lag_std*dt).
    (c) Graph assembly:     fully connected K4 graph over 4 locations,
        with node embedding shape and edge feature shape annotated.

  ROW 2 — Network forward pass (left -> right)
    CBLV (4, W) -> CBLVConvEncoder (3-branch CNN: plain / stride / dilate)
                  -> 96-d   ----+
    Aux  (5)    -> AuxBranch MLP (5 -> 64 -> 32) -> 32-d --+
                                                            |
    concat(96, 32) = 128-d node embedding ------------------+
                  -> GraphEdgeAttention (uses 3-d DTW edges) -> 256-d
                  -> Classifier MLP (256 -> 128 -> 64 -> 32 -> out)
                  -> 6 output heads (3 regression + 3 classification)

All channel widths and kernel/stride/dilation sizes are pulled from the
real ``stephy/config.py`` so the figure stays in sync with the model.

Usage:
    python3 architecture.py
    python3 architecture.py --out_pdf /tmp/architecture.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
    Circle,
    PathPatch,
)
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------
# Publication-quality matplotlib defaults (match conceptual.py).
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


# ---------------------------------------------------------------------
# Pull architecture hyperparameters from stephy/config.py so the figure
# remains in sync with the actual model.
# ---------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'stephy'))
from config import MODEL_ARGS, DATA_ARGS  # noqa: E402

CHAN_PLAIN  = MODEL_ARGS['phy_channel_plain']
CHAN_STRIDE = MODEL_ARGS['phy_channel_stride']
CHAN_DILATE = MODEL_ARGS['phy_channel_dilate']
KERN_PLAIN  = MODEL_ARGS['phy_kernel_plain']
KERN_STRIDE = MODEL_ARGS['phy_kernel_stride']
KERN_DILATE = MODEL_ARGS['phy_kernel_dilate']
STRD_STRIDE = MODEL_ARGS['phy_stride_stride']
DILT_DILATE = MODEL_ARGS['phy_dilate_dilate']
AUX_HIDDEN  = MODEL_ARGS['aux_hidden']
AUX_OUTPUT  = MODEL_ARGS['aux_output']
EDGE_DIM    = MODEL_ARGS['edge_dim']
ATTN_DIM    = MODEL_ARGS['attn_dim']
LBL_CHAN    = MODEL_ARGS['lbl_channel']
CNN_OUT     = CHAN_PLAIN[-1] + CHAN_STRIDE[-1] + CHAN_DILATE[-1]   # 96
NODE_DIM    = CNN_OUT + AUX_OUTPUT                                 # 128
GAT_OUT     = NODE_DIM * 2                                         # 256


# ---------------------------------------------------------------------
# Palette + font hierarchy.
# ---------------------------------------------------------------------
LOC_COLORS = {
    'a': '#FFC107',   # gold (matches index-case star used in conceptual.py)
    'b': '#4E79A7',
    'c': '#59A14F',
    'd': '#E15759',
}
CBLV_CMAP   = 'viridis'
DTW_CMAP    = 'magma_r'
AUX_FILL    = '#CFE0F0'
EDGE_FILL   = '#EFD9C7'
HID_FILL    = '#E8E8E8'
ATTN_FILL   = '#D9C9E0'
OUT_FILL    = '#F4F4F4'
INK         = '#1f1f1f'
INK_SOFT    = '#555555'
HEAD_REG    = '#3A6EA5'
HEAD_CLS    = '#A8423A'
PATH_RED    = '#D62728'

# Font sizes (uniform with conceptual.py)
FS_ROW_TITLE = 24
FS_PANEL     = 20
FS_SUBHDR    = 16
FS_LABEL     = 14
FS_TENSOR    = 13
FS_ANNOT     = 12
FS_TINY      = 10


# ---------------------------------------------------------------------
# Generic drawing helpers.
# ---------------------------------------------------------------------

def _set_panel_axes(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('auto')
    ax.axis('off')


def _round_box(ax, x, y, w, h, *, fc=OUT_FILL, ec=INK, lw=1.2,
               rounding=0.04, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, xy_from, xy_to, *, color=INK, lw=2.0, mut=18, ls='-',
           zorder=3, head_w=0.45, head_l=0.7):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=f'-|>,head_width={head_w},head_length={head_l}',
        mutation_scale=mut, lw=lw, color=color, ls=ls,
        zorder=zorder, capstyle='round', shrinkA=0.0, shrinkB=0.0,
    )
    ax.add_patch(arr)
    return arr


def _label_block(ax, x, y, w, h, title, sub=None, *, fc=HID_FILL,
                 ec=INK, lw=1.2, fs_title=FS_TENSOR, fs_sub=FS_TINY,
                 zorder=2, italic_sub=True):
    _round_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, zorder=zorder)
    if sub is None:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=fs_title, color=INK, zorder=zorder + 1)
    else:
        ax.text(x + w/2, y + h*0.66, title, ha='center', va='center',
                fontsize=fs_title, color=INK, zorder=zorder + 1,
                fontweight='bold')
        ax.text(x + w/2, y + h*0.30, sub, ha='center', va='center',
                fontsize=fs_sub, color=INK_SOFT, zorder=zorder + 1,
                style='italic' if italic_sub else 'normal')


# =====================================================================
# ROW 1 — input representations
# =====================================================================

def _draw_mini_tree(ax, x0, y0, w, h, *, highlight='a'):
    """Stylized phylogeny with the highlighted location's virtual subtree drawn
    in saturated gold over a dimmed background tree."""
    root_y = y0 + 0.05 * h
    top_y  = y0 + 0.95 * h
    cx     = x0 + w * 0.5

    tips = [
        ('a', 0.10), ('b', 0.22), ('a', 0.34), ('c', 0.46),
        ('a', 0.58), ('d', 0.70), ('a', 0.82), ('b', 0.94),
    ]
    branch_x = lambda f: x0 + f * w  # noqa: E731

    spine_top = top_y - 0.05 * h
    n = len(tips)
    depths = [root_y + (spine_top - root_y) * (0.15 + 0.85 * i / (n - 1))
              for i in range(n)]

    # Visual roles
    HL_PATH  = '#E69500'   # saturated gold for the virtual subtree
    DIM_INK  = '#BFBFBF'   # dimmed background topology

    # Spine: gold from root up to the topmost highlighted tip's depth, dim above.
    h_idx = [i for i, (loc, _) in enumerate(tips) if loc == highlight]
    h_hi_depth = max(depths[i] for i in h_idx)
    ax.plot([cx, cx], [root_y, h_hi_depth],
            color=HL_PATH, lw=3.2, zorder=3, solid_capstyle='round')
    ax.plot([cx, cx], [h_hi_depth, spine_top],
            color=DIM_INK, lw=1.2, zorder=1)

    # Tip branches + tip markers.
    for i, (loc, frac) in enumerate(tips):
        depth = depths[i]
        tx    = branch_x(frac)
        is_h  = (loc == highlight)
        if is_h:
            ax.plot([cx, tx], [depth, depth], color=HL_PATH, lw=2.6,
                    zorder=3, solid_capstyle='round')
            ax.plot([tx, tx], [depth, top_y], color=HL_PATH, lw=2.6,
                    zorder=3, solid_capstyle='round')
            ax.scatter([tx], [top_y], s=170, marker='o',
                       facecolor=LOC_COLORS[loc], edgecolor=PATH_RED,
                       lw=2.2, zorder=5)
        else:
            ax.plot([cx, tx], [depth, depth], color=DIM_INK, lw=1.0,
                    zorder=1)
            ax.plot([tx, tx], [depth, top_y], color=DIM_INK, lw=1.0,
                    zorder=1)
            ax.scatter([tx], [top_y], s=70, marker='o',
                       facecolor=LOC_COLORS[loc], edgecolor=INK_SOFT,
                       lw=0.7, zorder=4, alpha=0.55)

    # Caption — "virtual subtree" is now the visually dominant story.
    ax.text(cx, root_y - 0.05 * h,
            f"virtual subtree of Loc_{highlight}",
            ha='center', va='top', fontsize=FS_TINY,
            color=HL_PATH, style='italic', fontweight='bold')


def _draw_cblv_heatmap(ax, x0, y0, w, h, W=10):
    """A 4 x W block of viridis-coloured cells representing CBLV."""
    rng = np.random.default_rng(42)
    cblv = rng.random((4, W))
    # Make the visual interesting: enforce row-wise gradients matching how
    # CBLV encodes (depths/branch lengths along subtree positions).
    cblv[0] = np.linspace(0.1, 0.9, W)
    cblv[1] = np.linspace(0.7, 0.2, W)
    cblv[2] = 0.5 + 0.4 * np.sin(np.linspace(0, np.pi, W))
    cblv[3] = 0.4 + 0.3 * np.cos(np.linspace(0, 2 * np.pi, W))

    cell_w = w / W
    cell_h = h / 4
    cmap = mpl.colormaps[CBLV_CMAP]
    for i in range(4):
        for j in range(W):
            ax.add_patch(Rectangle((x0 + j * cell_w, y0 + (3 - i) * cell_h),
                                   cell_w, cell_h,
                                   facecolor=cmap(cblv[i, j]),
                                   edgecolor='white', lw=0.4, zorder=2))
    # Border
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor='none',
                           edgecolor=INK, lw=1.0, zorder=3))
    # Row labels (4 CBLV channels)
    row_labels = ['ch0', 'ch1', 'ch2', 'ch3']
    for i, lab in enumerate(row_labels):
        ax.text(x0 - 0.005, y0 + (3 - i) * cell_h + cell_h / 2, lab,
                ha='right', va='center', fontsize=FS_TINY, color=INK_SOFT)
    # Column annotation
    ax.annotate('', xy=(x0 + w, y0 - 0.012), xytext=(x0, y0 - 0.012),
                arrowprops=dict(arrowstyle='<->', color=INK_SOFT, lw=0.8))
    ax.text(x0 + w / 2, y0 - 0.030, 'subtree position (W)',
            ha='center', va='top', fontsize=FS_TINY, color=INK_SOFT,
            style='italic')
    ax.text(x0 - 0.035, y0 + h / 2, '4',
            ha='right', va='center', fontsize=FS_TINY, color=INK_SOFT)


def _draw_aux_vector(ax, x0, y0, w, h, names):
    """5 named cells representing the aux vector."""
    n = len(names)
    cell_w = w / n
    for i, nm in enumerate(names):
        rect = Rectangle((x0 + i * cell_w, y0), cell_w, h,
                         facecolor=AUX_FILL, edgecolor=INK, lw=1.0,
                         zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + i * cell_w + cell_w / 2, y0 + h / 2, nm,
                ha='center', va='center', fontsize=FS_TINY, color=INK)


def draw_panel_node_features(ax):
    """Panel (a): subtree -> CBLV heatmap + 5-d aux vector."""
    _set_panel_axes(ax)

    # Panel title
    ax.text(0.5, 0.96, '(a) Per-node features',
            ha='center', va='top', fontsize=FS_PANEL, fontweight='bold')
    ax.text(0.5, 0.905,
            r'one node = one location  $\rightarrow$  '
            r'CBLV (4$\times$W) + 5 aux statistics',
            ha='center', va='top', fontsize=FS_SUBHDR, style='italic',
            color=INK_SOFT)

    # Mini tree on the left, highlight Loc_a's virtual subtree.
    _draw_mini_tree(ax, x0=0.04, y0=0.22, w=0.38, h=0.55, highlight='a')

    # Encoder arrow from tree highlight to CBLV.
    _arrow(ax, (0.43, 0.50), (0.55, 0.66), color=INK_SOFT, lw=1.5, mut=14)
    ax.text(0.49, 0.62, 'VirtualSubtreeEncoder',
            ha='center', va='bottom', fontsize=FS_TINY, color=INK_SOFT,
            style='italic')

    # CBLV heatmap (right).
    _draw_cblv_heatmap(ax, x0=0.56, y0=0.55, w=0.40, h=0.20, W=10)
    ax.text(0.76, 0.78, r'CBLV $\in \mathbb{R}^{4\times W}$',
            ha='center', va='bottom', fontsize=FS_LABEL, fontweight='bold')

    # Aux vector below CBLV.
    aux_names = [r'$d_{\mathrm{mrca}}$', r'$t_{\min}$',
                 r'$t_{\max}$', r'$\bar{b}$', r'$n_{\mathrm{tips}}$']
    _draw_aux_vector(ax, x0=0.56, y0=0.30, w=0.40, h=0.10, names=aux_names)
    ax.text(0.76, 0.43, r'aux $\in \mathbb{R}^{5}$',
            ha='center', va='bottom', fontsize=FS_LABEL, fontweight='bold')

    # Aux feature legend
    ax.text(0.76, 0.235,
            r'mrca depth, first/last tip time, mean branch length, n_tips',
            ha='center', va='top', fontsize=FS_TINY, color=INK_SOFT,
            style='italic')


# ---------------------------------------------------------------------
# Panel (b) — DTW worked example.
# ---------------------------------------------------------------------

def _gauss_bump(t, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _dtw_matrix(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D


def _dtw_path(D):
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i, j)]
    while i > 0 and j > 0:
        diag, up, left = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i = i - 1
        else:
            j = j - 1
        path.append((i, j))
    while i > 0:
        i -= 1
        path.append((i, j))
    while j > 0:
        j -= 1
        path.append((i, j))
    return path[::-1]


def draw_panel_edge_features(fig, gs_slot):
    """Panel (b): two KDE curves + DTW grid + 3-d edge feature vector."""
    inner = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_slot,
        height_ratios=[1.0, 1.7], width_ratios=[1.0, 0.42],
        wspace=0.20, hspace=0.30,
    )
    ax_curves = fig.add_subplot(inner[0, 0])
    ax_grid   = fig.add_subplot(inner[1, 0])
    ax_text   = fig.add_subplot(inner[:, 1])

    # ---- Curves: two same-shape Gaussian bumps lagged in time -------
    n = 30
    t = np.linspace(0, 1, n)
    a_curve = _gauss_bump(t, mu=0.30, sigma=0.10, amp=1.0) + 0.02
    b_curve = _gauss_bump(t, mu=0.55, sigma=0.10, amp=1.0) + 0.02

    ax_curves.plot(t, a_curve, color=LOC_COLORS['a'], lw=2.0,
                   label='Loc_i KDE  $f_i(t)$')
    ax_curves.plot(t, b_curve, color=LOC_COLORS['b'], lw=2.0,
                   label='Loc_j KDE  $f_j(t)$')
    ax_curves.fill_between(t, 0, a_curve, color=LOC_COLORS['a'], alpha=0.15)
    ax_curves.fill_between(t, 0, b_curve, color=LOC_COLORS['b'], alpha=0.15)
    ax_curves.set_xlim(0, 1)
    ax_curves.set_ylim(0, 1.15)
    ax_curves.set_xticks([])
    ax_curves.set_yticks([])
    for s in ('top', 'right'):
        ax_curves.spines[s].set_visible(False)
    ax_curves.spines['left'].set_color(INK_SOFT)
    ax_curves.spines['bottom'].set_color(INK_SOFT)
    ax_curves.set_xlabel('shared time grid (200 pts)',
                         fontsize=FS_TINY, color=INK_SOFT, labelpad=2)
    ax_curves.legend(loc='upper right', fontsize=FS_TINY, frameon=False,
                     handlelength=1.2, borderpad=0.2, labelspacing=0.2)
    ax_curves.set_title('per-location tip-time KDE curves',
                        fontsize=FS_LABEL, pad=4)

    # ---- DTW grid + warping path ------------------------------------
    D = _dtw_matrix(a_curve, b_curve)
    path = _dtw_path(D)
    # Visualize cell costs (excluding the +inf border) — use the local
    # squared-difference cost rather than cumulative D for clarity.
    local = (a_curve[:, None] - b_curve[None, :]) ** 2
    im = ax_grid.imshow(local, origin='lower', cmap=DTW_CMAP,
                         aspect='equal')
    # Path overlay (skip the (0,0) phantom cell from the +inf border).
    pxs = [j - 1 for (i, j) in path if i > 0 and j > 0]
    pys = [i - 1 for (i, j) in path if i > 0 and j > 0]
    # White halo first so the path stays readable over dark magma_r corners.
    ax_grid.plot(pxs, pys, color='white', lw=4.2, zorder=4,
                 solid_capstyle='round')
    ax_grid.plot(pxs, pys, color=PATH_RED, lw=2.4, zorder=5,
                 solid_capstyle='round')
    ax_grid.scatter(pxs[::3], pys[::3], color=PATH_RED, s=14,
                    edgecolor='white', lw=0.6, zorder=6)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_xlabel(r'$j$  (Loc_j time index)',
                       fontsize=FS_TINY, color=INK_SOFT, labelpad=2)
    ax_grid.set_ylabel(r'$i$  (Loc_i time index)',
                       fontsize=FS_TINY, color=INK_SOFT, labelpad=2)
    for s in ax_grid.spines.values():
        s.set_color(INK_SOFT)
    ax_grid.set_title('DTW cost matrix + optimal warping path',
                      fontsize=FS_LABEL, pad=4)

    # Mini colorbar
    cb = fig.colorbar(im, ax=ax_grid, fraction=0.045, pad=0.02,
                      shrink=0.85)
    cb.outline.set_color(INK_SOFT)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=FS_TINY - 1)
    cb.set_label(r'$(f_i(t_i)-f_j(t_j))^2$',
                 fontsize=FS_TINY, color=INK_SOFT)

    # ---- Right column: edge feature explanation ---------------------
    _set_panel_axes(ax_text)
    ax_text.text(0.5, 0.97, '(b) Per-edge features',
                 ha='center', va='top', fontsize=FS_PANEL,
                 fontweight='bold')
    ax_text.text(0.5, 0.905,
                 r'one edge = one ordered $(i,j)$ pair',
                 ha='center', va='top', fontsize=FS_SUBHDR,
                 style='italic', color=INK_SOFT)

    # Edge feature vector — 3 colored cells.
    cell_w = 0.78 / 3
    cell_h = 0.10
    x0 = 0.11
    y0 = 0.62
    feat_labels = [r'$d_{ij}$', r'$\bar{\ell}_{ij}$', r'$\sigma_{\ell,ij}$']
    feat_long   = ['DTW distance',
                   r'mean lag $\cdot \Delta t$',
                   r'lag std $\cdot \Delta t$']
    for k in range(3):
        rect = Rectangle((x0 + k * cell_w, y0), cell_w, cell_h,
                         facecolor=EDGE_FILL, edgecolor=INK, lw=1.0)
        ax_text.add_patch(rect)
        ax_text.text(x0 + k * cell_w + cell_w / 2, y0 + cell_h / 2,
                     feat_labels[k], ha='center', va='center',
                     fontsize=FS_LABEL)
        ax_text.text(x0 + k * cell_w + cell_w / 2, y0 - 0.025,
                     feat_long[k], ha='center', va='top',
                     fontsize=FS_TINY, color=INK_SOFT, style='italic')

    ax_text.text(0.5, y0 + cell_h + 0.04,
                 r'edge_feat $\in \mathbb{R}^{3}$',
                 ha='center', va='bottom',
                 fontsize=FS_LABEL, fontweight='bold')

    # Recipe text
    recipe_y = 0.42
    ax_text.text(0.5, recipe_y,
                 'how it is built',
                 ha='center', va='top', fontsize=FS_LABEL,
                 fontweight='bold')
    bullets = [
        r'1. tip times per location $\rightarrow$ Gaussian KDE',
        r'2. evaluate on shared grid (200 pts)',
        r'3. DTW between curves $f_i, f_j$:',
        r'      $D[i,j]\!=\!(a_i\!-\!b_j)^2 +$',
        r'      $\quad \min\{D[i\!-\!1,j\!-\!1], D[i\!-\!1,j], D[i,j\!-\!1]\}$',
        r'4. $d_{ij}\!=\!D[n,m]$;  '
        r'$\bar{\ell}_{ij}, \sigma_{\ell,ij}$ from path',
    ]
    line_y = recipe_y - 0.05
    for line in bullets:
        ax_text.text(0.07, line_y, line, ha='left', va='top',
                     fontsize=FS_TINY, color=INK)
        line_y -= 0.055


# ---------------------------------------------------------------------
# Panel (c) — graph assembly.
# ---------------------------------------------------------------------

def draw_panel_graph_assembly(ax):
    """Panel (c): K4 graph with annotated node + edge tensors."""
    _set_panel_axes(ax)
    ax.text(0.5, 0.96, '(c) Graph assembly',
            ha='center', va='top', fontsize=FS_PANEL, fontweight='bold')
    ax.text(0.5, 0.905,
            r'fully connected DGL graph (no self-loops)',
            ha='center', va='top', fontsize=FS_SUBHDR, style='italic',
            color=INK_SOFT)

    # Diamond layout for the 4 nodes (axis coords 0..1).
    pos = {
        'a': (0.50, 0.78),
        'b': (0.86, 0.45),
        'c': (0.50, 0.13),
        'd': (0.14, 0.45),
    }

    # Edges (undirected pairs but we draw both arrowheads to show ordered).
    pairs = [('a', 'b'), ('a', 'c'), ('a', 'd'),
             ('b', 'c'), ('b', 'd'), ('c', 'd')]
    for u, v in pairs:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=INK_SOFT, lw=1.0,
                zorder=1, alpha=0.7)

    # Nodes
    R = 0.07
    for loc, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), R, facecolor=LOC_COLORS[loc],
                            edgecolor=INK, lw=1.4, zorder=3))
        ax.text(x, y, f'Loc_{loc}', ha='center', va='center',
                fontsize=FS_LABEL, fontweight='bold',
                color='black' if loc == 'a' else 'white', zorder=4)

    # Annotation: per-node embedding + per-edge tensor.
    ax.text(0.5, 0.985, '', ha='center')  # spacing anchor

    # Node legend
    _round_box(ax, 0.04, 0.005, 0.42, 0.085, fc=HID_FILL, ec=INK_SOFT,
               lw=0.8, rounding=0.02)
    ax.text(0.25, 0.07, 'each node carries',
            ha='center', va='center', fontsize=FS_TINY,
            color=INK_SOFT, style='italic')
    ax.text(0.25, 0.035,
            r'CBLV (4$\times$W)  +  aux ($\mathbb{R}^{5}$)',
            ha='center', va='center', fontsize=FS_TENSOR,
            fontweight='bold')

    # Edge legend
    _round_box(ax, 0.54, 0.005, 0.42, 0.085, fc=EDGE_FILL, ec=INK_SOFT,
               lw=0.8, rounding=0.02)
    ax.text(0.75, 0.07, 'each edge carries',
            ha='center', va='center', fontsize=FS_TINY,
            color=INK_SOFT, style='italic')
    ax.text(0.75, 0.035,
            r'edge_feat ($\mathbb{R}^{3}$, DTW)',
            ha='center', va='center', fontsize=FS_TENSOR,
            fontweight='bold')


# =====================================================================
# ROW 2 — network forward pass
# =====================================================================

def _draw_layer_stack(ax, x, y, layers, *, w=0.10, h=0.06, gap=0.012,
                      fc=HID_FILL, ec=INK, fs=FS_TINY,
                      label_above=None, label_below=None,
                      label_above_fs=FS_TENSOR):
    """Draw a vertical stack of labeled rectangles (top -> bottom)."""
    n = len(layers)
    total_h = n * h + (n - 1) * gap
    top_y = y + total_h / 2
    for i, lab in enumerate(layers):
        ry = top_y - (i + 1) * h - i * gap
        _round_box(ax, x, ry, w, h, fc=fc, ec=ec, lw=1.0,
                   rounding=0.01, zorder=2)
        ax.text(x + w / 2, ry + h / 2, lab,
                ha='center', va='center', fontsize=fs, color=INK)
    if label_above is not None:
        ax.text(x + w / 2, y + total_h / 2 + 0.018, label_above,
                ha='center', va='bottom', fontsize=label_above_fs,
                fontweight='bold', color=INK)
    if label_below is not None:
        ax.text(x + w / 2, y - total_h / 2 - 0.018, label_below,
                ha='center', va='top', fontsize=FS_TINY,
                style='italic', color=INK_SOFT)
    return ((x + w / 2, y + total_h / 2),
            (x + w / 2, y - total_h / 2))


def _draw_layer_row(ax, x_left, y_center, layers, *,
                    card_w=0.075, card_h=0.055, gap=0.013,
                    fc=HID_FILL, ec=INK, fs=FS_TINY,
                    label_left=None, label_right=None,
                    label_left_fs=FS_TENSOR,
                    label_left_color=INK):
    """Draw a horizontal row of labeled rectangles (left -> right).

    Returns (left_anchor_xy, right_anchor_xy) at the row's vertical center.
    """
    if label_left is not None:
        ax.text(x_left - 0.008, y_center, label_left,
                ha='right', va='center',
                fontsize=label_left_fs, fontweight='bold',
                color=label_left_color)
    x = x_left
    for lab in layers:
        ry = y_center - card_h / 2
        _round_box(ax, x, ry, card_w, card_h, fc=fc, ec=ec, lw=1.0,
                   rounding=0.012, zorder=2)
        ax.text(x + card_w / 2, y_center, lab,
                ha='center', va='center', fontsize=fs, color=INK)
        x += card_w + gap
    x_right = x - gap
    if label_right is not None:
        ax.text(x_right + 0.010, y_center, label_right,
                ha='left', va='center',
                fontsize=FS_TENSOR, color=INK)
    return ((x_left, y_center), (x_right, y_center))


def _node_input_block(ax, x, y, w, h):
    """Tiny CBLV (4xW) + Aux(5) input panel for a single node."""
    # CBLV thumbnail
    cblv_h = h * 0.55
    cblv_w = w * 0.92
    cx = x + (w - cblv_w) / 2
    cy = y + h - cblv_h - 0.015
    _draw_cblv_heatmap(ax, cx, cy, cblv_w, cblv_h, W=8)
    ax.text(x + w / 2, y + h - 0.005,
            r'CBLV $\in \mathbb{R}^{4\times W}$',
            ha='center', va='top', fontsize=FS_TENSOR, fontweight='bold')
    # Aux row
    aux_y = y + 0.018
    aux_h = h * 0.16
    _draw_aux_vector(ax, cx, aux_y, cblv_w, aux_h,
                     names=[r'$d_{m}$', r'$t_{\min}$', r'$t_{\max}$',
                            r'$\bar b$', r'$n$'])
    ax.text(x + w / 2, aux_y - 0.012, r'aux $\in \mathbb{R}^{5}$',
            ha='center', va='top', fontsize=FS_TENSOR, fontweight='bold')


def _cnn_branch_row(ax, x_left, y_center, channels, kernels, *,
                    strides=None, dilations=None, branch_name='plain',
                    color=HID_FILL, card_w=0.063, card_h=0.060,
                    gap=0.010):
    """Draw a CNN branch as a horizontal row of layer cards."""
    layers = []
    for k, (out_ch, kernel) in enumerate(zip(channels, kernels)):
        in_ch = channels[k - 1] if k > 0 else 4
        if strides is not None:
            layers.append(
                f"Conv1d\n{in_ch}$\\to${out_ch}\nk={kernel}, s={strides[k]}")
        elif dilations is not None:
            layers.append(
                f"Conv1d\n{in_ch}$\\to${out_ch}\nk={kernel}, d={dilations[k]}")
        else:
            layers.append(
                f"Conv1d\n{in_ch}$\\to${out_ch}\nk={kernel}")
    layers.append('Adaptive\nAvgPool1d')
    return _draw_layer_row(
        ax, x_left, y_center, layers,
        card_w=card_w, card_h=card_h, gap=gap, fc=color, ec=INK,
        fs=FS_TINY - 1,
        label_left=branch_name,
        label_right=f'$\\to {channels[-1]}$-d',
    )


def _aux_branch_row(ax, x_left, y_center, *, card_w=0.063,
                    card_h=0.060, gap=0.010):
    layers = [
        f'Linear\n5$\\to${AUX_HIDDEN}',
        'ReLU',
        f'Linear\n{AUX_HIDDEN}$\\to${AUX_OUTPUT}',
        'ReLU',
    ]
    return _draw_layer_row(
        ax, x_left, y_center, layers,
        card_w=card_w, card_h=card_h, gap=gap, fc=AUX_FILL, ec=INK,
        fs=FS_TINY - 1,
        label_left='AuxBranch',
        label_right=f'$\\to {AUX_OUTPUT}$-d',
    )


def _classifier_row(ax, x_left, y_center, *, card_w=0.063,
                    card_h=0.060, gap=0.010):
    layers = [
        f'Linear\n{GAT_OUT}$\\to${LBL_CHAN[0]}',
        f'Linear\n{LBL_CHAN[0]}$\\to${LBL_CHAN[1]}',
        f'Linear\n{LBL_CHAN[1]}$\\to${LBL_CHAN[2]}',
        f'Linear\n{LBL_CHAN[2]}$\\to$out',
    ]
    return _draw_layer_row(
        ax, x_left, y_center, layers,
        card_w=card_w, card_h=card_h, gap=gap, fc=HID_FILL, ec=INK,
        fs=FS_TINY - 1,
        label_left='Classifier',
        label_right='',
    )


def _attention_box(ax, x, y_center, w=0.20, h=0.30):
    """The GraphEdgeAttention block — show a tiny K4 + the attention math."""
    x0 = x - w / 2
    y0 = y_center - h / 2
    _round_box(ax, x0, y0, w, h, fc=ATTN_FILL, ec=INK, lw=1.4,
               rounding=0.02, zorder=2)
    ax.text(x, y0 + h - 0.014, 'GraphEdgeAttention',
            ha='center', va='top', fontsize=FS_TENSOR, fontweight='bold')

    # Mini K4 graph in the upper part of the box.
    cx, cy = x, y0 + h * 0.68
    radius = 0.044
    angles = {'a': 90, 'b': 0, 'c': 270, 'd': 180}
    pts = {}
    for loc, ang in angles.items():
        rad = np.deg2rad(ang)
        pts[loc] = (cx + radius * np.cos(rad), cy + radius * np.sin(rad))
    for u, v in [('a', 'b'), ('a', 'c'), ('a', 'd'),
                 ('b', 'c'), ('b', 'd'), ('c', 'd')]:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]],
                color=INK_SOFT, lw=0.7, alpha=0.7, zorder=3)
    for loc, (px, py) in pts.items():
        ax.add_patch(Circle((px, py), 0.011,
                            facecolor=LOC_COLORS[loc],
                            edgecolor=INK, lw=0.8, zorder=4))

    # Attention math underneath the K4 mini-graph.
    math_y = y0 + h * 0.30
    ax.text(x, math_y,
            r'$\alpha_{ij} = \mathrm{softmax}_j\,\mathrm{MLP}_{a}'
            r'(\mathrm{edge}_{ij})$',
            ha='center', va='center', fontsize=FS_TINY)
    ax.text(x, math_y - 0.026,
            r'$\mathrm{MLP}_a:\ 3 \to ' + str(ATTN_DIM) + r' \to 1$',
            ha='center', va='center', fontsize=FS_TINY - 1,
            color=INK_SOFT, style='italic')
    ax.text(x, math_y - 0.052,
            r'$\mathrm{agg}_i = \sum_{j} \alpha_{ij}\, h_j$',
            ha='center', va='center', fontsize=FS_TINY)
    ax.text(x, math_y - 0.078,
            r'$h^{\prime}_i = [\, h_i \,\Vert\, \mathrm{agg}_i \,]$',
            ha='center', va='center', fontsize=FS_TINY)

    return (x0, x0 + w, y_center)


def _output_heads(ax, x_left, y_center, *, head_w=0.092, head_h=0.045,
                  gap=0.014):
    """Six output heads (3 reg + 3 cls), color-coded."""
    heads = [
        ('reg_r0',  HEAD_REG, 'per-loc R0'),
        ('reg_rr',  HEAD_REG, 'per-loc recovery rate'),
        ('reg_sss', HEAD_REG, 'per-loc source/sink'),
        ('cls_r0',  HEAD_CLS, 'argmax R0 location'),
        ('cls_sss', HEAD_CLS, 'biggest exporter'),
        ('cls_as',  HEAD_CLS, 'index-case location'),
    ]
    n = len(heads)
    total_h = n * head_h + (n - 1) * gap
    top = y_center + total_h / 2
    anchors = []
    for k, (name, color, sub) in enumerate(heads):
        ry = top - (k + 1) * head_h - k * gap
        _round_box(ax, x_left, ry, head_w, head_h, fc=color, ec=INK,
                   lw=1.0, rounding=0.012, zorder=2)
        ax.text(x_left + head_w / 2, ry + head_h * 0.62, name,
                ha='center', va='center', fontsize=FS_TENSOR,
                color='white', fontweight='bold')
        ax.text(x_left + head_w / 2, ry + head_h * 0.22, sub,
                ha='center', va='center', fontsize=FS_TINY - 1,
                color='white', style='italic')
        anchors.append((x_left, ry + head_h / 2))
    return anchors


def draw_row2_network(ax):
    """Row 2: full network forward pass.

    Internal split (axis [0,1] coords):

      SUB-ROW A  (y 0.55 - 0.98) : per-node featurization
        per-node input | 3 CNN branches | concat96 | AuxBranch | concat128
      SUB-ROW B  (y 0.02 - 0.50) : graph processing + classification
        128-d hand-off | GraphEdgeAttention (DTW) | 256-d | Classifier | heads
    """
    _set_panel_axes(ax)
    # Faint horizontal separator between the two sub-rows.
    ax.plot([0.02, 0.98], [0.515, 0.515], color=INK_SOFT, lw=0.6,
            ls=':', alpha=0.6, zorder=0)
    # Row title placed inside the axis so it cannot collide with panel (b)
    # of the input row above.
    ax.text(0.5, 1.005, 'Network forward pass',
            ha='center', va='bottom',
            fontsize=FS_ROW_TITLE, fontweight='bold')

    # ================================================================
    # SUB-ROW A — per-node featurization (top half).
    # ================================================================
    A_label_y = 0.97
    ax.text(0.18, A_label_y, 'A.  per-node featurization',
            ha='left', va='top', fontsize=FS_LABEL,
            fontweight='bold', color=INK_SOFT)

    # --- Per-node input block ---------------------------------------
    in_x, in_y, in_w, in_h = 0.012, 0.60, 0.135, 0.32
    _round_box(ax, in_x, in_y, in_w, in_h, fc='#FAFAFA', ec=INK_SOFT,
               lw=1.0, rounding=0.02)
    ax.text(in_x + in_w / 2, in_y + in_h + 0.012,
            'per-node input', ha='center', va='bottom',
            fontsize=FS_TENSOR, fontweight='bold')
    _node_input_block(ax, in_x + 0.008, in_y + 0.020,
                      in_w - 0.016, in_h - 0.040)

    # --- CNN branch rows + AuxBranch row ----------------------------
    branch_xL = 0.20
    y_plain   = 0.88
    y_stride  = 0.78
    y_dilate  = 0.68
    y_aux     = 0.56

    # CNN encoder title above plain branch
    ax.text(branch_xL + 0.13, A_label_y,
            f'CBLVConvEncoder   ({CNN_OUT}-d)',
            ha='left', va='top', fontsize=FS_TENSOR,
            fontweight='bold')

    _, plain_right = _cnn_branch_row(
        ax, branch_xL, y_plain, CHAN_PLAIN, KERN_PLAIN,
        branch_name='plain', card_w=0.058, card_h=0.060, gap=0.008)
    _, stride_right = _cnn_branch_row(
        ax, branch_xL, y_stride, CHAN_STRIDE, KERN_STRIDE,
        strides=STRD_STRIDE, branch_name='stride',
        card_w=0.058, card_h=0.060, gap=0.008)
    _, dilate_right = _cnn_branch_row(
        ax, branch_xL, y_dilate, CHAN_DILATE, KERN_DILATE,
        dilations=DILT_DILATE, branch_name='dilate',
        card_w=0.058, card_h=0.060, gap=0.008)

    # CNN grouping rectangle around the 3 branches.
    cnn_box_x = branch_xL - 0.060
    cnn_box_y = y_dilate - 0.045
    cnn_box_w = (plain_right[0] + 0.062) - cnn_box_x
    cnn_box_h = (y_plain + 0.045) - cnn_box_y
    _round_box(ax, cnn_box_x, cnn_box_y, cnn_box_w, cnn_box_h,
               fc='none', ec=INK_SOFT, lw=0.8, rounding=0.012,
               zorder=1)

    _, aux_right = _aux_branch_row(
        ax, branch_xL, y_aux,
        card_w=0.058, card_h=0.060, gap=0.008)

    # Fan-out arrows from input box to each branch.
    in_right_x = in_x + in_w
    for yb in (y_plain, y_stride, y_dilate):
        _arrow(ax, (in_right_x, in_y + in_h * 0.75),
               (branch_xL - 0.060, yb), color=INK_SOFT, lw=1.1, mut=10)
    _arrow(ax, (in_right_x, in_y + in_h * 0.20),
           (branch_xL - 0.010, y_aux),
           color=INK_SOFT, lw=1.1, mut=10)

    # --- Concat marker after CNN branches (96-d) --------------------
    rightmost_cnn_x = max(plain_right[0], stride_right[0], dilate_right[0])
    cat96_x = rightmost_cnn_x + 0.040
    cat96_y = y_stride
    _round_box(ax, cat96_x - 0.012, cat96_y - 0.028, 0.024, 0.056,
               fc='#FFF4D6', ec=INK, lw=1.0, rounding=0.01)
    ax.text(cat96_x, cat96_y, '$\\Vert$',
            ha='center', va='center', fontsize=FS_LABEL, fontweight='bold')
    ax.text(cat96_x, cat96_y - 0.046, f'{CNN_OUT}-d',
            ha='center', va='top', fontsize=FS_TINY, color=INK_SOFT,
            style='italic')
    for (cx, cy) in (plain_right, stride_right, dilate_right):
        _arrow(ax, (cx + 0.005, cy),
               (cat96_x - 0.012, cat96_y),
               color=INK_SOFT, lw=1.1, mut=10)

    # --- Concat marker for 128-d node embedding ---------------------
    cat128_x = cat96_x + 0.085
    cat128_y = (cat96_y + y_aux) / 2 + 0.04
    _round_box(ax, cat128_x - 0.016, cat128_y - 0.030, 0.032, 0.060,
               fc='#FFE5B4', ec=INK, lw=1.2, rounding=0.012)
    ax.text(cat128_x, cat128_y, '$\\Vert$',
            ha='center', va='center', fontsize=FS_LABEL, fontweight='bold')
    ax.text(cat128_x, cat128_y + 0.046,
            f'$h \\in \\mathbb{{R}}^{{{NODE_DIM}}}$',
            ha='center', va='bottom',
            fontsize=FS_TENSOR, fontweight='bold')
    ax.text(cat128_x, cat128_y - 0.048,
            'node embedding',
            ha='center', va='top', fontsize=FS_TINY,
            color=INK_SOFT, style='italic')
    _arrow(ax, (cat96_x + 0.014, cat96_y),
           (cat128_x - 0.016, cat128_y),
           color=INK_SOFT, lw=1.2, mut=10)
    _arrow(ax, (aux_right[0] + 0.005, aux_right[1]),
           (cat128_x - 0.016, cat128_y),
           color=INK_SOFT, lw=1.2, mut=10)

    # ================================================================
    # SUB-ROW B — graph processing + classification (bottom half).
    # Starts at the LEFT and flows fully across so the classifier and
    # output heads all fit inside the [0,1] axis.
    # ================================================================
    B_label_y = 0.48
    ax.text(0.012, B_label_y,
            'B.  graph processing  $\\rightarrow$  classification',
            ha='left', va='top', fontsize=FS_LABEL,
            fontweight='bold', color=INK_SOFT)

    # Hand-off marker on the sub-row A side: a small downward-pointing
    # arrow under cat128 with text "carries h to step B".
    _arrow(ax, (cat128_x, cat128_y - 0.030),
           (cat128_x, cat128_y - 0.075),
           color=INK_SOFT, lw=1.2, mut=12)
    ax.text(cat128_x + 0.020, cat128_y - 0.058,
            r'(carries $h \in \mathbb{R}^{' + str(NODE_DIM) + r'}$ '
            r'per node into step B)',
            ha='left', va='center',
            fontsize=FS_TINY, style='italic', color=INK_SOFT)

    # 128-d input anchor at the left of sub-row B (with "from A" label).
    handoff_dst_x = 0.045
    handoff_dst_y = 0.20
    _round_box(ax, handoff_dst_x - 0.022, handoff_dst_y - 0.034,
               0.090, 0.068, fc='#FFE5B4', ec=INK, lw=1.4, rounding=0.012)
    ax.text(handoff_dst_x + 0.023, handoff_dst_y + 0.008,
            f'$h \\in \\mathbb{{R}}^{{{NODE_DIM}}}$',
            ha='center', va='center',
            fontsize=FS_TENSOR, fontweight='bold')
    ax.text(handoff_dst_x + 0.023, handoff_dst_y - 0.018,
            'per node, from step A',
            ha='center', va='center',
            fontsize=FS_TINY, color=INK_SOFT, style='italic')

    anchor_x = handoff_dst_x + 0.068   # right edge of 128-d block
    anchor_y = handoff_dst_y

    # --- GraphEdgeAttention block -----------------------------------
    gat_x = anchor_x + 0.18
    gat_y = anchor_y
    x0_gat, x1_gat, _ = _attention_box(ax, gat_x, gat_y, w=0.20, h=0.30)
    _arrow(ax, (anchor_x, anchor_y),
           (x0_gat - 0.005, gat_y),
           color=INK_SOFT, lw=1.4, mut=14)

    # DTW edge_feat input box (above GAT, kept inside sub-row B).
    edge_box_cx = gat_x
    edge_box_cy = gat_y + 0.22         # 0.42 — well below the 0.515 separator
    edge_box_w  = 0.22
    edge_box_h  = 0.048
    _round_box(ax, edge_box_cx - edge_box_w / 2,
               edge_box_cy - edge_box_h / 2,
               edge_box_w, edge_box_h, fc=EDGE_FILL, ec=INK, lw=1.0,
               rounding=0.012)
    ax.text(edge_box_cx, edge_box_cy,
            r'edge_feat $\in \mathbb{R}^{3}$   (DTW)',
            ha='center', va='center', fontsize=FS_TENSOR)
    _arrow(ax, (edge_box_cx, edge_box_cy - edge_box_h / 2),
           (gat_x, gat_y + 0.15), color=INK_SOFT, lw=1.2, mut=12)

    # GAT output 256-d label
    gat_out_x = x1_gat + 0.012
    ax.text(gat_out_x + 0.020, gat_y + 0.06,
            f'$\\to \\mathbb{{R}}^{{{GAT_OUT}}}$',
            ha='left', va='center', fontsize=FS_TENSOR,
            fontweight='bold')

    # --- Classifier row ---------------------------------------------
    cls_xL = gat_out_x + 0.060
    cls_y = gat_y
    _, cls_right = _classifier_row(
        ax, cls_xL, cls_y,
        card_w=0.052, card_h=0.060, gap=0.008)
    _arrow(ax, (x1_gat + 0.005, gat_y), (cls_xL - 0.005, gat_y),
           color=INK_SOFT, lw=1.4, mut=14)
    ax.text((cls_xL + cls_right[0]) / 2, cls_y + 0.060,
            'Classifier MLP', ha='center', va='bottom',
            fontsize=FS_TENSOR, fontweight='bold')

    # --- Output heads -----------------------------------------------
    head_x = cls_right[0] + 0.025
    head_y = cls_y
    anchors = _output_heads(ax, head_x, head_y, head_w=0.072,
                            head_h=0.042, gap=0.010)
    cls_right_anchor_x = cls_right[0] + 0.005
    for (hx, hy) in anchors:
        _arrow(ax, (cls_right_anchor_x, cls_y), (hx - 0.003, hy),
               color=INK_SOFT, lw=1.0, mut=9)

    # Group brackets for output heads
    reg_top = anchors[0][1] + 0.022
    reg_bot = anchors[2][1] - 0.022
    cls_top = anchors[3][1] + 0.022
    cls_bot = anchors[-1][1] - 0.025
    bx = head_x + 0.072 + 0.010
    ax.plot([bx, bx + 0.010, bx + 0.010, bx],
            [reg_top, reg_top, reg_bot, reg_bot],
            color=HEAD_REG, lw=1.0)
    ax.text(bx + 0.014, (reg_top + reg_bot) / 2, 'regression',
            ha='left', va='center', fontsize=FS_TINY,
            color=HEAD_REG, fontweight='bold', rotation=270)
    ax.plot([bx, bx + 0.010, bx + 0.010, bx],
            [cls_top, cls_top, cls_bot, cls_bot],
            color=HEAD_CLS, lw=1.0)
    ax.text(bx + 0.014, (cls_top + cls_bot) / 2, 'classification',
            ha='left', va='center', fontsize=FS_TINY,
            color=HEAD_CLS, fontweight='bold', rotation=270)


# =====================================================================
# Compose the figure.
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_pdf',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'architecture.pdf'),
        help='Output PDF path (default: alongside this script)',
    )
    args = parser.parse_args()

    fig = plt.figure(figsize=(30, 24))
    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.00, 1.10],
        hspace=0.18,
        top=0.94, bottom=0.04, left=0.025, right=0.975,
    )

    # ROW 1 — three sub-panels.
    row1 = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], width_ratios=[1.0, 1.20, 0.85],
        wspace=0.12,
    )

    ax_a = fig.add_subplot(row1[0, 0])
    draw_panel_node_features(ax_a)

    draw_panel_edge_features(fig, row1[0, 1])

    ax_c = fig.add_subplot(row1[0, 2])
    draw_panel_graph_assembly(ax_c)

    # ROW 2 — single wide axis for the network.
    ax_net = fig.add_subplot(outer[1])
    draw_row2_network(ax_net)

    # "Input representations" — sits in the top margin of the figure;
    # "Network forward pass" is rendered inside the network axis itself
    # (see draw_row2_network) so it cannot collide with row 1 content.
    fig.text(0.5, 0.965, 'Input representations',
             ha='center', va='center',
             fontsize=FS_ROW_TITLE, fontweight='bold')

    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f"Saved: {args.out_pdf}")


if __name__ == "__main__":
    main()
