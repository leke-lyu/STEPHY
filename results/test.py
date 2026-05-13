#!/usr/bin/env python3
"""LGAT-style message-passing visualization (one round, left-to-right),
with an MLP head appended.

    layer l (K4)  -->  m_neigh + m_self  -->  layer l+1 (K4)  -->  MLP  -->  y_hat
                    \\                 /
                     alpha_neigh, alpha_self

Self node = Loc_a (top of the diamond, red). Neighbors = Loc_b/c/d
(gray). Layer planes are drawn as thin 3D slabs (top face + front
strip + side strip) so the perspective reads as a floating plate
rather than a flat parallelogram. The MLP head (4 amber 3D blocks
256 -> 128 -> 64 -> 32) takes h_a as input and emits y_hat.

Usage:
    python3 test.py
    python3 test.py --out test.pdf
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Ellipse, FancyArrowPatch, Polygon, Rectangle,
)


mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.linewidth':   0.8,
})

C_HL    = '#C0392B'
C_DIM   = '#BDBDBD'
C_EDGE  = '#444'
C_PLANE_TOP   = '#EDF1F6'   # plate top face (where graph sits)
C_PLANE_FRONT = '#C9D1DC'   # plate front (thickness)
C_PLANE_SIDE  = '#B6BFCD'   # plate right side (thickness)
C_PLANE_EDGE  = '#7d8794'
C_MSG       = '#FFFFFF'
C_MSG_EDGE  = '#444'
C_ALPHA     = '#3a1f6b'
C_MLP       = '#F2A93B'      # gold for MLP blocks
C_MLP_EDGE  = '#7a5310'


def _draw_panel_3d(ax, x_left, y_bottom, thick=0.7, height=4.0,
                   depth_x=2.6, depth_y=1.0, side='right'):
    """3D panel cuboid in the same style as the MLP blocks
    (front + top + side faces visible). The THIN (1xH) face is
    what the viewer sees head-on; the LARGE (HxH) 'K4 face' recedes
    into the picture along the depth axis, where the K4 is painted.

    side='right' (use for the LEFT panel of the figure):
        front face on the LEFT, K4 face recedes back-right.
    side='left' (use for the RIGHT panel of the figure):
        front face on the RIGHT, K4 face recedes back-left.

    Returns (origin, u_vec, v_vec) — the affine basis on the K4
    face. K4 nodes at UV in [0,1]^2 are placed at
    origin + u*u_vec + v*v_vec."""
    if side == 'right':
        front_pts = [
            (x_left,                 y_bottom),
            (x_left + thick,         y_bottom),
            (x_left + thick,         y_bottom + height),
            (x_left,                 y_bottom + height),
        ]
        face_pts = [
            (x_left + thick,                 y_bottom),
            (x_left + thick + depth_x,       y_bottom + depth_y),
            (x_left + thick + depth_x,       y_bottom + depth_y + height),
            (x_left + thick,                 y_bottom + height),
        ]
        top_pts = [
            (x_left,                         y_bottom + height),
            (x_left + thick,                 y_bottom + height),
            (x_left + thick + depth_x,       y_bottom + height + depth_y),
            (x_left + depth_x,               y_bottom + height + depth_y),
        ]
        origin = (x_left + thick, y_bottom)
        u_vec  = (depth_x, depth_y)
        v_vec  = (0.0, height)
    else:  # side == 'left'
        front_x = x_left + depth_x
        front_pts = [
            (front_x,                 y_bottom),
            (front_x + thick,         y_bottom),
            (front_x + thick,         y_bottom + height),
            (front_x,                 y_bottom + height),
        ]
        face_pts = [
            (front_x,                 y_bottom),
            (front_x - depth_x,       y_bottom + depth_y),
            (front_x - depth_x,       y_bottom + depth_y + height),
            (front_x,                 y_bottom + height),
        ]
        top_pts = [
            (front_x - depth_x,       y_bottom + height + depth_y),
            (front_x,                 y_bottom + height),
            (front_x + thick,         y_bottom + height),
            (front_x + thick - depth_x, y_bottom + height + depth_y),
        ]
        origin = (front_x, y_bottom)
        u_vec  = (-depth_x, depth_y)
        v_vec  = (0.0, height)

    # Draw faces back-to-front: top, K4 face, front.
    ax.add_patch(Polygon(top_pts, closed=True,
                         facecolor=C_PLANE_FRONT,
                         edgecolor=C_PLANE_EDGE,
                         lw=1.0, alpha=1.0, zorder=0))
    ax.add_patch(Polygon(face_pts, closed=True,
                         facecolor=C_PLANE_TOP,
                         edgecolor=C_PLANE_EDGE,
                         lw=1.2, alpha=1.0, zorder=1))
    ax.add_patch(Polygon(front_pts, closed=True,
                         facecolor=C_PLANE_SIDE,
                         edgecolor=C_PLANE_EDGE,
                         lw=1.0, alpha=1.0, zorder=2))
    return origin, u_vec, v_vec


def _draw_sphere_node(ax, cx, cy, rx, ry,
                      face_color, edge_color,
                      label, label_color='white',
                      label_fontsize=10, zorder_base=5):
    """Draw a 3D-shaded ellipsoidal node — main ellipse + smaller
    light highlight in the upper-left quadrant + faint dark crescent
    in the lower-right (gives a soft sphere look)."""
    # Soft drop shadow.
    ax.add_patch(Ellipse((cx + rx * 0.10, cy - ry * 0.16),
                         width=2 * rx * 1.05, height=2 * ry * 1.05,
                         facecolor='#000', edgecolor='none',
                         alpha=0.10, zorder=zorder_base - 1))
    # Main body.
    ax.add_patch(Ellipse((cx, cy), width=2 * rx, height=2 * ry,
                         facecolor=face_color, edgecolor=edge_color,
                         lw=1.2, zorder=zorder_base))
    # Dark shading crescent (lower-right) — slightly offset darker
    # ellipse clipped by the body.
    ax.add_patch(Ellipse((cx + rx * 0.08, cy - ry * 0.10),
                         width=2 * rx * 0.94, height=2 * ry * 0.94,
                         facecolor='#000', edgecolor='none',
                         alpha=0.10, zorder=zorder_base + 1))
    # Highlight (upper-left).
    ax.add_patch(Ellipse((cx - rx * 0.32, cy + ry * 0.30),
                         width=2 * rx * 0.45, height=2 * ry * 0.45,
                         facecolor='white', edgecolor='none',
                         alpha=0.40, zorder=zorder_base + 2))
    # Label.
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=label_fontsize, fontweight='bold',
            color=label_color, zorder=zorder_base + 3)


def _draw_k4(ax, origin, u_vec, v_vec,
             self_loc='a', label_template='Loc',
             draw_directed=True, edge_attention=None,
             k4_uv_radius=0.30, node_uv_radius=0.10):
    """Diamond K4 painted on a 3D face. (origin, u_vec, v_vec) is
    the affine basis of the K4 face (returned by _draw_panel_3d).
    K4 nodes live at UV coordinates on the face; their projected
    screen ellipses inherit the face's perspective shape."""
    def _uv_to_xy(u, v):
        return (origin[0] + u * u_vec[0] + v * v_vec[0],
                origin[1] + u * u_vec[1] + v * v_vec[1])
    # K4 diamond in UV (face-local) coordinates. Center = (0.5, 0.5),
    # radius = k4_uv_radius (in face units).
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

    # Projected node ellipse half-axes. v-axis is purely vertical
    # (height = v_vec[1]), so vertical radius = r_uv * v_vec[1].
    # u-axis projects to (depth_x, depth_y); horizontal extent of a
    # circle of UV radius r_uv ~ |r_uv * u_vec[0]| (we ignore the
    # small y component of u for simplicity — gives a clean
    # axis-aligned ellipse).
    node_ry = node_uv_radius * abs(v_vec[1])
    node_rx = node_uv_radius * abs(u_vec[0])

    if edge_attention is not None:
        shrink = 14   # in points; tuned to keep arrows clear of node ellipses
        offset_d = 0.06
        for src in locs:
            for dst in locs:
                if src == dst:
                    continue
                w = edge_attention.get((src, dst), 0.5)
                lw = 0.8 + 1.8 * w
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
                    arrowprops=dict(arrowstyle='-|>', color=C_EDGE,
                                    lw=lw, mutation_scale=10,
                                    shrinkA=shrink, shrinkB=shrink,
                                    zorder=3),
                )
    elif draw_directed:
        for i in range(len(locs)):
            for j in range(i + 1, len(locs)):
                x0, y0 = pos[locs[i]]
                x1, y1 = pos[locs[j]]
                ax.plot([x0, x1], [y0, y1],
                        color='#888', lw=1.4, zorder=2,
                        solid_capstyle='round')

    for l in locs:
        x, y = pos[l]
        fc = C_HL if l == self_loc else C_DIM
        ec = '#7a1f1f' if l == self_loc else '#5a5a5a'
        if label_template == 'Loc':
            text = f'Loc$_{l}$'
        else:
            text = f'${label_template}_{l}$'
        _draw_sphere_node(ax, x, y, node_rx, node_ry,
                          face_color=fc, edge_color=ec,
                          label=text, label_color='white',
                          label_fontsize=9, zorder_base=5)
    return pos


def _draw_block(ax, x_left, y_center, width, height, depth=0.32,
                color=C_MLP, edgecolor=C_MLP_EDGE, alpha=0.92):
    """3D-ish amber block (front + top + right faces)."""
    front = Rectangle((x_left, y_center - height / 2),
                      width, height,
                      facecolor=color, edgecolor=edgecolor,
                      lw=1.2, alpha=alpha, zorder=4)
    ax.add_patch(front)
    top_pts = [
        (x_left,                 y_center + height / 2),
        (x_left + depth,         y_center + height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,         y_center + height / 2),
    ]
    ax.add_patch(Polygon(top_pts, facecolor=color, edgecolor=edgecolor,
                         lw=1.0, alpha=alpha * 0.75, zorder=4))
    right_pts = [
        (x_left + width,         y_center - height / 2),
        (x_left + width + depth, y_center - height / 2 + depth * 0.6),
        (x_left + width + depth, y_center + height / 2 + depth * 0.6),
        (x_left + width,         y_center + height / 2),
    ]
    ax.add_patch(Polygon(right_pts, facecolor=color, edgecolor=edgecolor,
                         lw=1.0, alpha=alpha * 0.55, zorder=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='test.pdf',
                    help='Output filename (saved next to this script).')
    args = ap.parse_args()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.out,
    )

    fig, ax = plt.subplots(figsize=(18, 5.0))
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal', adjustable='box')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Incoming-to-'a' attention (b->a, c->a, d->a) is spread more
    # widely so the per-arrow width differences are unambiguously
    # visible in the gather arrows.
    eatt = {
        ('a', 'b'): 0.45, ('b', 'a'): 0.08,
        ('a', 'c'): 0.30, ('c', 'a'): 0.45,
        ('a', 'd'): 0.20, ('d', 'a'): 0.28,
        ('b', 'c'): 0.55, ('c', 'b'): 0.35,
        ('b', 'd'): 0.40, ('d', 'b'): 0.65,
        ('c', 'd'): 0.25, ('d', 'c'): 0.45,
    }

    panel_thick   = 0.6
    panel_height  = 4.0
    panel_depth_x = 2.6
    panel_depth_y = 1.0

    # ----- Layer l (LEFT) ------------------------------------------------
    # Left panel: thin front face on the LEFT, K4 face recedes
    # back-RIGHT (toward the messages in the middle).
    pl_l_x, pl_l_y = 0.8, 1.3
    origin_l, u_l, v_l = _draw_panel_3d(
        ax, pl_l_x, pl_l_y,
        thick=panel_thick, height=panel_height,
        depth_x=panel_depth_x, depth_y=panel_depth_y,
        side='right',
    )
    pos_l = _draw_k4(ax, origin_l, u_l, v_l,
                     self_loc='a', label_template='h',
                     edge_attention=eatt,
                     k4_uv_radius=0.30, node_uv_radius=0.10)
    # Layer label sits on the front face (the visible thin "1xH" face).
    ax.text(pl_l_x + panel_thick / 2, pl_l_y + 0.20,
            r'$\ell$',
            ha='center', va='bottom', fontsize=14,
            color='#666', style='italic')

    # ----- Aggregation + self pass-through (MIDDLE) ----------------------
    # In the actual STEPHY GAT (stephy/model.py:36-45), each node v's
    # update is:
    #     agg_v = sum_u alpha_uv * h_u            (over ALL u incl. self)
    #     h_v^(l+1) = concat(h_v^(l), agg_v)      (128 || 128 -> 256)
    # The "self vs neighbor" split lives in the CONCAT, not in two
    # separate alpha-gated messages. So we draw:
    #   TOP   ellipse: h_a^(l)   — the self pass-through (128-d).
    #   BOTTOM ellipse: agg_a    — attention-weighted sum from all
    #                              four K4 nodes (incl. a via self-loop).
    # The two streams merge through a 'concat' operator, producing
    # the 256-d h_a^(l+1).
    msg_x = 8.4
    msg_w, msg_h = 1.7, 0.95
    panel_mid_y = pl_l_y + panel_height / 2
    # Center gap of 1.85 leaves ~0.90 of clear space between the two
    # ellipse edges — enough for the "concat" text to sit cleanly
    # in the middle without overlapping either body.
    self_xy = (msg_x, panel_mid_y + 1.30)
    agg_xy  = (msg_x, panel_mid_y - 0.55)
    for cx_e, cy_e, label in [
        (self_xy[0], self_xy[1],
         r'$h_{a}^{(\ell)}$'),
        (agg_xy[0],  agg_xy[1],
         r'$\mathrm{agg}_{a}$'),
    ]:
        ax.add_patch(Ellipse((cx_e, cy_e), msg_w, msg_h,
                             facecolor=C_MSG, edgecolor=C_MSG_EDGE,
                             lw=1.4, zorder=4))
        ax.text(cx_e, cy_e, label, ha='center', va='center',
                fontsize=14, color='#222', zorder=5)
    # Caption under agg_a giving the formula. STEPHY's K4 has NO
    # self-loop (stephy/data.py:5), so the sum runs over neighbors
    # u != a only — three terms for K4.
    ax.text(agg_xy[0], agg_xy[1] - msg_h / 2 - 0.30,
            r'$\sum_{u \neq a}\, \alpha_{au}\, h_{u}^{(\ell)}$',
            ha='center', va='top',
            fontsize=11, color='#444', style='italic')

    agg_left  = (agg_xy[0] - msg_w / 2 - 0.02, agg_xy[1])
    self_left = (self_xy[0] - msg_w / 2 - 0.02, self_xy[1])
    # NEIGHBORS only (b, c, d) feed into agg_a — STEPHY's K4 has
    # no self-loops. Each arrow is a solid PURPLE line whose width
    # encodes the incoming attention weight alpha_au (= weight on
    # the directed edge u -> a).
    rad_for = {'b': -0.20, 'c': -0.04, 'd': -0.20}
    incoming_alpha = {l: eatt[(l, 'a')] for l in ('b', 'c', 'd')}
    for l in ('b', 'c', 'd'):
        sx, sy = pos_l[l]
        w = incoming_alpha[l]
        # Aggressive linear mapping: w in [0.10, 0.90] -> lw in
        # [0.8, 7.0]. Gives a clearly visible width spread for the
        # three gather arrows.
        lw = 0.8 + (w - 0.10) / 0.80 * 6.2
        ax.add_patch(FancyArrowPatch(
            (sx, sy), agg_left,
            connectionstyle=f'arc3,rad={rad_for[l]}',
            shrinkA=14, shrinkB=4,
            arrowstyle='-|>', color=C_ALPHA,
            lw=lw, mutation_scale=11, zorder=4,
        ))
    # Self pass-through arrow: a -> h_a^(l) ellipse. Solid (not
    # dashed) since it's the original embedding carried through
    # unchanged, not a learned aggregation.
    sx, sy = pos_l['a']
    ax.add_patch(FancyArrowPatch(
        (sx, sy), self_left,
        connectionstyle='arc3,rad=-0.18',
        shrinkA=14, shrinkB=4,
        arrowstyle='-|>', color='#444',
        lw=1.2, mutation_scale=10, zorder=4,
    ))

    # ----- Layer l+1 (MIDDLE-RIGHT) --------------------------------------
    # Same orientation as layer l: thin front face on the LEFT, K4
    # face recedes back-right.
    pl_r_x, pl_r_y = 10.5, pl_l_y
    origin_r, u_r, v_r = _draw_panel_3d(
        ax, pl_r_x, pl_r_y,
        thick=panel_thick, height=panel_height,
        depth_x=panel_depth_x, depth_y=panel_depth_y,
        side='right',
    )
    pos_r = _draw_k4(ax, origin_r, u_r, v_r,
                     self_loc='a', label_template='h',
                     edge_attention=eatt,
                     k4_uv_radius=0.30, node_uv_radius=0.10)
    # Layer label on the front face (matches layer l placement).
    ax.text(pl_r_x + panel_thick / 2, pl_r_y + 0.20,
            r'$\ell\!+\!1$',
            ha='center', va='bottom', fontsize=14,
            color='#666', style='italic')

    # h_a^(l) and agg_a -> "concat" -> h_a^(l+1) on the L+1 K4.
    # Plain neutral text in the gap between the two ellipses, then
    # a single curved arrow into h_a in layer l+1.
    tx, ty = pos_r['a']
    concat_x = self_xy[0]
    concat_y = 0.5 * (self_xy[1] + agg_xy[1])
    ax.text(concat_x, concat_y, 'concat',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#333', style='italic', zorder=6)
    ax.add_patch(FancyArrowPatch(
        (concat_x + 0.55, concat_y), (tx, ty),
        connectionstyle='arc3,rad=-0.18',
        arrowstyle='-|>', color='#444',
        lw=1.6, mutation_scale=14,
        shrinkA=2, shrinkB=22, zorder=4,
    ))

    # ----- MLP head (RIGHT) ----------------------------------------------
    dims    = [256, 128, 64,  32]
    heights = [2.4, 1.85, 1.4, 1.0]
    blk_w   = 0.55
    blk_dep = 0.22
    spacing = 0.85
    centers_y = panel_mid_y
    n_blocks = len(dims)
    total_w = n_blocks * blk_w + (n_blocks - 1) * spacing
    mlp_x_left0 = 17.6
    mlp_x_end   = mlp_x_left0 + total_w + blk_dep

    left_edges, right_edges = [], []
    for i, (d, h) in enumerate(zip(dims, heights)):
        x_left = mlp_x_left0 + i * (blk_w + spacing)
        _draw_block(ax, x_left=x_left, y_center=centers_y,
                    width=blk_w, height=h, depth=blk_dep,
                    color=C_MLP, edgecolor=C_MLP_EDGE)
        ax.text(x_left + blk_w / 2, centers_y, str(d),
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='#222',
                zorder=6)
        left_edges.append(x_left)
        right_edges.append(x_left + blk_w)

    # Inter-block ReLU arrows.
    for src, dst in [(0, 1), (1, 2), (2, 3)]:
        x_src = right_edges[src] + blk_dep + 0.04
        x_dst = left_edges[dst] - 0.04
        ax.annotate('',
                    xy=(x_dst, centers_y),
                    xytext=(x_src, centers_y),
                    arrowprops=dict(arrowstyle='->', mutation_scale=12,
                                    lw=1.2, color='#444'))
        ax.text(0.5 * (x_src + x_dst),
                centers_y + max(heights) / 2 + 0.20,
                'ReLU', ha='center', va='bottom',
                fontsize=10, color='#666', style='italic')

    # h_a → MLP first block (gray arrow leaving the right-K4 self node).
    h_in_x = mlp_x_left0 - 0.04
    ax.annotate(
        '', xy=(h_in_x, centers_y),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle='-|>', color='#444',
                        lw=1.6, mutation_scale=14,
                        shrinkA=18, shrinkB=4, zorder=4),
    )
    # Caption above this transit arrow (placed at the arrow's
    # midpoint, slightly above the line).
    arrow_mid_x = 0.5 * (tx + h_in_x)
    arrow_mid_y = 0.5 * (ty + centers_y)
    ax.text(arrow_mid_x, arrow_mid_y + 0.30,
            r'$h_a$',
            ha='center', va='center',
            fontsize=13, color='#222', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.92))

    # MLP final → y_hat caption.
    x_final_src = right_edges[-1] + blk_dep + 0.04
    x_yhat = x_final_src + 1.1
    ax.annotate('',
                xy=(x_yhat - 0.05, centers_y),
                xytext=(x_final_src, centers_y),
                arrowprops=dict(arrowstyle='-|>', mutation_scale=12,
                                lw=1.4, color='#444'))
    ax.text(x_yhat, centers_y, r'$\hat{y}$',
            ha='left', va='center',
            fontsize=16, fontweight='bold', color='#222')

    # Output captions below MLP stack (matches panel f convention).
    ax.text(mlp_x_left0 + total_w / 2,
            centers_y - max(heights) / 2 - 0.50,
            r'out_dim = 1  (point estimate)',
            ha='center', va='center',
            fontsize=10, color='#222')
    ax.text(mlp_x_left0 + total_w / 2,
            centers_y - max(heights) / 2 - 1.00,
            r'out_dim = 3  (CQR quantiles)',
            ha='center', va='center',
            fontsize=10, color='#222')

    # Title at the bottom.
    ax.text(13.5, 0.55,
            'message passing through one layer (LGAT)  +  MLP head',
            ha='center', va='center',
            fontsize=14, fontweight='bold', color='#222')

    fig.savefig(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
