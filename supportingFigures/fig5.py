#!/usr/bin/env python3
"""
Figure 4 — Denmark SARS-CoV-2 5-clade composite (ML + bootstrap).

Five-row x five-column grid, one row per Nextstrain clade
(20I, 21I, 21J, 21K, 21L):

  Col 0:  Exploded ML timetree, branches colored by Danish division.
  Col 1:  ML Source-Sink Score choropleth.
  Col 2:  Bootstrap distribution of SSS per region (40 replicates) + ML star.
  Col 3:  ML R_0 choropleth.
  Col 4:  Bootstrap distribution of R_0 per region (40 replicates) + ML star.

Tree topology comes from stephy_input/<clade>/newick/ml.nwk; division and
numdate are joined onto each node from the parent lineage's augur outputs
(traits.json + branch_lengths.json) by node name. Predictions come from
stephy_output/<clade>/regression.tsv (ml row + 40 replicate rows).

Usage:
    python3 fig5.py
    python3 fig5.py --base_dir /path/to/bootstrap_uncertainty --geojson /path/to/gadm41_DNK_1.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import baltic as bt
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from shapely.geometry import box, MultiPolygon

# ---------------------------------------------------------------------------
# Publication defaults (mirror denmark_old)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLADES = ['20I', '21I', '21J', '21K', '21L']
CLADE_LABELS = {
    '20I': 'Alpha 20I',
    '21I': 'Delta 21I',
    '21J': 'Delta 21J',
    '21K': 'Omicron 21K',
    '21L': 'Omicron 21L',
}
CLADE_TO_LINEAGE = {
    '20I': 'Alpha',
    '21I': 'Delta',
    '21J': 'Delta',
    '21K': 'Omicron',
    '21L': 'Omicron',
}

DIVISION_COLORS = {
    'Hovedstaden': '#4C90C0',
    'Midtjylland': '#E68133',
    'Nordjylland': '#75B681',
    'Sjaelland':   '#D4534E',
    'Syddanmark':  '#9B59B6',
}
DEFAULT_COLOR = '#AAAAAA'
BRANCH_COLOR = '#888888'   # tree branches: neutral grey; tips keep division color

SSS_CMAP = LinearSegmentedColormap.from_list(
    'sss_diverging',
    ['#2166AC', '#67A9CF', '#D1E5F0', '#FAFAFA',
     '#FDDBC7', '#EF8A62', '#B2182B'],
)
R0_CMAP = LinearSegmentedColormap.from_list(
    'r0_sequential',
    ['#FFFFD4', '#FEE391', '#FEC44F', '#FE9929',
     '#EC7014', '#CC4C02', '#8C2D04'],
)

REGION_NAME_MAP = {'Sjælland': 'Sjaelland'}

# Region order for the bootstrap violins: left -> right by population (matches
# stephy_lib.DIVISION_TO_LOC and the reference violin_r0_sss.png).
REGIONS_BY_POP = ['Hovedstaden', 'Midtjylland', 'Syddanmark',
                  'Sjaelland', 'Nordjylland']
SSS_YLIM = (-1.05, 1.05)
R0_YLIM = (0.8, 3.0)
VIOLIN_FACE = '#A9C5DE'
ML_COLOR = '#D62728'

FIG_WIDTH = 3.39
FIG_RATIO = 2
Y_GAP = 500        # vertical y-data gap between stacked trees (must fit a label)
LABEL_OFFSET = 80  # label baseline above each tree's top tip (y-data units)
HORIZ_LW = 0.4
TIP_SIZE = 1


# ---------------------------------------------------------------------------
# stdout tee — diagnostic output saved next to the script
# ---------------------------------------------------------------------------
class _Tee:
    """Mirror writes across multiple streams (used to tee stdout to fig5.out)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def division_color(node):
    """Return the color for a node based on its division trait."""
    return DIVISION_COLORS.get(node.traits.get('division', ''), DEFAULT_COLOR)


def decimal_to_datetime(dec_year):
    """Convert decimal year to datetime."""
    year = int(dec_year)
    remainder = dec_year - year
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return start + timedelta(seconds=(end - start).total_seconds() * remainder)


def node_name(k):
    """Return the best-effort name string for a baltic node (tip or internal)."""
    name = getattr(k, 'name', None)
    if name:
        return name
    return k.traits.get('label')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clade_tree(base_dir, clade):
    """Load a clade's ML topology and decorate nodes with division + numdate.

    Topology comes from the per-clade Newick under stephy_input/<clade>/newick/
    so we get a clean clade-scoped tree (no outgroup, no sibling clades).
    Node-level division and absolute time come from the parent lineage's
    augur outputs (traits.json, branch_lengths.json) and are joined by node
    name (DCGC tip IDs and NODE_xxxxx internal labels match across files).
    """
    lineage = CLADE_TO_LINEAGE[clade]
    tree_path = base_dir / 'stephy_input' / clade / 'newick' / 'ml.nwk'
    bl_path = base_dir / lineage / 'ml_point_estimate' / 'branch_lengths.json'
    traits_path = base_dir / lineage / 'ml_point_estimate' / 'traits.json'

    tree = bt.loadNewick(str(tree_path), absoluteTime=False)
    tree.traverse_tree()
    tree.sortBranches()

    with open(bl_path) as f:
        bl_nodes = json.load(f)['nodes']
    with open(traits_path) as f:
        traits_nodes = json.load(f)['nodes']

    missing_time = 0
    missing_div = 0
    for k in tree.Objects:
        name = node_name(k)
        if name and name in bl_nodes and 'numdate' in bl_nodes[name]:
            k.absoluteTime = float(bl_nodes[name]['numdate'])
        else:
            missing_time += 1
        if name and name in traits_nodes:
            k.traits['division'] = traits_nodes[name].get('division', '')
        else:
            missing_div += 1

    # Root absoluteTime is needed for the chronological sort + label placement.
    if tree.root.absoluteTime is None:
        # Fall back to the earliest descendant tip time minus a small epsilon.
        tip_times = [k.absoluteTime for k in tree.getExternal()
                     if k.absoluteTime is not None]
        if tip_times:
            tree.root.absoluteTime = min(tip_times) - 0.01

    if missing_time or missing_div:
        print(f'  ({missing_time} nodes missing numdate, '
              f'{missing_div} missing division)')
    return tree


def compute_subtree_sizes(tree):
    """Count descendant tips per internal node (drives vertical branch width)."""
    sizes = {id(k): 1 for k in tree.getExternal()}
    remaining = list(tree.getInternal())
    while remaining:
        still = []
        for k in remaining:
            if all(id(ch) in sizes for ch in k.children):
                sizes[id(k)] = sum(sizes[id(ch)] for ch in k.children)
            else:
                still.append(k)
        if len(still) == len(remaining):
            break  # safeguard against malformed topology
        remaining = still
    return sizes


def load_clade_predictions(base_dir, clade):
    """Return ML + bootstrap distributions for SSS and R0 from regression.tsv.

    Returns:
        sss_ml, r0_ml:    {region: float}                — ML point estimates.
        sss_boot, r0_boot:{region: 1D np.ndarray}        — per-replicate values.
    """
    reg_path = base_dir / 'stephy_output' / clade / 'regression.tsv'
    df = pd.read_csv(reg_path, sep='\t')
    ml = df[df['source'] == 'ml']
    boot = df[df['source'].astype(str).str.startswith('replicate_')]

    sss_ml = dict(zip(ml['location_name'], ml['sss_point']))
    r0_ml = dict(zip(ml['location_name'], ml['r0_point']))
    sss_boot = {region: g['sss_point'].to_numpy()
                for region, g in boot.groupby('location_name')}
    r0_boot = {region: g['r0_point'].to_numpy()
               for region, g in boot.groupby('location_name')}
    return sss_ml, r0_ml, sss_boot, r0_boot


def load_regions(geojson_path):
    """Load Denmark regions GeoJSON, clipped to mainland (Bornholm excluded)."""
    gdf = gpd.read_file(geojson_path)
    gdf['region'] = gdf['NAME_1'].replace(REGION_NAME_MAP)
    clip_box = box(7.5, 54.0, 12.65, 58.0)
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.intersection(clip_box)

    def filter_small(geom, min_area=0.005):
        if isinstance(geom, MultiPolygon):
            parts = [p for p in geom.geoms if p.area > min_area]
            if not parts:
                return geom
            return MultiPolygon(parts) if len(parts) > 1 else parts[0]
        return geom

    gdf['geometry'] = gdf['geometry'].apply(filter_small)
    return gdf


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tree(ax, tree, y_offset, subtree_sizes, log_max):
    """Draw a single phylogeny via LineCollection.

    Horizontal branches: thin, neutral grey (BRANCH_COLOR).
    Vertical (joining) branches: grey, log-scaled width by descendant count.
    Tips: division-colored (the only color-coded element).

    log_max is shared across all clades (log2 of the largest subclade in the
    whole figure), so a subclade of N tips renders at the same width in every
    tree and the widths are comparable across clades.
    """
    def vert_lw(n):
        # log2(subclade tips) mapped to 0.05-1.0 pt on a GLOBAL scale shared by
        # all 5 clades: a 2-tip cherry -> 0.05 pt, the largest subclade in the
        # figure -> 1.0 pt, linear in log2 between.
        frac = (np.log2(max(n, 2)) - 1) / (log_max - 1) if log_max > 1 else 0.0
        return 0.05 + 0.95 * frac

    h_segments, h_colors = [], []
    v_by_lw = {}

    for k in tree.Objects:
        if k.absoluteTime is None:
            continue
        c = BRANCH_COLOR
        x_node = mdates.date2num(decimal_to_datetime(k.absoluteTime))
        if (k.parent and k.parent != 'Root'
                and getattr(k.parent, 'absoluteTime', None) is not None):
            x_parent = mdates.date2num(
                decimal_to_datetime(k.parent.absoluteTime))
        else:
            x_parent = x_node
        y_node = k.y + y_offset

        h_segments.append([(x_parent, y_node), (x_node, y_node)])
        h_colors.append(c)

        if k.branchType == 'node' and k.children:
            ys = [ch.y + y_offset for ch in k.children]
            lw_key = round(vert_lw(subtree_sizes.get(id(k), 2)), 2)
            v_by_lw.setdefault(lw_key, ([], []))
            v_by_lw[lw_key][0].append([(x_node, min(ys)), (x_node, max(ys))])
            v_by_lw[lw_key][1].append(c)

    if h_segments:
        ax.add_collection(LineCollection(
            h_segments, colors=h_colors, linewidths=HORIZ_LW,
            zorder=1, capstyle='round'))
    for lw_key, (segs, cols) in v_by_lw.items():
        ax.add_collection(LineCollection(
            segs, colors=cols, linewidths=lw_key,
            zorder=2, capstyle='round'))

    tips_by_color = {}
    for k in tree.getExternal():
        if k.absoluteTime is None:
            continue
        c = division_color(k)
        tips_by_color.setdefault(c, ([], []))
        tips_by_color[c][0].append(
            mdates.date2num(decimal_to_datetime(k.absoluteTime)))
        tips_by_color[c][1].append(k.y + y_offset)

    for c, (xs, ys) in tips_by_color.items():
        ax.scatter(xs, ys, s=TIP_SIZE, color=c, zorder=100, linewidths=0,
                   alpha=0.6)


def draw_violin(ax, ml_values, boot_values, ylim, ylabel,
                regions=REGIONS_BY_POP, show_xticks=False):
    """Per-clade bootstrap violin: distribution + 95% CI + median + ML star.

    Mirrors stephy_output/violin_r0_sss.png — light-blue body with a soft
    edge, black 95% interval, black median tick, red ML star. Regions
    ordered left -> right by population (Hovedstaden -> Nordjylland). The
    glyph legend is drawn once per column on the topmost violin (see main()).
    """
    positions = np.arange(len(regions))
    boot_data = [boot_values.get(r, np.array([])) for r in regions]
    ml_pts = [ml_values.get(r, np.nan) for r in regions]

    valid = [(i, d) for i, d in enumerate(boot_data) if len(d) > 0]
    if valid:
        parts = ax.violinplot(
            [d for _, d in valid],
            positions=[i for i, _ in valid],
            widths=0.78, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(VIOLIN_FACE)
            pc.set_edgecolor('#5B7E9E')
            pc.set_linewidth(0.6)
            pc.set_alpha(0.85)

    for i, d in enumerate(boot_data):
        if len(d) == 0:
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        med = np.median(d)
        ax.vlines(i, lo, hi, color='#222222', linewidth=0.8, zorder=3,
                  capstyle='round')
        ax.hlines(med, i - 0.22, i + 0.22,
                  color='#222222', linewidth=1.0, zorder=4,
                  capstyle='round')

    for i, v in enumerate(ml_pts):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.scatter([i], [v], marker='*', s=35, color=ML_COLOR,
                   edgecolors='black', linewidths=0.4, zorder=5)

    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(positions)
    if show_xticks:
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=5)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(axis='y', labelsize=5)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def draw_choropleth(ax, gdf, values, label, norm, cmap):
    """Draw a Denmark choropleth map for one clade (no ancestor highlight)."""
    merged = gdf.copy()
    merged['val'] = merged['region'].map(values)
    merged.plot(ax=ax, column='val', cmap=cmap, norm=norm,
                edgecolor='#333333', linewidth=0.4)

    ax.set_xlim(7.9, 12.7)
    ax.set_ylim(54.45, 57.85)
    ax.set_aspect(1 / np.cos(np.radians(56)))
    ax.axis('off')
    ax.set_title(label, fontsize=7, fontweight='bold',
                 color='#333333', pad=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build fig5.pdf — 5-clade trees + ML SSS / R_0 choropleths + bootstrap violins.

    Loads each clade's ML tree (decorated with division + numdate from the
    parent lineage's augur outputs), reads ML and bootstrap predictions from
    stephy_output/<clade>/regression.tsv, then assembles a 5x5 panel grid
    with one row per clade and shared colorbars under the map columns.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', type=str,
        default='/Users/lukelyu/Desktop/denmark_case/nextstrain/'
                'bootstrap_uncertainty',
        help='Root containing <Lineage>/ml_point_estimate/, stephy_input/, '
             'stephy_output/.')
    parser.add_argument(
        '--geojson', type=str,
        default='/Users/lukelyu/Desktop/vault/denmark_old/figure/'
                'gadm41_DNK_1.json',
        help='GADM Denmark regions GeoJSON (reused from denmark_old).')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    log = open(out_dir / 'fig5.out', 'w')
    sys.stdout = _Tee(sys.__stdout__, log)

    # --- Load trees ---
    loaded = []
    for clade in CLADES:
        print(f'Loading {clade}...', end=' ', flush=True)
        tree = load_clade_tree(base_dir, clade)
        n_tips = len(tree.getExternal())
        sizes = compute_subtree_sizes(tree)
        print(f'{n_tips} tips')
        loaded.append((tree, CLADE_LABELS[clade], n_tips, sizes, clade))

    # Chronological: earliest clade at top
    loaded.sort(key=lambda t: (t[0].root.absoluteTime, -t[2]), reverse=True)
    visual_order = list(reversed(loaded))

    # --- Load per-clade ML + bootstrap predictions ---
    clade_data = {}
    for clade in CLADES:
        sss_ml, r0_ml, sss_boot, r0_boot = load_clade_predictions(
            base_dir, clade)
        clade_data[CLADE_LABELS[clade]] = {
            'sss_ml': sss_ml, 'r0_ml': r0_ml,
            'sss_boot': sss_boot, 'r0_boot': r0_boot,
        }
    sss_data = {lbl: d['sss_ml'] for lbl, d in clade_data.items()}
    r0_data = {lbl: d['r0_ml'] for lbl, d in clade_data.items()}

    gdf = load_regions(args.geojson)
    sss_norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    r0_norm = plt.Normalize(vmin=0.8, vmax=3.0)

    # --- Figure layout: tree | SSS map | SSS violin | R0 map | R0 violin ---
    fig_height = FIG_WIDTH * FIG_RATIO
    map_width = fig_height / 5
    violin_width = map_width
    fig = plt.figure(
        figsize=(FIG_WIDTH + 2 * (map_width + violin_width), fig_height),
        facecolor='w')
    gs = gridspec.GridSpec(
        5, 5, figure=fig,
        width_ratios=[FIG_WIDTH, map_width, violin_width,
                      map_width, violin_width],
        wspace=0.40, hspace=0.18)

    # --- Left: stacked trees ---
    ax_tree = fig.add_subplot(gs[:, 0])
    # Global vertical-width scale: largest subclade across ALL clades maps to
    # 2.0 pt, so branch widths are comparable between trees (a 7k-tip stem in
    # 21K reads thicker than the 571-tip 21I stem).
    global_max = max(max(sizes.values()) for _, _, _, sizes, _ in loaded)
    log_max = np.log2(max(global_max, 2))
    cumulative_y = 0
    # Omicron clades emerge late in time → root_x sits near the right edge of
    # the tree column. Left-shift their labels so the text doesn't extend past
    # the column into the SSS panels.
    LABEL_X_SHIFT_DAYS = {'21K': 90, '21L': 90}
    for tree, label, n_tips, sizes, clade in loaded:
        draw_tree(ax_tree, tree, cumulative_y, sizes, log_max)
        root_x = decimal_to_datetime(tree.root.absoluteTime)
        label_x = root_x - timedelta(days=LABEL_X_SHIFT_DAYS.get(clade, 0))
        ax_tree.text(
            label_x, cumulative_y + tree.ySpan + LABEL_OFFSET,
            f'{label} (n={n_tips:,})',
            ha='left', va='bottom', fontsize=6, fontweight='bold',
            color='#333333')
        cumulative_y += tree.ySpan + Y_GAP

    ax_tree.autoscale_view()
    for spine in ('top', 'right', 'left'):
        ax_tree.spines[spine].set_visible(False)
    ax_tree.spines['bottom'].set_linewidth(0.4)
    ax_tree.tick_params(axis='x', direction='out', length=2)
    ax_tree.tick_params(axis='y', size=0)
    ax_tree.set_yticklabels([])
    ax_tree.grid(axis='x', ls='--', alpha=0.3, linewidth=0.3)
    ax_tree.set_ylim(-2, cumulative_y)
    ax_tree.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax_tree.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax_tree.get_xticklabels(), rotation=45, ha='right')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=5, label=div)
               for div, c in DIVISION_COLORS.items()]
    ax_tree.legend(handles=handles, loc='lower left', frameon=False, ncol=1)
    ax_tree.text(-0.02, 1.0, 'a', transform=ax_tree.transAxes,
                 fontsize=14, fontweight='bold', va='top', ha='right')

    # --- SSS / R0 choropleth columns (1, 3) — panel letters span map+violin ---
    map_configs = [
        (1, sss_data, sss_norm, SSS_CMAP, 'bcdef'),
        (3, r0_data,  r0_norm,  R0_CMAP,  'ghijk'),
    ]
    last_ax = {}
    for col, data, norm, cmap, panel_letters in map_configs:
        for i, (_, label, _, _, _) in enumerate(visual_order):
            ax = fig.add_subplot(gs[i, col])
            values = data.get(label)
            if not values:
                ax.axis('off')
                ax.set_title(label, fontsize=7, fontweight='bold',
                             color='#333333')
                continue
            draw_choropleth(ax, gdf, values, label, norm, cmap)
            ax.text(-0.08, 1.05, panel_letters[i], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')
        last_ax[col] = ax

    # --- SSS / R0 bootstrap violin columns (2, 4) — share panel letter with
    #     the adjacent map (map+violin = one panel) ---
    violin_configs = [
        (2, 'sss_ml', 'sss_boot', SSS_YLIM, 'SSS'),
        (4, 'r0_ml',  'r0_boot',  R0_YLIM,  r'$R_0$'),
    ]
    n_rows = len(visual_order)
    for col, ml_key, boot_key, ylim, ylabel in violin_configs:
        for i, (_, label, _, _, _) in enumerate(visual_order):
            ax = fig.add_subplot(gs[i, col])
            d = clade_data.get(label, {})
            ml = d.get(ml_key, {})
            boot = d.get(boot_key, {})
            if not ml and not boot:
                ax.axis('off')
                continue
            draw_violin(ax, ml, boot, ylim, ylabel,
                        show_xticks=(i == n_rows - 1))

    # --- Colorbars: width = map column only; fewer ticks for breathing room ---
    fig.canvas.draw()
    tree_pos = ax_tree.get_position()
    sss_map_pos = last_ax[1].get_position()
    r0_map_pos  = last_ax[3].get_position()
    cbar_y = tree_pos.y0 - 0.03

    sm_sss = plt.cm.ScalarMappable(cmap=SSS_CMAP, norm=sss_norm)
    cbar_ax = fig.add_axes([sss_map_pos.x0 + 0.005, cbar_y,
                            sss_map_pos.width - 0.01, 0.008])
    cbar = fig.colorbar(sm_sss, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('SSS', fontsize=6)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(labelsize=5)

    sm_r0 = plt.cm.ScalarMappable(cmap=R0_CMAP, norm=r0_norm)
    cbar_ax = fig.add_axes([r0_map_pos.x0 + 0.005, cbar_y,
                            r0_map_pos.width - 0.01, 0.008])
    cbar = fig.colorbar(sm_r0, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$R_0$', fontsize=6)
    cbar.set_ticks([0.8, 2.0, 3.0])
    cbar.ax.tick_params(labelsize=5)

    out = out_dir / 'fig5.pdf'
    out_png = out_dir / 'fig5.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'Saved: {out}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
