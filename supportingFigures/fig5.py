#!/usr/bin/env python3
"""
Figure 5 — Denmark SARS-CoV-2 5-clade composite (ML + bootstrap).

Panels a-m. A five-row x five-column grid, one row per Nextstrain clade in
clade order (20I, 21I, 21J, 21K, 21L):

  Col 0:  Exploded ML timetree, branches colored by Danish division  (a)
  Col 1:  ML Source-Sink Score choropleth (STEPHY)                   (b-f)
  Col 2:  Paired bootstrap violins of SSS per region                 (b-f)
  Col 3:  ML R_0 choropleth (STEPHY)                                 (g-k)
  Col 4:  Paired bootstrap violins of R_0 per region                 (g-k)

Each map shares its panel letter with the violin beside it. Below the grid sit
two square scatters of STEPHY against the ablation, SSS (l) and R_0 (m).

The violins carry two models: STEPHY offset left of each region's tick, the
graph-free CBLV-CNN ablation right, so the ablation reads as a reference
against the paper's result rather than as a second result. Each body shows the
central 95% of its replicates; the choropleths stay STEPHY-only.

Tree topology comes from stephy_input/<clade>/newick/ml.nwk; division and
numdate are joined onto each node from the parent lineage's augur outputs
(traits.json + branch_lengths.json) by node name. Predictions come from
stephy_output/<clade>/regression.tsv and cblv-cnn_output/<clade>/regression.tsv
(ml row + 40 replicate rows each).

Usage:
    python3 fig5.py
    python3 fig5.py --base_dir /path/to/bootstrap_uncertainty --geojson /path/to/gadm41_DNK_1.json
    python3 fig5.py --cnn_dir /path/to/cblv-cnn_output
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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from shapely.geometry import box, MultiPolygon

from _paths import under

# ---------------------------------------------------------------------------
# Publication defaults (mirror denmark_old)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6,
    # One source of truth per text role, so sizes cannot drift between panels:
    # axis labels 6, tick labels 5, legends 5. Set here rather than at each
    # call site — the three legends had drifted to 6 / 5 / 5 that way, and the
    # 7 pt axes.labelsize was dead, overridden to 6 at every use.
    'axes.labelsize': 6,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
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

# Okabe-Ito, shared verbatim with fig4.py so a region keeps one colour across
# the paper — the two figures previously disagreed, with green meaning
# Nordjylland here and Syddanmark there. Ordered by population to match
# REGIONS_BY_POP and the violin x-axis, which also keeps green (#009E73) and
# reddish-purple (#CC79A7) — the weakest deuteranope pair — non-adjacent in
# the legend.
DIVISION_COLORS = {
    'Hovedstaden': '#0072B2',
    'Midtjylland': '#E69F00',
    'Syddanmark':  '#009E73',
    'Sjaelland':   '#D55E00',
    'Nordjylland': '#CC79A7',
}
DEFAULT_COLOR = '#AAAAAA'
BRANCH_COLOR = '#888888'   # tree branches: neutral grey; tips keep division color

# ColorBrewer RdBu stops, but with the three near-white steps pulled in tight
# around the midpoint instead of sitting at even sixths. Evenly spaced they
# gave every |SSS| below ~0.3 an almost white fill, which is most of the Alpha
# and Delta rows; compressing the neutral band lets those rows show their sign
# while 0 still reads as "neither source nor sink". Positions stay mirrored
# about 0.5 so the two arms keep equal steps.
SSS_CMAP = LinearSegmentedColormap.from_list('sss_diverging', list(zip(
    [0.0, 0.25, 0.42, 0.5, 0.58, 0.75, 1.0],
    ['#2166AC', '#67A9CF', '#D1E5F0', '#FAFAFA',
     '#FDDBC7', '#EF8A62', '#B2182B'])))
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
# One fixed range per metric, shared by that metric's choropleth and its
# violins, so a map colour and a violin height mean the same number. Round
# endpoints are chosen over data-derived ones because they read cleanly on the
# colorbar and stay put as predictions are regenerated. SSS's is its full
# definitional range; R_0's covers the bootstrap spread with headroom.
SSS_LIM = (-1.0, 1.0)
R0_LIM = (0.8, 3.0)

# Hue is reserved for region across the whole figure (panels a, l, m), so the
# two model colours must avoid DIVISION_COLORS and the two colormap ramps.
# #7A0177 is the highest-chroma colour satisfying both: it keeps the region
# palette's own worst-pair baseline under the dataviz validator (--pairs all),
# where bright magentas collapse onto Nordjylland's pink under deuteranopia and
# crimson lands ΔE 3.5 from the SSS ramp's red end — indistinguishable from the
# map drawn beside it. Only a *dark* neutral pairs with it; light greys collapse
# onto the same pink. Rejected candidates are catalogued in tasks/next.md.
C_STEPHY = '#7A0177'
C_CNN = '#4D4D4D'
VIOLIN_WIDTH = 0.32
LEGEND_LOC = 'upper right'   # same corner in both violin columns; see add_model_legend

# Weight as well as hue separates the two, so the paper's result reads first:
# STEPHY near-opaque with black statistics and a large star, the ablation faint
# with grey ones and a small diamond, sitting behind it as a reference.
MODEL_STYLES = {
    'STEPHY':   dict(offset=-0.18, color=C_STEPHY, alpha=0.70, stat='#222222',
                     ml_marker='*', ml_size=22, ml_edge='black'),
    'CBLV-CNN': dict(offset=+0.18, color=C_CNN, alpha=0.28, stat='#909090',
                     ml_marker='D', ml_size=6, ml_edge='#6E6E6E'),
}

# Marker shape carries clade in the scatter panels, where fill carries region.
# Same assignment as the case study's compare figure.
CLADE_MARKERS = {'20I': 'o', '21I': 's', '21J': '^', '21K': 'D', '21L': 'v'}

FIG_WIDTH = 3.39
FIG_RATIO = 2
# Height of the scatter row, in inches, added below the 5x5 grid. The grid's
# own geometry is computed from FIG_WIDTH/FIG_RATIO alone, so growing this
# lengthens the figure without touching its width.
SCATTER_H = 3.05
# Clearance between the 5x5 grid and the scatter row, in inches. It has to
# hold the bottom violins' rotated region labels *and* the two colorbars.
GAP_H = 1.0
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


def load_clade_predictions(pred_dir, clade):
    """Return ML + bootstrap distributions for SSS and R0 from regression.tsv.

    `pred_dir` is a directory of per-clade predictions — either STEPHY's
    stephy_output/ or the CBLV-CNN ablation's cblv-cnn_output/. Both are
    written by the same inference script with the same columns, so one reader
    serves both and the two models cannot diverge through their loading code.

    Returns:
        sss_ml, r0_ml:    {region: float}                — ML point estimates.
        sss_boot, r0_boot:{region: 1D np.ndarray}        — per-replicate values.
    """
    reg_path = pred_dir / clade / 'regression.tsv'
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


def load_paired(stephy_dir, cnn_dir, clade, columns):
    """Both models' predictions for one clade, aligned replicate by replicate.

    The scatter panels put one point per (region, replicate) at the two models'
    predictions *for the same resampled tree*, so the pairing has to be by
    replicate id. Row order happens to agree between the two files today, but
    relying on it would silently scramble the pairing if either were ever
    regenerated in a different order — and a scrambled cloud looks entirely
    plausible. So the join is explicit and the replicate sets are checked.

    Both metrics are extracted from one pass, since they share the file.

    Returns {column: {region: (ml_s, ml_c, boot_s, boot_c)}}.
    """
    def frame(d):
        return pd.read_csv(d / clade / 'regression.tsv', sep='\t')

    fs, fc = frame(stephy_dir), frame(cnn_dir)
    out = {col: {} for col in columns}
    for region in REGIONS_BY_POP:
        def split(df, col):
            m = df['location_name'] == region
            rep = df[m & df['source'].astype(str).str.startswith('replicate_')]
            ml = df[m & (df['source'] == 'ml')]
            return ml[col].iloc[0], rep.set_index('source')[col]

        for col in columns:
            ml_s, rep_s = split(fs, col)
            ml_c, rep_c = split(fc, col)
            if set(rep_s.index) != set(rep_c.index):
                only = sorted(set(rep_s.index) ^ set(rep_c.index))
                sys.exit(f"{clade}/{region}: the two models cover different "
                         f"replicates ({len(rep_s)} vs {len(rep_c)}); "
                         f"not in both: {only[:5]}")
            idx = sorted(rep_s.index)
            out[col][region] = (ml_s, ml_c,
                                rep_s.loc[idx].to_numpy(),
                                rep_c.loc[idx].to_numpy())
    return out


def observed_range(*model_data, ml_key, boot_key):
    """Return (min, max) over every ML point and bootstrap replicate given.

    The axis ranges are fixed round numbers (SSS_LIM, R0_LIM), so nothing here
    sets them; this exists so each run prints what the data actually spans
    beside what the axis shows. A metric whose values drift outside its fixed
    range would otherwise only surface as a clipped violin.
    """
    values = []
    for data in model_data:
        for d in data.values():
            values.extend(d[ml_key].values())
            for replicates in d[boot_key].values():
                values.extend(replicates.tolist())
    return min(values), max(values)


def report_bootstrap_coverage(clade_data, labels, metrics):
    """Print how often each ML estimate falls inside its bootstrap spread.

    One cell is one (clade, region, metric). A cell counts as covered when the
    ML point lies inside the replicates' central 95% interval, and separately
    when it lies anywhere inside their full range. Both are drawn — the CI bar
    and the ML star — so any single cell is checkable by eye; this table is the
    whole grid at once, written to fig5.out so the figure and the number quoted
    alongside it cannot drift apart.
    """
    print('\nML estimate vs bootstrap spread (cells inside 95% CI / full range)')
    totals = {name: [0, 0, 0] for _, _, name in metrics}
    for label in labels:
        d = clade_data[label]
        row = []
        for ml_key, boot_key, name in metrics:
            in_ci = in_range = n = 0
            for region in REGIONS_BY_POP:
                ml, replicates = d[ml_key][region], d[boot_key][region]
                lo, hi = np.percentile(replicates, [2.5, 97.5])
                n += 1
                in_ci += lo <= ml <= hi
                in_range += replicates.min() <= ml <= replicates.max()
            row.append(f'{name} {in_ci}/{n} · {in_range}/{n}')
            totals[name][0] += in_ci
            totals[name][1] += in_range
            totals[name][2] += n
        print(f'  {label:<12} ' + '   '.join(row))

    overall = [sum(t[i] for t in totals.values()) for i in range(3)]
    for name, (ci, rng, n) in totals.items():
        print(f'  {"total " + name:<12} {ci}/{n} ({ci / n:.0%}) · '
              f'{rng}/{n} ({rng / n:.0%})')
    print(f'  {"all cells":<12} {overall[0]}/{overall[2]} '
          f'({overall[0] / overall[2]:.0%}) · {overall[1]}/{overall[2]} '
          f'({overall[1] / overall[2]:.0%})')


def report_clipped(records):
    """Print anything a violin still draws outside its axis. Should be empty.

    Bodies are trimmed to their 95% interval, which brings every drawn element
    inside the fixed axes, so this is a regression check rather than a routine
    report: if it ever prints, the figure is silently cutting something off.
    """
    if not records:
        print('\nNothing drawn outside the violin axes.')
        return
    print('\n*** Drawn outside the violin axes — the panel is cutting these '
          'off silently:')
    for metric, clade, model, region, values in records:
        shown = ', '.join(f'{v:.3f}' for v in np.sort(values))
        print(f'  {metric:<4} {clade:<12} {model:<9} {region:<12} {shown}')


def report_out_of_range(models, ml_key, boot_key, metric, valid):
    """Print predictions outside a metric's definitional range, per model.

    SSS is (exports - imports) / (exports + imports), so no true value can
    leave [-1, 1] — but the regression head is an unbounded nn.Linear, so a
    prediction can. Trimming the violins to 95% keeps such a value from
    distorting the panel, which also means the figure no longer shows it: this
    check is what keeps it on the record. It reads the full prediction set, not
    what was drawn, so it is unaffected by any plotting choice.
    """
    print(f'\n{metric} predictions outside its definitional range {valid}:')
    for name, data in models:
        offenders = []
        for clade, d in data.items():
            for region in REGIONS_BY_POP:
                for v in np.append(d[boot_key][region], d[ml_key][region]):
                    if v < valid[0] or v > valid[1]:
                        offenders.append((clade, region, v))
        total = sum(len(d[boot_key][r]) + 1
                    for d in data.values() for r in REGIONS_BY_POP)
        print(f'  {name:<9} {len(offenders)}/{total}')
        for clade, region, v in offenders:
            print(f'      {clade} / {region}: {v:.4f}  (not drawn anywhere: '
                  'outside the violins\' 95% trim and the scatter axis)')


def load_regions(geojson_path):
    """Load Denmark regions GeoJSON, clipped to mainland (Bornholm excluded).

    The boundary file spells one region with a Danish letter (Sjælland) where
    the rest of the pipeline uses ASCII (Sjaelland), so names are normalised
    through REGION_NAME_MAP. An unmatched name would otherwise survive to the
    choropleth join and silently render that region blank, so the region set is
    checked against the pipeline's here instead.
    """
    gdf = gpd.read_file(geojson_path)
    gdf['region'] = gdf['NAME_1'].replace(REGION_NAME_MAP)

    expected, found = set(REGIONS_BY_POP), set(gdf['region'])
    if expected != found:
        sys.exit(
            f"Region names in {geojson_path} do not match the pipeline's.\n"
            f"  missing   : {sorted(expected - found) or 'none'}\n"
            f"  unexpected: {sorted(found - expected) or 'none'}\n"
            "Add the needed entries to REGION_NAME_MAP. Boundary files vary in "
            "how they spell Danish letters and whether they prefix names with "
            "'Region'.")

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


def draw_violin(ax, models, ylim, ylabel,
                regions=REGIONS_BY_POP, show_xticks=False):
    """Per-clade bootstrap violins for one metric, one violin per model.

    `models` is a list of (name, ml_values, boot_values). Each region's tick
    carries a pair of translucent violins, STEPHY offset left and the CBLV-CNN
    ablation right, styled per MODEL_STYLES. This mirrors panels c and d of the
    case study's compare_stephy_vs_cnn figure, so the paper's two
    model-comparison panels use one visual language.

    Regions run left -> right by population (Hovedstaden -> Nordjylland).
    `ylim` is the metric's fixed shared scale (SSS_LIM / R0_LIM).

    **Each body is trimmed to the central 95% of its replicates**, so a violin
    spans exactly the interval its black bar marks. The density itself is still
    estimated from all 40 replicates — only the drawn extent is bounded — so
    the shape inside the interval is unaffected by the trim. Trimming is what
    keeps a single wild replicate from setting a violin's height: the CBLV-CNN
    ablation predicts SSS = 1.312 for one 21L / Hovedstaden tree, which is
    outside SSS's definitional [-1, 1] and 4.3 sd above that cell's other
    replicates. Such values are excluded from the figure and reported instead;
    see report_out_of_range.

    Returns the (model, region, values) records whose *drawn* elements still
    fall outside `ylim` — a safety net that should stay empty now that bodies
    are trimmed, and which prints rather than silently clipping if it does not.

    A region absent from either dict would otherwise be skipped silently,
    leaving a gap that reads as "no data" rather than as an error, so both are
    checked against `regions` first. Every (clade, source) in either output
    covers all five regions, so a gap always means a bug.
    """
    positions = np.arange(len(regions))
    offscale = []

    for name, ml_values, boot_values in models:
        style = MODEL_STYLES[name]
        missing_boot = [r for r in regions if r not in boot_values]
        missing_ml = [r for r in regions if r not in ml_values]
        if missing_boot or missing_ml:
            sys.exit(
                f"{ylabel} / {name}: incomplete region coverage.\n"
                f"  no bootstrap values : {missing_boot or 'none'}\n"
                f"  no ML value         : {missing_ml or 'none'}\n"
                f"  available           : "
                f"{sorted(set(boot_values) | set(ml_values))}\n"
                "Left unfixed these regions would be omitted from the panel "
                "without warning.")

        for i, region in enumerate(regions):
            d = boot_values[region]
            ml = ml_values[region]
            pos = i + style['offset']

            lo, hi = np.percentile(d, [2.5, 97.5])

            parts = ax.violinplot([d], positions=[pos], widths=VIOLIN_WIDTH,
                                  showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                # Clamp the body's outline to the 95% interval. The KDE behind
                # it still saw every replicate, so this bounds the drawn extent
                # without reshaping the density; the ends square off flush with
                # the interval bar drawn over them.
                verts = pc.get_paths()[0].vertices
                verts[:, 1] = np.clip(verts[:, 1], lo, hi)
                pc.set_facecolor(style['color'])
                pc.set_edgecolor(style['color'])
                pc.set_linewidth(0.5)
                pc.set_alpha(style['alpha'])

            ax.vlines(pos, lo, hi, color=style['stat'], linewidth=0.6, zorder=3)
            ax.hlines(np.median(d), pos - 0.12, pos + 0.12,
                      color=style['stat'], linewidth=0.8, zorder=4)
            ax.scatter([pos], [ml], marker=style['ml_marker'],
                       s=style['ml_size'], color=style['color'],
                       edgecolors=style['ml_edge'], linewidths=0.3, zorder=5)

            drawn = np.array([lo, hi, ml])
            outside = drawn[(drawn < ylim[0]) | (drawn > ylim[1])]
            if outside.size:
                offscale.append((name, region, outside))

    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(positions)
    if show_xticks:
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=5)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='y', labelsize=5)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    return offscale


def draw_choropleth(ax, gdf, values, label, norm, cmap):
    """Draw a Denmark choropleth map for one clade (no ancestor highlight)."""
    merged = gdf.copy()
    merged['val'] = merged['region'].map(values)
    if merged['val'].isna().any():
        blank = sorted(merged.loc[merged['val'].isna(), 'region'])
        sys.exit(f"No {label} value for region(s): {blank}. "
                 f"Available: {sorted(values)}. Left unfixed these regions "
                 "would be drawn blank with no warning.")
    merged.plot(ax=ax, column='val', cmap=cmap, norm=norm,
                edgecolor='#333333', linewidth=0.4)

    ax.set_xlim(7.9, 12.7)
    ax.set_ylim(54.45, 57.85)
    ax.set_aspect(1 / np.cos(np.radians(56)))
    ax.axis('off')
    ax.set_title(label, fontsize=7, fontweight='bold',
                 color='#333333', pad=2)


def draw_scatter(ax, paired, lim, name):
    """One metric, STEPHY on x against the CBLV-CNN ablation on y.

    Every (clade, region) cell contributes one filled ML marker and a cloud of
    40 hollow points, each positioned by the two models' predictions for the
    *same* resampled tree. The cloud therefore shows the joint distribution
    rather than two marginal intervals: elongated along the identity line means
    the models move together across replicates, elongated across it means they
    do not — a distinction the violins in columns 2 and 4 cannot make, because
    they show each model's spread separately.

    Fill is region and shape is clade, matching the tree legend in panel a, so
    hue means the same thing here as everywhere else in the figure.

    `lim` is the metric's shared scale, reused from the maps and violins rather
    than fitted to the cloud. For SSS that bounds the panel at the metric's own
    [-1, 1] definition, which excludes the ablation's single impossible 1.312
    replicate; report_out_of_range keeps it on the record.
    """
    xs, ys = [], []
    for clade, cells in paired.items():
        for region, (ml_s, ml_c, boot_s, boot_c) in cells.items():
            # Hollow, because 1,000 filled markers this size merge into blobs;
            # they still carry the clade shape so a cloud can be traced back.
            ax.scatter(boot_s, boot_c, marker=CLADE_MARKERS[clade], s=4,
                       facecolor='none', edgecolor=DIVISION_COLORS[region],
                       alpha=0.55, linewidths=0.25, zorder=1)
            ax.scatter([ml_s], [ml_c], marker=CLADE_MARKERS[clade], s=11,
                       facecolor=DIVISION_COLORS[region], edgecolor='#222222',
                       linewidths=0.3, zorder=4)
            xs.append(ml_s)
            ys.append(ml_c)

    xs, ys = np.array(xs), np.array(ys)
    ax.plot(lim, lim, ls='--', lw=0.5, color='#444444', zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')

    # r / MAE / bias describe the 25 ML markers only, never the cloud; the
    # replicate share is the one line that uses all 1,000 pairs. Both are
    # computed over every value, including any the axis excludes.
    #
    # Bottom right, because that corner is empty in both panels — measured, not
    # assumed: no point of either metric falls in the block's footprint there,
    # where the top-left corner it used to occupy overlaps 20 points on SSS.
    above = np.mean([c > s for cells in paired.values()
                     for _, _, bs, bc in cells.values()
                     for s, c in zip(bs, bc)])
    ax.text(0.97, 0.03,
            f'r = {np.corrcoef(xs, ys)[0, 1]:.2f}\n'
            f'MAE = {np.abs(xs - ys).mean():.3f}\n'
            f'bias = {np.mean(ys - xs):+.3f}\n'
            f'replicates above y=x: {above:.0%}',
            transform=ax.transAxes, va='bottom', ha='right', ma='left',
            fontsize=6, color='#333333', linespacing=1.4)
    ax.set_xlabel(f'{name} — STEPHY')
    ax.set_ylabel(f'{name} — CBLV-CNN')
    ax.tick_params(labelsize=5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def add_model_legend(ax):
    """Key the two violin series on the top panel of a violin column.

    Each column gets its own copy rather than sharing one, so neither the SSS
    nor the R_0 block depends on the reader having looked at the other, and
    both sit in the same corner of the same row so the pair reads as one
    decision rather than two.

    LEGEND_LOC is upper right because that is the only corner free in both
    columns at the top row: measured against the data on Alpha 20I, the right
    half leaves 22% of the SSS panel and 79% of the R_0 panel clear, where
    upper left leaves only 13% on SSS and the lower corners leave 3-5% on R_0.

    The handles carry both encodings at once — fill colour and ML marker — and
    the labels stay bare model names: spelling out "left/right of each pair",
    as the case-study comparison figure does, overflows a panel this narrow and
    pushes the bbox-trimmed figure past Nature's 183 mm width.
    """
    ax.legend(
        handles=[Line2D([0], [0], marker=s['ml_marker'], color='w',
                        markerfacecolor=s['color'],
                        markeredgecolor=s['ml_edge'], markeredgewidth=0.3,
                        markersize=6 if s['ml_marker'] == '*' else 3.5,
                        label=name)
                 for name, s in MODEL_STYLES.items()],
        loc=LEGEND_LOC, frameon=False, handlelength=1.0,
        borderpad=0.1, labelspacing=0.25, handletextpad=0.4,
        borderaxespad=0.2)


def add_colorbar(fig, map_ax, y, cmap, norm, lim, label):
    """Draw a metric's horizontal colorbar under its map column.

    The bar is inset to the map column's own width rather than spanning the
    map and violin together, so it reads as belonging to the choropleths.
    Three ticks — the two ends and the midpoint of the shared scale — is as
    many as fit at this width without the labels colliding.
    """
    pos = map_ax.get_position()
    cax = fig.add_axes([pos.x0 + 0.005, y, pos.width - 0.01, 0.008])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                        cax=cax, orientation='horizontal')
    cbar.set_label(label)
    cbar.set_ticks(np.round(np.linspace(lim[0], lim[1], 3), 1))
    cbar.ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build fig5.pdf — 5-clade trees + ML SSS / R_0 choropleths + bootstrap violins.

    Loads each clade's ML tree (decorated with division + numdate from the
    parent lineage's augur outputs), reads ML and bootstrap predictions from
    stephy_output/<clade>/regression.tsv and the CBLV-CNN ablation's
    cblv-cnn_output/<clade>/regression.tsv, then assembles a 5x5 panel grid
    with one row per clade and shared colorbars under the map columns. The
    choropleths stay STEPHY-only; the violins carry both models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', type=str,
        default=under('denmark', 'nextstrain', 'bootstrap_uncertainty'),
        help='Root containing <Lineage>/ml_point_estimate/, stephy_input/, '
             'stephy_output/.')
    parser.add_argument(
        '--cnn_dir', type=str,
        default=under('denmark', 'nextstrain', 'bootstrap_uncertainty',
                      'cblv-cnn_output'),
        help='Per-clade predictions from the CBLV-CNN ablation, drawn as the '
             'right half of each violin. Same layout as stephy_output/.')
    parser.add_argument(
        '--geojson', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'gadm41_DNK_1.json'),
        help='GADM level-1 boundaries for Denmark. Not redistributed here; '
             'download gadm41_DNK_1.json from https://gadm.org/download_country.html '
             'and place it beside this script, or pass an explicit path.')
    args = parser.parse_args()

    if not os.path.exists(args.geojson):
        sys.exit(f"GADM boundaries not found: {args.geojson}\n"
                 "Download gadm41_DNK_1.json (Denmark, level 1) from "
                 "https://gadm.org/download_country.html and place it beside "
                 "this script, or pass --geojson explicitly. GADM data is not "
                 "redistributable, so it is not bundled with this repository.")
    if not os.path.isdir(args.base_dir):
        sys.exit(f"Case-study data not found: {args.base_dir}\n"
                 "Set DENMARK_CASE to the Denmark data tree, or pass "
                 "--base_dir explicitly.")
    if not os.path.isdir(args.cnn_dir):
        sys.exit(f"CBLV-CNN predictions not found: {args.cnn_dir}\n"
                 "Every violin draws both models, so this is required. "
                 "Regenerate it with the case study's "
                 "07_run_stephy.py --family cblv-cnn, or pass --cnn_dir "
                 "explicitly.")

    base_dir = Path(args.base_dir)
    out_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    log = open(out_dir / 'fig5.out', 'w')
    sys.stdout = _Tee(sys.__stdout__, log)

    # Validate the boundary file before the expensive tree loading, so a
    # region-name mismatch surfaces in a second rather than after five trees.
    gdf = load_regions(args.geojson)

    # --- Load trees ---
    loaded = []
    for clade in CLADES:
        print(f'Loading {clade}...', end=' ', flush=True)
        tree = load_clade_tree(base_dir, clade)
        n_tips = len(tree.getExternal())
        sizes = compute_subtree_sizes(tree)
        print(f'{n_tips} tips')
        loaded.append((tree, CLADE_LABELS[clade], n_tips, sizes, clade))

    # Row order top-to-bottom in every column is clade name (20I, 21I, 21J,
    # 21K, 21L), i.e. CLADES as declared. An earlier version sorted by root
    # date instead, which put 21J above 21I because Delta's two clades root a
    # few days apart — a distinction no reader is tracking, and one that made
    # the panel letters run out of clade order. The trees are drawn in reverse,
    # because they stack upward from y=0; their true root dates still set their
    # x positions, so nothing about the timeline changes.
    loaded.sort(key=lambda t: CLADES.index(t[4]))

    # --- Load per-clade ML + bootstrap predictions, per model ---
    def load_model(pred_dir):
        out = {}
        for clade in CLADES:
            sss_ml, r0_ml, sss_boot, r0_boot = load_clade_predictions(
                pred_dir, clade)
            out[CLADE_LABELS[clade]] = {
                'sss_ml': sss_ml, 'r0_ml': r0_ml,
                'sss_boot': sss_boot, 'r0_boot': r0_boot,
            }
        return out

    clade_data = load_model(base_dir / 'stephy_output')
    cnn_data = load_model(Path(args.cnn_dir))
    sss_data = {lbl: d['sss_ml'] for lbl, d in clade_data.items()}
    r0_data = {lbl: d['r0_ml'] for lbl, d in clade_data.items()}

    sss_lim, r0_lim = SSS_LIM, R0_LIM
    sss_norm = TwoSlopeNorm(vcenter=0, vmin=sss_lim[0], vmax=sss_lim[1])
    r0_norm = plt.Normalize(vmin=r0_lim[0], vmax=r0_lim[1])
    print('\nShared scales (axis vs the data it has to hold)')
    for name, lim, ml_key, boot_key in (('SSS', sss_lim, 'sss_ml', 'sss_boot'),
                                        ('R_0', r0_lim, 'r0_ml', 'r0_boot')):
        lo, hi = observed_range(clade_data, cnn_data,
                                ml_key=ml_key, boot_key=boot_key)
        print(f'  {name:<4} axis [{lim[0]:.2f}, {lim[1]:.2f}]   '
              f'both models span [{lo:.3f}, {hi:.3f}]')

    report_bootstrap_coverage(
        clade_data, [lbl for _, lbl, _, _, _ in loaded],
        [('sss_ml', 'sss_boot', 'SSS'), ('r0_ml', 'r0_boot', 'R_0')])

    # --- Figure layout ---------------------------------------------------
    # Grid block: tree | SSS map | SSS violin | R0 map | R0 violin.
    # Scatter block below it: SSS scatter | R_0 scatter, halving the width.
    #
    # The grid's geometry still derives from FIG_WIDTH/FIG_RATIO alone, so the
    # scatter row lengthens the figure without widening it — the width is at
    # Nature's 183 mm cap and has to stay there.
    grid_height = FIG_WIDTH * FIG_RATIO
    map_width = grid_height / 5
    violin_width = map_width
    widths = [FIG_WIDTH, map_width, violin_width, map_width, violin_width]
    fig = plt.figure(
        figsize=(sum(widths), grid_height + GAP_H + SCATTER_H), facecolor='w')
    # Two stacked blocks rather than one 6-row grid: hspace is uniform within a
    # GridSpec, so a 6th row could not be given more clearance than the five
    # above it — and it needs much more, because row f/k's rotated region
    # labels hang below the grid and the colorbars sit under that again.
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[grid_height, SCATTER_H],
        hspace=2 * GAP_H / (grid_height + SCATTER_H))
    gs = gridspec.GridSpecFromSubplotSpec(
        5, 5, subplot_spec=outer[0], width_ratios=widths,
        wspace=0.40, hspace=0.18)
    # The scatter row ignores the grid's column widths and simply halves the
    # figure: l and m are square by construction (equal aspect on a shared
    # scale), so giving each half the width is what makes them as large as the
    # page allows. Their clade key goes under the row instead of beside it.
    gs_scatter = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], wspace=0.28)

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
    for tree, label, n_tips, sizes, clade in reversed(loaded):
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
    ax_tree.grid(axis='x', alpha=0.3, linewidth=0.3)
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
        for i, (_, label, _, _, _) in enumerate(loaded):
            ax = fig.add_subplot(gs[i, col])
            draw_choropleth(ax, gdf, data[label], label, norm, cmap)
            ax.text(-0.08, 1.05, panel_letters[i], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='right')
        last_ax[col] = ax

    # --- SSS / R0 bootstrap violin columns (2, 4) — share panel letter with
    #     the adjacent map (map+violin = one panel) ---
    violin_configs = [
        (2, 'sss_ml', 'sss_boot', sss_lim, 'SSS', 'SSS'),
        (4, 'r0_ml',  'r0_boot',  r0_lim,  r'$R_0$', 'R_0'),
    ]
    clipped = []
    for col, ml_key, boot_key, ylim, ylabel, metric in violin_configs:
        for i, (_, label, _, _, _) in enumerate(loaded):
            ax = fig.add_subplot(gs[i, col])
            models = [(name, d[label][ml_key], d[label][boot_key])
                      for name, d in (('STEPHY', clade_data),
                                      ('CBLV-CNN', cnn_data))]
            offscale = draw_violin(ax, models, ylim, ylabel,
                                   show_xticks=(i == len(loaded) - 1))
            clipped += [(metric, label, *rec) for rec in offscale]
            if i == 0:
                add_model_legend(ax)

    report_clipped(clipped)
    report_out_of_range([('STEPHY', clade_data), ('CBLV-CNN', cnn_data)],
                        'sss_ml', 'sss_boot', 'SSS', (-1.0, 1.0))

    # --- Scatter row: STEPHY vs the ablation, one per metric (panels l, m) ---
    paired = {clade: load_paired(base_dir / 'stephy_output', Path(args.cnn_dir),
                                 clade, ('sss_point', 'r0_point'))
              for clade in CLADES}
    ax_scatter = None
    for slot, column, lim, name, letter in (
            (0, 'sss_point', sss_lim, 'SSS', 'l'),
            (1, 'r0_point', r0_lim, r'$R_0$', 'm')):
        ax = fig.add_subplot(gs_scatter[0, slot])
        draw_scatter(ax, {c: paired[c][column] for c in CLADES}, lim, name)
        ax.text(-0.15, 1.0, letter, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='right')
        ax_scatter = ax_scatter or ax

    # --- Colorbars: width = map column only; fewer ticks for breathing room ---
    # Seated near the top of the gap between the grid and the scatter row. It
    # can sit high because it spans a map column, while the rotated region
    # labels that hang into this gap belong to the violin columns beside it —
    # they never share an x range. Low placement instead collides with the
    # scatter panels, whose inset statistics start at their top-left.
    fig.canvas.draw()
    grid_bottom = ax_tree.get_position().y0
    scatter_top = ax_scatter.get_position().y1
    cbar_y = grid_bottom - 0.28 * (grid_bottom - scatter_top)
    add_colorbar(fig, last_ax[1], cbar_y, SSS_CMAP, sss_norm, sss_lim, 'SSS')
    add_colorbar(fig, last_ax[3], cbar_y, R0_CMAP, r0_norm, r0_lim, r'$R_0$')

    # Clade-shape key for l and m, in one row under both panels so neither
    # loses width to it. Anchored below the scatter axes with room for their
    # tick labels and x-label; fill colour is not repeated here, because it is
    # region and panel a already keys it.
    fig.legend(
        handles=[Line2D([0], [0], marker=CLADE_MARKERS[c], color='w',
                        markerfacecolor='#BBBBBB', markeredgecolor='#333333',
                        markeredgewidth=0.3, markersize=4,
                        label=CLADE_LABELS[c]) for c in CLADES],
        loc='upper center',
        bbox_to_anchor=(0.5, ax_scatter.get_position().y0 - 0.038),
        ncol=len(CLADES), frameon=False,
        handletextpad=0.4, columnspacing=1.6)

    out = out_dir / 'fig5.pdf'
    out_png = out_dir / 'fig5.png'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'Saved: {out}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
