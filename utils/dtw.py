#!/usr/bin/env python3
"""Add DTW-based edge features to edge CSV."""

import sys
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import gaussian_kde


def parse_tree(tree_file, tree_idx=0):
    """Extract tree string from BEAST2 .trees file."""
    with open(tree_file) as f:
        trees = re.findall(r'tree STATE_\d+ = (.+?)(?=\ntree |\nEnd;|$)', f.read(), re.DOTALL)
    return trees[tree_idx].strip() if trees else None


def extract_tip_times(tree_string):
    """Extract tip times grouped by location from tree string."""
    tips = defaultdict(list)
    for loc, time in re.findall(r'\d+\[&type="I\{(\d+)\}",samp="sample",time=([\d.]+)\]', tree_string):
        tips[int(loc)].append(float(time))
    return dict(tips)


def tips_to_curves(tips_by_loc, n_points=200):
    """Convert tip times to KDE-smoothed epidemic curves."""
    all_times = [t for times in tips_by_loc.values() for t in times]
    if not all_times:
        return None, {}, 0

    t_min, t_max = min(all_times), max(all_times)
    t_grid = np.linspace(t_min, t_max, n_points)
    dt = (t_max - t_min) / (n_points - 1)

    curves = {}
    for loc, times in tips_by_loc.items():
        curves[loc] = gaussian_kde(times)(t_grid) * len(times) if len(times) >= 2 else np.zeros(n_points)
    return t_grid, curves, dt


def dtw_edge_features(curve_a, curve_b):
    """Compute DTW distance and lag statistics between two curves."""
    n, m = len(curve_a), len(curve_b)
    cost = (curve_a[:, None] - curve_b[None, :]) ** 2

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i-1, j-1] + min(dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1])

    # Backtrack for lags
    lags = []
    i, j = n, m
    while i > 0 and j > 0:
        lags.append(j - i)
        step = np.argmin([dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]])
        if step == 0: i, j = i-1, j-1
        elif step == 1: i -= 1
        else: j -= 1

    lags = np.array(lags)
    return dtw[n, m], lags.mean(), lags.std()


def compute_dtw_matrices(tree_file, tree_idx=0):
    """Compute DTW feature matrices for all location pairs."""
    tree_string = parse_tree(tree_file, tree_idx)
    if not tree_string:
        return None, None, None, []

    tips_by_loc = extract_tip_times(tree_string)
    if not tips_by_loc:
        return None, None, None, []

    _, curves, dt = tips_to_curves(tips_by_loc)
    if not curves:
        return None, None, None, []

    locations = sorted(curves.keys())
    n_locs = len(locations)
    curve_arr = np.array([curves[loc] for loc in locations])

    dist_mat = np.zeros((n_locs, n_locs))
    lag_mat = np.zeros((n_locs, n_locs))
    lag_std_mat = np.zeros((n_locs, n_locs))

    for i in range(n_locs):
        for j in range(n_locs):
            if i != j:
                dist, lag, lag_std = dtw_edge_features(curve_arr[i], curve_arr[j])
                dist_mat[i, j] = dist
                lag_mat[i, j] = lag * dt
                lag_std_mat[i, j] = lag_std * dt

    return dist_mat, lag_mat, lag_std_mat, locations


def main():
    if len(sys.argv) < 3:
        print("Usage: python dtw.py <edge_csv_file> <tree_file>", file=sys.stderr)
        sys.exit(1)

    edge_csv_file, tree_file = sys.argv[1:3]
    edge_df = pd.read_csv(edge_csv_file)

    # Build DTW cache for all states
    dtw_cache = {}
    for state_suffix in edge_df['graph_id'].apply(lambda x: x.rsplit('_', 1)[-1]).unique():
        dist_mat, lag_mat, lag_std_mat, locations = compute_dtw_matrices(tree_file, int(state_suffix))
        if dist_mat is not None:
            dtw_cache[state_suffix] = {
                'matrices': (dist_mat, lag_mat, lag_std_mat),
                'loc_to_idx': {loc: i for i, loc in enumerate(locations)}
            }

    # Add DTW columns
    dtw_distances, dtw_lag_means, dtw_lag_stds = [], [], []

    for _, row in edge_df.iterrows():
        state_suffix = row['graph_id'].rsplit('_', 1)[-1]
        src, dst = int(row['src']), int(row['dst'])

        if src == dst or state_suffix not in dtw_cache:
            dtw_distances.append(np.nan)
            dtw_lag_means.append(np.nan)
            dtw_lag_stds.append(np.nan)
        else:
            cache = dtw_cache[state_suffix]
            loc_to_idx = cache['loc_to_idx']
            if src in loc_to_idx and dst in loc_to_idx:
                i, j = loc_to_idx[src], loc_to_idx[dst]
                dist_mat, lag_mat, lag_std_mat = cache['matrices']
                dtw_distances.append(dist_mat[i, j])
                dtw_lag_means.append(lag_mat[i, j])
                dtw_lag_stds.append(lag_std_mat[i, j])
            else:
                dtw_distances.append(np.nan)
                dtw_lag_means.append(np.nan)
                dtw_lag_stds.append(np.nan)

    edge_df['dtw_distance'] = dtw_distances
    edge_df['dtw_lag_mean'] = dtw_lag_means
    edge_df['dtw_lag_std'] = dtw_lag_stds

    edge_df.to_csv(edge_csv_file, index=False)
    print(f"Added DTW columns to {len(edge_df)} edges -> {edge_csv_file}")


if __name__ == '__main__':
    main()
