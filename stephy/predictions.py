"""Shared helpers for assembling test_predictions.csv across all pipelines.

Keeps metadata handling (batch / sim_id / tree_idx / location_idx / location_name)
and row-count alignment in one place so every pipeline and the batch test
runner produce consistent CSV schemas.
"""

import pandas as pd


def format_graph_id(meta):
    """Human-readable identifier for a graph (for logs and error messages)."""
    batch = meta.get('batch', '')
    return f"{batch}/{meta['sim_id']}_{meta['tree_idx']}" if batch else f"{meta['sim_id']}_{meta['tree_idx']}"


def build_metadata_df(test_graphs, is_classification):
    """Build a DataFrame of identifying columns aligned to test_predictions row order.

    Classification: 1 row per graph with columns [batch, sim_id, tree_idx].
    Regression:     N rows per graph (N = num_locations) with columns
                    [batch, sim_id, tree_idx, location_idx, location_name].
    """
    rows = []
    for g, meta, *_ in test_graphs:
        base = {
            'batch': meta.get('batch', ''),
            'sim_id': meta['sim_id'],
            'tree_idx': meta['tree_idx'],
        }
        if is_classification:
            rows.append(base)
        else:
            locations = g.ndata['location'].tolist()
            for loc_idx, loc_name in enumerate(locations):
                rows.append({**base, 'location_idx': loc_idx, 'location_name': loc_name})

    cols = ['batch', 'sim_id', 'tree_idx']
    if not is_classification:
        cols += ['location_idx', 'location_name']
    return pd.DataFrame(rows, columns=cols)


def attach_metadata(pred_df, test_graphs, is_classification):
    """Prepend metadata columns to ``pred_df``. Row counts must match."""
    meta_df = build_metadata_df(test_graphs, is_classification)
    if len(meta_df) != len(pred_df):
        raise ValueError(
            f"Metadata rows ({len(meta_df)}) != prediction rows ({len(pred_df)}). "
            f"Check that the test DataLoader uses shuffle=False and that the "
            f"CP output shape matches the expected task layout."
        )
    return pd.concat(
        [meta_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
        axis=1,
    )
