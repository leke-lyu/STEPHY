#!/usr/bin/env python3
"""
Configuration for the CBLV-GAT baseline (standard GAT, node-based attention).

All model dimensions are driven from this file so that config.py is the single
source of truth for architecture choices.

Architecture rationale -- standard GAT (DGL ``GATConv``):
    * Attention is computed from 128-dim node embeddings only (no edge features).
    * 4 attention heads x 64-dim each = 256-dim output, providing a good
      capacity/speed trade-off for graphs with ~5-20 nodes (locations).
    * Self-loops are added at load time in train.py, not baked into graphs.pt,
      so the same graph files can be shared with stephy.

Output files:
    training_history.csv  -- Loss in NORMALIZED scale (what the optimizer sees)
    test_predictions.csv  -- Predictions in TRUE (original) scale
    Test R2/MSE printed   -- TRUE scale
"""

# Model architecture
MODEL_ARGS = {
    # Note: 'subtree_width' is injected from CLI arguments in train.py

    # ---- CNN encoder branches for CBLV (phylogenetic) features ----
    # Three parallel branches capture different temporal scales of the CBLV
    # matrix.  Output: 48 + 24 + 24 = 96 per node (+ 32-dim aux = 128 total).
    'phy_channel_plain': [12, 24, 48],    # Plain branch: 3 layers, same padding
    'phy_channel_stride': [12, 24],        # Stride branch: 2 layers, hierarchical downsampling
    'phy_channel_dilate': [12, 24],        # Dilate branch: 2 layers, long-range dependencies

    'phy_kernel_plain': [3, 5, 7],
    'phy_kernel_stride': [7, 9],
    'phy_kernel_dilate': [3, 5],

    'phy_stride_stride': [3, 6],
    'phy_dilate_dilate': [3, 5],

    # ---- Aux branch: MLP on 5 tree statistics ----
    'aux_hidden': 64,    # Hidden layer dimension
    'aux_output': 32,    # Output dimension (concat with 96-dim CNN -> 128 total)

    # ---- Standard GAT layer (node-based attention, via DGL GATConv) ----
    # Unlike stephy's custom edge-attention GAT, attention here is computed
    # solely from the 128-dim node embeddings.  4 heads provide diverse
    # attention patterns while keeping the total output at 256-dim.
    'gat_num_heads': 4,    # Number of attention heads
    'gat_out_dim': 64,     # Output dimension per head (4 * 64 = 256 total)

    # ---- Classifier MLP ----
    # Input is gat_num_heads * gat_out_dim = 256-dim
    'lbl_channel': [128, 64, 32],   # 256 -> 128 -> 64 -> 32 -> 1

    # Activation
    'activation_func': 'relu',
}

# Training parameters
TRAIN_ARGS = {
    'learning_rate': 0.001,
    'batch_size': 32,       # Graph-level batching
    'num_epochs': 500,
    'early_stopping_patience': 25,

    # Data split (remaining is split 50/50 into val/test, or 3-way if CP on)
    'train_ratio': 0.8,
    # Random seed
    'random_seed': 42,

    # Conformal prediction
    'conformal_prediction': True,    # Master switch: True enables CQR/RAPS
    'cp_alpha': 0.05,                # Miscoverage rate (1-alpha = coverage)
    'cqr_quantiles': [0.025, 0.5, 0.975],  # Quantiles for CQR (regression)
    'raps_lambda': 0.01,             # RAPS regularization strength (classification)
    'raps_k_reg': 2,                 # RAPS: penalty starts after k classes
}

# Data processing parameters
DATA_ARGS = {
    'cblv_scale': 'tree_height',  # CBLV scaling: 'tree_height' (divide by tree height -> [0,1]) or 'log1p' (log(x+1))
}

def get_config():
    """Get full configuration dictionary.

    Note: 'subtree_width' and 'num_locations' are added by train.py from CLI args.
    """
    return {
        'model': MODEL_ARGS,
        'train': TRAIN_ARGS,
        'data': DATA_ARGS,
    }


if __name__ == '__main__':
    import json
    config = get_config()
    print(json.dumps(config, indent=2))
