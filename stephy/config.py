#!/usr/bin/env python3
"""
Configuration for STEPHY (Phylogeny-only Model).

Uses phylogenetic (CBLV) + auxiliary tree statistics for spatial transmission estimation.
Node embedding: 96-dim CNN (48+24+24) + 32-dim aux branch = 128-dim.

Output files:
- training_history.csv: Loss in NORMALIZED scale (what optimizer sees)
- test_predictions.csv: Predictions in TRUE scale (for interpretation)
- Test R2/MSE printed: TRUE scale
"""

# Model architecture
#
# Design rationale:
# - The 96-dim CNN output (48+24+24) balances capacity across three parallel
#   branches that capture different scales of phylogenetic structure: local
#   (plain), hierarchical (stride), and long-range (dilate).
# - The 32-dim aux branch is intentionally smaller since it encodes only 5
#   scalar statistics; this prevents the aux signal from dominating the
#   richer CBLV representation.
# - Together, CNN (96) + aux (32) = 128-dim node embedding. After GAT
#   concatenation of self + aggregated neighbor, the 256-dim representation
#   is halved through a 3-layer classifier (128->64->32->1) to smoothly
#   compress spatial context into per-node predictions.
# - The attention dimension (16) is kept small because edge features are only
#   3-dimensional (DTW distance, lag mean, lag std); a larger attention MLP
#   would overfit these few features.
MODEL_ARGS = {
    # Note: 'subtree_width' is injected from CLI arguments in train.py

    # CNN encoder branches for CBLV (phylogenetic) features
    # Three branches capture complementary patterns from the CBLV matrix:
    # - Plain: standard convolutions for local feature extraction
    # - Stride: strided convolutions for hierarchical / multi-scale patterns
    # - Dilate: dilated convolutions for long-range dependencies
    # Output: 48 + 24 + 24 = 96 per node (+ 32-dim aux branch = 128 total)
    'phy_channel_plain': [12, 24, 48],    # 3 layers, widest -- captures most detail
    'phy_channel_stride': [12, 24],        # 2 layers, narrower -- coarser hierarchy
    'phy_channel_dilate': [12, 24],        # 2 layers, narrower -- sparse receptive field

    'phy_kernel_plain': [3, 5, 7],         # Increasing kernel sizes for growing receptive field
    'phy_kernel_stride': [7, 9],           # Larger kernels pair with strides for downsampling
    'phy_kernel_dilate': [3, 5],           # Smaller kernels; dilation expands effective field

    'phy_stride_stride': [3, 6],           # Aggressive downsampling to compress temporal axis
    'phy_dilate_dilate': [3, 5],           # Dilation factors widen receptive field without pooling

    # Aux branch: MLP on 5 tree statistics (mrca_depth, earliest/latest tip
    # times, avg branch length, n_tips). Kept small to avoid dominating CBLV.
    'aux_hidden': 64,    # Hidden layer dimension
    'aux_output': 32,    # Output dimension (contributes 32 of the 128-dim node embedding)

    # GAT layer -- edge-attention mechanism using DTW-derived features
    'edge_dim': 3,      # DTW features: distance, lag_mean, lag_std
    'attn_dim': 16,     # Attention hidden dimension (small; only 3 input features)

    # Classifier: 256 -> 128 -> 64 -> 32 -> 1
    # Input is 256-dim: 128 (96 CNN + 32 aux) * 2 (self + neighbor_agg from GAT)
    # Gradual compression avoids information bottleneck
    'lbl_channel': [128, 64, 32],

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
    'conformal_prediction': False,   # Master switch: True enables CQR/RAPS
    'cp_alpha': 0.1,                 # Miscoverage rate (1-alpha = coverage)
    'cqr_quantiles': [0.05, 0.5, 0.95],  # Quantiles for CQR (regression)
    'raps_lambda': 0.01,             # RAPS regularization strength (classification)
    'raps_k_reg': 2,                 # RAPS: penalty starts after k classes
}

# Data processing parameters
DATA_ARGS = {
    'dtw_num_points': 200,  # KDE grid resolution for DTW curves
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
