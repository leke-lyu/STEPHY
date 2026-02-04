#!/usr/bin/env python3
"""
Configuration for STEPHY+ (Phylogeny + Epidemiological Data Model).

Uses phylogenetic (CBLV) and epidemiological features for spatial transmission estimation.

Output files:
- training_history.csv: Loss in NORMALIZED scale (what optimizer sees)
- test_predictions.csv: Predictions in TRUE scale (for interpretation)
- Test R2/MSE printed: TRUE scale
"""

# Model architecture
MODEL_ARGS = {
    # Note: 'subtree_width' is injected from CLI arguments in train.py

    # CNN encoder branches for CBLV (phylogenetic) features
    # Output: 48 + 24 + 24 = 96 per node
    'phy_channel_plain': [12, 24, 48],    # 3 layers, ends at 48
    'phy_channel_stride': [12, 24],        # 2 layers, ends at 24
    'phy_channel_dilate': [12, 24],        # 2 layers, ends at 24

    'phy_kernel_plain': [3, 5, 7],
    'phy_kernel_stride': [7, 9],
    'phy_kernel_dilate': [3, 5],

    'phy_stride_stride': [3, 6],
    'phy_dilate_dilate': [3, 5],

    # Epidemiological branch
    # Dense: 4 -> 64 -> 32
    'epi_dim': 4,                # 4 epi features
    'epi_channel': [64, 32],     # Dense: 4 -> 64 -> 32

    # GAT layer
    'edge_dim': 3,      # DTW features: distance, lag_mean, lag_std
    'attn_dim': 16,     # Attention hidden dimension

    # Classifier: 256 -> 128 -> 64 -> 32 -> 1
    # Input is 256-dim: (96+32)*2 = 128*2 (self + neighbor_agg)
    'lbl_channel': [128, 64, 32],

    # Activation
    'activation_func': 'relu',
}

# Training parameters
TRAIN_ARGS = {
    'learning_rate': 0.001,
    'batch_size': 64,       # Graph-level batching (64 graphs per batch)
    'num_epochs': 500,
    'early_stopping_patience': 25,

    # Data split (remaining 40% is split 50/50 into val/test)
    'train_ratio': 0.6,
    # Random seed
    'random_seed': 42,
}

# Data processing parameters
DATA_ARGS = {
    'dtw_num_points': 200,  # KDE grid resolution for DTW curves
    # Normalization options: 'zscore' or 'none'
    'edge_norm': 'zscore',  # Edge feature normalization
    'label_norm': 'zscore', # Label normalization (regression only; ignored for classification)
    'epi_log': True,        # Log transform epi features before normalization
    'epi_norm': 'zscore',   # Z-score normalize epi features
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
