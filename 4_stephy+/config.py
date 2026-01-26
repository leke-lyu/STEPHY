#!/usr/bin/env python3
"""
Configuration for STEPHY+ (Phylogeny + Epidemiological Data).

Combines phylogenetic (CBLV) and epidemiological features for spatial transmission estimation.

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
    'phy_channel_plain': [12, 24, 48],    # 3 layers, ends at 48 (was 64)
    'phy_channel_stride': [12, 24],        # 2 layers, ends at 24 (was 32)
    'phy_channel_dilate': [12, 24],        # 2 layers, ends at 24 (was 32)

    'phy_kernel_plain': [3, 5, 7],
    'phy_kernel_stride': [7, 9],
    'phy_kernel_dilate': [3, 5],

    'phy_stride_stride': [3, 6],
    'phy_dilate_dilate': [3, 5],

    # GAT layer
    'edge_dim': 3,      # DTW features: distance, lag_mean, lag_std
    'attn_dim': 16,     # Attention hidden dimension

    # Epi branch: 4 epidemiological features from trajectory data
    # Features: Initial_Population, Epidemic_Peak, Peak_Timing, Accumulated_Infections
    # All features use log(x) transformation before Z-score normalization
    'epi_dim': 4,
    'epi_channel': [64, 32],         # Dense layer sizes -> output 32-dim

    # Classifier: 256 -> 128 -> 64 -> 32 -> num_labels
    # Input is 256-dim: (96 CBLV + 32 epi) * 2 (self + neighbor_agg)
    'lbl_channel': [128, 64, 32],

    # Activation
    'activation_func': 'relu',
}

# Training parameters
TRAIN_ARGS = {
    'learning_rate': 0.001,
    'batch_size': 64,       # Graph-level batching (64 graphs per batch)
    'num_epochs': 300,
    'early_stopping_patience': 15,

    # Data split
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,

    # Random seed
    'random_seed': 42,
}

# Data processing parameters
DATA_ARGS = {
    'dtw_num_points': 200,  # KDE grid resolution for DTW curves
    # Normalization options: 'zscore' or 'none'
    'epi_log': True,        # Apply log to epi features before normalization
    'epi_norm': 'zscore',   # Epi feature normalization
    'edge_norm': 'zscore',  # Edge feature normalization
    'label_norm': 'zscore', # Label normalization
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
