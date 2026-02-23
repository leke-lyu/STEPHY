#!/usr/bin/env python3
"""
Configuration for CBLV-GAT2 (Standard GAT with Node-Based Attention).

Uses phylogenetic (CBLV) + auxiliary tree statistics for spatial transmission estimation.
Node embedding: 96-dim CNN (48+24+24) + 32-dim aux branch = 128-dim.
Standard GAT: 4 heads x 64-dim = 256-dim output.

Output files:
- training_history.csv: Loss in NORMALIZED scale (what optimizer sees)
- test_predictions.csv: Predictions in TRUE scale (for interpretation)
- Test R2/MSE printed: TRUE scale
"""

# Model architecture
MODEL_ARGS = {
    # Note: 'subtree_width' is injected from CLI arguments in train.py

    # CNN encoder branches for CBLV (phylogenetic) features
    # Output: 48 + 24 + 24 = 96 per node (+ 32-dim aux branch = 128 total)
    'phy_channel_plain': [12, 24, 48],    # 3 layers, ends at 48
    'phy_channel_stride': [12, 24],        # 2 layers, ends at 24
    'phy_channel_dilate': [12, 24],        # 2 layers, ends at 24

    'phy_kernel_plain': [3, 5, 7],
    'phy_kernel_stride': [7, 9],
    'phy_kernel_dilate': [3, 5],

    'phy_stride_stride': [3, 6],
    'phy_dilate_dilate': [3, 5],

    # Aux branch: MLP on 5 tree statistics
    'aux_hidden': 64,    # Hidden layer dimension
    'aux_output': 32,    # Output dimension

    # Standard GAT layer (node-based attention)
    'gat_num_heads': 4,    # Number of attention heads
    'gat_out_dim': 64,     # Output dimension per head (4 * 64 = 256 total)

    # Classifier: 256 -> 128 -> 64 -> 32 -> 1
    # Input is 256-dim: gat_num_heads * gat_out_dim
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

    # Data split (remaining is split 50/50 into val/test)
    'train_ratio': 0.8,
    # Random seed
    'random_seed': 42,
}

def get_config():
    """Get full configuration dictionary.

    Note: 'subtree_width' and 'num_locations' are added by train.py from CLI args.
    """
    return {
        'model': MODEL_ARGS,
        'train': TRAIN_ARGS,
    }


if __name__ == '__main__':
    import json
    config = get_config()
    print(json.dumps(config, indent=2))
