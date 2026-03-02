#!/usr/bin/env python3
"""
Configuration for CBLV-CNN2 (CNN + Aux Branch baseline, no graph structure).

Uses phylogenetic (CBLV) + auxiliary tree statistics for spatial transmission estimation.
Node embedding: 192-dim CNN (96+48+48) + 64-dim aux branch = 256-dim.
Ablation of stephy2: same encoder, no GAT layer.

Output files:
- training_history.csv: Loss in NORMALIZED scale (what optimizer sees)
- test_predictions.csv: Predictions in TRUE scale (for interpretation)
- Test R2/MSE printed: TRUE scale
"""

# Model architecture
# Wider CNN than stephy2 (192-dim vs stephy2's 96-dim) to maintain a 256-dim
# classifier input without the GAT layer. Aux branch is also doubled (64 vs 32).
MODEL_ARGS = {
    # Note: 'subtree_width' is injected from CLI arguments in train.py

    # CNN encoder branches for CBLV (phylogenetic) features
    # Output: 96 + 48 + 48 = 192 per node (+ 64-dim aux branch = 256 total)
    # (stephy2 uses 48 + 24 + 24 = 96-dim CNN + 32-dim aux, then GAT fills the gap)
    'phy_channel_plain': [24, 48, 96],    # 3 layers, ends at 96
    'phy_channel_stride': [24, 48],        # 2 layers, ends at 48
    'phy_channel_dilate': [24, 48],        # 2 layers, ends at 48

    'phy_kernel_plain': [3, 5, 7],
    'phy_kernel_stride': [7, 9],
    'phy_kernel_dilate': [3, 5],

    'phy_stride_stride': [3, 6],
    'phy_dilate_dilate': [3, 5],

    # Aux branch: MLP on 5 tree statistics
    'aux_hidden': 128,   # Hidden layer dimension
    'aux_output': 64,    # Output dimension

    # Classifier: 256 -> 128 -> 64 -> 32 -> 1
    # Input is 256-dim: 192 CNN + 64 aux (no GAT layer)
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
