import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Base path
base_path = "/Users/lukelyu/Library/CloudStorage/OneDrive-Emory/emory/phyloGNN/3_phyddle"

# Dataset names
datasets = ["500_1_MM0.002", "500_1_MM0.0025", "500_1_MM0.003", "500_1_MM0.005"]

# Create figure with 1 row, 4 columns
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, dataset in enumerate(datasets):
    # Load true and estimated values
    true_path = f"{base_path}/{dataset}/estimate_output/r0_est.test_true.labels_num.csv"
    est_path = f"{base_path}/{dataset}/estimate_output/r0_est.test_est.labels_num.csv"

    true_df = pd.read_csv(true_path)
    est_df = pd.read_csv(est_path)

    # Extract true values (all log_R0_* columns) and exponentiate
    true_cols = [col for col in true_df.columns if col.startswith('log_R0_')]
    true_values = np.exp(true_df[true_cols].values.flatten())

    # Extract estimated values (all log_R0_*_value columns) and exponentiate
    est_cols = [col for col in est_df.columns if col.endswith('_value')]
    est_values = np.exp(est_df[est_cols].values.flatten())

    # Calculate statistics
    r, _ = stats.pearsonr(true_values, est_values)
    r_squared = r ** 2
    mse = np.mean((true_values - est_values) ** 2)
    n_samples = len(true_values)

    # Plot
    ax = axes[i]
    ax.scatter(true_values, est_values, alpha=0.5, s=20, c='#6baed6', edgecolors='none')

    # Add diagonal line
    min_val = min(true_values.min(), est_values.min())
    max_val = max(true_values.max(), est_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    # Add stats box (show number of test trees, not total points)
    n_trees = len(true_df)
    stats_text = f'n = {n_trees} samples\nR² = {r_squared:.4f}\nr = {r:.4f}\nMSE = {mse:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Labels
    ax.set_xlabel('True', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(dataset, fontsize=14)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

# Add row label on the left
fig.text(0.02, 0.5, 'CNN - R0', va='center', rotation='vertical', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig(f'{base_path}/cnn_r0_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{base_path}/cnn_r0_comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved to cnn_r0_comparison.pdf and cnn_r0_comparison.png")
