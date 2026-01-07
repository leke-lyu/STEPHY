#!/usr/bin/env python3
"""
Visualize phyddle R0 estimation results.
Format matches GNN training output style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ============================================================
# Load data
# ============================================================

history_df = pd.read_csv('./train_output/r0_est.train_history.csv')
test_true_df = pd.read_csv('./estimate_output/r0_est.test_true.labels_num.csv')
test_est_df = pd.read_csv('./estimate_output/r0_est.test_est.labels_num.csv')

# Get validation loss per epoch (use mse_value due to phyddle bug in loss_value recording)
val_loss = history_df[(history_df['dataset'] == 'validation') &
                       (history_df['metric'] == 'mse_value')]
epochs = val_loss['epoch'].values
losses = val_loss['value'].values

# Find best epoch and check for early stopping
best_idx = np.argmin(losses)
best_epoch = epochs[best_idx]
best_val = losses[best_idx]
final_epoch = epochs[-1]

# Calculate test MSE
all_test_true, all_test_pred = [], []
for i in range(16):
    true_r0 = np.exp(test_true_df[f'log_R0_{i}'].values)
    pred_r0 = np.exp(test_est_df[f'log_R0_{i}_value'].values)
    all_test_true.extend(true_r0)
    all_test_pred.extend(pred_r0)

all_test_true = np.array(all_test_true)
all_test_pred = np.array(all_test_pred)
test_mse = np.mean((all_test_true - all_test_pred)**2)

# ============================================================
# Print training progress (like your GNN format)
# ============================================================

print(f"\n{'='*50}")
print("Training for: R0 (16 locations)")
print(f"{'='*50}")
print("\nPhyddle CNN:")

# Print every 10 epochs
for epoch, loss in zip(epochs, losses):
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:>3} | Val: {loss:.4f}")

# Early stopping message
if final_epoch < 499:  # Assuming max 500 epochs
    print(f"  Early stopping at epoch {final_epoch + 1}")

# Final summary line
print(f"  Best Val: {best_val:.4f} | Test: {test_mse:.4f}")

# ============================================================
# Plot training curve
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

train_loss = history_df[(history_df['dataset'] == 'train') &
                         (history_df['metric'] == 'mse_value')]

ax.plot(train_loss['epoch'] + 1, train_loss['value'], 'b-', alpha=0.7, label='Train')
ax.plot(val_loss['epoch'] + 1, val_loss['value'], 'r-', linewidth=2, label='Validation')
ax.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.7,
           label=f'Best ({best_epoch + 1})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('Phyddle CNN: R0 Estimation', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./train_output/training_curve.pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============================================================
# Test scatter plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(all_test_true, all_test_pred, alpha=0.6, color='steelblue', s=50, edgecolor='white')

min_val = min(all_test_true.min(), all_test_pred.min())
max_val = max(all_test_true.max(), all_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

ax.set_xlabel('True', fontsize=12)
ax.set_ylabel('Predicted', fontsize=12)
ax.set_title('CNN - R0', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Stats - always show n, R², r, MSE
n_samples = len(test_true_df)
r2 = r2_score(all_test_true, all_test_pred)
corr = np.corrcoef(all_test_true, all_test_pred)[0, 1]
stats = f'n = {n_samples} samples\nR² = {r2:.4f}\nr = {corr:.4f}\nMSE = {test_mse:.4f}'

ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./estimate_output/r0_test.pdf', bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSaved: training_curve.pdf, r0_test.pdf")
