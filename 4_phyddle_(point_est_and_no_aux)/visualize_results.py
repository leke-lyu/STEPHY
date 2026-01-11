#!/usr/bin/env python3
"""
Visualize simplified training results.

Usage:
    python3 visualize_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def main():
    # Load data
    history_df = pd.read_csv('./output/training_history.csv')
    results_df = pd.read_csv('./output/test_predictions.csv')

    epochs = history_df['epoch'].values + 1
    train_loss = history_df['train_loss'].values
    val_loss = history_df['val_loss'].values

    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_val = val_loss[best_idx]

    # Get true and predicted values (convert from log scale)
    true_cols = sorted([c for c in results_df.columns if c.endswith('_true')])
    pred_cols = sorted([c for c in results_df.columns if c.endswith('_pred')])

    all_true = np.concatenate([np.exp(results_df[c].values) for c in true_cols])
    all_pred = np.concatenate([np.exp(results_df[c].values) for c in pred_cols])
    test_mse = np.mean((all_true - all_pred) ** 2)

    # Compute stats for plots
    n_samples = len(results_df)
    r2 = r2_score(all_true, all_pred)
    corr = np.corrcoef(all_true, all_pred)[0, 1]

    # Training curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation')
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Simplified CNN: R0 Estimation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./output/training_curve.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Test scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_true, all_pred, alpha=0.6, color='steelblue', s=50, edgecolor='white')
    min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('True R0', fontsize=12)
    ax.set_ylabel('Predicted R0', fontsize=12)
    ax.set_title('Simplified CNN - R0', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    stats = f'n = {n_samples} samples\nR² = {r2:.4f}\nr = {corr:.4f}\nMSE = {test_mse:.4f}'
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig('./output/r0_test.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: output/training_curve.pdf, output/r0_test.pdf")


if __name__ == '__main__':
    main()
