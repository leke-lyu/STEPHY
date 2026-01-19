#!/usr/bin/env python3
"""
Summarize and compare results from three R0 estimation pipelines.

Pipelines:
    - 3_phyddle: Full Phyddle library (CNN + CPI)
    - 4_phyddle: Simplified Phyddle (CNN, point estimates only)
    - 5_stephy:  CBLV-GAT (CNN + Graph Attention Network)

Outputs:
    - scatter_combined.pdf:         True vs Predicted R0 (3 rows × N cols)
    - training_curves_combined.pdf: Training/Validation MSE curves (3 rows × N cols)
    - metrics_summary.csv:          R², Pearson r, MSE for all pipeline-dataset pairs

Usage:
    python3 summarize_results.py --base_dir /path/to/epidata --output_dir /path/to/summary
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# =============================================================================
# Configuration
# =============================================================================

PIPELINES = ['3_phyddle', '4_phyddle', '5_stephy']

PIPELINE_LABELS = {
    '3_phyddle': 'Phyddle (Full)',
    '4_phyddle': 'Phyddle (Simplified)',
    '5_stephy': 'STEPHY (CBLV-GAT)'
}

PIPELINE_COLORS = {
    '3_phyddle': '#1f77b4',
    '4_phyddle': '#ff7f0e',
    '5_stephy': '#2ca02c'
}

NUM_LOCATIONS = 16

# =============================================================================
# Data Loaders
# =============================================================================

def load_predictions(pipeline, result_dir):
    """Load test predictions for a pipeline. Returns (true_R0, pred_R0) in original scale."""

    if pipeline == '3_phyddle':
        # Files: estimate_output/r0_est.test_true.labels_num.csv, r0_est.test_est.labels_num.csv
        true_file = os.path.join(result_dir, 'estimate_output', 'r0_est.test_true.labels_num.csv')
        pred_file = os.path.join(result_dir, 'estimate_output', 'r0_est.test_est.labels_num.csv')
        if not os.path.exists(true_file) or not os.path.exists(pred_file):
            return None, None
        true_df, pred_df = pd.read_csv(true_file), pd.read_csv(pred_file)
        true_log = true_df[[f'log_R0_{i}' for i in range(NUM_LOCATIONS)]].values.flatten()
        pred_log = pred_df[[f'log_R0_{i}_value' for i in range(NUM_LOCATIONS)]].values.flatten()
        return np.exp(true_log), np.exp(pred_log)

    elif pipeline == '4_phyddle':
        # File: output/test_predictions.csv (columns: log_R0_{i}_true, log_R0_{i}_pred)
        pred_file = os.path.join(result_dir, 'output', 'test_predictions.csv')
        if not os.path.exists(pred_file):
            return None, None
        df = pd.read_csv(pred_file)
        true_log = df[[f'log_R0_{i}_true' for i in range(NUM_LOCATIONS)]].values.flatten()
        pred_log = df[[f'log_R0_{i}_pred' for i in range(NUM_LOCATIONS)]].values.flatten()
        return np.exp(true_log), np.exp(pred_log)

    elif pipeline == '5_stephy':
        # File: test_predictions.csv (columns: true_R0, pred_R0 - already in R0 scale)
        pred_file = os.path.join(result_dir, 'test_predictions.csv')
        if not os.path.exists(pred_file):
            return None, None
        df = pd.read_csv(pred_file)
        return df['true_R0'].values, df['pred_R0'].values

    return None, None


def load_history(pipeline, result_dir):
    """Load training history. Returns DataFrame with (epoch, train_loss, val_loss, [val_loss_combined])."""

    if pipeline == '3_phyddle':
        # Long format: (epoch, dataset, metric, value). Uses loss_combined for early stopping.
        hist_file = os.path.join(result_dir, 'train_output', 'r0_est.train_history.csv')
        if not os.path.exists(hist_file):
            return None
        df = pd.read_csv(hist_file)
        train = df[(df['dataset'] == 'train') & (df['metric'] == 'mse_value')][['epoch', 'value']].copy()
        val = df[(df['dataset'] == 'validation') & (df['metric'] == 'mse_value')][['epoch', 'value']].copy()
        val_comb = df[(df['dataset'] == 'validation') & (df['metric'] == 'loss_combined')][['epoch', 'value']].copy()
        train.columns, val.columns, val_comb.columns = ['epoch', 'train_loss'], ['epoch', 'val_loss'], ['epoch', 'val_loss_combined']
        history = train.merge(val, on='epoch').merge(val_comb, on='epoch')
        return history.sort_values('epoch').reset_index(drop=True)

    elif pipeline == '4_phyddle':
        # Wide format: (epoch, train_loss, val_loss)
        hist_file = os.path.join(result_dir, 'output', 'training_history.csv')
        return pd.read_csv(hist_file) if os.path.exists(hist_file) else None

    elif pipeline == '5_stephy':
        # Wide format without epoch column: (train_loss, val_loss)
        hist_file = os.path.join(result_dir, 'training_history.csv')
        if not os.path.exists(hist_file):
            return None
        df = pd.read_csv(hist_file)
        df['epoch'] = range(len(df))
        return df[['epoch', 'train_loss', 'val_loss']]

    return None

# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(true_R0, pred_R0):
    """Compute R², Pearson r, and MSE. Returns dict or None if invalid."""
    if true_R0 is None or pred_R0 is None:
        return None
    mask = ~(np.isnan(true_R0) | np.isnan(pred_R0))
    true_R0, pred_R0 = true_R0[mask], pred_R0[mask]
    if len(true_R0) == 0:
        return None
    return {
        'R2': r2_score(true_R0, pred_R0),
        'Pearson_r': pearsonr(true_R0, pred_R0)[0],
        'MSE': mean_squared_error(true_R0, pred_R0),
        'N_samples': len(true_R0)
    }

# =============================================================================
# Plotting
# =============================================================================

def plot_training_curves(all_histories, output_dir):
    """Plot training curves grid: rows=pipelines, cols=datasets."""
    datasets = sorted(all_histories.keys())
    n_datasets, n_pipelines = len(datasets), len(PIPELINES)
    fig, axes = plt.subplots(n_pipelines, n_datasets, figsize=(4 * n_datasets, 4 * n_pipelines))
    if n_pipelines == 1: axes = axes.reshape(1, -1)
    if n_datasets == 1: axes = axes.reshape(-1, 1)

    for i, pipeline in enumerate(PIPELINES):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            history = all_histories[dataset].get(pipeline)

            if history is None or len(history) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                epochs = history['epoch'].values
                train_loss, val_loss = history['train_loss'].values, history['val_loss'].values

                # Best epoch: 3_phyddle uses loss_combined, others use val_loss
                if pipeline == '3_phyddle' and 'val_loss_combined' in history.columns:
                    best_idx = np.argmin(history['val_loss_combined'].values)
                else:
                    best_idx = np.argmin(val_loss)
                best_epoch, best_val_mse = epochs[best_idx], val_loss[best_idx]

                # Plot curves
                ax.plot(epochs, train_loss, 'b-', lw=1.5, label='Train')
                ax.plot(epochs, val_loss, 'r-', lw=1.5, label='Validation')
                ax.axvline(x=best_epoch, color='green', ls='--', lw=1.5, label='Best')
                ax.legend(loc='upper right', fontsize=7)

                # Info box
                info = f'Val MSE: {best_val_mse:.3f}\nBest: {best_epoch}\nTotal: {len(epochs)}'
                ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8, va='top',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                ax.set_ylim(0, 1.2)
                ax.set_xlim(0, max(300, epochs.max()))

            # Labels
            if j == 0:
                ax.set_ylabel(f'{PIPELINE_LABELS[pipeline]}\nLoss (MSE)', fontsize=10)
            else:
                ax.set_ylabel('Loss (MSE)', fontsize=10)
            if i == n_pipelines - 1:
                ax.set_xlabel('Epoch', fontsize=10)
            if i == 0:
                ax.set_title(dataset.replace('500_1_', '500_1_MM'), fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_combined.pdf'), bbox_inches='tight')
    plt.close()


def plot_scatter(all_predictions, output_dir):
    """Plot scatter grid: rows=pipelines, cols=datasets."""
    datasets = sorted(all_predictions.keys())
    n_datasets, n_pipelines = len(datasets), len(PIPELINES)
    fig, axes = plt.subplots(n_pipelines, n_datasets, figsize=(4 * n_datasets, 4 * n_pipelines))
    if n_pipelines == 1: axes = axes.reshape(1, -1)
    if n_datasets == 1: axes = axes.reshape(-1, 1)

    for i, pipeline in enumerate(PIPELINES):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            true_R0, pred_R0 = all_predictions[dataset].get(pipeline, (None, None))

            if true_R0 is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.scatter(true_R0, pred_R0, alpha=0.3, s=5, color=PIPELINE_COLORS[pipeline])
                lims = [min(true_R0.min(), pred_R0.min()), max(true_R0.max(), pred_R0.max())]
                ax.plot(lims, lims, 'k--', lw=1)

                # Metrics info box
                m = compute_metrics(true_R0, pred_R0)
                if m:
                    info = f"R² = {m['R2']:.3f}\nr  = {m['Pearson_r']:.3f}\nMSE = {m['MSE']:.3f}"
                    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=9, va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Labels
            if j == 0:
                ax.set_ylabel(f'{PIPELINE_LABELS[pipeline]}\nPredicted R0', fontsize=10)
            else:
                ax.set_ylabel('Predicted R0', fontsize=10)
            if i == n_pipelines - 1:
                ax.set_xlabel('True R0', fontsize=10)
            if i == 0:
                ax.set_title(dataset.replace('500_1_', '500_1_MM'), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_combined.pdf'), bbox_inches='tight')
    plt.close()

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Summarize R0 estimation results from 3 pipelines')
    parser.add_argument('--base_dir', type=str, default='/Users/lukelyu/Desktop/epidata',
                        help='Base directory containing 3_phyddle, 4_phyddle, 5_stephy folders')
    parser.add_argument('--output_dir', type=str, default='/Users/lukelyu/Desktop/epidata/summary',
                        help='Output directory for summary files (PDF and CSV)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover datasets from 3_phyddle folder
    sample_dir = os.path.join(args.base_dir, '3_phyddle')
    datasets = sorted([d for d in os.listdir(sample_dir)
                       if os.path.isdir(os.path.join(sample_dir, d)) and d.startswith('500_')])

    print(f"Found {len(datasets)} datasets: {datasets}")
    print(f"Output: {args.output_dir}\n")

    # Collect data
    all_predictions, all_histories, all_metrics = {}, {}, {}

    for dataset in datasets:
        print(f"{'='*60}\nProcessing: {dataset}\n{'='*60}")
        predictions, histories, metrics = {}, {}, {}

        for pipeline in PIPELINES:
            result_dir = os.path.join(args.base_dir, pipeline, dataset)
            true_R0, pred_R0 = load_predictions(pipeline, result_dir)
            predictions[pipeline] = (true_R0, pred_R0)
            histories[pipeline] = load_history(pipeline, result_dir)
            m = compute_metrics(true_R0, pred_R0)
            metrics[pipeline] = m

            if m:
                print(f"\n{PIPELINE_LABELS[pipeline]}:  R²={m['R2']:.4f}  r={m['Pearson_r']:.4f}  MSE={m['MSE']:.4f}")
            else:
                print(f"\n{PIPELINE_LABELS[pipeline]}: No data")

        all_predictions[dataset] = predictions
        all_histories[dataset] = histories
        all_metrics[dataset] = metrics

    # Generate plots
    print(f"\n{'='*60}\nGenerating plots...\n{'='*60}")
    plot_scatter(all_predictions, args.output_dir)
    print("  - scatter_combined.pdf")
    plot_training_curves(all_histories, args.output_dir)
    print("  - training_curves_combined.pdf")

    # Save metrics CSV
    rows = []
    for dataset in datasets:
        for pipeline in PIPELINES:
            m = all_metrics[dataset].get(pipeline)
            if m:
                rows.append({
                    'dataset': dataset,
                    'migration_rate': float(dataset.split('_')[-1]),
                    'pipeline': pipeline,
                    'pipeline_label': PIPELINE_LABELS[pipeline],
                    'R2': m['R2'],
                    'Pearson_r': m['Pearson_r'],
                    'MSE': m['MSE']
                })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics_summary.csv'), index=False)
    print("  - metrics_summary.csv")

    # Print summary table
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for metric in ['R2', 'Pearson_r', 'MSE']:
        print(f"\n{metric}:")
        print(metrics_df.pivot(index='dataset', columns='pipeline_label', values=metric).to_string())

    print(f"\n{'='*60}\nDone. Outputs saved to: {args.output_dir}\n{'='*60}")


if __name__ == '__main__':
    main()
