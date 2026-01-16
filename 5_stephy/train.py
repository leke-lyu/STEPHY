#!/usr/bin/env python3
"""
Training script for CBLV-GAT.

Usage:
    # Step 1: Analyze dataset to get parameters
    python3 analyze_trees.py /path/to/data

    # Step 2: Train with required parameters
    python3 train.py --num_locations ... --subtree_width ... \\
                     --input_dir /path/to/data --output_dir ./results
"""

import argparse
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader

from model import CBLV_GAT, count_parameters
from data import build_all_graphs, normalize_edge_features
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train CBLV-GAT model')
    # Required arguments
    parser.add_argument('--input_dir', required=True, help='Input data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_locations', type=int, required=True,
                        help='Number of locations (from analyze_trees.py)')
    parser.add_argument('--subtree_width', type=int, required=True,
                        help='Max tips per location (from analyze_trees.py)')
    # Optional overrides
    parser.add_argument('--epochs', type=int, default=None, help='Override num_epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning_rate')
    parser.add_argument('--seed', type=int, default=None, help='Override random_seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def normalize_labels(train_graphs, val_graphs, test_graphs):
    """
    Z-score normalize R0 labels using training set statistics.

    Args:
        train_graphs: List of (graph, id, locs, height) tuples for training
        val_graphs: List of (graph, id, locs, height) tuples for validation
        test_graphs: List of (graph, id, locs, height) tuples for testing

    Returns:
        label_norm: Dict with 'mean' and 'std' tensors
    """
    # Compute mean/std from training set only
    train_r0 = torch.cat([g.ndata['R0'] for g, *_ in train_graphs])
    mean = train_r0.mean()
    std = train_r0.std()

    # Avoid division by zero
    if std < 1e-8:
        std = torch.tensor(1.0)

    # Normalize all graphs
    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata['R0'] = (g.ndata['R0'] - mean) / std

    return {'mean': mean, 'std': std}


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch using batched graphs."""
    model.train()
    total_loss = 0
    total_nodes = 0

    for batched_g in dataloader:
        batched_g = batched_g.to(device)
        node_cblv = batched_g.ndata['cblv']
        edge_feat = batched_g.edata['feat']
        labels = batched_g.ndata['R0']

        optimizer.zero_grad()
        pred = model(batched_g, node_cblv, edge_feat)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batched_g.num_nodes()
        total_nodes += batched_g.num_nodes()

    return total_loss / total_nodes


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on batched graphs."""
    model.eval()
    total_loss = 0
    total_nodes = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_g in dataloader:
            batched_g = batched_g.to(device)
            node_cblv = batched_g.ndata['cblv']
            edge_feat = batched_g.edata['feat']
            labels = batched_g.ndata['R0']

            pred = model(batched_g, node_cblv, edge_feat)
            loss = criterion(pred, labels)

            total_loss += loss.item() * batched_g.num_nodes()
            total_nodes += batched_g.num_nodes()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_nodes
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    r2 = r2_score(all_labels, all_preds)
    corr = np.corrcoef(all_labels, all_preds)[0, 1]

    return avg_loss, r2, corr, all_preds, all_labels


def plot_results(train_history, test_preds, test_labels, output_dir, best_epoch):
    """Generate result plots."""
    output_dir = Path(output_dir)

    # Training curve
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_history['train_loss']) + 1)
    ax.plot(epochs, train_history['train_loss'], 'b-', alpha=0.7, label='Train')
    ax.plot(epochs, train_history['val_loss'], 'r-', linewidth=2, label='Validation')
    ax.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.7,
               label=f'Best ({best_epoch + 1})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('CBLV-GAT Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Test scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(test_labels, test_preds, alpha=0.6, color='steelblue', s=50, edgecolor='white')
    min_val = min(test_labels.min(), test_preds.min())
    max_val = max(test_labels.max(), test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('True R0', fontsize=12)
    ax.set_ylabel('Predicted R0', fontsize=12)
    ax.set_title('CBLV-GAT - R0 Prediction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    r2 = r2_score(test_labels, test_preds)
    corr = np.corrcoef(test_labels, test_preds)[0, 1]
    mse = np.mean((test_labels - test_preds) ** 2)
    n_samples = len(test_labels)

    stats = f'n = {n_samples} samples\nR² = {r2:.4f}\nr = {corr:.4f}\nMSE = {mse:.4f}'
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'r0_test.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {output_dir}/training_curve.pdf, {output_dir}/r0_test.pdf")


def main():
    args = parse_args()
    config = get_config()

    # Set required parameters from CLI
    config['model']['subtree_width'] = args.subtree_width
    config['num_locations'] = args.num_locations

    # Override config with optional command line arguments
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    if args.lr:
        config['train']['learning_rate'] = args.lr
    if args.seed:
        config['train']['random_seed'] = args.seed

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")
    print(f"num_locations: {args.num_locations}")
    print(f"subtree_width: {args.subtree_width}")

    set_seed(config['train']['random_seed'])

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Build graphs
    print("\nBuilding graphs...")
    subtree_width = config['model']['subtree_width']
    all_graphs = build_all_graphs(args.input_dir, subtree_width, verbose=True)
    print(f"  Total graphs: {len(all_graphs)}")

    # Validate num_locations
    for g, graph_id, locs, _ in all_graphs:
        if g.num_nodes() != args.num_locations:
            raise ValueError(
                f"Graph {graph_id}: expected {args.num_locations} locations, got {g.num_nodes()}. "
                f"Check your data or --num_locations value."
            )
    print(f"  All graphs have {args.num_locations} locations ✓")

    # Normalize edge features
    all_graphs, edge_norm = normalize_edge_features(all_graphs)

    # Save normalization params
    torch.save(edge_norm, output_dir / 'edge_norm.pt')

    # Train/val/test split
    train_ratio = config['train']['train_ratio']
    val_ratio = config['train']['val_ratio']

    train_graphs, temp_graphs = train_test_split(
        all_graphs, test_size=1-train_ratio, random_state=config['train']['random_seed']
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=0.5, random_state=config['train']['random_seed']
    )

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Normalize R0 labels using training set statistics
    label_norm = normalize_labels(train_graphs, val_graphs, test_graphs)
    print(f"  R0 normalization: mean={label_norm['mean']:.4f}, std={label_norm['std']:.4f}")

    # Save normalization params
    torch.save(label_norm, output_dir / 'label_norm.pt')

    # Extract just graphs for DataLoader (graphs are tuples: (g, id, locs, height))
    train_g = [g for g, *_ in train_graphs]
    val_g = [g for g, *_ in val_graphs]
    test_g = [g for g, *_ in test_graphs]

    # Create DataLoaders with batching
    batch_size = config['train']['batch_size']
    train_loader = GraphDataLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_g, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_g, batch_size=batch_size, shuffle=False)

    print(f"  Batch size: {batch_size}")

    # Create model
    model = CBLV_GAT(config['model']).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    num_epochs = config['train']['num_epochs']
    patience = config['train']['early_stopping_patience']

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2, val_corr, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val R²: {val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest epoch: {best_epoch+1} (val_loss: {best_val_loss:.4f})")

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_loss_norm, _, _, test_preds_norm, test_labels_norm = evaluate(
        model, test_loader, criterion, device
    )

    # Denormalize predictions and labels for evaluation
    label_mean = label_norm['mean'].item()
    label_std = label_norm['std'].item()
    test_preds = test_preds_norm * label_std + label_mean
    test_labels = test_labels_norm * label_std + label_mean

    # Compute metrics on original scale
    test_mse = np.mean((test_preds - test_labels) ** 2)
    test_r2 = r2_score(test_labels, test_preds)
    test_corr = np.corrcoef(test_labels, test_preds)[0, 1]

    print(f"\nTest Results (denormalized):")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  R²:  {test_r2:.4f}")
    print(f"  r:   {test_corr:.4f}")

    # Save results
    torch.save(best_state, output_dir / 'best_model.pt')

    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)

    results_df = pd.DataFrame({
        'true_R0': test_labels,
        'pred_R0': test_preds
    })
    results_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    summary = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'test_mse': float(test_mse),
        'test_r2': float(test_r2),
        'test_corr': float(test_corr),
        'label_norm_mean': label_mean,
        'label_norm_std': label_std,
        'num_train': len(train_graphs),
        'num_val': len(val_graphs),
        'num_test': len(test_graphs),
        'num_parameters': count_parameters(model),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    plot_results(history, test_preds, test_labels, output_dir, best_epoch)

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
