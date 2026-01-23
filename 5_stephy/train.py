#!/usr/bin/env python3
"""
Training script for CBLV-GAT.

All settings are controlled via config.py.

Usage:
    # Step 1: Analyze dataset to get parameters
    python3 analyze_trees.py /path/to/data

    # Step 2: Train with required parameters
    python3 train.py --num_locations ... --subtree_width ... \\
                     --input_dir /path/to/data --output_dir ./results
"""

import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from dgl.dataloading import GraphDataLoader

from model import CBLV_GAT, count_parameters
from data import build_all_graphs
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
    # Runtime options
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def apply_label_transform(graphs, label_scale):
    """Apply label transformation (log or linear) to R0 values."""
    if label_scale == 'log':
        for g, *_ in graphs:
            r0 = g.ndata['R0']
            r0 = torch.clamp(r0, min=1e-8)
            g.ndata['R0'] = torch.log(r0)


def normalize_labels(train_graphs, val_graphs, test_graphs):
    """Z-score normalize R0 labels using training set statistics."""
    train_r0 = torch.cat([g.ndata['R0'] for g, *_ in train_graphs])
    mean = train_r0.mean()
    std = train_r0.std()

    if std < 1e-8:
        std = torch.tensor(1.0)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata['R0'] = (g.ndata['R0'] - mean) / std

    return {'mean': mean, 'std': std}


def normalize_aux_features(train_graphs, val_graphs, test_graphs):
    """Z-score normalize auxiliary features using training set statistics."""
    train_aux = torch.cat([g.ndata['aux'] for g, *_ in train_graphs], dim=0)

    mean = train_aux.mean(dim=0)
    std = train_aux.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata['aux'] = (g.ndata['aux'] - mean) / std

    return {'mean': mean, 'std': std}


def normalize_edge_features(train_graphs, val_graphs, test_graphs):
    """Z-score normalize edge features using training set statistics."""
    train_edge = torch.cat([g.edata['feat'] for g, *_ in train_graphs], dim=0)

    mean = train_edge.mean(dim=0)
    std = train_edge.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.edata['feat'] = (g.edata['feat'] - mean) / std

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
    """Evaluate model on batched graphs. Returns loss and predictions."""
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
    return avg_loss, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    config = get_config()

    # Set required parameters from CLI
    config['model']['subtree_width'] = args.subtree_width
    config['num_locations'] = args.num_locations

    # Get settings from config
    label_scale = config['data'].get('label_scale', 'log')

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")
    print(f"num_locations: {args.num_locations}")
    print(f"subtree_width: {args.subtree_width}")

    set_seed(config['train']['random_seed'])

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

    # Train/val/test split
    train_ratio = config['train']['train_ratio']
    train_graphs, temp_graphs = train_test_split(
        all_graphs, test_size=1-train_ratio, random_state=config['train']['random_seed']
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs, test_size=0.5, random_state=config['train']['random_seed']
    )

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Apply label transform (log or linear)
    print(f"  Label scale: {label_scale}")
    if label_scale == 'log':
        apply_label_transform(train_graphs, label_scale)
        apply_label_transform(val_graphs, label_scale)
        apply_label_transform(test_graphs, label_scale)

    # Normalize auxiliary features using training set statistics
    aux_norm = normalize_aux_features(train_graphs, val_graphs, test_graphs)
    print(f"  Aux normalization: mean shape={aux_norm['mean'].shape}, std shape={aux_norm['std'].shape}")

    # Normalize edge features using training set statistics
    edge_norm = normalize_edge_features(train_graphs, val_graphs, test_graphs)
    print(f"  Edge normalization: mean shape={edge_norm['mean'].shape}, std shape={edge_norm['std'].shape}")

    # Normalize R0 labels using training set statistics
    label_norm = normalize_labels(train_graphs, val_graphs, test_graphs)
    label_norm['scale'] = label_scale
    print(f"  R0 normalization: mean={label_norm['mean']:.4f}, std={label_norm['std']:.4f}")

    # Save normalization params
    torch.save(aux_norm, output_dir / 'aux_norm.pt')
    torch.save(edge_norm, output_dir / 'edge_norm.pt')
    torch.save(label_norm, output_dir / 'label_norm.pt')

    # Extract just graphs for DataLoader
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

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

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
    _, test_preds_norm, test_labels_norm = evaluate(model, test_loader, criterion, device)

    # Denormalize predictions and labels
    label_mean = label_norm['mean'].item()
    label_std = label_norm['std'].item()
    test_preds = test_preds_norm * label_std + label_mean
    test_labels = test_labels_norm * label_std + label_mean

    # If log scale was used, convert back to original scale
    if label_scale == 'log':
        test_preds = np.exp(test_preds)
        test_labels = np.exp(test_labels)

    # Save results
    torch.save(best_state, output_dir / 'best_model.pt')
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    pd.DataFrame({
        'true_R0': test_labels,
        'pred_R0': test_preds
    }).to_csv(output_dir / 'test_predictions.csv', index=False)

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
