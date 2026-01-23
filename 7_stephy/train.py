#!/usr/bin/env python3
"""
Training script for CBLV-GAT with Epidemiological Features (7_stephy).

Single-task prediction: either R0 or Source_Sink_Score (controlled by config.py).

Usage:
    # Step 1: Analyze dataset to get parameters
    python3 analyze_trees.py /path/to/data

    # Step 2: Train with required parameters
    python3 train.py --num_locations ... --subtree_width ... \\
                     --input_dir /path/to/data --output_dir ./results

Requirements:
    - *_beast2.trees files (phylogenetic trees)
    - *_nf.csv files (node features + labels)
"""

import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from dgl.dataloading import GraphDataLoader

from model import CBLV_GAT, count_parameters
from data import build_all_graphs
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train CBLV-GAT model with epi features')
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


def apply_epi_log_transform(graphs):
    """
    Apply log transform to epidemiological features.

    Epi features (Initial_Population, Epidemic_Peak, Peak_Timing, Accumulated_Infections)
    are all positive, so we use log(x) directly (no +1 needed).
    """
    for g, *_ in graphs:
        epi = g.ndata['epi']
        epi = torch.clamp(epi, min=1e-8)
        g.ndata['epi'] = torch.log(epi)


def apply_label_log_transform(graphs, label_name):
    """Apply log transform to labels."""
    for g, *_ in graphs:
        labels = g.ndata[label_name]
        labels = torch.clamp(labels, min=1e-8)
        g.ndata[label_name] = torch.log(labels)


def normalize_epi_features(train_graphs, val_graphs, test_graphs):
    """Z-score normalize epi features using training set statistics."""
    train_epi = torch.cat([g.ndata['epi'] for g, *_ in train_graphs], dim=0)

    mean = train_epi.mean(dim=0)
    std = train_epi.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata['epi'] = (g.ndata['epi'] - mean) / std

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


def normalize_labels(train_graphs, val_graphs, test_graphs, label_name):
    """Z-score normalize labels using training set statistics."""
    train_labels = torch.cat([g.ndata[label_name] for g, *_ in train_graphs])
    mean = train_labels.mean()
    std = train_labels.std()

    if std < 1e-8:
        std = torch.tensor(1.0)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata[label_name] = (g.ndata[label_name] - mean) / std

    return {'mean': mean, 'std': std}


def train_epoch(model, dataloader, optimizer, criterion, device, label_name):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_nodes = 0

    for batched_g in dataloader:
        batched_g = batched_g.to(device)
        node_cblv = batched_g.ndata['cblv']
        edge_feat = batched_g.edata['feat']

        optimizer.zero_grad()
        predictions = model(batched_g, node_cblv, edge_feat)
        labels = batched_g.ndata[label_name]

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        n_nodes = batched_g.num_nodes()
        total_loss += loss.item() * n_nodes
        total_nodes += n_nodes

    return total_loss / total_nodes


def evaluate(model, dataloader, criterion, device, label_name):
    """Evaluate model. Returns loss and predictions."""
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

            predictions = model(batched_g, node_cblv, edge_feat)
            labels = batched_g.ndata[label_name]

            loss = criterion(predictions, labels)

            n_nodes = batched_g.num_nodes()
            total_loss += loss.item() * n_nodes
            total_nodes += n_nodes

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'loss': total_loss / total_nodes,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
    }


def main():
    args = parse_args()
    config = get_config()

    # Set required parameters from CLI
    config['model']['subtree_width'] = args.subtree_width
    config['num_locations'] = args.num_locations

    # Get settings from config
    label_name = config['label']  # 'R0' or 'Source_Sink_Score'
    label_scale = config['data'].get('label_scale', 'log')
    epi_scale = config['data'].get('epi_scale', 'log')

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")
    print(f"num_locations: {args.num_locations}")
    print(f"subtree_width: {args.subtree_width}")
    print(f"Label: {label_name}")

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

    # Apply transforms BEFORE normalization
    print(f"\n  Epi scale: {epi_scale}")
    if epi_scale == 'log':
        apply_epi_log_transform(train_graphs)
        apply_epi_log_transform(val_graphs)
        apply_epi_log_transform(test_graphs)

    # Apply log transform to labels if needed (only for R0)
    apply_log_to_label = (label_name == 'R0' and label_scale == 'log')
    print(f"  Label scale: {label_scale}")
    if apply_log_to_label:
        apply_label_log_transform(train_graphs, label_name)
        apply_label_log_transform(val_graphs, label_name)
        apply_label_log_transform(test_graphs, label_name)

    # Normalize features using training set statistics
    epi_norm = normalize_epi_features(train_graphs, val_graphs, test_graphs)
    epi_norm['scale'] = epi_scale
    print(f"  Epi normalization: mean shape={epi_norm['mean'].shape}, std shape={epi_norm['std'].shape}")

    edge_norm = normalize_edge_features(train_graphs, val_graphs, test_graphs)
    print(f"  Edge normalization: mean shape={edge_norm['mean'].shape}, std shape={edge_norm['std'].shape}")

    # Normalize labels
    label_norm = normalize_labels(train_graphs, val_graphs, test_graphs, label_name)
    label_norm['scale'] = label_scale if label_name == 'R0' else 'linear'
    print(f"  Label normalization: mean={label_norm['mean']:.4f}, std={label_norm['std']:.4f}")

    # Save normalization params
    torch.save(epi_norm, output_dir / 'epi_norm.pt')
    torch.save(edge_norm, output_dir / 'edge_norm.pt')
    torch.save(label_norm, output_dir / 'label_norm.pt')

    # Extract just graphs for DataLoader
    train_g = [g for g, *_ in train_graphs]
    val_g = [g for g, *_ in val_graphs]
    test_g = [g for g, *_ in test_graphs]

    # Create DataLoaders
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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, label_name)
        val_results = evaluate(model, val_loader, criterion, device, label_name)
        val_loss = val_results['loss']

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
    test_results = evaluate(model, test_loader, criterion, device, label_name)

    # Denormalize predictions and labels
    label_mean = label_norm['mean'].item()
    label_std = label_norm['std'].item()
    test_preds = test_results['preds'] * label_std + label_mean
    test_labels = test_results['labels'] * label_std + label_mean

    if apply_log_to_label:
        test_preds = np.exp(test_preds)
        test_labels = np.exp(test_labels)

    # Compute metrics
    r2 = r2_score(test_labels, test_preds)
    mse = mean_squared_error(test_labels, test_preds)

    print(f"\nTest Results ({label_name}):")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")

    # Save results
    torch.save(best_state, output_dir / 'best_model.pt')
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    pd.DataFrame({
        f'true_{label_name}': test_labels,
        f'pred_{label_name}': test_preds,
    }).to_csv(output_dir / 'test_predictions.csv', index=False)

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
