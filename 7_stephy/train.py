#!/usr/bin/env python3
"""
Training script for CBLV-GAT with Epidemiological Features (7_stephy).

Changes from 5_stephy:
- Epi features: log(x) transform before Z-score normalization
- Dual output: R0 + Source_Sink_Score
- Separate loss weighting for each label

Usage:
    # Step 1: Preprocess dataset to generate *_nd.csv files
    python3 preprocess.py /path/to/data

    # Step 2: Analyze dataset to get parameters
    python3 analyze_trees.py /path/to/data

    # Step 3: Train with required parameters
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
    parser = argparse.ArgumentParser(description='Train CBLV-GAT model with epi features')
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
    parser.add_argument('--label_scale', choices=['linear', 'log'], default=None,
                        help='Override R0 label scale (linear or log)')
    parser.add_argument('--r0_weight', type=float, default=1.0,
                        help='Weight for R0 loss (default: 1.0)')
    parser.add_argument('--sss_weight', type=float, default=1.0,
                        help='Weight for Source_Sink_Score loss (default: 1.0)')
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

    Args:
        graphs: List of (graph, id, locs, height) tuples
    """
    for g, *_ in graphs:
        epi = g.ndata['epi']
        # Clamp to avoid log(0) - shouldn't happen but safety first
        epi = torch.clamp(epi, min=1e-8)
        g.ndata['epi'] = torch.log(epi)


def apply_r0_log_transform(graphs):
    """
    Apply log transform to R0 labels.

    Args:
        graphs: List of (graph, id, locs, height) tuples
    """
    for g, *_ in graphs:
        r0 = g.ndata['R0']
        r0 = torch.clamp(r0, min=1e-8)
        g.ndata['R0'] = torch.log(r0)


def normalize_epi_features(train_graphs, val_graphs, test_graphs):
    """
    Z-score normalize epi features using training set statistics.

    Args:
        train_graphs: List of (graph, id, locs, height) tuples for training
        val_graphs: List of (graph, id, locs, height) tuples for validation
        test_graphs: List of (graph, id, locs, height) tuples for testing

    Returns:
        epi_norm: Dict with 'mean' and 'std' tensors (shape: 4,)
    """
    # Collect all epi features from training set
    train_epi = torch.cat([g.ndata['epi'] for g, *_ in train_graphs], dim=0)  # (total_train_nodes, 4)

    # Compute per-feature mean and std
    mean = train_epi.mean(dim=0)  # (4,)
    std = train_epi.std(dim=0)    # (4,)

    # Avoid division by zero
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    # Normalize all graphs
    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata['epi'] = (g.ndata['epi'] - mean) / std

    return {'mean': mean, 'std': std}


def normalize_edge_features(train_graphs, val_graphs, test_graphs):
    """
    Z-score normalize edge features using training set statistics.

    Returns:
        edge_norm: Dict with 'mean' and 'std' tensors (shape: 3,)
    """
    train_edge = torch.cat([g.edata['feat'] for g, *_ in train_graphs], dim=0)

    mean = train_edge.mean(dim=0)
    std = train_edge.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.edata['feat'] = (g.edata['feat'] - mean) / std

    return {'mean': mean, 'std': std}


def normalize_labels(train_graphs, val_graphs, test_graphs, label_name):
    """
    Z-score normalize a specific label using training set statistics.

    Args:
        label_name: 'R0' or 'Source_Sink_Score'

    Returns:
        norm: Dict with 'mean' and 'std' values
    """
    train_labels = torch.cat([g.ndata[label_name] for g, *_ in train_graphs])
    mean = train_labels.mean()
    std = train_labels.std()

    if std < 1e-8:
        std = torch.tensor(1.0)

    for graph_list in [train_graphs, val_graphs, test_graphs]:
        for g, *_ in graph_list:
            g.ndata[label_name] = (g.ndata[label_name] - mean) / std

    return {'mean': mean, 'std': std}


def train_epoch(model, dataloader, optimizer, criterion, device, r0_weight=1.0, sss_weight=1.0):
    """Train for one epoch using batched graphs with dual output."""
    model.train()
    total_loss = 0
    total_r0_loss = 0
    total_sss_loss = 0
    total_nodes = 0

    for batched_g in dataloader:
        batched_g = batched_g.to(device)
        node_cblv = batched_g.ndata['cblv']
        edge_feat = batched_g.edata['feat']
        r0_labels = batched_g.ndata['R0']
        sss_labels = batched_g.ndata['Source_Sink_Score']

        optimizer.zero_grad()
        r0_pred, sss_pred = model(batched_g, node_cblv, edge_feat)

        # Compute weighted loss
        r0_loss = criterion(r0_pred, r0_labels)
        sss_loss = criterion(sss_pred, sss_labels)
        loss = r0_weight * r0_loss + sss_weight * sss_loss

        loss.backward()
        optimizer.step()

        n_nodes = batched_g.num_nodes()
        total_loss += loss.item() * n_nodes
        total_r0_loss += r0_loss.item() * n_nodes
        total_sss_loss += sss_loss.item() * n_nodes
        total_nodes += n_nodes

    return (total_loss / total_nodes,
            total_r0_loss / total_nodes,
            total_sss_loss / total_nodes)


def evaluate(model, dataloader, criterion, device, r0_weight=1.0, sss_weight=1.0):
    """Evaluate model on batched graphs. Returns losses and predictions."""
    model.eval()
    total_loss = 0
    total_r0_loss = 0
    total_sss_loss = 0
    total_nodes = 0

    all_r0_preds = []
    all_r0_labels = []
    all_sss_preds = []
    all_sss_labels = []

    with torch.no_grad():
        for batched_g in dataloader:
            batched_g = batched_g.to(device)
            node_cblv = batched_g.ndata['cblv']
            edge_feat = batched_g.edata['feat']
            r0_labels = batched_g.ndata['R0']
            sss_labels = batched_g.ndata['Source_Sink_Score']

            r0_pred, sss_pred = model(batched_g, node_cblv, edge_feat)

            r0_loss = criterion(r0_pred, r0_labels)
            sss_loss = criterion(sss_pred, sss_labels)
            loss = r0_weight * r0_loss + sss_weight * sss_loss

            n_nodes = batched_g.num_nodes()
            total_loss += loss.item() * n_nodes
            total_r0_loss += r0_loss.item() * n_nodes
            total_sss_loss += sss_loss.item() * n_nodes
            total_nodes += n_nodes

            all_r0_preds.extend(r0_pred.cpu().numpy())
            all_r0_labels.extend(r0_labels.cpu().numpy())
            all_sss_preds.extend(sss_pred.cpu().numpy())
            all_sss_labels.extend(sss_labels.cpu().numpy())

    return {
        'loss': total_loss / total_nodes,
        'r0_loss': total_r0_loss / total_nodes,
        'sss_loss': total_sss_loss / total_nodes,
        'r0_preds': np.array(all_r0_preds),
        'r0_labels': np.array(all_r0_labels),
        'sss_preds': np.array(all_sss_preds),
        'sss_labels': np.array(all_sss_labels),
    }


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
    if args.label_scale:
        config['data']['label_scale'] = args.label_scale

    # Get settings
    label_scale = config['data'].get('label_scale', 'log')
    epi_scale = config['data'].get('epi_scale', 'log')
    r0_weight = args.r0_weight
    sss_weight = args.sss_weight

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")
    print(f"num_locations: {args.num_locations}")
    print(f"subtree_width: {args.subtree_width}")
    print(f"R0 weight: {r0_weight}, SSS weight: {sss_weight}")

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

    print(f"  R0 label scale: {label_scale}")
    if label_scale == 'log':
        apply_r0_log_transform(train_graphs)
        apply_r0_log_transform(val_graphs)
        apply_r0_log_transform(test_graphs)

    # Normalize features using training set statistics
    epi_norm = normalize_epi_features(train_graphs, val_graphs, test_graphs)
    epi_norm['scale'] = epi_scale
    print(f"  Epi normalization: mean shape={epi_norm['mean'].shape}, std shape={epi_norm['std'].shape}")

    edge_norm = normalize_edge_features(train_graphs, val_graphs, test_graphs)
    print(f"  Edge normalization: mean shape={edge_norm['mean'].shape}, std shape={edge_norm['std'].shape}")

    # Normalize labels
    r0_norm = normalize_labels(train_graphs, val_graphs, test_graphs, 'R0')
    r0_norm['scale'] = label_scale
    print(f"  R0 normalization: mean={r0_norm['mean']:.4f}, std={r0_norm['std']:.4f}")

    sss_norm = normalize_labels(train_graphs, val_graphs, test_graphs, 'Source_Sink_Score')
    print(f"  SSS normalization: mean={sss_norm['mean']:.4f}, std={sss_norm['std']:.4f}")

    # Save normalization params
    torch.save(epi_norm, output_dir / 'epi_norm.pt')
    torch.save(edge_norm, output_dir / 'edge_norm.pt')
    torch.save(r0_norm, output_dir / 'r0_norm.pt')
    torch.save(sss_norm, output_dir / 'sss_norm.pt')

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

    history = {'train_loss': [], 'val_loss': [],
               'train_r0_loss': [], 'val_r0_loss': [],
               'train_sss_loss': [], 'val_sss_loss': []}

    for epoch in range(num_epochs):
        train_loss, train_r0, train_sss = train_epoch(
            model, train_loader, optimizer, criterion, device, r0_weight, sss_weight
        )
        val_results = evaluate(model, val_loader, criterion, device, r0_weight, sss_weight)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['train_r0_loss'].append(train_r0)
        history['val_r0_loss'].append(val_results['r0_loss'])
        history['train_sss_loss'].append(train_sss)
        history['val_sss_loss'].append(val_results['sss_loss'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} (R0:{train_r0:.4f}, SSS:{train_sss:.4f}) | "
                  f"Val: {val_results['loss']:.4f} (R0:{val_results['r0_loss']:.4f}, SSS:{val_results['sss_loss']:.4f})")

        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
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
    test_results = evaluate(model, test_loader, criterion, device, r0_weight, sss_weight)

    # Denormalize predictions and labels
    # R0
    r0_mean = r0_norm['mean'].item()
    r0_std = r0_norm['std'].item()
    test_r0_preds = test_results['r0_preds'] * r0_std + r0_mean
    test_r0_labels = test_results['r0_labels'] * r0_std + r0_mean

    if label_scale == 'log':
        test_r0_preds = np.exp(test_r0_preds)
        test_r0_labels = np.exp(test_r0_labels)

    # Source_Sink_Score
    sss_mean = sss_norm['mean'].item()
    sss_std = sss_norm['std'].item()
    test_sss_preds = test_results['sss_preds'] * sss_std + sss_mean
    test_sss_labels = test_results['sss_labels'] * sss_std + sss_mean

    # Save results
    torch.save(best_state, output_dir / 'best_model.pt')

    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)

    results_df = pd.DataFrame({
        'true_R0': test_r0_labels,
        'pred_R0': test_r0_preds,
        'true_Source_Sink_Score': test_sss_labels,
        'pred_Source_Sink_Score': test_sss_preds,
    })
    results_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    # Print test metrics
    from sklearn.metrics import r2_score, mean_squared_error

    r0_r2 = r2_score(test_r0_labels, test_r0_preds)
    r0_mse = mean_squared_error(test_r0_labels, test_r0_preds)
    sss_r2 = r2_score(test_sss_labels, test_sss_preds)
    sss_mse = mean_squared_error(test_sss_labels, test_sss_preds)

    print(f"\nTest Results:")
    print(f"  R0:                 R²={r0_r2:.4f}, MSE={r0_mse:.4f}")
    print(f"  Source_Sink_Score:  R²={sss_r2:.4f}, MSE={sss_mse:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
